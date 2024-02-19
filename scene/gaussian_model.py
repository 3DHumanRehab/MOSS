#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import open3d as o3d
import cv2
import torch
import itertools
from collections import defaultdict
from sklearn.neighbors import KDTree
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from knn_cuda import KNN
import pickle
import torch.nn.functional as F
# from gaussian_renderer import render
from nets.mlp_delta_body_pose import BodyPoseRefiner,Autoregression
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder,CrossAttention_lbs
from grid_put import mipmap_linear_grid_put_2d

from pytorch3d.transforms import matrix_to_quaternion
 
class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, transform=None):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if transform is not None:
                actual_covariance = transform @ actual_covariance
                actual_covariance = actual_covariance @ transform.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, smpl_type : str, motion_offset_flag : bool, actor_gender: str):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.origin_scaling = self._scaling
        self.origin_rotation = self._rotation
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.device=torch.device('cuda', torch.cuda.current_device())
        # load SMPL model
        if smpl_type == 'smpl':
            neutral_smpl_path = os.path.join('assets', f'SMPL_{actor_gender.upper()}.pkl')
            self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=self.device)
        elif smpl_type == 'smplx':
            # neutral_smpl_path = os.path.join('assets/models/smplx', f'SMPLX_{actor_gender.upper()}.npz')
            neutral_smpl_path = "/HOME/HOME/Caixiang/SMPLX_NEUTRAL_2020.npz"
            params_init = dict(np.load(neutral_smpl_path, allow_pickle=True))
            self.SMPL_NEUTRAL = SMPL_to_tensor(params_init, device=self.device)

        # load knn module
        self.knn = KNN(k=1, transpose_mode=True)
        self.knn_near_2 = KNN(k=2, transpose_mode=True)
        self.knn_near_5 = KNN(k=5, transpose_mode=True)

        self.motion_offset_flag = motion_offset_flag
        if self.motion_offset_flag:
            # load pose correction module
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
            self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2)
            self.pose_decoder.to(self.device)

            # load lbs weight module
            self.weight_offset_decoder = LBSOffsetDecoder(total_bones=total_bones)
            self.weight_offset_decoder.to(self.device)
            
            # auto regression
            self.auto_regression = Autoregression(device=self.device)
            self.auto_regression.to(self.device)
            
            self.cross_attention_lbs = CrossAttention_lbs()
            self.cross_attention_lbs.to(self.device)
            
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.pose_decoder,
            self.weight_offset_decoder,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.pose_decoder,
        self.weight_offset_decoder) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    
    @property
    def get_origin_scaling(self):
        return self.scaling_activation(self.origin_scaling)
    
    @property
    def get_origin_rotation(self):
        return self.rotation_activation(self.origin_rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, transform=None):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, transform)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if not self.motion_offset_flag:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        else:
            # Highlight_lr  # TODO:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                # {'params': self.auto_regression.parameters(),'lr':training_args.pose_refine_lr*5,"name":"auto_regression"},
                {'params': self.auto_regression.parameters(),'lr':training_args.auto_regression,"name":"auto_regression"},
                # {'params': self.auto_regression.parameters(),'lr':training_args.pose_refine_lr,"name":"auto_regression"},
                # {'params': self.auto_regression.parameters(),'lr':0.00001,"name":"auto_regression"},
                {'params': self.weight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "weight_offset_decoder"},
                {'params': self.cross_attention_lbs.parameters(), 'lr': training_args.cross_attention_lbs, "name": "cross_attention_lbs"},
            ] 

        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def get_camera_extrinsics_zju_mocap_refine(self,view_index, camera_view_num=36):
        def norm_np_arr(arr):
            return arr / np.linalg.norm(arr)

        def lookat(eye, at, up):
            zaxis = norm_np_arr(at - eye)
            xaxis = norm_np_arr(np.cross(zaxis, up))
            yaxis = np.cross(xaxis, zaxis)
            _viewMatrix = np.array([
                [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
                [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
                [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
                [0       , 0       , 0       , 1     ]
            ])
            return _viewMatrix
        
        def fix_eye(phi, theta):
            camera_distance = 2
            return np.array([
                camera_distance * np.sin(theta) * np.cos(phi),
                camera_distance * np.sin(theta) * np.sin(phi),
                camera_distance * np.cos(theta)
            ])

        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
        return extrinsics

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path,point_cloud=None):
        mkdir_p(os.path.dirname(path))
        
        if point_cloud=None:
            xyz = self._xyz.detach().cpu().numpy()
        else:
            xyz = point_cloud.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_tensor(self,path,data):
        mkdir_p(os.path.dirname(path))
        # torchvision.utils.save_image(data,path)
        data = data.squeeze(0).cpu().detach().numpy()
        data=data*255/data.max()
        data=np.uint8(data)
        data = cv2.applyColorMap(data, 2)
        cv2.imwrite(path,data)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # if len(group["params"]) == 1:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def kl_densify_and_clone(self, grads,rot_joint,scl_joint, grad_threshold, scene_extent, kl_threshold=0.4):
        if self._xyz.shape[0]>45695:
            print("==================================")
            print("self._xyz.shape:",self._xyz.shape)
            return
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # for each gaussian point, find its nearest 2 points and return the distance
        self.kl_selected_pts_mask = self.cal_kl(kl_threshold)
        
        # rotation_0_q = self._rotation.detach()
        # scaling_diag_0 = self.get_scaling.detach()

        # rotation_1_q = self.origin_rotation.detach()
        # scaling_diag_1 = self.get_origin_scaling.detach()

        # kl_div = self.kl_div(self._xyz, rotation_0_q, scaling_diag_0, self._xyz, rotation_1_q, scaling_diag_1,torch.stack([rot_joint,rot_joint],dim=1).detach())

        # kl_selected_pts_mask_2 =  kl_div > 0.4
        
        # selected_idx_mask = torch.where(torch.norm(grads, dim=-1) >= 0.0003, True, False)
        # selected_idx_mask = torch.logical_and(selected_idx_mask,
        #                                       torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # _, point_ids = self.knn_near_5(self._xyz[None].detach(),self._xyz[selected_idx_mask][None].detach())
        # point_ids = torch.unique(point_ids)
        # knn_selected_pts_mask = torch.zeros(self.kl_selected_pts_mask.shape, dtype=torch.bool,device='cuda')

        # knn_selected_pts_mask[point_ids] = True

        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask & knn_selected_pts_mask | kl_selected_pts_mask_2
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask & knn_selected_pts_mask & kl_selected_pts_mask_2
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask 
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask & knn_selected_pts_mask
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask & knn_selected_pts_mask & kl_selected_pts_mask_2
        
        
        # FIXME: normal_angle_mask
        normals = self.compute_normals_co3d(self._xyz)
        # mean_angle = compute_mean_angle(points, normals)
        # normal_angle_mask = mean_angle > angle_threshold
        angle_threshold = 0.1
        distance_threshold = 0.05
        normal_angle_mask = self.compute_angle_change_rate(self._xyz,normals,angle_threshold,distance_threshold)
        print("normal_angle_mask",normal_angle_mask.sum())
        
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask 
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask | normal_angle_mask
        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask & normal_angle_mask  # ME
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask 
        
        print("[kl clone]: ", (selected_pts_mask).sum().item())

        # FIXME: density get_scaling
        stds = self.get_scaling[selected_pts_mask]
        # stds = scl_joint[selected_pts_mask]*self.get_scaling[selected_pts_mask]
 
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])  # (*,3,3)

        # FIXME: density rots
        rots = rots
        # rots = rots @ rot_joint[selected_pts_mask].reshape(-1,3,3)
        # rots = rot_joint[selected_pts_mask].reshape(-1,3,3) @ rots  # ME
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        # FIXME: GS scale
        # new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask])
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] * scl_joint[selected_pts_mask])
        
        # FIXME: GS rot
        # new_rotation = self._rotation[selected_pts_mask]
        new_rotation = matrix_to_quaternion(rot_joint[selected_pts_mask].reshape(-1,3,3)) * self._rotation[selected_pts_mask] 

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
 
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def kl_densify_and_split(self, grads,origin_rot_joint,origin_scl_joint, grad_threshold, scene_extent, kl_threshold=0.4, N=2):
        #  grad_threshold 0.0002
        if self._xyz.shape[0]>45695:
            print("==================================")
            print("self._xyz.shape:",self._xyz.shape)
            return
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condi tion
        # FIXME: Fisher
        rot_joint = torch.zeros((n_init_points,3,3), device="cuda")
        rot_joint[:grads.shape[0]] = origin_rot_joint
        scl_joint = torch.zeros((n_init_points,3), device="cuda")
        scl_joint[:grads.shape[0]] = origin_scl_joint
        
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # for each gaussian point, find its nearest 2 points and return the distance
        self.kl_selected_pts_mask = self.cal_kl(kl_threshold) 
        
        # FIXME: normal_angle_mask
        # normals = self.compute_normals_co3d(self._xyz)
        # # mean_angle = compute_mean_angle(points, normals)
        # # normal_angle_mask = mean_angle > angle_threshold
        # angle_threshold = 0.1
        # distance_threshold = 0.05
        # normal_angle_mask = self.compute_angle_change_rate(self._xyz,normals,angle_threshold,distance_threshold)
        # print("normal_angle_mask",normal_angle_mask.sum())
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask & normal_angle_mask  # ME
        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask | normal_angle_mask  # ME
        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask 

        # selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl split]: ", (selected_pts_mask).sum().item())

        # FIXME: density get_scaling
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # stds = (scl_joint[selected_pts_mask]*self.get_scaling[selected_pts_mask]).repeat(N,1)
        
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        
        # FIXME: density rots
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # rots = build_rotation(self._rotation[selected_pts_mask])
        # rots = (rot_joint[selected_pts_mask].reshape(-1,3,3) @ rots).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # FIXME: GS scale
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        #new_scaling = self.scaling_inverse_activation((scl_joint[selected_pts_mask] * self.get_scaling[selected_pts_mask] / (0.8*N)).repeat(N,1) )

        # FIXME: GS rot
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        #new_rotation = (matrix_to_quaternion(rot_joint[selected_pts_mask].reshape(-1,3,3)) * self._rotation[selected_pts_mask]).repeat(N,1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def kl_merge(self, grads,grad_threshold, scene_extent, kl_threshold=0.1):
        if self._xyz.shape[0]>45695:
            print("==================================")
            print("self._xyz.shape:",self._xyz.shape)
            return
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # for each gaussian point, find its nearest 2 points and return the distance
        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())     
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div < kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl merge]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

        if selected_pts_mask.sum() >= 1:

            selected_point_ids = point_ids[0][selected_pts_mask]
            new_xyz = self.get_xyz[selected_point_ids].mean(1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_point_ids][:,0] / 0.8)
            new_rotation = self._rotation[selected_point_ids][:,0]
            new_features_dc = self._features_dc[selected_point_ids].mean(1)
            new_features_rest = self._features_rest[selected_point_ids].mean(1)
            new_opacity = self._opacity[selected_point_ids].mean(1)

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

            selected_pts_mask[selected_point_ids[:,1]] = True
            # prune_filter = torch.cat((selected_pts_mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(new_xyz.shape[0], device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad,joint_F,lbs_weights, min_opacity, extent, max_screen_size, kl_threshold=0.4, t_vertices=None, iter=None):
        grads = self.xyz_gradient_accum / self.denom
        
        # FIXME: Fisher
        
        # rot_joint = None
        # scl_joint = None
        
        joint_F =  joint_F / self.denom[0]
        lbs_weights = lbs_weights / self.denom[0]
        
        with torch.no_grad():
            joint_U, joint_S, joint_V = torch.svd(joint_F)
            det_joint_U, det_joint_V = torch.det(joint_U).to('cuda'), torch.det(joint_V).to('cuda')  # (bsize,), (bsize,)

            # Ensure that U_proper and V_proper are rotation matrices (orthogonal with det = 1).
            joint_U[:, :, 2] *= det_joint_U.unsqueeze(-1)
            joint_V[:, :, 2] *= det_joint_V.unsqueeze(-1)
        
        rot_joint = torch.matmul(joint_U, joint_V.transpose(dim0=-1, dim1=-2))
        # rot_joint = rot_joint.reshape(23,9)   # TODO:使用target rot  debug 查看根节点的旋转矩阵
        # rot_joint = (lbs_weights[0,:,1:]@rot_joint).reshape(-1,3,3)
        
        rot_joint = torch.cat([torch.ones(1,3,3).cuda(),rot_joint],dim=0).reshape(24,9)   # TODO:使用target rot  debug 查看根节点的旋转矩阵
        rot_joint = (lbs_weights[0]@rot_joint).reshape(-1,3,3)

        scl_joint = torch.cat([torch.ones(1,3).cuda(),joint_S],dim=0) 
        scl_joint = (lbs_weights[0]@scl_joint)
        grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)
        self.kl_densify_and_clone(grads,rot_joint,scl_joint, max_grad, extent, kl_threshold)
        self.kl_densify_and_split(grads,rot_joint,scl_joint, max_grad, extent, kl_threshold)
        self.kl_merge(grads, max_grad, extent, 0.1)

        # TODO: min_opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # TODO: 剔除离群点
        # use smpl prior to prune points 
        distance, _ = self.knn(t_vertices[None], self._xyz[None].detach())
        distance = distance.view(distance.shape[0], -1)
        threshold = 0.05
        pts_mask = (distance > threshold).squeeze()
        # from sklearn.ensemble import IsolationForest
        # clf = IsolationForest(random_state=0).fit(self._xyz.detach().cpu().reshape(-1, 3))
        # outliers = self._xyz[clf.predict(self._xyz.detach().cpu().reshape(-1, 3)) == -1]
        # print("Outliers:", outliers)
        
        prune_mask = prune_mask | pts_mask
        print('total points num: ', self._xyz.shape[0], 'prune num: ', prune_mask.sum().item())        
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # import numpy as np
        # points =  t_vertices.detach().cpu().numpy()

        # # 创建一个3D坐标系
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # 画点
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        # # 在每个点周围画圈
        # for point in points:
        #     # 创建一个球面
        #     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        #     x = np.cos(u) * np.sin(v)
        #     y = np.sin(u) * np.sin(v)
        #     z = np.cos(v)
            
        #     # 缩放和平移球面来创建圆圈
        #     radius = 0.05
        #     x = radius * x + point[0]
        #     y = radius * y + point[1]
        #     z = radius * z + point[2]
            
        #     # 画圆圈
        #     ax.plot_wireframe(x, y, z, color="r")

        # # 设置坐标轴标签
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_zlabel('Z Axis')

        # plt.savefig("3d_plot.png", dpi=300)  # 保存为PNG文件，分辨率300dpi

        # # 关闭图形，释放资源
        # plt.close(fig)


    def compute_normals_co3d(self,points, radius=0.1, max_nn=5):
        point_3d = points.detach().cpu().numpy()
        # 将numpy数组转换为Open3D的点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_3d)

        # 计算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

        # 将法向量从Open3D的点云对象转换回numpy数组
        normals = np.asarray(pcd.normals)
        normals = torch.tensor(normals, dtype=torch.float32 ,device ="cuda")
        return normals

    def compute_normals(self,points, k=5, use_centroid=True, external_ref_point=None):
            # 确保点云数据存在
            # print("points",points)
            points = points.detach().cpu().numpy()
            normals = np.zeros_like(points)

            # 计算点云的中心或使用外部参考点
            if use_centroid:
                centroid = np.mean(points, axis=0)
            elif external_ref_point is not None:
                centroid = external_ref_point
            else:
                raise ValueError("需要提供点云的中心或外部参考点")

            # 使用k-NN找到邻域点
            tree = KDTree(points)
            for i, point in enumerate(points):
                _, idx = tree.query(point, k=k)
                if idx.any() >= len(points):
                    print("point",point)
                    print("idx",i)
                neighbors = points[idx]
                # 平面拟合
                mean = np.mean(neighbors, axis=0)
                cov = np.cov((neighbors - mean).T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                normal = eigenvectors[:, np.argmin(eigenvalues)]

                # 判断并调整法向量方向
                if np.dot(normal, centroid - point) > 0:
                    normal = -normal  # 反转法向量

                normals[i] = normal

            # 转换回torch.Tensor并存储
            normals = torch.tensor(normals, dtype=torch.float32)
            return normals


    def compute_angle_change_rate(self,positions, normals,threshold,distance_threshold = 0.1):
        # 将输入转换为 numpy
        positions_np = positions.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()

        # 使用 KDTree 找到每个点的最近邻
        tree = KDTree(positions_np)
        _, indices = tree.query(positions_np, k=5)

        # 初始化一个空的变化率列表
        mask = []
        # 对每个点和其邻居进行处理
        for idx in indices:
            # 获取邻域内的法向量和位置
            neighborhood_normals = normals_np[idx]
            neighborhood_positions = positions_np[idx]

            # 计算邻域内所有点的法向量之间的角度和距离
            angles = []
            distances = []
            for pair in itertools.combinations(range(len(neighborhood_normals)), 2):
                i, j = pair
                v1, v2 = neighborhood_normals[i], neighborhood_normals[j]
                p1, p2 = neighborhood_positions[i], neighborhood_positions[j]
                distance = np.linalg.norm(p1 - p2)
                if distance < distance_threshold:  # 当距离小于阈值时，跳过此次循环
                    continue
                dot_product = np.dot(v1, v2)
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                cos_angle = dot_product / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
                distances.append(distance)

            # 计算角度变化率
            angles = np.array(angles)
            distances = np.array(distances)
            sorted_indices = np.argsort(distances)
            sorted_angles = angles[sorted_indices]
            angle_change_rate = np.diff(sorted_angles) / np.diff(distances[sorted_indices])
            # print("angle_change_rate",angle_change_rate)
            # 将变化率添加到列表中
            mean_change_rate = np.mean(angle_change_rate)

            # 若均值超过阈值，则保留该点的掩码
            mask.append(mean_change_rate > threshold)

            #转为tensor
        mask = torch.tensor(mask, dtype=torch.bool,device="cuda")
        return mask

    def cal_kl(self, kl_threshold):
        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())     
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()   # torch.Size([6890, 4])
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)

        return kl_div > kl_threshold

    def kl_div(self, mu_0, rotation_0_q, scaling_0_diag, mu_1, rotation_1_q, scaling_1_diag):

        # claculate cov_0
        rotation_0 = build_rotation(rotation_0_q)
        scaling_0 = build_scaling(scaling_0_diag)
        
        L_0 = rotation_0 @ scaling_0
        cov_0 = L_0 @ L_0.transpose(1, 2)

        # claculate inverse of cov_1
        rotation_1 = build_rotation(rotation_1_q)
        scaling_1_inv = build_scaling(1/scaling_1_diag)

        L_1_inv = rotation_1 @ scaling_1_inv
        cov_1_inv = L_1_inv @ L_1_inv.transpose(1, 2)

        # difference of mu_1 and mu_0
        mu_diff = mu_1 - mu_0

        # calculate kl divergence
        # kl_div_0 = torch.diagonal(torch.trace(cov_1_inv @ cov_0, dim1=-2, dim2=-1), dim1=-1, dim2=-2)
        # kl_div_0 = torch.vmap(torch.trace)(cov_1_inv @ cov_0)

        product = cov_1_inv @ cov_0
        try:
            kl_div_0 = torch.empty(product.shape[0]).to('cuda')
        except:
            import pdb
            pdb.set_trace()
            print("===============product===============")
            print(product.shape)

        for i in range(product.shape[0]):
            kl_div_0[i] = torch.trace(product[i])

        kl_div_1 = mu_diff[:,None].matmul(cov_1_inv).matmul(mu_diff[..., None]).squeeze()
        kl_div_2 = torch.log(torch.prod((scaling_1_diag/scaling_0_diag)**2, dim=1))
        kl_div = 0.5 * (kl_div_0 + kl_div_1 + kl_div_2 - 3)
        return kl_div

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def coarse_deform_c2source(self, query_pts, params, t_params, t_vertices, lbs_weights=None, correct_Rs=None, return_transl=False):
        bs = query_pts.shape[0]
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]
        vertices_num = t_vertices.shape[1]
        # Find nearest smpl vertex
        smpl_pts = t_vertices

        _, vert_ids = self.knn(smpl_pts.float(), query_pts.float())
        if lbs_weights is None:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)#.cuda() # [bs, points_num, joints_num]
        else:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
            bweights = torch.log(bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)

        ### From Big To T Pose
        big_pose_params = t_params
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        query_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        query_pts = torch.matmul(R_inv, query_pts[..., None]).squeeze(-1)

        # transforms from Big To T Pose
        transforms = R_inv

        # translation from Big To T Pose
        translation = None
        if return_transl: 
            translation = -A[..., :3, 3]
            translation = torch.matmul(R_inv, translation[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float() # torch.Size([6890, 3, 207])   23*9
            pose_ = big_pose_params['poses']                        # torch.Size([1, 72])          24*3
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])  # 24,3,3
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            # torch.Size([1, 6890, 3])   
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts - pose_offsets

            if return_transl: 
                translation -= pose_offsets

            # From mean shape to normal shape  torch.Size([6890, 3, 10])
            shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :params['shapes'].shape[-1]]#.cuda()  
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'].cuda(), (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + shape_offset

            if return_transl: 
                translation += shape_offset

            posedirs = self.SMPL_NEUTRAL['posedirs']#.cuda().float()
            # torch.Size([6890, 3, 207])
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
 
            if correct_Rs is not None:
                rot_mats_no_root = rot_mats[:, 1:]
                rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, joints_num-1, 3, 3)
                rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)
    
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])  #.cuda()
    
            # torch.Size([1, 207])    #  posedirs torch.Size([1, 207, 20670])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + pose_offsets

            if return_transl:
                translation += pose_offsets

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, rot_mats=rot_mats)

        self.s_A = A
        A = torch.matmul(bweights, self.s_A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]
        
        transforms = torch.matmul(A[..., :3, :3], transforms)

        if return_transl: 
            translation = torch.matmul(A[..., :3, :3], translation[..., None]).squeeze(-1) + A[..., :3, 3]

        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.matmul(smpl_src_pts, R_inv) + Th
 
        transforms = torch.matmul(R, transforms)
  
        if return_transl: 
            translation = torch.matmul(translation, R_inv).squeeze(-1) + Th

        return smpl_src_pts, world_src_pts, bweights, transforms, translation

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

# @profile
def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']
    # v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()
    v_shaped = v_template[None] + torch.sum(shapedirs[None][...,:betas.shape[-1]] * betas[:,None], axis=-1).float()

    if rot_mats is None:
        # add pose blend shapes
        poses = params['poses'].reshape(-1, 3)
        # bs x 24 x 3 x 3
        rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] 
    Th = params['Th'] 

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
