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

import torch
import torchvision
from collections import defaultdict
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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix_refine
from knn_cuda import KNN
import pickle
from PIL import Image
import torch.nn.functional as F
# from gaussian_renderer import render
from nets.mlp_delta_body_pose import BodyPoseRefiner
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder
from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import gaussian_3d_coeff
from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
from mesh_utils import decimate_mesh, clean_mesh
from gs_renderer import Renderer, MiniCam

#TODO image feature
class CrossAttention_lbs(nn.Module):
    def __init__(self, feature_dim=24,mesh_dim = 3,rot_dim = 3, num_heads=3,):
        super(CrossAttention_lbs, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.query = nn.Linear(mesh_dim, feature_dim)
        self.key = nn.Linear(rot_dim, feature_dim)
        self.value = nn.Linear(rot_dim, feature_dim)
        self.out_layer = nn.Linear(feature_dim,feature_dim)
        init_val =1e-5
        self.out_layer.weight.data.uniform_(-init_val, init_val)
        self.out_layer.bias.data.zero_()

    def forward(self, query, key):
        value = key
        # 这里仅为示例，实际应用中可能需要根据query, key, value的实际情况调整维度
        Q = self.query(query)  # [batch_size, seq_len, feature_dim]
        K = self.key(key)      # [batch_size, seq_len, feature_dim]
        V = self.value(value)  # [batch_size, seq_len, feature_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        # 应用注意力分数到value上
        output = torch.matmul(attention, V)
        output = self.out_layer(output)

        return output

class CrossAttention_pos(nn.Module):
    def __init__(self, feature_dim=9,mesh_dim = 3,rot_dim = 3, num_heads=1):
        super(CrossAttention_pos, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.query = nn.Linear(rot_dim, feature_dim)
        self.key = nn.Linear(mesh_dim, feature_dim)
        self.value = nn.Linear(mesh_dim, feature_dim)
        self.out_layer = nn.Linear(feature_dim,feature_dim)
        init_val = 1e-5
        self.out_layer.weight.data.uniform_(-init_val, init_val)
        self.out_layer.bias.data.zero_()

    def forward(self, query, key):
        value = key
        # 这里仅为示例，实际应用中可能需要根据query, key, value的实际情况调整维度
        Q = self.query(query)  # [batch_size, seq_len, feature_dim]
        K = self.key(key)      # [batch_size, seq_len, feature_dim]
        V = self.value(value)  # [batch_size, seq_len, feature_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        # 应用注意力分数到value上
        output = torch.matmul(attention, V)
        output = self.out_layer(output)

        return output


# Highlight

if False:
    class Autoregression(nn.Module):
        def __init__(self,device='cuda'):
            super(Autoregression,self).__init__()
            self.device = device
            embed_dim = 64
            self.num_glob_params = 24*3
            self.num_cam_params = 0
            # self.num_cam_params = 3
            # self.num_shape_params= 10
            self.num_shape_params= 0
            self.joint_dim = 9  # 3
            self.parents_dict = self.immediate_parent_to_all_ancestors() # SMPL.parents.tolist()
            self.fc_embed = nn.Linear(self.num_shape_params  + self.num_glob_params + self.num_cam_params,
                                    embed_dim)
            self.activation = nn.ELU()
            self.num_joints = 23

            self.fc_pose = nn.ModuleList()
            init_val = 1e-5
            self.rodriguez = RodriguesModule()
            for joint in range(self.num_joints):
                num_parents = len(self.parents_dict[joint])
                input_dim = embed_dim + num_parents * (9 + 3 + 9) 
                fc = nn.Sequential(nn.Linear(input_dim, embed_dim // 2),
                                                self.activation,
                                                nn.Linear(embed_dim // 2, self.joint_dim))
                                                #   nn.Linear(embed_dim // 2, 3))
                                                #   nn.Linear(embed_dim // 2, 9)))
                fc[-1].weight.data.uniform_(-init_val, init_val)
                fc[-1].bias.data.zero_()
                self.fc_pose.append(fc)
                
        def immediate_parent_to_all_ancestors(self,immediate_parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]):
            """
            :param immediate_parents: list with len = num joints, contains index of each joint's parent.
                    - includes root joint, but its parent index is -1.
            :return: ancestors_dict: dict of lists, dict[joint] is ordered list of parent joints.
                    - DOES NOT INCLUDE ROOT JOINT! Joint 0 here is actually joint 1 in SMPL.
            """
            ancestors_dict = defaultdict(list)
            for i in range(1, len(immediate_parents)):  # Excluding root joint
                joint = i - 1
                immediate_parent = immediate_parents[i] - 1
                if immediate_parent >= 0:
                    ancestors_dict[joint] += [immediate_parent] + ancestors_dict[immediate_parent]
            return ancestors_dict

        # def forward(self,shape_params, glob, cam):
        def forward(self, glob):
            # Pose
            embed = self.activation(self.fc_embed(torch.cat([glob], dim=1)))  # (bsize, embed dim)
            batch_size = embed.shape[0]
            pose_F = torch.zeros(batch_size, self.num_joints, 3, 3, device=self.device)  # (bsize, 23, 3, 3)
            pose_U = torch.zeros(batch_size, self.num_joints, 3, 3, device=self.device)  # (bsize, 23, 3, 3)
            pose_S = torch.zeros(batch_size, self.num_joints, 3, device=self.device)  # (bsize, 23, 3)
            pose_V = torch.zeros(batch_size, self.num_joints, 3, 3, device=self.device)  # (bsize, 23, 3, 3)
            pose_U_proper = torch.zeros(batch_size, self.num_joints, 3, 3, device=self.device)  # (bsize, 23, 3, 3)
            pose_S_proper = torch.zeros(batch_size, self.num_joints, 3, device=self.device)  # (bsize, 23, 3)
            pose_rotmats_mode = torch.zeros(batch_size, self.num_joints, 3, 3, device=self.device)  # (bsize, 23, 3, 3)
            for joint in range(self.num_joints):
                parents = self.parents_dict[joint]
                fc_joint = self.fc_pose[joint]
                if len(parents) > 0:
                    parents_U_proper = pose_U_proper[:, parents, :, :].view(batch_size, -1)  # (bsize, num parents * 3 * 3)
                    parents_S_proper = pose_S_proper[:, parents, :].view(batch_size, -1)  # (bsize, num parents * 3)
                    parents_mode = pose_rotmats_mode[:, parents, :, :].view(batch_size, -1)  # (bsize, num parents * 3 * 3)

                    joint_F = fc_joint(torch.cat([embed, parents_U_proper, parents_S_proper, parents_mode], dim=1))
                else:
                    joint_F = fc_joint(embed)
                # joint_F = self.rodriguez(joint_F)
                joint_F = joint_F.view(-1,3,3)

                joint_U, joint_S, joint_V = torch.svd(joint_F.view(-1,3,3).cpu())  # (bsize, 3, 3), (bsize, 3), (bsize, 3, 3)

                with torch.no_grad():
                    det_joint_U, det_joint_V = torch.det(joint_U).to(self.device), torch.det(joint_V).to(self.device)  # (bsize,), (bsize,)
                joint_U, joint_S, joint_V = joint_U.to(self.device), joint_S.to(self.device), joint_V.to(self.device)

                # "Proper" SVD
                joint_U_proper = joint_U.clone()
                joint_S_proper = joint_S.clone()
                joint_V_proper = joint_V.clone()
                # Ensure that U_proper and V_proper are rotation matrices (orthogonal with det = 1).
                joint_U_proper[:, :, 2] *= det_joint_U.unsqueeze(-1)
                joint_S_proper[:, 2] *= det_joint_U * det_joint_V
                joint_V_proper[:, :, 2] *= det_joint_V.unsqueeze(-1)

                joint_rotmat_mode = torch.matmul(joint_U_proper, joint_V_proper.transpose(dim0=-1, dim1=-2))

                pose_F[:, joint, :, :] = joint_F
                pose_U[:, joint, :, :] = joint_U
                pose_S[:, joint, :] = joint_S
                pose_V[:, joint, :, :] = joint_V
                pose_U_proper[:, joint, :, :] = joint_U_proper
                pose_S_proper[:, joint, :] = joint_S_proper
                pose_rotmats_mode[:, joint, :, :] = joint_rotmat_mode

            return {
                "Rs": pose_F,
                "pose_U":pose_U,
                "pose_S":pose_S,
                "pose_V":pose_V,
            }
else:
    class Autoregression(nn.Module):
        def __init__(self,device='cuda'):
            super(Autoregression,self).__init__()
            self.device = device

            mlp_depth=2
            self.num_joints = 23
            embedding_size = 69
            # mlp_width = 128+9 * self.num_joints
            mlp_width = 128
            block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
            
            # TODO: change heavy network  罗德里格斯公式
            for _ in range(0, mlp_depth-1):
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

            block_mlps += [nn.Linear(mlp_width, 3 * self.num_joints)] 

            self.block_mlps = nn.Sequential(*block_mlps)

            # init the weights of the last layer as very small value
            # -- at the beginning, we hope the rotation matrix can be identity 
            init_val = 1e-5
            last_layer = self.block_mlps[-1]
            last_layer.weight.data.uniform_(-init_val, init_val)
            last_layer.bias.data.zero_()
            self.rodriguez = RodriguesModule()
            
            
        def forward(self,feature):
            # joint_F = self.block_mlps(feature).view(-1, 3, 3)  # (bsize, 3, 3)
            joint_F = self.block_mlps(feature[:, 3:]).view(-1, 3)  # (Joints, 3, 3)
            
            
            joint_F = self.rodriguez(joint_F)

            joint_U, joint_S, joint_V = torch.svd(joint_F)  # (Joints, 3, 3), (Joints, 3), (Joints, 3, 3)

            return {
                "Rs": joint_F,
                "pose_U":joint_U,
                "pose_S":joint_S,
                "pose_V":joint_V,
            }

class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)
 
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
            
            self.cross_attention_pos = CrossAttention_pos()
            self.cross_attention_pos.to(self.device)

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
            # Highlight
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                {'params': self.auto_regression.parameters(),'lr':training_args.pose_refine_lr*5,"name":"auto_regression"},
                # {'params': self.auto_regression.parameters(),'lr':training_args.pose_refine_lr,"name":"auto_regression"},
                # {'params': self.auto_regression.parameters(),'lr':0.00001,"name":"auto_regression"},
                {'params': self.weight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "weight_offset_decoder"},
                {'params': self.cross_attention_lbs.parameters(), 'lr': 0.0001, "name": "cross_attention_lbs"},
                {'params': self.cross_attention_pos.parameters(), 'lr': 0.001, "name": "cross_attention_pos"},
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

    @torch.no_grad()
    def save_model(self, mode='geo',render=None,viewpoint_camera=None,cur_out=None,background=None, pipe=None, texture_size=1024):
        
        self.outdir = '/root/workspace/Caixiang/GauHuman-main/result'
        self.save_path = 'me'
        self.density_thresh = 1
        self.mesh_format = 'obj'
        self.radius = 2
        self.near=0.001
        self.far=1000
        self.scale = 1
        self.perspective = np.array([[ 2.189235,  0.      ,  0.      ,  0.      ],
                                [ 0.      , -2.189235,  0.      ,  0.      ],
                                [ 0.      ,  0.      , -1.0002  , -0.020002],
                                [ 0.      ,  0.      , -1.      ,  0.      ]])
        os.makedirs(self.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.outdir, self.save_path + '_mesh.ply')
            mesh = self.extract_mesh(path, self.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.outdir, self.save_path + '_mesh.' + self.mesh_format)
            mesh = self.extract_mesh(path, self.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            import nvdiffrast.torch as dr            
            glctx = dr.RasterizeCudaContext()
            
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]
            vers = [0 for i in range(len(hors))]
            render_resolution = 512
            
            temp_video =  copy.deepcopy(viewpoint_camera)
            # viewpoint_camera = copy.deepcopy(temp_video)

            for ver, hor in zip(vers, hors):
                # render image
                # world_view_transform = torch.tensor(getWorld2View2(viewpoint_camera.R, viewpoint_camera.T, viewpoint_camera.trans, self.scale)).transpose(0, 1).cuda()
                # projection_matrix = getProjectionMatrix_refine(torch.Tensor(viewpoint_camera.K).cuda(), render_resolution, render_resolution, self.near, self.far).transpose(0, 1)
                # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                # camera_center = world_view_transform.inverse()[3, :3]
                pose = orbit_camera(ver, hor, self.radius,is_degree=True, target=None, opengl=True)
                # # cur_cam = MiniCam(pose,render_resolution,render_resolution,\
                # #     viewpoint_camera.FoVy,viewpoint_camera.FoVx,self.near,self.far)
                
                # viewpoint_camera.world_view_transform = world_view_transform
                # viewpoint_camera.full_proj_transform = full_proj_transform
                # viewpoint_camera.camera_center = camera_center
                # cur_out = render(viewpoint_camera, self, pipe, background)
                # rgbs = cur_out["render"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                # print('over')
                from math import cos,sin
                
                # Sita = 1
                # Rz = np.array([[cos(Sita), -sin(Sita), 0],
                #                [sin(Sita), cos(Sita), 0],
                #                [0, 0, 1]])
                # Rx = np.array([[cos(Sita), -sin(Sita), 0],
                #                [sin(Sita), cos(Sita), 0],
                #                [0, 0, 1]])

                viewpoint_camera = copy.deepcopy(temp_video)
                # Rx = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]])
                # Ry = np.array([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])
                # Rz = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
                pose = np.matmul(np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]), self.get_camera_extrinsics_zju_mocap_refine(viewpoint_camera.uid))
                R = pose[:3,:3]
                T = pose[:3, 3]
                world_view_transform = torch.tensor(getWorld2View2(R, T, viewpoint_camera.trans, self.scale)).transpose(0, 1).cuda()
                # world_view_transform = torch.tensor(getWorld2View2(viewpoint_camera.R@Rz, viewpoint_camera.T, viewpoint_camera.trans, self.scale)).transpose(0, 1).cuda()
                projection_matrix = getProjectionMatrix_refine(torch.Tensor(viewpoint_camera.K).cuda(), render_resolution, render_resolution, self.near, self.far).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                viewpoint_camera.world_view_transform = world_view_transform
                viewpoint_camera.full_proj_transform = full_proj_transform
                viewpoint_camera.camera_center = camera_center
                cur_out = render(viewpoint_camera, self, pipe, background)
                rgbs = cur_out["render"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                print('over')

                # enhance texture quality with zero123 [not working well]
                # if self.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)

                proj = torch.from_numpy(self.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.outdir, self.save_path + '_model.ply')
            self.save_ply(path)

        print(f"[INFO] save model to {path}.")

    @torch.no_grad()
    def extract_fields(self, resolution=512, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        # kiui.lo(occ, verbose=1)

        return occ
    
    def extract_mesh(self, path, density_thresh=1, resolution=512, decimate_target=1e5):
    # def extract_mesh(self, path, density_thresh=1, resolution=512, decimate_target=0):

        #Local Density Query
        density_thresh=1
        resolution=512
        decimate_target=1e5
        os.makedirs(os.path.dirname(path), exist_ok=True)
        #获取每个块内信息
        occ = self.extract_fields(resolution,  num_blocks=16, relax_ratio=1.5).detach().cpu().numpy()
        #使用Marching Cubes算法在每个块内查询8^3的密集网格，最终得到128^3的密集网格
        import mcubes
        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        #将3D空间划分为16^3个块，并归一化
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.scale + self.center.detach().cpu().numpy()
        #在网格中的每个查询位置x处，对每个保留的3D高斯的加权不透明度进行求和，然后通过清理和减少网格的步骤来处理和优化网格数据。
        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

        mesh = Mesh(v=v, f=f, device='cuda')
        mesh.write_ply(path)
        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        print("========")

        # density_thresh=1
        # resolution=1024
        # decimate_target=1e5
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # #获取每个块内信息
        # occ = self.extract_fields(resolution,  num_blocks=16, relax_ratio=1.5).detach().cpu().numpy()
        # #使用Marching Cubes算法在每个块内查询8^3的密集网格，最终得到128^3的密集网格
        # import mcubes
        # vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        # #将3D空间划分为16^3个块，并归一化
        # vertices = vertices / (resolution - 1.0) * 2 - 1

        # # transform back to the original space
        # vertices = vertices / self.scale + self.center.detach().cpu().numpy()
        # #在网格中的每个查询位置x处，对每个保留的3D高斯的加权不透明度进行求和，然后通过清理和减少网格的步骤来处理和优化网格数据。
        # vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        # if decimate_target > 0 and triangles.shape[0] > decimate_target:
        #     vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        # v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        # f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

        # print(
        #     f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        # )

        # mesh = Mesh(v=v, f=f, device='cuda')
        
        # mesh.write_ply(path)



        return mesh

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

    def save_ply_with_mesh(self, path, mesh):
        mkdir_p(os.path.dirname(path))

        xyz = mesh.detach().cpu().numpy()
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

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
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
        torchvision.utils.save_image(data,path)

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

    def kl_densify_and_clone(self, grads, grad_threshold, scene_extent, kl_threshold=0.4):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
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
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl clone]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask])
        new_rotation = self._rotation[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def kl_densify_and_split(self, grads, grad_threshold, scene_extent, kl_threshold=0.4, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

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
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl split]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

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

    def kl_merge(self, grads, grad_threshold, scene_extent, kl_threshold=0.1):
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

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, kl_threshold=0.4, t_vertices=None, iter=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)
        self.kl_densify_and_clone(grads, max_grad, extent, kl_threshold)
        self.kl_densify_and_split(grads, max_grad, extent, kl_threshold)
        self.kl_merge(grads, max_grad, extent, 0.1)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # use smpl prior to prune points 
        distance, _ = self.knn(t_vertices[None], self._xyz[None].detach())
        distance = distance.view(distance.shape[0], -1)
        threshold = 0.05
        pts_mask = (distance > threshold).squeeze()

        prune_mask = prune_mask | pts_mask

        print('total points num: ', self._xyz.shape[0], 'prune num: ', prune_mask.sum().item())
        
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

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

        kl_div_0 = torch.empty(product.shape[0]).to(cov_0.device)

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
            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float() #torch.Size([6890, 3, 207])   23*9
            pose_ = big_pose_params['poses']    # torch.Size([1, 72])
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

            # From mean shape to normal shape
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
            
            # torch.Size([1, 207])
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



