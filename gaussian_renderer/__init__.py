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

import torch
from torch import nn
import torch.nn.functional as F
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from smplx.lbs import batch_rodrigues

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_smpl_rot=False, transforms=None, translation=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    pose_out = None
    if not pc.motion_offset_flag:
        _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], 
            viewpoint_camera.smpl_param,
            viewpoint_camera.big_pose_smpl1_param,
            viewpoint_camera.big_pose_world_vertex[None])
        
    else:
        if transforms is None:
            # highlight_train
            # Ours
            pose_out = pc.auto_regression(viewpoint_camera.smpl_param['poses'],)       # torch.Size([1, 72])
            pose_out['target_R'] = viewpoint_camera.smpl_param['pose_rotmats']
            
            pose_ = viewpoint_camera.smpl_param['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            rot_mats_no_root = rot_mats[:, 1:]
            correct_Rs = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), pose_out['Rs'].reshape(-1, 3, 3)).reshape(-1, 23, 3, 3)

            lbs_weights = pc.cross_attention_lbs(means3D[None],correct_Rs)
            correct_Rs =pose_out['Rs'].reshape(1,23,3,3)

            # Baseline
            # pose_out = pc.pose_decoder(viewpoint_camera.smpl_param['poses'][:, 3:])  
            # correct_Rs = pose_out['Rs']

            # lbs_weights = pc.weight_offset_decoder(means3D[None].detach()) # torch.Size([1, 6890, 3])
            # lbs_weights = lbs_weights.permute(0,2,1)                       # torch.Size([1, 6890, 24])

            # transform points
            # torch.Size([1, 6890, 3])

            _, means3D, bweights, transforms, translation = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,viewpoint_camera.big_pose_smpl_param,viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
        else:
            bweights = None
            correct_Rs = None
            lbs_weights = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation

    means3D = means3D.squeeze()  # torch.Size([6890, 3])
    means2D = screenspace_points  # torch.Size([6890, 3])
    opacity = pc.get_opacity       # torch.Size([6890, 1])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier, transforms.squeeze())
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features     # torch.Size([6890, 16, 3])
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,                        # torch.Size([6890, 3])
        means2D = means2D,                        # torch.Size([6890, 3])  3D空间中高斯模型的中心点经过投影变换后的2D坐标。
        shs = shs,                                # torch.Size([6890, 16, 3])
        colors_precomp = colors_precomp,          # None
        opacities = opacity,                      # torch.Size([6890, 1])
        scales = scales,                          # None
        rotations = rotations,                    # None
        cov3D_precomp = cov3D_precomp)            #torch.Size([6890, 6])

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": depth,
            "render_alpha": alpha,
            "viewspace_points": screenspace_points, # means2D 3D空间中高斯模型的中心点经过投影变换后的2D坐标。
            "visibility_filter" : radii > 0,  
            "radii": radii,           # 屏幕空间中高斯模型的“半径”。在3D渲染中，当3D对象被投影到2D屏幕上时，它们的尺寸（如半径）可能会根据视角和深度而变化。radii 反映的是这些投影后的尺寸，它可以用于确定每个高斯模型在屏幕上占据的空间大小
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,
            "pose_out":pose_out,
            # "lbs_weights":lbs_weights,
            "lbs_weights":bweights,
            "means3D":means3D}