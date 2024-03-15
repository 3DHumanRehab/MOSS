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
from scene import Scene
import os
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    ply_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ply")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Load data (deserialize)
    with open(model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
        smpl_rot = pickle.load(handle)

    rgbs = []
    rgbs_gt = []
    elapsed_time = 0
    # import copy
    # temp_view = copy.deepcopy(views[0])

    for index, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask
        transforms, translation = smpl_rot[name][view.pose_id]['transforms'], smpl_rot[name][view.pose_id]['translation']
 
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        render_output = render(view, gaussians, pipeline, background, transforms=transforms, translation=translation)
        rendering = render_output["render"]

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rgbs.append(rendering)
        rgbs_gt.append(gt)
        
        # FIXME: remove these codes
        # if not index%10:
            # gaussians.save_ply_with_mesh(os.path.join(ply_path, '{0:05d}'.format(index) + ".ply"),render_output["means3D"])
            # gaussians.save_tensor(os.path.join(depth_path, '{0:05d}'.format(index) + ".png"),render_output["render_depth"])
            # gaussians.save_tensor(os.path.join(alpha_path, '{0:05d}'.format(index) + ".png"),render_output["render_alpha"])


    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))

        # metrics
        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)
    ssims /= len(views)
    lpipss /= len(views)

    # evalution metrics
    print("==========="*8)
    print("\n[ITER {}] Evaluating {} #{}: PSNR  SSIM   LPIPS  ".format(iteration, name, len(views)))
    print(f"{psnrs.item(), ssims.item(), lpipss.item()*1000}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    args.actor_gender='neutral'
    args.compute_cov3D_python=True
    args.convert_SHs_python=False
    args.data_device='cuda'
    args.debug=False
    args.eval=True
 
    name_list = ['377','386','387','392','393','394']
    log_name = 'best_4'
    iteration_list = [2700,2700,3000,3000,2500,2700]

    
    for iteration,data_name in zip(iteration_list,name_list):
        args.data_name = data_name
        args.exp_name=f'/home/tom/fsas/workspace/Caixiang/MOSS/output/zju_mocap_refine/my_{args.data_name}_{log_name}'
        args.iteration=iteration
        args.images='images'
        # args.model_path=f'output/{args.exp_name}'
        args.model_path=f'{args.exp_name}'
        args.motion_offset_flag=True
        args.quiet=False
        args.resolution=-1
        args.sh_degree=3
        args.skip_test=False
        args.skip_train=True
        args.smpl_type='smpl'
        args.source_path=f'/home/tom/fsas/workspace/dataset/ZJU_monocap/my_{args.data_name}'
        args.white_background=False

        # print("=====================================")
        print("Rendering " + args.exp_name)
        # print(args)
        # print("=====================================")

        # safe_state(args.quiet)

        # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

