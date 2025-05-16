import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_analyse
try:
    from gaussian_renderer.solid_render import render_analyse as render_analyse_solid
except:
    print("Solid render not available, using default render")
import torchvision
import cv2
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def save_depth_as_colormap(depth_tensor, save_path):
    """
    将深度图 Tensor 转换为伪彩色图并保存为 PNG。
    
    Args:
        depth_tensor (torch.Tensor): 单通道深度图，形状为 (H, W) 或 (1, H, W)，值域任意。
        save_path (str): 保存路径，文件名需带 .png 后缀。
    """
    if depth_tensor.ndim == 3:
        depth_tensor = depth_tensor.squeeze(0)  
    

    depth_np = depth_tensor.cpu().numpy()
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    color_mapped = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET) 

    color_tensor = torch.from_numpy(color_mapped).permute(2, 0, 1).float() / 255.0 

    torchvision.utils.save_image(color_tensor, save_path)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = rendering_pkg["render"]
        rendering_depth = rendering_pkg["depth"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        save_depth_as_colormap(rendering_depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, opacity_analyse:bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if opacity_analyse:
            gaussians.save_app_opacity(os.path.join(dataset.model_path, "point_cloud/iteration_30000/opacity_analysis"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--solid", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # if args.analyse:
    render = render_analyse
    logger.info("Rendering with analysis")
    if args.solid:
        logger.info("Rendering with solid analysis")
        render = render_analyse_solid

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,True)