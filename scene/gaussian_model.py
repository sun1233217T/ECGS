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
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
from tqdm import tqdm

from mtools import logger, do_once, debug, auto_counter, debug_after_n_iters
logger.set_logger_skip_iters(logger, 100)
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass
from diff_gaussian_rasterization_analyse import s_curve
import logging
from utils.quantize_tools import quantize_small_to_ufp6_fp32

class FSQQuantizer(nn.Module):
    def __init__(self):
        super(FSQQuantizer, self).__init__()
        self.quent_range = 2

    # def forward(self, x):
    #     # 前向做离散量化
    #     x_q = torch.where(x < -0.5, torch.full_like(x, -1),
    #           torch.where(x > 0.5, torch.full_like(x, 1),
    #                       torch.zeros_like(x)))
        
    #     # 使用 Straight-Through Estimator：反向直接使用 x 的梯度
    #     return x + (x_q - x).detach()

    def forward(self, x):
        # 先对 x 四舍五入到最接近的整数，再裁剪到 [-2, 2]
        x_q = torch.clamp(torch.round(x), -self.quent_range, self.quent_range)
        return x + (x_q - x).detach()


class FSQDecoder(nn.Module):
    def __init__(self, input_dim=8, inner_dim=32, output_dim=3):
        """
        input_dim: 量化输入的通道数，可选 8 或 16
        """
        super(FSQDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, inner_dim)  # 第一层，全连接
        self.fc2 = nn.Linear(inner_dim, inner_dim // 2)         # 第二层，降维
        # io_dim = inner_dim // 2
        # if io_dim >= 24:                         # 如果 io_dim 大于 24，增加一层
        #     self.fc3 = nn.Sequential(
        #         nn.Linear(io_dim, io_dim // 2),
        #         nn.ReLU(),
        #         nn.Linear(io_dim // 2, output_dim) 
        #     )
        # else:
        self.fc3 = nn.Linear(inner_dim // 2, output_dim)    # 输出层，单个 float32 值
    
    def forward(self, x):
        x = x.float()  # 确保输入为 float 类型
        x = F.relu(self.fc1(x))  # ReLU 激活函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层无激活，输出 float32 值
        return x
    
class InverseDecoder(nn.Module):
    def __init__(self, output_dim=8, inner_dim = 32, input_dim = 3):  # 必须知道原来的维度
        super().__init__()
        self.fc1 = nn.Linear(input_dim, inner_dim // 2)
        self.fc2 = nn.Linear(inner_dim // 2, inner_dim)
        self.fc3 = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出范围控制在 [-1, 1]，接近 FSQ 范围
    
class ScaleStatFSQQuantizer(nn.Module):
    def __init__(self,self_quantizer=False,up_thresh = 0.75, down_thresh = 0.25, temperature =0.5):
        super(ScaleStatFSQQuantizer, self).__init__()
        self.en = FSQDecoder(input_dim=3, inner_dim=32, output_dim=3) if self_quantizer else lambda x: x
        self.sm = F.sigmoid if self_quantizer else lambda x: x
        self.self_quantizer = self_quantizer
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.temperature = temperature

    def STE_forward(self, x, last_x = None, only_last_stat = False):
        x = self.en(x)
        x = self.sm(x) # 归一化到 [0,1]
        
        # 前向做离散量化, 量化{0,1}
        if last_x is None:
            x_q = torch.where(x > 0.5, torch.full_like(x, 1),
                          torch.zeros_like(x))
        elif only_last_stat:
            assert last_x is not None
            return x + (last_x - x).detach()
        else:
            # 滞后离散化逻辑
            # 如果 last_x == 1, 只有 x < down_thresh 才变成 0
            turn_1 = (last_x == 0) & (x >= self.up_thresh)
            # 如果 last_x == 0, 只有 x > up_thresh 才变成 1
            turn_0 = (last_x == 1) & (x <= self.down_thresh)

            x_q = torch.where(turn_1, torch.ones_like(x),
                torch.where(turn_0, torch.zeros_like(x), last_x))
            
        
        # 使用 Straight-Through Estimator：反向直接使用 x 的梯度
        return x + (x_q - x).detach()
    
    def sample_gumbel(self, shape, eps=1e-10):
        """从标准 Gumbel 分布采样"""
        U = torch.rand(shape, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, x, last_x=None, only_last_stat=False):
        #Gumbel Softmax 量化
        # 提取特征和归一化到 [0,1]
        x = self.en(x)
        x = torch.sigmoid(x)  # 这里 x 被解释为取 1 的概率

        # 当 only_last_stat 为 True 时，直接用上一步状态（不改变梯度路径）
        if last_x is not None and only_last_stat:
            return x + (last_x - x).detach()

        # 构造二分类问题的 logits
        # 对于二值量化，我们认为类别 1 的概率为 x，类别 0 的概率为 (1-x)
        eps = 1e-10  # 防止对数不稳定
        # logits 分量：log(1-x) 与 log(x)
        logits_0 = torch.log(1 - x + eps)
        logits_1 = torch.log(x + eps)

        # 加入 Gumbel 噪声：分别为类别 0 和类别 1 采样
        gumbel_0 = self.sample_gumbel(x.shape, eps)
        gumbel_1 = self.sample_gumbel(x.shape, eps)

        # 加入噪声并除以温度（temperature 决定平滑程度，训练初期可设置较大，后期退火）
        noisy_logits_0 = (logits_0 + gumbel_0) / self.temperature
        noisy_logits_1 = (logits_1 + gumbel_1) / self.temperature

        # 将两种 logits 组合为二维张量，最后一维表示类别 0 和 1
        logits = torch.stack([noisy_logits_0, noisy_logits_1], dim=-1)
        # 使用 softmax 得到近似 one-hot 的概率分布
        probs = F.softmax(logits, dim=-1)
        # “离散量化”结果取类别 1 的概率，当温度趋于 0 时，该概率趋向于 {0,1}
        x_q_candidate = probs[..., 1]

        # 如果提供 last_x，则可使用延迟更新逻辑以抑制频繁跳变
        if last_x is not None:
            # 这里延迟逻辑与之前类似：
            # 如果 last_x 接近 0，只有当候选值超过上阈值时才更新为 1
            turn_1 = (last_x < 0.5) & (x_q_candidate >= self.up_thresh)
            # 如果 last_x 接近 1，只有当候选值低于下阈值时才更新为 0
            turn_0 = (last_x >= 0.5) & (x_q_candidate <= self.down_thresh)
            # 根据延迟逻辑，决定最终的离散化状态（注意：此处用 torch.where 做硬更新，若希望保持完全连续可微，可考虑引入软门限）
            x_q = torch.where(turn_1, torch.ones_like(x),
                      torch.where(turn_0, torch.zeros_like(x), last_x))
        else:
            x_q = x_q_candidate

        # 使用类似 STE 的技巧：
        # 前向返回离散化的结果，但反向传播梯度还是从连续采样 x 得到
        return x + (x_q - x).detach()

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # self.scaling_activation = torch.exp
        # self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default",cat_low_app_opc = False,scale_dim = False, scle_inner_dim = 32):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
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

        self.if_erank_trace = False
        self.app_opacity = None
        self.app_opacity_mask = None
        self.cat_low_app_opc = cat_low_app_opc
        self.app_opc_decaly_ratio = 0.99
        self.view_length = -1
        self.denom_app = torch.empty(0)
        self.scale_quant = True if scale_dim > 0 else False
        if self.scale_quant:
            scale_dim = int(scale_dim)
            self.scale_dim = scale_dim
            scle_inner_dim = int(scle_inner_dim)
            logger.info("Using FSQ quantization for scaling, with dim: {}".format(scale_dim))
            self.scale_fsq = FSQDecoder(input_dim=scale_dim, inner_dim=scle_inner_dim).cuda()
            self.scale_quantizer = FSQQuantizer().cuda()
            self.scale_inverse_fsq = InverseDecoder(output_dim=scale_dim, inner_dim=scle_inner_dim).cuda()
        self.scale_stat_quant = True if scale_dim == -1 else False
        self.scale_stat_self_quant = True if scale_dim == -2 else False
        if self.scale_stat_quant:
            self.scale_stat = torch.empty(0)
            self.last_scale_stat = torch.empty(0)
            self.scale_stat_quantizer = ScaleStatFSQQuantizer().cuda()
        if self.scale_stat_self_quant:
            self.scale_stat_quantizer = ScaleStatFSQQuantizer(self_quantizer=True).cuda()

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
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def scaling_activation(self,x):
        if self.scale_quant:
            x = self.scale_quantizer(x)
            x = self.scale_fsq(x)
            return torch.exp(-x)
        elif self.scale_stat_quant:
            x = self.scale_stat_quantizer(self.scale_stat, self.last_scale_stat, True) * x
            return torch.exp(x)
        elif self.scale_stat_self_quant:
            o = self.scale_stat_quantizer(x) * x
            return torch.exp(o)
        else:
            return torch.exp(x)
    
    def scaling_inverse_activation(self,x):
        if self.scale_quant:
            x = self.scale_inverse_fsq(x)
        return torch.log(x)
    
    def train_inverse_scaling(self, x):
        logger.error("not implemented yet!")
        exit()
        _x = self.scale_quantizer(x)
        loss = F.mse_loss(self.scaling_inverse_activation(_x), x)
        return loss
    
    def scale_start_train(self,target):
        target = torch.exp(target).requires_grad_(False)
        opt = torch.optim.Adam([{"params": self.scale_fsq.parameters()},{"params": [self._scaling]}],lr=0.01)
        self.scale_fsq.train()
        logger.info("Start training scale")
        # tbar = tqdm(range(100), desc="Training scale")
        for i in range(100):
            opt.zero_grad()
            self.scale_fsq.zero_grad()
            loss = F.mse_loss(self.get_scaling, target)
            loss.backward()
            opt.step()
            # logger.debug("Training scale, with loss: {}".format(loss.item()))
            # tbar.set_postfix({"loss": loss.item()})
        logger.info("Finish training scale, with loss: {}".format(loss.item()))

    def scale_quant_train(self):
        opt = torch.optim.Adam([{"params": self.scale_stat_quantizer.parameters()}],lr=0.0001)
        self.scale_stat_quantizer.train()
        logger.info("Start training scale")
        # tbar = tqdm(range(100), desc="Training scale")
        for i in range(100):
            opt.zero_grad()
            self.scale_stat_quantizer.zero_grad()
            # debug()
            loss = F.mse_loss(self.scale_stat_quantizer(self._scaling), torch.ones_like(self.get_scaling))
            loss.backward()
            opt.step()
            logger.debug("Training scale, with loss: {}".format(loss.item()))
            # tbar.set_postfix({"loss": loss.item()})
        logger.info("Finish training scale, with loss: {}".format(loss.item()))

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
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
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

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        scale_shape = scales.shape[0]
        if self.scale_quant:
            self._scaling = nn.Parameter(torch.randn(scale_shape, self.scale_dim, device="cuda").requires_grad_(True))
            self.scale_start_train(scales)
        elif self.scale_stat_quant:
            self.scale_stat = nn.Parameter(torch.ones(scale_shape, 3, device="cuda").requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self.last_scale_stat = torch.ones_like(self.scale_stat).cuda()
        elif self.scale_stat_self_quant:
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self.scale_quant_train()
        else:
            self._scaling = nn.Parameter(scales.requires_grad_(True))
        # debug()
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_app = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.scale_quant:
            self.scale_fsq.train()
            self.scale_quantizer.train()
            self.scale_inverse_fsq.train()
        elif self.scale_stat_self_quant:
            self.scale_stat_quantizer.train()

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.scale_quant:
            l.append({'params': self.scale_fsq.parameters(), 'lr': training_args.scaling_quntization_lr, "name": "scale_fsq"})
        elif self.scale_stat_quant:
            l.append({'params': self.scale_stat, 'lr': training_args.scaling_lr, "name": "scale_stat"})
        elif self.scale_stat_self_quant:
            l.append({'params': self.scale_stat_quantizer.parameters(), 'lr': training_args.scaling_quntization_lr, "name": "scale_fsq"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

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
        for i in range(self.get_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        if self.scale_quant:
            scale = torch.log(self.get_scaling).detach().cpu().numpy()
        elif self.scale_stat_quant:
            scale_stat = self.scale_stat_quantizer(self.scale_stat, self.last_scale_stat, True).detach().cpu().numpy().astype(np.bool_)
            scale = self._scaling.detach().cpu().numpy()
            scale[~scale_stat] = -10000
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        if self.if_erank_trace:
            self.save_erank_trace(path[:-4] + "_erank_trace")

        if self.app_opacity is not None:
            self.save_app_opacity(path[:-4] + "_app_opacity")

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # opacities_new = self.inverse_opacity_activation(torch.ones_like(self.get_opacity)*0.05) #on ufp6
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

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
        if self.scale_quant:
            logger.error("scale witg fsq cannot be laoded!")
            exit(123)
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
            if group["name"] == "scale_fsq":
                continue
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
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_app = self.denom_app[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask] if self.tmp_radii is not None else None

        if self.if_erank_trace:
            self.choosed_points_mask = self.choosed_points_mask[valid_points_mask]
        if self.app_opacity != None:
            self.app_opacity = self.app_opacity[valid_points_mask]

        if self.scale_stat_quant:
            self.scale_stat = optimizable_tensors["scale_stat"]
            self.last_scale_stat = self.last_scale_stat[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "scale_fsq":
                continue
            assert len(group["params"]) == 1
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii = None, new_denom = None, new_grad_accum = None, new_scale_stat = None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.scale_stat_quant:
            d["scale_stat"] = new_scale_stat
            if new_scale_stat == None:
                logger.error("new_scale_stat should not be None!")
                exit(122)

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if new_scale_stat is not None:
            self.scale_stat = optimizable_tensors["scale_stat"]

        if new_tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_app = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
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
        if self.scale_quant:
            new_scaling = self._scaling[selected_pts_mask].repeat(N,1)
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        if self.if_erank_trace:
            self.choosed_points_mask = torch.cat((self.choosed_points_mask, self.choosed_points_mask[selected_pts_mask].repeat(N)))
        if self.app_opacity != None:
            new_app_opacity = self.app_opacity[selected_pts_mask].repeat(N,1)
            self.app_opacity = torch.cat((self.app_opacity,new_app_opacity),dim = 0)
        if self.scale_stat_quant:
            new_scale_stat = self.scale_stat[selected_pts_mask].repeat(N,1)
            new_last_stat = self.last_scale_stat[selected_pts_mask].repeat(N,1)
            self.last_scale_stat = torch.cat((self.last_scale_stat, new_last_stat), dim=0)
        else:
            new_scale_stat = None


        # self.tmp_save(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_app_opacity)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii,new_scale_stat = new_scale_stat)

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

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        if self.if_erank_trace:
            self.choosed_points_mask = torch.cat((self.choosed_points_mask, self.choosed_points_mask[selected_pts_mask]))
        if self.app_opacity != None:
            new_app_opacity = self.app_opacity[selected_pts_mask]
            self.app_opacity = torch.cat((self.app_opacity,new_app_opacity),dim = 0)
        if self.scale_stat_quant:
            new_scale_stat = self.scale_stat[selected_pts_mask]
            new_last_stat = self.last_scale_stat[selected_pts_mask]
            self.last_scale_stat = torch.cat((self.last_scale_stat, new_last_stat), dim=0)
        else:
            new_scale_stat = None
 
        # debug_after_n_iters(50)()

        # self.tmp_save(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_app_opacity)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_scale_stat = new_scale_stat)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, grad_demon_beta = 0.01):

        # debug_after_n_iters(10)()
        # denom = self.denom_app.floor().float()
        # denom = denom / (denom.max() / 100)
        # denom = denom.ceil().float() * 20 # 20 comes from self.denom.sum() / denom.sum()
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # correlation_matrix = torch.corrcoef(torch.stack([grads[grads>0], self.denom[grads>0]]))
        # pearson_r = correlation_matrix[0, 1]
        # logger.debug(f"Correlation between grads and denom: {pearson_r}")

        '''
        统计去趋势(Residual Correction)
        '''

        # 假设 grads 和 denom 已经过滤过无效的部分，并展平为一维向量
        valid_mask = grads > 0
        grads_valid = grads[valid_mask]
        denom_valid = self.denom[valid_mask].unsqueeze(1)  # 转为列向量

        # 加入偏置项
        X = torch.cat([denom_valid, torch.ones_like(denom_valid)], dim=1)  # [N, 2]
        # 求解线性回归参数 (正规方程)
        theta = torch.linalg.lstsq(X, grads_valid.unsqueeze(1)).solution

        # 计算拟合值并求残差
        fitted = (X @ theta).squeeze()
        residuals = grads_valid - fitted

        # 标准化 denom（只对有效数据进行）
        denom_norm = (self.denom[valid_mask] - self.denom[valid_mask].mean()) / self.denom[valid_mask].std()
        beta = grad_demon_beta * (residuals.std()) 

        # 用 residuals 替换 grads_valid 来获得去趋势后的梯度
        grads_corrected = grads.clone()
        grads_corrected[valid_mask] = residuals + beta * denom_norm

        # correlation_matrix = torch.corrcoef(torch.stack([grads_corrected[grads>0], self.denom[grads>0]]))
        # pearson_r = correlation_matrix[0, 1]
        # logger.debug(f"Correlation between corrected_grads and denom: {pearson_r}")


        self.tmp_radii = radii
        self.densify_and_clone(grads_corrected, max_grad, extent)
        self.densify_and_split(grads_corrected, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.app_opacity is not None and self.cat_low_app_opc:
            prune_mask2 = self.app_opacity_cat_main()
            prune_mask = torch.logical_or(prune_mask, prune_mask2)
            logger.debug(f"Pruning {prune_mask2.sum()} in {prune_mask.sum()} points due to app_opacity.")
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        if self.scale_stat_quant:
            # 始终保持scale_stat在0到1之间
            new_scale_stat = torch.clamp(self.scale_stat, min=0.0, max=1.0)
            optimizable_tensors = self.replace_tensor_to_optimizer(new_scale_stat, "scale_stat")
            self.scale_stat = optimizable_tensors["scale_stat"]
            self.last_scale_stat = self.scale_stat_quantizer(self.scale_stat, self.last_scale_stat).detach().clone()

        torch.cuda.empty_cache()

    def densify_and_split_v2(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if self.scale_quant:
            new_scaling = self._scaling[selected_pts_mask].repeat(N,1)
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        if self.app_opacity != None:
            new_app_opacity = self.app_opacity[selected_pts_mask].repeat(N,1)
            self.app_opacity = torch.cat((self.app_opacity,new_app_opacity),dim = 0)

        self.tmp_save(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_app_opacity)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_v2(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if self.app_opacity != None:
            new_app_opacity = self.app_opacity[selected_pts_mask]
            self.app_opacity = torch.cat((self.app_opacity,new_app_opacity),dim = 0)
        
        self.tmp_save(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_app_opacity)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune_v2(self, max_grad, min_opacity, extent, max_screen_size,radii = None):
        self.tmp_radii = None
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        before = self._xyz.shape[0]
        self.densify_and_clone_v2(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        
        self.densify_and_split_v2(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.app_opacity is not None and self.cat_low_app_opc:
            prune_mask2 = self.app_opacity_cat_main()
            prune_mask = torch.logical_or(prune_mask, prune_mask2)
            logger.debug(f"Pruning {prune_mask2.sum()} in {prune_mask.sum()} points due to app_opacity.")
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        torch.cuda.empty_cache()
        # logger.debug(f"Before: {before}, Clone: {clone}, Split: {split}, Prune: {prune}")
        return clone - before, split - clone, split - prune
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter, app_opacity = None):
        update_filter = update_filter.squeeze()
        
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))

        self.denom[update_filter] += 1
        # debug()
        self.denom_app += app_opacity

        if self.scale_stat_quant:
            self.last_scale_stat = self.scale_stat_quantizer(self.scale_stat, self.last_scale_stat, True).clone()

    def cat_low_app_opacity(self,radii):
        self.tmp_radii = radii
        prune_mask = self.app_opacity_cat_main()
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        logger.debug(f"Pruning {prune_mask.sum()} points due to app_opacity.")
        torch.cuda.empty_cache()

    def app_opacity_cat_main(self):
        cat_ratio = self.cat_low_app_opc
        # debug()
        # (self.get_opacity - self.app_opacity) > 0.7
        return (self.app_opacity < self.get_opacity * cat_ratio).squeeze()    #new_appcat_1
        
    def add_densification_stats_old(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def get_erank(self, epsilon = 1e-5):
        A = self.get_scaling
        S = torch.pow(A, 2).sum(dim=-1) + 1e-10
        q = torch.pow(A, 2) / S[..., None] + epsilon
        H = - torch.sum(q * torch.log(q), dim=-1)
        if H.mean().isnan():
            logger.error("H is nan!")
        return torch.exp(H)

    def get_erank_loss(self, lambda_erank = 0.001, epsilon = 1e-5, use_2d_restrict = False, mode = 0):
        if use_2d_restrict:
            logger.logonce("Using 2D restrict in erank loss!")
        else:
            logger.logonce(f"Erank loss mode {mode}!")
        erank = self.get_erank()
        if mode == 0:
            s3 = self.get_scaling.min(dim=-1).values if use_2d_restrict else 0
            # 计算 L_erank
            log_term = torch.log(erank - 1 + epsilon)  # 计算 log(erank(Gk) - 1 + epsilon)
            penalty = torch.clamp(-log_term, min=0)  # max(-log_term, 0)
            L_erank = (lambda_erank * penalty + s3).mean()  # 损失函数求和
        elif mode == 1:
            penalty = torch.pow(erank, 4) - 6 * torch.pow(erank, 3) + 13 * torch.pow(erank, 2) - 12 * (erank) + 4 - 0.05 * torch.pow(erank - 2, 2) * torch.pow(erank - 0.4, 2)
            L_erank = (lambda_erank * penalty).mean()
        return L_erank
    
    def get_erank_at(self,temp = 1):
        scale = self.get_scaling
        D = (scale*scale)**(1/temp)
        _sum = D.sum(dim=1, keepdim=True)
        pD = D / _sum
        ordered_scale, _ = torch.sort(scale, descending=True)
        try:
            entropy = -torch.sum(pD*torch.log(pD), dim=1)
            erank = torch.exp(entropy) 
        except Exception as e:
            print(e)
            pass
        return erank, ordered_scale
    
    def get_erank_loss_at(self, lambda_erank = 0.01):
        erank, ordered_scale = self.get_erank_at()
        return lambda_erank * torch.clamp(-torch.log(erank-1+1e-7), 0).mean() + ordered_scale[:,2].mean()
    
    def get_2d_loss(self, lambda_erank = 0.01):
        s3 = self.get_scaling.min(dim=-1).values
        return s3.mean()
    
    def erank_trance_init(self, traced_points_start_num = 10):
        self.if_erank_trace = True
        logger.logonce("Erank trace init!")

        n_points = self.get_xyz.shape[0]
        device = self.get_xyz.device

        self.choosed_points_mask = torch.zeros(n_points, dtype=torch.int8, device=device)
        random_idx = torch.randint(0, n_points, (traced_points_start_num,))
        for i in range(traced_points_start_num):
            self.choosed_points_mask[random_idx[i]] = i + 1
        
        self.traced_erank_list = []

        return 0

    def cal_erank_trace(self):
        do_once(self.erank_trance_init)

        mask = self.choosed_points_mask > 0
        assert mask.shape[0] == self.get_erank().shape[0]

        # logger.skip_info(f"Tracing erank, {mask.sum()} points selected.")
        self.traced_erank_list.append([self.get_erank()[mask].detach().cpu().numpy(), self.choosed_points_mask[mask].detach().cpu().numpy()])

        return 0
    
    def save_erank_trace(self, path):
        assert self.if_erank_trace
        np.save(path, self.traced_erank_list)
        return 0
    

    def get_my_scalse_loss(self, lambda_scale = 0.01, epsilon = 1e-8, needle_shift = 0.5, plane_shift = 2, lambda_n = 0.9):
        def get_nedratio(scales, mask = None):
            if mask is not None:
                if mask.sum() != 0:
                    A = scales[mask] 
                else:
                    A = scales
            else:
                A = scales
            Asort = torch.sort(A, dim=1)[0]
            Amax = Asort[:, -1]
            Amin = Asort[:, 0] + epsilon
            Amid = Asort[:, 1]
            pr = (Amid / Amin) * (Amid / Amax)
            nr = Amax / Amid

            pr = torch.log10(pr)
            nr = torch.log10(nr)

            return pr , nr, Amax
        
        planeratio, nedratio, Amax = get_nedratio(self.get_scaling)

        loss_n = torch.clamp((nedratio - needle_shift), min=0).mean()
        loss_p = torch.clamp((plane_shift - planeratio), min=0).mean()

        loss = lambda_n * loss_n + (1 - lambda_n + 0.5) * loss_p

        # max_scale_loss = torch.clamp(1e-2 - Amax, min=0).mean()

        return loss * lambda_scale
    
    def get_opacity_loss(self, lambda_opacity = 0.0001):
        def f(x):
            left = (1 - torch.cos((torch.pi/0.1)*x)) / 2
            right = (1 + torch.cos(torch.pi*(x-0.1)/0.9)) / 2
            return torch.where(x <= 0.1, left, right)

        opacity = self.get_opacity

        # loss 1
        # loss = f(opacity).mean() * lambda_opacity

        # loss 2
        loss = 1 - torch.pow((0.1 - opacity),2)
        loss = loss.mean() * lambda_opacity

        return loss
    
    def init_app_opacity(self):
        logger.logonce("App opacity init!")
        self.app_opacity = torch.zeros_like(self._opacity)
        self.app_opacity_mask = torch.zeros_like(self._opacity)

    def set_decaly_ratio(self, view_length, confidance = 0.90, resave_ratio = 0.5):
        self.view_length = view_length
        decaly_ratio = resave_ratio ** (1 / (math.log(1 - confidance) / math.log(1 - 1 / view_length)))
        self.app_opc_decaly_ratio = decaly_ratio
        logger.info(f"Decaly ratio set to {decaly_ratio}")

    def store_app_opacity(self, app_opacity):
        do_once(self.init_app_opacity)
        if self.app_opacity.shape[0] != app_opacity.shape[0]:
            logger.error("app_opacity.shape error! reinit app_opacity")
            self.init_app_opacity()
        self.app_opacity = self.app_opacity
        self.app_opacity = self.app_opacity * self.app_opc_decaly_ratio
        #fix app_opacity that update later than opacity
        mask2 = self.app_opacity > self.get_opacity
        self.app_opacity = torch.where(mask2, self.get_opacity, self.app_opacity).contiguous().detach().requires_grad_(False)

        self.app_opacity_mask = app_opacity > self.app_opacity
        self.app_opacity = torch.where(self.app_opacity_mask, app_opacity, self.app_opacity).contiguous().detach().requires_grad_(False)

        self.app_opacity_mask = s_curve((self.app_opacity - app_opacity).detach().requires_grad_(False))
        
        # app_opacity = torch.cat([app_opacity,self.app_opacity.detach() * 0.99],dim = -1)
        # self.app_opacity = torch.max(app_opacity,dim = -1)[0].unsqueeze(-1).contiguous().detach().requires_grad_(False)

    def save_app_opacity(self, path):
        assert self.app_opacity != None
        np.save(path, self.app_opacity.detach().cpu().numpy())
        logger.info(f"App opacity saved to {path}")
        return 0
    
    def new_shape_loss(self, lamda = 0.1, lambda_sdl = 0.9):
        def cal_V(scales):
            s1 = scales[:, 0]
            s2 = scales[:, 1]
            s3 = scales[:, 2]
            V = s1 * s2 * s3 * 4 / 3
            return V
        
        def cal_S(scales):
            p = 1.6075
            s1 = scales[:, 0]
            s2 = scales[:, 1]
            s3 = scales[:, 2]
            S = 4 * torch.pow(torch.pow(s1 * s2, p) + torch.pow(s1 * s3, p) + torch.pow(s2 * s3, p) + 1e-8, 1/p)
            return S
        
        def cal_L(scales):
            L = torch.sum(scales, dim = -1)
            return L
        
        SdL = torch.clamp(cal_S(self.get_scaling) / (torch.pow(cal_L(self.get_scaling),2) + 1e-8), min=0, max=1)
        VdS = torch.pow(cal_V(self.get_scaling),2/3) / (cal_S(self.get_scaling) + 1e-8)

        loss = (1 - SdL) * lambda_sdl + VdS * (1 - lambda_sdl)
        loss = loss.mean() * lamda
        # logger.info(f"Shape loss: {loss.item()}")
        return loss
    
    @auto_counter
    def tmp_save(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_app_opacity, count = 0):
        path = r"tmp_save/{}.ply".format(count)

        xyz = new_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = new_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = new_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = new_opacities.detach().cpu().numpy()
        scale = new_scaling.detach().cpu().numpy()
        rotation = new_rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        app_path = r"tmp_save/{}_app.ply".format(count)
        np.save(app_path, new_app_opacity.detach().cpu().numpy())

        assert new_app_opacity.shape[0] == new_xyz.shape[0]

        logger.info("tmp save done!")

        return 0

    def logger_stat(self):
        if logger.level == logging.DEBUG:
            # scale = self.get_scaling
            # qscale = quantize_small_to_ufp6_fp32(scale)
            # qstat = qscale > 0
            # qstat_hist = torch.histc(qstat.sum(-1), bins=4, min=0, max=3)
            # logger.skip_debug(f"qstat_hist: {qstat_hist}")

            # scale_stat = self.scale_stat_quantizer(self._scaling)
            # scale_stat = self.scale_stat_quantizer(self.scale_stat, self.last_scale_stat)
            # scale_stat = self.last_scale_stat
            # logger.debug(f"scale_stat sum: {scale_stat.sum()} \t scale_stat mean: {scale_stat.mean()} \t std: {scale_stat.std()}")
            # gaussian_sclae_stat = scale_stat.sum(-1)
            # # debug_after_n_iters(100)()
            # gaussian_sclae_stat_hist = torch.histc(gaussian_sclae_stat, bins=4, min=0, max=3).long()
            # logger.debug(f"gaussian_sclae_stat_hist: {gaussian_sclae_stat_hist}")
            pass
        
        # debug_after_n_iters(100)()

        pass
