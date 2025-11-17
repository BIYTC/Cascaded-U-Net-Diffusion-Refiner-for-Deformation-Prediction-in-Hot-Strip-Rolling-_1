import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
from typing import Optional, Tuple
import numpy as np

# Added: 集中配置参数
class ModelConfig:
    # 通道倍数配置（可调节模型规模）
    channel_mult = [2, 4, 8, 16, 32]  # 原始为[1, 2, 4, 8, 16]
    num_res_blocks = 3  # 每层的残差块数量
    time_dim = 512  # 时间嵌入维度
    attn_layers = [2, 3]  # 添加注意力机制的层索引
    dropout = 0.1
    use_adaGN = True  # 使用自适应归一化


class DiffusionUtils:
    def __init__(self, timesteps=1000, ddim_timesteps=50, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.ddim_timesteps = ddim_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        # Extract the appropriate t index for a batch of indices
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        # Forward diffusion process: q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, features, t, noise=None, loss_type="l1"):
        # Calculate loss for training
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, features, t)

        if loss_type == 'l1':
            # loss = F.l1_loss(noise[:,0:1,:,:], predicted_noise[:,0:1,:,:])
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.huber_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, features, t, t_index):
        # Reverse diffusion process: p(x_{t-1} | x_t)
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, features, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_ddim(self, model, x, features, t, t_index, eta=0.0):
        # DDIM sampling
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Predict noise
        pred_noise = model(x, features, t)

        # Calculate x0
        x0 = sqrt_recip_alphas_t * (x - sqrt_one_minus_alphas_cumprod_t * pred_noise)

        if t_index == 0:
            return x0

        # Calculate direction pointing to xt
        sigma_t = eta * torch.sqrt((1 - self.alphas_cumprod[t_index - 1]) / (1 - self.alphas_cumprod[t_index]) *
                                   (1 - self.alphas_cumprod[t_index] / self.alphas_cumprod[t_index - 1]))

        noise = torch.randn_like(x)
        x_prev = torch.sqrt(self.alphas_cumprod[t_index - 1]) * x0 + \
                 torch.sqrt(1 - self.alphas_cumprod[t_index - 1] - sigma_t ** 2) * pred_noise + \
                 sigma_t * noise

        return x_prev

    @torch.no_grad()
    def sample(self, model, unet_model, features, shape, device, ddim=False, eta=0.0, denoise_step=5):
        # Sample from the model using DDPM or DDIM
        b = shape[0]
        # img = torch.randn(shape, device=device)
        img = unet_model(features)
        _t_list = None
        if ddim:
            """生成均匀间隔的时间步序列"""
            step_ratio = self.timesteps // self.ddim_timesteps
            _t_list = list(range(0, self.timesteps, step_ratio))
            for i in tqdm(reversed(range(0, self.ddim_timesteps)), desc='sampling loop time step', total=self.ddim_timesteps):
                _t = _t_list[i]
                t = torch.full((b,), _t, device=device, dtype=torch.long)  # ddim采用跳步采样
                img = self.p_sample_ddim(model, img, features, t, i, eta=eta)
        else:
            for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
                if i < denoise_step:
                    t = torch.full((b,), i, device=device, dtype=torch.long)  # ddpm采用原来的时间步采样
                    img = self.p_sample(model, img, features, t, i)
        return img