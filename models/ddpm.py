"""
DDPM model and Diffusion model
Latest Update: 2021-07-22
Author: Linge Wang
reference: https://github.com/SingleZombie/DL-Demos
"""

import torch
import math
from diffusers import DDPMScheduler, UNet2DModel

class DDPM():
    """
    DDPM class for managing hyperparameters and sampling
    Args:
        n_steps (int): number of steps
        device (torch.device): device
        min_beta (float): min beta
        max_beta (float): max beta
        beta_schedule_type (str): beta schedule type, 'linear' or 'cosine'
    """
    def __init__(self,
                 n_steps: int,
                 device=torch.device('cpu'),
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02,
                 beta_schedule_type: str = 'linear'):
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.device = device
        betas = self.beta_schedule(n_steps, beta_schedule_type)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)       #alpha的累乘, alpha_bar_t
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        alpha_prev = torch.empty_like(alpha_bars)   #alpha_bars_t-1
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        
        # q(x_t-1| x_t, x_0) ~ N(x_t-1; coef1 * x_t + coef2 * x_0, betas[t]*(1 - alpha_bars[t-1])/(1 - alpha_bars[t]))
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)
    
    def beta_schedule(self, timesteps, schedule_type = 'linear', s = 0.008):
        if schedule_type == 'linear':
            betas = torch.linspace(self.min_beta, self.max_beta, timesteps).to(self.device)
        elif schedule_type == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas =  torch.clip(betas, 0, 0.999)
        else:
            raise NotImplementedError
        return betas
    
    def sample_forward(self, x, t, eps=None):
        """
        given x_0 = x, steps = t, sample x_t

        Args:
            x (tensor): x_0, (n, c, h, w)
            t (int): steps
            eps (tensor): (n, c, h, w), default None, if None, eps = N(0, I)
        return:
            res: x_t, (n, c, h, w) 
        """
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #(n, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        
        # x_t = eps * sqrt(1 - alpha_bar) + sqrt(alpha_bar) * x_0
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res
    
    
    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):
        """
        given x_t, steps = t, trained model net to predict eps,  sample x_t-1

        Args:
            x_t (tensor): x_t, (n, c, h, w)
            t (int): steps
            net (nn.Module): trained model to predict eps
            simple_var: if True, use betas[t] as variance, else use (1 - alpha_bars[t-1]) / (1 - alpha_bars[t]) * betas[t]
            clip_x0: if True, clip x_0 to [-1, 1]

        Returns:
            x_t-1 (tensor): x_t-1, (n, c, h, w), x_t-1 = mean + eps * sqrt(var)
        """
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t
    
