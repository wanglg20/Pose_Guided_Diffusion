"""
Unet model for time series forecasting with epipolar attention
Latest Update: 2021-07-24
Author: Linge Wang
reference: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
import torch.nn.functional as F
from diffusion_unet import * 
from epipolar import *

def Source_feature_Downsample(source_resolution: int, attn_resolution: int,
                              source_feature: torch.Tensor):
    """
    Downsampling source view feature map to the same resolution as attention map
    Note that default source_resolution has the same height and width 
    

    Args:
        source_resolution (int): source view feature map resolution, 
        attn_resolution (int): attention map resolution
        source_feature (batch_size, channel, source_resolution, source_resolution): source view feature map, shape: 

    Returns:
        source_feature (batch_size, channel, attn_resolution, attn_resolution): source view feature map, shape: 
    """
    scale_factor = attn_resolution / source_resolution
    source_feature = F.interpolate(source_feature, scale_factor=scale_factor, mode='bilinear')
    return source_feature


class epipolar_Attention_Block(nn.Module):
    """
    Args:
        f_channels (int): number of channels of source view feature map
        u_channels (int): number of channels of intermedia UNet feature map
        n_heads (int): number of attention heads
        d_k (int): dimension of key
        n_groups (int): number of groups for group normalization
        attn_resolution (int): attention map resolution 
    """
    
    def __init__(self, f_channels: int, u_channels: int,
                 n_heads: int = 1, 
                 d_k: int = None, n_groups: int = 32,
                 attn_resolution: int = 32,
                 weight_Mat: torch.Tensor = None
                 ):
        super().__init__()
        # Default `d_k`
        if d_k is None:
            d_k = u_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, u_channels)
        # Projections for query, key and values
        #self.projection_key = nn.Linear(f_channels, n_heads * d_k)
        #self.projection_query = nn.Linear(u_channels, n_heads * d_k)
        #self.projection_value = nn.Linear(f_channels, n_heads * d_k)
        # Linear layer for final transformation
        #self.output = nn.Linear(n_heads * d_k, u_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.attn_resolution = attn_resolution
        self.epipolar_affinity = epipolar_Affinity_Mat()
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, key: torch.Tensor, query: torch.Tensor, t: Optional[torch.Tensor], weight_Mat: torch.Tensor):
        key = Source_feature_Downsample(key.shape[2], self.attn_resolution, key)
        # Common Cross Attention
        _ = t
        batch_size, n_channels, height, width = key.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        k, q, v = key, query, key       # (B, C, H, W)
        Affinity_Mat = epipolar_Affinity_Mat(k, q)
        B,C, HW, HW = Affinity_Mat.shape
        Weigh_Mat = weight_Mat.unsqueeze(0).unsqueeze(0).expand(B, C, HW, HW)
        Affinity_Mat = torch.dot(Affinity_Mat, Weigh_Mat)
        
    
        v = v.view(batch_size, n_channels, -1)      # (B, C, HW)
        Affinity_Mat = self.softmax(Affinity_Mat)   # (B, C, HW, HW), v: (B, C, H, W)
        attn = torch.einsum('bijk,bik->bij', Affinity_Mat, v)
        res = attn.softmax(dim=2)
        
        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.view(batch_size, n_channels, height, width)
        # res operation
        res += query

        return res    
    
class epipolar_MiddleBlock(nn.Module):
    """
    Middle block of the UNet
    resBlock + self-attention + epipolar attention
        
    """
    def __init__(self, n_channels: int, time_channels: int, f_channels: int,
                 source_resolution: int, attn_resolution: int):
        super().__init__()
        
        self.source_resolution = source_resolution
        self.attn_resolution = attn_resolution
        self.n_channels = n_channels
        self.time_channels = time_channels
        
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.self_attn = Self_AttentionBlock(n_channels, n_channels, n_channels)
        self.epipolar_attn = epipolar_Attention_Block(f_channels=f_channels, u_channels=n_channels, 
                                                      n_heads = 1, attn_resolution=attn_resolution)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, f: torch.Tensor, weight_Mat: torch.Tensor):
        x = self.res1(x, t)
        x = self.self_attn(x)
        x = self.epipolar_attn(key=f, query=x, t=t, weight_Mat = weight_Mat)
        return x


class epipolar_DownBlock(nn.Module):
    """
     Downsample + self-attention + epipolar attention
    """
    def __init__(self, n_channels: int, out_channels: int, time_channels: int, f_channels: int,
                 source_resolution: int, attn_resolution: int):
        super().__init__()
        self.source_resolution = source_resolution
        self.attn_resolution = attn_resolution
        self.n_channels = n_channels
        self.time_channels = time_channels
        
        self.res1 = ResidualBlock(n_channels, out_channels, time_channels)
        self.Downsample = Downsample()
        self.self_attn = Self_AttentionBlock(n_channels, n_channels, n_channels)
        self.epipolar_attn = epipolar_Attention_Block(f_channels=f_channels, u_channels=n_channels, 
                                                      n_heads = 1, attn_resolution=attn_resolution)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, f: torch.Tensor, weight_Mat: torch.Tensor):
        x = self.res1(x, t)
        x = self.Downsample(x)
        x = self.self_attn(x)
        x = self.epipolar_attn(key=f, query=x, t=t, weight_Mat = weight_Mat)
        return x

class epipolar_UpBlock(nn.Module):
    """
    Upsample + self-attention + epipolar attention
    """
    def __init__(self, n_channels: int, out_channels: int, time_channels: int, f_channels: int,
                 source_resolution: int, attn_resolution: int):
        super().__init__()
        self.source_resolution = source_resolution
        self.attn_resolution = attn_resolution
        self.n_channels = n_channels
        self.time_channels = time_channels
        
        self.res1 = ResidualBlock(n_channels, out_channels, time_channels)
        self.Upsample = Upsample()
        self.self_attn = Self_AttentionBlock(n_channels, n_channels, n_channels)
        self.epipolar_attn = epipolar_Attention_Block(f_channels=f_channels, u_channels=n_channels, 
                                                      n_heads = 1, attn_resolution=attn_resolution)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, f: torch.Tensor, weight_mat: torch.Tensor):
        x = self.res1(x, t)
        x = self.Upsample(x)
        x = self.self_attn(x)
        x = self.epipolar_attn(key=f, query=x, t=t, weight_mat = weight_mat)
        return x

class epipolar_attn_Unet(nn.Module):
    """
    Unet model described in posed guided diffusion
    
    Args:
       in_channels (int): number of channels of input image
       latten_c (int): number of channels in the first layer
       f_channels (int): number of channels of source view feature map
       channel_multiples (tuple): the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
       time_channels_mul (int): the number of time channels
       attn_resolution (tuple): the list of attention map resolution at each resolution
       source_resolution (int): source view feature map resolution
       dropout (float): dropout rate using for training
       is_train (bool): whether the model is in training mode
    """
    def __init__(self,
                 in_channels=3,
                 latten_c=128,
                 f_channels = 3,
                 head_channels = 64,
                 channel_multiples = (1,2,3,4),
                 time_channels_mul = 4,
                 attn_resolution = (32, 16, 8),
                 source_resolution = 256,
                 dropout = 0.1,
                 is_train = True
                 ):
        super().__init__()
        n_blocks = len(channel_multiples)
        
        self.image_conv = nn.Conv2d(in_channels, latten_c, kernel_size=(3, 3), padding=(1, 1))
        self.time_embedding = TimeEmbedding(latten_c * time_channels_mul)
        self.time_channels = latten_c * time_channels_mul
        self.attn_resolution = attn_resolution
        
        if is_train:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        # encoder
        self.DownBlock_1 = nn.Sequential(
            ResidualBlock(latten_c, latten_c, self.time_channels),
            ResidualBlock(latten_c, latten_c, self.time_channels),
            ResidualBlock(latten_c, latten_c * channel_multiples[0], self.time_channels),
            Downsample()
        )
        
        self.DownBlock_2 = nn.Sequential(
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, f_channels, source_resolution, attn_resolution[0]),
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, f_channels, source_resolution, attn_resolution[0]),
            epipolar_DownBlock(latten_c * channel_multiples[0], latten_c * channel_multiples[1], self.time_channels, f_channels, source_resolution, attn_resolution[1]),
        )

        self.DownBlock_3 = nn.Sequential(
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, f_channels, source_resolution, attn_resolution[1]),
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, f_channels, source_resolution, attn_resolution[1]),
            epipolar_DownBlock(latten_c * channel_multiples[1], latten_c * channel_multiples[2], self.time_channels, f_channels, source_resolution, attn_resolution[2]),
        )
        
        
        # middle block
        self.middle_process = nn.Sequential(
            epipolar_MiddleBlock(latten_c * channel_multiples[2], self.time_channels, f_channels, source_resolution, attn_resolution[2]),
            epipolar_MiddleBlock(latten_c * channel_multiples[2], self.time_channels, f_channels, source_resolution, attn_resolution[2]),
            epipolar_MiddleBlock(latten_c * channel_multiples[2], self.time_channels, f_channels, source_resolution, attn_resolution[2]),
        )
        
        # decoder
        self.UpBlock_1 = nn.Sequential(
            epipolar_UpBlock(latten_c * channel_multiples[2], latten_c * channel_multiples[1], self.time_channels, f_channels, source_resolution, attn_resolution[1]),
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, f_channels, source_resolution, attn_resolution[1]),
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, f_channels, source_resolution, attn_resolution[1]),
        )
        
        self.UpBlock_2 = nn.Sequential(
            epipolar_UpBlock(latten_c * channel_multiples[1], latten_c * channel_multiples[0], self.time_channels, f_channels, source_resolution, attn_resolution[0]),
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, f_channels, source_resolution, attn_resolution[0]),
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, f_channels, source_resolution, attn_resolution[0]),
        )
        
        self.UpBlock_3 = nn.Sequential(
            Upsample(),
            ResidualBlock(latten_c * channel_multiples[0], latten_c * channel_multiples[0], self.time_channels),
            ResidualBlock(latten_c * channel_multiples[0], latten_c * channel_multiples[0], self.time_channels),
            ResidualBlock(latten_c * channel_multiples[0], latten_c * channel_multiples[0], self.time_channels),
        )
        
        self.norm = nn.GroupNorm(8, latten_c)
        self.act = Swish()
        self.output_conv = nn.Conv2d(latten_c, in_channels, kernel_size=(3, 3), padding=(1, 1))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, f: torch.Tensor, K: torch.Tensor, R: torch.Tensor, trans: torch.Tensor):
        # Get time-step embeddings
        step_embedding = self.time_emb(t)
        
        x = self.image_conv(x)
        
        weight_mat_1 = epipolar_weight_Mat(self.attn_resolution[0])
        weight_mat_2 = epipolar_weight_Mat(self.attn_resolution[1])
        weight_mat_3 = epipolar_weight_Mat(self.attn_resolution[2])
        
        
        # encoder
        x1 = self.DownBlock_1(x, step_embedding, f)
        x2 = self.DownBlock_2(x1, step_embedding, f, weight_mat_1)
        x3 = self.DownBlock_3(x2, step_embedding, f, weight_mat_2)
        
        # middle
        x4 = self.middle_process(x3, step_embedding, f, weight_mat_3)
        
        # decoder
        x5 = torch.cat([x4, x3], dim=1)
        x5 = self.UpBlock_1(x5, step_embedding, f, weight_mat_2)
        
        x6 = torch.cat([x5, x2], dim=1)
        x6 = self.UpBlock_2(x6, step_embedding, f, weight_mat_1)
        
        x7 = torch.cat([x6, x1], dim=1), 
        x7 = self.UpBlock_3(x7, step_embedding, f)
        
        x7 = self.dropout(x7)
        res =  self.output_conv(self.act(self.norm(x)))
        
        return res