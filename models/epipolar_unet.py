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

from models.unet import *
from models.diffusion_unet import * 
from models.epipolar import *

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

class Weighted_QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, Weight_Mat):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)       # each: (N, H*C, T)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        #print(weight.shape)     #(N*H, T, T)

        weight = Weight_Mat * weight
        
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)  #(N*H, T, T)
        v = v.reshape(bs * self.n_heads, ch, length)
        # v:(N*H, C, T )
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)



class epipolar_Attention_Block(nn.Module):
    """
    Args:
        query_channels (int): number of channels of source view feature map
        key_channels (int): number of channels of intermedia UNet feature map
        n_heads (int): number of attention heads
        d_k (int): dimension of key
        n_groups (int): number of groups for group normalization
        attn_resolution (int): attention map resolution 
    """
    
    def __init__(self, 
                 query_channels: int, 
                 key_channels: int,
                 num_heads: int = 1, 
                 num_head_channels: int = -1,
                 attn_resolution: int = 32
                 ):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                query_channels % num_head_channels == 0
            ), f"query channels {query_channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = query_channels // num_head_channels

        self.norm = normalization(query_channels)
        
        self.key_proj = nn.Conv2d(key_channels, query_channels, kernel_size=1, bias=False)
        
        self.qkv = conv_nd(1, query_channels*2, query_channels * 3, 1)
        self.attention = Weighted_QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, query_channels, query_channels, 1))
        # Projections for query, key and values
        #self.kv = nn.Conv2d(query_channels, query_channels * 2, kernel_size=1, bias=False)

        self.attn_resolution = attn_resolution
        
        
    def forward(self, key: torch.Tensor, query: torch.Tensor, t: Optional[torch.Tensor], Weight_Mat: torch.Tensor):
        """
        Args:
            key (torch.Tensor):  source view feature  (b, k_c, source_resolution, source_resolution)
            query (torch.Tensor): intermedia UNet feature (b, q_c, attn_resolution, attn_resolution)
            t (Optional[torch.Tensor]): time step embedding (b, time_channels)
            Weight_Mat (torch.Tensor): epipolar weight matrix (b, attn_resolution**2, attn_resolution**2)

        """
        
        #print(self.attn_resolution)
        key = Source_feature_Downsample(key.shape[2], self.attn_resolution, key)    # (b, f_c, attn_resolution, attn_resolution)
        
        scale = (key.shape[2] * key.shape[3]) / Weight_Mat.shape[1]
        Weight_Mat = Weight_Mat.unsqueeze(0)        
        Weight_Mat = F.interpolate(Weight_Mat, scale_factor=scale, mode='bilinear')#(attn_resolution**2, attn_resolution**2)
        Weight_Mat = Weight_Mat.squeeze(0)                          #(attn_resolution**2, attn_resolution**2)
        #print(Weight_Mat.shape)
        key = self.key_proj(key)    # (b, q_c, attn_resolution, attn_resolution)
        # version 1
        # # Common Cross Attention
        # _ = t
        # batch_size, n_channels, height, width = key.shape
        # # Change `x` to shape `[batch_size, seq, n_channels]`
        # k, q, v = key, query, key       # (B, C, H, W)
        # Affinity_Mat = epipolar_Affinity_Mat(k, q)
        
        # B,C, HW, HW = Affinity_Mat.shape
        # Weight_Mat = Weight_Mat.unsqueeze(0).unsqueeze(0).expand(B, C, HW, HW)
        # Affinity_Mat = torch.dot(Affinity_Mat, Weight_Mat)
        
        
        # v = v.view(batch_size, n_channels, -1)      # (B, C, HW)
        # Affinity_Mat = self.softmax(Affinity_Mat)   # (B, C, HW, HW), v: (B, C, H, W)
        # attn = torch.einsum('bijk,bik->bij', Affinity_Mat, v)
        # res = attn.softmax(dim=2)
        
        # # Change to shape `[batch_size, in_channels, height, width]`
        # res = res.view(batch_size, n_channels, height, width)
        # # res operation
        # res += query
        # return res   
        
        # version 2
        b, q_channels, *q_spatial = query.shape
        b, k_channels, *k_spatial = key.shape
        
        key = key.view(b, k_channels, -1)   # (b, k_c, attn_resolution**2)
        query = query.view(b, q_channels, -1)   # (b, q_c, attn_resolution**2)
        query = self.norm(query)
        key = self.norm(key)
        
        qkv_input = torch.cat((query, key), dim=1)
        qkv = self.qkv(qkv_input)
        h = self.attention(qkv, Weight_Mat)   # (b, q_c, attn_resolution**2
        h = self.proj_out(h)
        return (query + h).reshape(b, q_channels, *q_spatial)
    
class epipolar_MiddleBlock(nn.Module):
    """
    Middle block of the UNet
    resBlock + self-attention + epipolar attention
        
    """
    def __init__(self, n_channels: int, time_channels: int, query_channels: int,
                 source_resolution: int, attn_resolution: int):
        super().__init__()
        
        self.source_resolution = source_resolution
        self.attn_resolution = attn_resolution
        self.n_channels = n_channels
        self.time_channels = time_channels
        
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.self_attn = Self_AttentionBlock(n_channels, n_channels, n_channels)
        self.epipolar_attn = epipolar_Attention_Block(query_channels=query_channels, key_channels=n_channels, 
                                                      n_heads = 1, attn_resolution=attn_resolution)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, f: torch.Tensor, Weight_Mat: torch.Tensor):
        x = self.res1(x, t)
        x = self.self_attn(x)
        x = self.epipolar_attn(key=f, query=x, t=t, Weight_Mat = Weight_Mat)
        return x


class epipolar_DownBlock(nn.Module):
    """
     Downsample + self-attention + epipolar attention
    """
    def __init__(self, n_channels: int, out_channels: int, time_channels: int, query_channels: int,
                 source_resolution: int, attn_resolution: int):
        super().__init__()
        self.source_resolution = source_resolution
        self.attn_resolution = attn_resolution
        self.n_channels = n_channels
        self.time_channels = time_channels
        
        self.res1 = ResidualBlock(n_channels, out_channels, time_channels)
        self.Downsample = Downsample()
        self.self_attn = Self_AttentionBlock(n_channels, n_channels, n_channels)
        self.epipolar_attn = epipolar_Attention_Block(query_channels=query_channels, key_channels=n_channels, 
                                                      n_heads = 1, attn_resolution=attn_resolution)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, f: torch.Tensor, Weight_Mat: torch.Tensor):
        x = self.res1(x, t)
        x = self.Downsample(x)
        x = self.self_attn(x)
        x = self.epipolar_attn(key=f, query=x, t=t, Weight_Mat = Weight_Mat)
        return x

class epipolar_UpBlock(nn.Module):
    """
    Upsample + self-attention + epipolar attention
    """
    def __init__(self, n_channels: int, out_channels: int, time_channels: int, query_channels: int,
                 source_resolution: int, attn_resolution: int):
        super().__init__()
        self.source_resolution = source_resolution
        self.attn_resolution = attn_resolution
        self.n_channels = n_channels
        self.time_channels = time_channels
        
        self.res1 = ResidualBlock(n_channels, out_channels, time_channels)
        self.Upsample = Upsample()
        self.self_attn = Self_AttentionBlock(n_channels, n_channels, n_channels)
        self.epipolar_attn = epipolar_Attention_Block(query_channels=query_channels, key_channels=n_channels, 
                                                      n_heads = 1, attn_resolution=attn_resolution)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, f: torch.Tensor, Weight_Mat: torch.Tensor):
        x = self.res1(x, t)
        x = self.Upsample(x)
        x = self.self_attn(x)
        x = self.epipolar_attn(key=f, query=x, t=t, Weight_Mat = Weight_Mat)
        return x

##############################################
#       Original Epipolar Unet Model         #
##############################################


class epipolar_attn_Unet(nn.Module):
    """
    Unet model described in posed guided diffusion
    
    Args:
       in_channels (int): number of channels of input image
       latten_c (int): number of channels in the first layer
       query_channels (int): number of channels of source view feature map
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
                 query_channels = 3,
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
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, query_channels, source_resolution, attn_resolution[0]),
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, query_channels, source_resolution, attn_resolution[0]),
            epipolar_DownBlock(latten_c * channel_multiples[0], latten_c * channel_multiples[1], self.time_channels, query_channels, source_resolution, attn_resolution[1]),
        )

        self.DownBlock_3 = nn.Sequential(
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, query_channels, source_resolution, attn_resolution[1]),
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, query_channels, source_resolution, attn_resolution[1]),
            epipolar_DownBlock(latten_c * channel_multiples[1], latten_c * channel_multiples[2], self.time_channels, query_channels, source_resolution, attn_resolution[2]),
        )
        
        
        # middle block
        self.middle_process = nn.Sequential(
            epipolar_MiddleBlock(latten_c * channel_multiples[2], self.time_channels, query_channels, source_resolution, attn_resolution[2]),
            epipolar_MiddleBlock(latten_c * channel_multiples[2], self.time_channels, query_channels, source_resolution, attn_resolution[2]),
            epipolar_MiddleBlock(latten_c * channel_multiples[2], self.time_channels, query_channels, source_resolution, attn_resolution[2]),
        )
        
        # decoder
        self.UpBlock_1 = nn.Sequential(
            epipolar_UpBlock(latten_c * channel_multiples[2], latten_c * channel_multiples[1], self.time_channels, query_channels, source_resolution, attn_resolution[1]),
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, query_channels, source_resolution, attn_resolution[1]),
            epipolar_MiddleBlock(latten_c * channel_multiples[1], self.time_channels, query_channels, source_resolution, attn_resolution[1]),
        )
        
        self.UpBlock_2 = nn.Sequential(
            epipolar_UpBlock(latten_c * channel_multiples[1], latten_c * channel_multiples[0], self.time_channels, query_channels, source_resolution, attn_resolution[0]),
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, query_channels, source_resolution, attn_resolution[0]),
            epipolar_MiddleBlock(latten_c * channel_multiples[0], self.time_channels, query_channels, source_resolution, attn_resolution[0]),
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
        
        Weight_Mat_1 = epipolar_weight_Mat(self.attn_resolution[0], K, R, t)
        Weight_Mat_2 = epipolar_weight_Mat(self.attn_resolution[1], K, R, t)
        Weight_Mat_3 = epipolar_weight_Mat(self.attn_resolution[2], K, R, t)
        
        
        # encoder
        x1 = self.DownBlock_1(x, step_embedding, f)
        x2 = self.DownBlock_2(x1, step_embedding, f, Weight_Mat_1)
        x3 = self.DownBlock_3(x2, step_embedding, f, Weight_Mat_2)
        
        # middle
        x4 = self.middle_process(x3, step_embedding, f, Weight_Mat_3)
        
        # decoder
        x5 = torch.cat([x4, x3], dim=1)
        x5 = self.UpBlock_1(x5, step_embedding, f, Weight_Mat_2)
        
        x6 = torch.cat([x5, x2], dim=1)
        x6 = self.UpBlock_2(x6, step_embedding, f, Weight_Mat_1)
        
        x7 = torch.cat([x6, x1], dim=1), 
        x7 = self.UpBlock_3(x7, step_embedding, f)
        
        x7 = self.dropout(x7)
        res =  self.output_conv(self.act(self.norm(x)))
        
        return res
    
    
##############################################
#    modified Epipolar Unet Model            #
##############################################

class pretrained_epipolar_Unet(UNetModel):
    """
    We modified epippoar attention unet model to use pretrained UNet 
    Note that modified model is slightly different from original model from paper
    
    The model is based on https://github.com/openai/guided-diffusion
    """
    def __init__(self,
        image_size,
        in_channels,
        feature_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        epipolar_distance_threshold = 0.5):
        super(pretrained_epipolar_Unet, self).__init__(image_size, in_channels, model_channels, 
                                                       out_channels, num_res_blocks, attention_resolutions, 
                                                       dropout, channel_mult, conv_resample, dims, num_classes, 
                                                       use_checkpoint, use_fp16, num_heads, num_head_channels, 
                                                       num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order)
        self._feature_channels = feature_channels
        self.epipolar_distance_threshold = epipolar_distance_threshold
        
        self.down_epipolar_attn_list = nn.ModuleList()
        for idx, mul in enumerate(channel_mult[1:]):
            for i in range(num_res_blocks):
                attn = epipolar_Attention_Block(
                    query_channels=model_channels * mul,
                    key_channels = self._feature_channels,
                    num_heads= num_heads,
                    num_head_channels=num_head_channels,
                    attn_resolution= int(image_size / attention_resolutions[idx])
                )
                self.down_epipolar_attn_list.append(attn)
    
        
        self.up_epipolar_attn_list = nn.ModuleList()
        
        for idx, mul in enumerate(reversed(channel_mult[1:])):
            for i in range(num_res_blocks + 1):
                attn = epipolar_Attention_Block(
                    query_channels=model_channels * mul,
                    key_channels = self._feature_channels,
                    num_heads= num_heads,
                    num_head_channels=num_head_channels,
                    attn_resolution= int(image_size /  attention_resolutions[-idx-1])
                )
                self.up_epipolar_attn_list.append(attn)
        
        self.middle_epipolar_attn_list = nn.ModuleList()
        self.middle_epipolar_attn_list.append(
            epipolar_Attention_Block(
                query_channels=model_channels * channel_mult[-1],
                key_channels = self._feature_channels,
                num_heads= num_heads,
                num_head_channels=num_head_channels,
                attn_resolution= int(image_size /  attention_resolutions[-1])
            )
        )
        # self.middle_block_layers = list(self.middle_block.children())
        # part1 = self.middle_block_layers[:2]
        # part2 = self.middle_block_layers[2:]
        
        
    #def forward(self, x, timesteps, f, Weight_Mat, y=None):
    def forward(self, x, timesteps, f, Weight_Mat):
        """
        Apply the model to an input batch.
        Arglist = [timesteps, f, Weight_Mat]
        
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param Weight_Mat: an [N x HW x HW] Tensor of weight matrix
        :param f: feature view embeddings
        
        :return: an [N x C x ...] Tensor of outputs.
        """
        #assert (y is not None) == (
        #    self.num_classes is not None
        #), "must specify y if and only if the model is class-conditional"
        #f, Weight_Mat = y['f'], y['Weight_Mat']
        
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        #if self.num_classes is not None:
        #    assert y.shape == (x.shape[0],)
        #    emb = emb + self.label_emb(y)
  
        h = x.type(self.dtype)
    
        #encoder:
        down_attn_idx = 0
        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if len(module) > 1:
                # print(h.shape)
                h = self.down_epipolar_attn_list[down_attn_idx](key = f, query = h, t = timesteps, Weight_Mat= Weight_Mat)
                down_attn_idx += 1
            hs.append(h)
            #print(idx, ':', h.shape,"len: of down block", len(module))
        
        #middle block
        middle_part1 = self.middle_block[:2]
        middle_part2 = self.middle_block[2:]
        h = middle_part1(h, emb)
        h = self.middle_epipolar_attn_list[0](key = f, query = h, t = timesteps, Weight_Mat = Weight_Mat)
        h = middle_part2(h, emb)
        
        up_attn_idx = 0
        for idx, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            if len(module) == 2:
                h = module(h, emb)
                h = self.up_epipolar_attn_list[up_attn_idx](key = f, query = h, t = timesteps, Weight_Mat = Weight_Mat)
                up_attn_idx += 1
            elif len(module) == 3:
                part1 = module[:2]
                part2 = module[2:]
                h = part1(h, emb)
                h = self.up_epipolar_attn_list[up_attn_idx](key = f, query = h, t = timesteps, Weight_Mat = Weight_Mat)
                up_attn_idx += 1
                h = part2(h, emb)
            else:
                h = module(h, emb)
            #print(idx, ':', h.shape,"len:", len(module))
                
            
        h = h.type(x.dtype)
        return self.out(h)