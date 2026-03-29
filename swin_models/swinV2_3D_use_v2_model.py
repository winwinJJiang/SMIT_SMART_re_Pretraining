""" Swin Transformer V2
A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/abs/2111.09883

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import math
from typing import Callable, Optional, Tuple, Union
import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from functools import partial
from timm.models.layers import  Mlp, DropPath, to_2tuple, trunc_normal_, _assert, ClassifierHead,to_3tuple
from eva.modeling_finetune import LayerNormWithForceFP32

#from ._features_fx import register_notrace_function

import torch.nn.functional as nnf
#Jue added and changed
from timm.models.registry import register_model#,generate_default_cfgs
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from .format import Format, nchw_to
import numpy as np 
_int_or_tuple_2_t = Union[int, Tuple[int, int]]

# Functions we want to autowrap (treat them as leaves)
_autowrap_functions = set()
def register_notrace_function(func: Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    _autowrap_functions.add(func)
    return func

# def window_partition(x, window_size: Tuple[int, int]):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
#     return windows

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    #print (' in windw_partition x size',x.shape)
    #print ('window size ',window_size)
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)
    #x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    #print ('info: after window_parti window size ',windows.shape)
    return windows



# def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
#     """
#     Args:
#         windows: (num_windows * B, window_size[0], window_size[1], C)
#         window_size (Tuple[int, int]): Window size
#         img_size (Tuple[int, int]): Image size

#     Returns:
#         x: (B, H, W, C)
#     """
#     H, W = img_size
#     C = windows.shape[-1]
#     x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
#     return x

@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, H, W, L, -1)
    return x


# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#         pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
#     """

#     def __init__(
#             self,
#             dim,
#             window_size,
#             num_heads,
#             qkv_bias=True,
#             attn_drop=0.,
#             proj_drop=0.,
#             pretrained_window_size=[0, 0],
#     ):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.pretrained_window_size = pretrained_window_size
#         self.num_heads = num_heads

#         self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

#         # mlp to generate continuous relative position bias
#         self.cpb_mlp = nn.Sequential(
#             nn.Linear(2, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, num_heads, bias=False)
#         )

#         # get relative_coords_table
#         relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
#         relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
#         relative_coords_table = torch.stack(torch.meshgrid([
#             relative_coords_h,
#             relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
#         if pretrained_window_size[0] > 0:
#             relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
#         else:
#             relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
#         relative_coords_table *= 8  # normalize to -8, 8
#         relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
#             torch.abs(relative_coords_table) + 1.0) / math.log2(8)

#         self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index, persistent=False)

#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         if qkv_bias:
#             self.q_bias = nn.Parameter(torch.zeros(dim))
#             self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
#             self.v_bias = nn.Parameter(torch.zeros(dim))
#         else:
#             self.q_bias = None
#             self.k_bias = None
#             self.v_bias = None
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask: Optional[torch.Tensor] = None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv_bias = None
#         if self.q_bias is not None:
#             qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
#         qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#         qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)

#         # cosine attention
#         attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
#         logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
#         attn = attn * logit_scale

#         relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
#         relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             num_win = mask.shape[0]
#             attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
            pretrained_window_size=[0, 0],
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_t = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)


        relative_coords_table = torch.stack(torch.meshgrid([
            relative_coords_h,
            relative_coords_w,
            relative_coords_t])).permute(1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1,2*Wt-1, 3

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)

        relative_coords_table *= 8  # normalize to -8, 8

        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt

        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        # ('info: relative_position_index ',relative_position_index.shape)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        # ('info: x size ',x.size())
        #print ('self.qkv.weight size ',self.qkv.weight.shape)
        #print ('qkv_bias size ',qkv_bias.shape)
        #print ('mask si ',mask)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        #print (q)
        # cosine attention
        q=q.float()
        k=k.float()
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        #print (attn)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        # ('self.relative_coords_table size ',self.relative_coords_table.shape)
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww,Wh*Ww,nH
            
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        attn=attn.half()
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerV2Block(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size.
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            pretrained_window_size: Window size in pretraining.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = to_3tuple(input_resolution)
        self.num_heads = num_heads
        ws, ss = self._calc_window_shift(window_size, shift_size)
        self.window_size: Tuple[int, int,int] = ws
        self.shift_size: Tuple[int, int,int] = ss


        self.window_area = self.window_size[0] * self.window_size[1]* self.window_size[2]

        self.mlp_ratio = mlp_ratio

        self.attn = WindowAttention(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_3tuple(pretrained_window_size),
        )

        #self.norm1 = norm_layer(dim)

        norm_layer1=partial(LayerNormWithForceFP32, eps=1e-6)
        self.norm1 = norm_layer1(dim)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        #self.norm2 = norm_layer(dim)

        norm_layer2=partial(LayerNormWithForceFP32, eps=1e-6)
        self.norm2 = norm_layer2(dim)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if any(self.shift_size):
            # calculate attention mask for SW-MSA

            print ('warning: computing shift size ')
            H, W,T = self.input_resolution

            Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
            Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
            Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]

            img_mask = torch.zeros((1, Hp, Wp, Tp, 1))  # 1 Hp Wp 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            t_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for t in t_slices:
                        img_mask[:, h, w, t, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _calc_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_3tuple(target_window_size)
        target_shift_size = to_3tuple(target_shift_size)
        #print ('info: target_window_size is ',target_window_size)
        #print ('info: target_shift_size is ',target_shift_size)

        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        #print ('info: here window_size si ',window_size)
        #print ('info: here shift_size si ',shift_size)

        return tuple(window_size), tuple(shift_size)

    def _attn(self, x):
        B, H, W,T, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        #print ('info: x_windows size ',x_windows.shape)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2])  # B H' W' C

        
        # reverse cyclic shift
        if has_shift:
            #x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        return x

    

    def forward(self, x):
        B, H, W, T,C = x.shape
        #print ('info: x size ',x.shape)
        #print ('info: self._attn(x) size ',self._attn(x).shape)
        x = x + self.drop_path1(self.norm1(self._attn(x)))
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = x.reshape(B, H, W,T, C)
        return x


# class PatchMerging(nn.Module):
#     """ Patch Merging Layer.
#     """

#     def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
#         """
#         Args:
#             dim (int): Number of input channels.
#             out_dim (int): Number of output channels (or 2 * dim if None)
#             norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#         """
#         super().__init__()
#         self.dim = dim
#         self.out_dim = out_dim or 2 * dim
#         self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
#         self.norm = norm_layer(self.out_dim)

#     def forward(self, x):
#         B, H, W, C = x.shape
#         _assert(H % 2 == 0, f"x height ({H}) is not even.")
#         _assert(W % 2 == 0, f"x width ({W}) is not even.")
#         x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
#         x = self.reduction(x)
#         x = self.norm(x)
#         return x

#V2 use BHWC input
class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(8 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):

        B, H, W, T,C = x.shape
        #print ('error: before patch merging size ',x.shape)
        _assert(H % 2 == 0, f"x height ({H}) is not even.")
        _assert(W % 2 == 0, f"x width ({W}) is not even.")
        _assert(T % 2 == 0, f"x width ({T}) is not even.")
        x = x.reshape(B, H // 2, 2, W // 2, 2, T//2, 2, C).permute(0, 1, 3,5, 4, 2, 6,7).flatten(4)
        #print ('error x afer resahpe ',x.shape)
        x = self.reduction(x)
        x = self.norm(x)
        #print ('error: after patch merging size ',x.shape)
        B, H, W, T,C = x.shape
        #x=x.view(B,-1,C)
        #x_out=torch.zeros(B, H, W, T,C ).cuda().half()
        return x#x_out
    

class SwinTransformerV2Stage(nn.Module):
    """ A Swin Transformer V2 Stage.
    """

    def __init__(
            self,
            dim,
            out_dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            downsample=False,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0,
            output_nchw=False,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop: Dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


# class SwinTransformerV2(nn.Module):
#     """ Swin Transformer V2

#     A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
#         - https://arxiv.org/abs/2111.09883
#     """

#     def __init__(
#             self,
#             img_size: _int_or_tuple_2_t = 224,
#             patch_size: int = 4,
#             in_chans: int = 3,
#             num_classes: int = 1000,
#             global_pool: str = 'avg',
#             embed_dim: int = 96,
#             depths: Tuple[int, ...] = (2, 2, 6, 2),
#             num_heads: Tuple[int, ...] = (3, 6, 12, 24),
#             window_size: _int_or_tuple_2_t = 7,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             drop_rate: float = 0.,
#             attn_drop_rate: float = 0.,
#             drop_path_rate: float = 0.1,
#             norm_layer: Callable = nn.LayerNorm,
#             pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
#             **kwargs,
#     ):
#         """
#         Args:
#             img_size: Input image size.
#             patch_size: Patch size.
#             in_chans: Number of input image channels.
#             num_classes: Number of classes for classification head.
#             embed_dim: Patch embedding dimension.
#             depths: Depth of each Swin Transformer stage (layer).
#             num_heads: Number of attention heads in different layers.
#             window_size: Window size.
#             mlp_ratio: Ratio of mlp hidden dim to embedding dim.
#             qkv_bias: If True, add a learnable bias to query, key, value.
#             drop_rate: Dropout rate.
#             attn_drop_rate: Attention dropout rate.
#             drop_path_rate: Stochastic depth rate.
#             norm_layer: Normalization layer.
#             patch_norm: If True, add normalization after patch embedding.
#             pretrained_window_sizes: Pretrained window sizes of each layer.
#             output_fmt: Output tensor format if not None, otherwise output 'NHWC' by default.
#         """
#         super().__init__()

#         self.num_classes = num_classes
#         assert global_pool in ('', 'avg')
#         self.global_pool = global_pool
#         self.output_fmt = 'NHWC'
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.feature_info = []

#         if not isinstance(embed_dim, (tuple, list)):
#             embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim[0],
#             norm_layer=norm_layer,
#             output_fmt='NHWC',
#         )

#         dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
#         layers = []
#         in_dim = embed_dim[0]
#         scale = 1
#         for i in range(self.num_layers):
#             out_dim = embed_dim[i]
#             layers += [SwinTransformerV2Stage(
#                 dim=in_dim,
#                 out_dim=out_dim,
#                 input_resolution=(
#                     self.patch_embed.grid_size[0] // scale,
#                     self.patch_embed.grid_size[1] // scale),
#                 depth=depths[i],
#                 downsample=i > 0,
#                 num_heads=num_heads[i],
#                 window_size=window_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[i],
#                 norm_layer=norm_layer,
#                 pretrained_window_size=pretrained_window_sizes[i],
#             )]
#             in_dim = out_dim
#             if i > 0:
#                 scale *= 2
#             self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]

#         self.layers = nn.Sequential(*layers)
#         self.norm = norm_layer(self.num_features)
#         self.head = ClassifierHead(
#             self.num_features,
#             num_classes,
#             pool_type=global_pool,
#             drop_rate=drop_rate,
#             input_fmt=self.output_fmt,
#         )

#         self.apply(self._init_weights)
#         for bly in self.layers:
#             bly._init_respostnorm()

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         nod = set()
#         for n, m in self.named_modules():
#             if any([kw in n for kw in ("cpb_mlp", "logit_scale", 'relative_position_bias_table')]):
#                 nod.add(n)
#         return nod

#     @torch.jit.ignore
#     def group_matcher(self, coarse=False):
#         return dict(
#             stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
#             blocks=r'^layers\.(\d+)' if coarse else [
#                 (r'^layers\.(\d+).downsample', (0,)),
#                 (r'^layers\.(\d+)\.\w+\.(\d+)', None),
#                 (r'^norm', (99999,)),
#             ]
#         )

#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         for l in self.layers:
#             l.grad_checkpointing = enable

#     @torch.jit.ignore
#     def get_classifier(self):
#         return self.head.fc

#     def reset_classifier(self, num_classes, global_pool=None):
#         self.num_classes = num_classes
#         self.head.reset(num_classes, global_pool)

#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         x = self.layers(x)
#         x = self.norm(x)
#         return x

#     def forward_head(self, x, pre_logits: bool = False):
#         return self.head(x, pre_logits=True) if pre_logits else self.head(x)

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.forward_head(x)
#         return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1],img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1]* self.grid_size[2]
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W,T = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        B,h,w,t,C=x.shape
        #x=x.view(B,-1,C)
        #print ('info: after patch emb ',x.shape)
        return x.view(B,-1,C)#,x

class SwinTransformerV2_MIM(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer stage (layer).
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            patch_norm: If True, add normalization after patch embedding.
            pretrained_window_sizes: Pretrained window sizes of each layer.
            output_fmt: Output tensor format if not None, otherwise output 'NHWC' by default.
        """
        super().__init__()

        self.num_classes = num_classes
        assert global_pool in ('', 'avg')
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        #self.mask_token = nn.Parameter(torch.zeros(2, 262144,  self.embed_dim))    
        
        #self.mask_token = torch.zeros(1, 1, self.embed_dim).cuda().half()   
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt='NHWC',
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerV2Stage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_sizes[i],
            )]

            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]

        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue

        self.encoder_stride=16
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1],out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]
        self.pt_size=self.encoder_stride
        self.feat_size= [int(img_size/self.encoder_stride),int(img_size/self.encoder_stride),int(img_size/self.encoder_stride)]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = set()
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale", 'relative_position_bias_table')]):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def proj_feat(self, x, hidden_size, feat_size):
        
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    
    def forward_features(self, x,mask):
        print ('mask size ',mask.shape)

        B, nc, w, h, t = x.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x)

        #x = self.patch_embed(x)

        #print ('info: after patch_embed x size ',x_reshape.shape)

        # For V1 use only
        """ x=x_reshape
        for i in range(self.num_layers):
            #print ('stage ',i)
            layer = self.layers[i]
            print ('*'*50)
            print (i)
            print ('before X size is ', x.size())
            x= layer(x)
            print ('after X size is ', x.size()) """

        B,L,C=x_reshape.shape

        H=round(math.pow(L,1/3.))
        W=H 
        T=H 


        


        mask_tokens = self.mask_token.expand(B, L, -1)
        #print (mask_tokens)
        #print ('mask size is ',mask.size())
        
        
        #print ('mask size is ',mask.size())
        #print ('x_3 size ',x_3.shape)
        
        #print ('mask_tokens size ',mask_tokens.shape)

        
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size ',w.shape)

        #print ('w size ',w.shape)
        #print ('x_reshape size ',x_reshape.shape)
        #print ('mask_tokens size ',mask_tokens.shape)

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        x_reshape=x_reshape.view(B,H,W,T,C)

        x = self.layers(x_reshape)
        #print (x)
        x = self.norm(x)
        #print (x)
        return x



    def forward(self, x_in,mask):
        x_for_rec = self.forward_features(x_in,mask)

        #print ('x_for_rec is ')
        #print(x_for_rec)
        #print ('before proj x size ',x_for_rec.size())
        x_for_rec=self.proj_feat(x_for_rec, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x_for_rec.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x_for_rec)
        
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 

        #print ('x_rec size ',x_rec.shape)
        
        
        #mask = mask.repeat_interleave(self.pt_size, 1).repeat_interleave(self.pt_size, 2).repeat_interleave(self.pt_size, 3).unsqueeze(1).contiguous()


        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        #print ('mask size ',mask.shape)

        x_in=x_in.float()
        x_rec=x_rec.float()
        #mask=mask.float()

        #print ('*'*50)
        #print(x_in)
        #print ('*'*50)
        #print(x_rec)
        #print ('*'*50)
        #print(mask)

        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        #loss_recon = F.l1_loss(x_in, x_rec)

        #print(loss_recon)
        

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)

        loss=loss.half()


        return loss, x_rec


def checkpoint_filter_fn(state_dict, model):
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    if 'head.fc.weight' in state_dict:
        return state_dict
    out_dict = {}
    import re
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'relative_coords_table')]):
            continue  # skip buffers that should not be persistent
        k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
        k = k.replace('head.', 'head.fc.')
        out_dict[k] = v
    return out_dict

 
def swinv2_3D_tiny_window4_96(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM(
       img_size=128,window_size=4, embed_dim=48,patch_size=2, depths=(2,2,18,2), num_heads=(4,4,8,16),qkv_bias=True,in_chans=1, **kwargs)


    
    return model
