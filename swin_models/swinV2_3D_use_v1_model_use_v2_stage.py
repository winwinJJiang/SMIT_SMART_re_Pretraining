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
from monai.losses import DiceLoss, DiceCELoss
from functools import partial
from timm.models.layers import  Mlp, DropPath, to_2tuple, trunc_normal_, _assert, ClassifierHead,to_3tuple
from eva.modeling_finetune import Block, _cfg, PatchEmbed, RelativePositionBias, DecoupledRelativePositionBias, LayerNormWithForceFP32

#from ._features_fx import register_notrace_function

import torch.nn.functional as nnf
#Jue added and changed
from timm.models.registry import register_model#,generate_default_cfgs
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from .format import Format, nchw_to
import numpy as np 
__all__ = ['SwinTransformerV2']  # model_registry will add each entrypoint fn to this

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


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     Args:
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, img_size,patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         patch_size = to_3tuple(patch_size)
#         img_size = to_3tuple(img_size)
#         self.patch_size = patch_size

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])

#     def forward(self, x):
#         """Forward function."""
#         # padding
#         _, _, H, W, T = x.size()
#         if W % self.patch_size[1] != 0:
#             x = nnf.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
#         if H % self.patch_size[0] != 0:
#             x = nnf.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
#         if T % self.patch_size[0] != 0:
#             x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

#         x = self.proj(x)  # B C Wh Ww Wt
#         if self.norm is not None:
#             Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
#             x = x.flatten(2).transpose(1, 2)
#             x = self.norm(x)
#             x_reshape = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
#         print (x_reshape.shape)
#         #x_reshape=torch.permute(x_reshape, (0,2,3,4,1))#.transpose
#         print (x_reshape.shape)
#         return x_reshape


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
        x=x.view(B,-1,C)
        #print ('info: after patch emb ',x.shape)
        return x
    
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


class WindowAttention_old(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)

        #print ('self.qkv ',self.qkv.weight.shape )
        #print ('x ',x.shape )

        d=self.qkv(x)
        #print ('error d shape ',d.shape) # ([1024, 64, 576]


        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        attn=attn.half()
        v=v.half()
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

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



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        #self.window_size=(4,4,4)
        #print ('min(self.shift_size ',self.shift_size)

        #print ('min(self.window_size) ',window_size )

        
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        norm_layer1=partial(LayerNormWithForceFP32, eps=1e-6)
        self.norm1 = norm_layer1(dim)

        #self.norm1 = norm_layer(dim)
        #self.attn = WindowAttention_old(
        #    dim, window_size=self.window_size, num_heads=num_heads,
        #    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        norm_layer2=partial(LayerNormWithForceFP32, eps=1e-6)
        self.norm2 = norm_layer2(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix=None):
        H, W, T = self.H, self.W, self.T
        # ('x shape', x.shape)
        #print ('H', self.H)

        B, L, C = x.shape

        
        #assert L == H * W * T, "input feature has wrong size"

        H=round(math.pow(L,1/3.))
        W=H 
        T=H 
        shortcut = x
        
        x = x.view(B, H, W, T, C)
        #print ('x size is ',x.size())
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x = self.norm1(x)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x
    

class SwinTransformerBlock_Change_Norm(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.window_size=(4,4,4)
        #print ('min(self.shift_size ',self.shift_size)

        #print ('min(self.window_size) ',window_size )

        
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim)
        #self.attn = WindowAttention_old(
        #    dim, window_size=self.window_size, num_heads=num_heads,
        #    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix=None):
        H, W, T = self.H, self.W, self.T
        # ('x shape', x.shape)
        #print ('H', self.H)

        B, L, C = x.shape

        
        #assert L == H * W * T, "input feature has wrong size"

        H=round(math.pow(L,1/3.))
        W=H 
        T=H 
        shortcut = x
        
        x = x.view(B, H, W, T, C)
        #print ('x size is ',x.size())
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = self.norm1(x)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))

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

        #self.attn = WindowAttention_old(
        #    dim,
        #    window_size=to_3tuple(self.window_size),
        #    num_heads=num_heads,
        #    qkv_bias=qkv_bias,
        #    attn_drop=attn_drop,
        #    proj_drop=drop,
            
        #)

        #
        norm_layer1=partial(LayerNormWithForceFP32, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        norm_layer2=partial(LayerNormWithForceFP32, eps=1e-6)
        self.norm2 = norm_layer2(dim)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if any(self.shift_size):
            # calculate attention mask for SW-MSA

            
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
        

        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]


        return tuple(window_size), tuple(shift_size)

    def _attn(self, x):
        """ B, H, W,T, C = x.shape

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
            x = shifted_x """

        B, H, W,T, C = x.shape
        #print ('x size is ',x.size())
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            #attn_mask = mask_matrix
        else:
            shifted_x = x
            #attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        return x

    def forward(self, x):
        # print ('info 0 : x size ',x.shape)
        # B_,L_,C_=x.shape
        # H_=round(math.pow(L_,1/3.))
        # W_=H_
        # T_=H_

        # x=self.norm1(x)
        # x=x.view(B_,H_,W_,T_,C_)

        # B, H, W, T,C = x.shape
        # print ('info: x size ',x.shape)
        # print ('info: self._attn(x) size ',self._attn(x).shape)
        # x = x + self.drop_path1(self._attn(x))
        # x = x.reshape(B, -1, C)

        # x=self.norm2(x)
        # x = x + self.drop_path2(self.mlp(x))
        #x = x.reshape(B, H, W,T, C)
        #print ('info: out block x size is ',x.shape)


        #H, W, T = self.H, self.W, self.T
        # ('x shape', x.shape)
        #print ('H', self.H)

        B, L, C = x.shape

        B_,L_,C_=x.shape
        H_=round(math.pow(L_,1/3.))
        W_=H_
        T_=H_

        #assert L == H * W * T, "input feature has wrong size"

        H=round(math.pow(L,1/3.))
        W=H 
        T=H 
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)
        #print ('x size is ',x.size())
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        #self.reduction = nn.Linear(8 * dim, (4//reduce_factor) * dim, bias=False)
        #self.reduction = nn.Linear(8 * dim, (4//reduce_factor) * dim, bias=False)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer( 2*dim)


    def forward(self, x):
        """
        x: B, H*W*T, C
        """
        # ('x before merge ',x.shape)
        B, L, C = x.shape

        H=round(math.pow(L,1/3.))
        W=H
        T=H
        #assert L == H * W * T, "input feature has wrong size"
        #assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)
        #print ('x reshape  ',x.shape)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C
        #print ('self.dim is ',self.dim)
        #print ('error x2 before reduction resahpe ',x.shape)
        x = self.reduction(x)
        #print ('error x3 after reduction  resahpe ',x.shape)
        x = self.norm(x)
        #print ('x after merge ',x.shape)  #B,L,C

        return x
    

#V2 use BHWC input
class PatchMerging_V2(nn.Module):
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
        print ('error: before patch merging size ',x.shape)
        _assert(H % 2 == 0, f"x height ({H}) is not even.")
        _assert(W % 2 == 0, f"x width ({W}) is not even.")
        _assert(T % 2 == 0, f"x width ({T}) is not even.")
        x = x.reshape(B, H // 2, 2, W // 2, 2, T//2, 2, C).permute(0, 1, 3,5, 4, 2, 6,7).flatten(4)
        #print ('error x afer resahpe ',x.shape)
        x = self.reduction(x)
        x = self.norm(x)
        #print ('error: after patch merging size ',x.shape)
        B, H, W, T,C = x.shape
        x=x.view(B,-1,C)
        #x_out=torch.zeros(B, H, W, T,C ).cuda().half()
        return x#x_out
    

#V2 use BLC input 
class PatchMerging_V2_2(nn.Module):
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
        self.reduction = nn.Linear(4* dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        
        B, L,C = x.shape
        H=round(math.pow(L,1/3.))
        W=H
        T=H
        # ('error: before patch merging size ',x.shape)
        
        x = x.reshape(B, H // 2, 2, W // 2, 2, T//2, 2, C).permute(0, 1, 3,5, 4, 2, 6,7).flatten(4)
        #print ('error x afer resahpe ',x.shape)
        x = self.reduction(x)
        x = self.norm(x)
        #print ('error: after patch merging size ',x.shape)
        B, H, W, T,C = x.shape
        x=x.view(B,-1,C)
        #x_out=torch.zeros(B, H, W, T,C ).cuda().half()
        return x#x_out
    
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
#         self.reduction = nn.Linear(8 * dim, self.out_dim, bias=False)
#         self.norm = norm_layer(self.out_dim)

#     def forward(self, x):
#         B, H, W,T, C = x.shape
#         print ('info: before patch merging size ',x.shape)
#         _assert(H % 2 == 0, f"x height ({H}) is not even.")
#         _assert(W % 2 == 0, f"x width ({W}) is not even.")
#         _assert(T % 2 == 0, f"x Slice_T ({T}) is not even.")

#         #x = x.reshape(B, H // 2, 2, W // 2, 2,T // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)

#         # padding
#         pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
#         if pad_input:
#             x = nnf.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))

#         x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
#         x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
#         x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
#         x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
#         x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
#         x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
#         x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
#         x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
#         x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
#         x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

#         x = self.reduction(x)
#         x = self.norm(x)
#         print ('info: after patch merging size ',x.shape)
#         x= x.reshape(B, H // 2, W // 2, T//2, -1)
#         print ('info: after patch merging size ',x.shape)
#         return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 input_resolution=(1,1,1),
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf

        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution

        # build blocks
        self.blocks = nn.ModuleList([
             SwinTransformerBlock(
                 dim=dim,
                 num_heads=num_heads,
                 window_size=window_size,
                 shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop=drop,
                 attn_drop=attn_drop,
                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                 norm_layer=norm_layer,)
             for i in range(depth)])

        """ self.blocks = nn.ModuleList([
              SwinTransformerV2Block(
                  dim=dim,
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
                  #pretrained_window_size=pretrained_window_size,
              )
             for i in range(depth)]) """


        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        B,L,C=x.shape
        #x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        #Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        #h=math.cbrt(L)
        H=round(math.pow(L,1/3.))
        W=H
        T=H
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
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
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            #x_down = self.downsample(x, H, W, T)
            x_down = self.downsample(x)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x_down#x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x#, H, W, T, x, H, W, T

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

        

        # build blocks
        """ self.blocks = nn.ModuleList([
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
             for i in range(depth)]) """


        # build blocks
        qk_scale=None
        window_size=(window_size,window_size,window_size)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,)
            for i in range(depth)]) 
        
        # patch merging / downsample layer
        if downsample:
            #V2 use 
            #self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)

            #V1 use 
            self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer)

        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

    def forward(self, x):
        # ('error: current dim is ',self.dim)
        

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                #print (' error x size before stage blk is ',x.shape)

                B,L,C=x.shape
                #x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
                #Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
                #h=math.cbrt(L)
                h=round(math.pow(L,1/3.))
                #x = blk(x,h,h,h)
                x=blk(x)

        x = self.downsample(x)

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
#             img_size: _int_or_tuple_2_t = 96,
#             patch_size: int = 2,
#             in_chans: int = 1,
#             num_classes: int = 1000,
#             global_pool: str = 'avg',
#             embed_dim: int = 48,
#             depths: Tuple[int, ...] = (2, 2, 6, 2),
#             num_heads: Tuple[int, ...] = (3, 6, 12, 24),
#             window_size: _int_or_tuple_2_t = 5,
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
#                     self.patch_embed.grid_size[1] // scale,
#                     self.patch_embed.grid_size[2] // scale,),
                    

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


class PatchEmbed_For_SSIM(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        if T % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x_reshape = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x,x_reshape
    
class SwinTransformer_Unetr_Mask_In_Seperate_only_feature_out_CaiT_All_3_loss(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 Cait_layer=2,
                 pat_merg_rf=2,):
        super().__init__()
        
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))        
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed_For_SSIM(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            #self.pos_embd = SinPositionalEncoding3D(96).cuda()#SinusoidalPositionEmbedding().cuda()
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        

        
        # add a norm layer for each output
        #for i_layer in out_indices:
        #    layer = norm_layer(num_features[i_layer])
        #    layer_name = f'norm{i_layer}'
        #    self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x,mask,mask_flag):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())  # b,1,96,96,96
        x,x_ful_size = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())  # b,48,48,48,48
        #print ('after patch embd x_ful_size size is  ',x_ful_size.size())  # b,48,48,48,48
        assert mask is not None
        #print ('self.mask_token size ',self.mask_token.size())
        B, L, _ = x.shape
        _,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        #print ('mask size is ',mask.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #temperary comments for debug
        #print ('x size is ',x.size())
        #print ('self.mask_token size is ',mask_tokens.size())


        # before patch embd x size is  torch.Size([3, 1, 96, 96, 96])
        # after patch embd x size is   torch.Size([3, 110592, 48])
        # after patch embd x_ful_size size is   torch.Size([3, 48, 48, 48, 48])
        # mask size is  torch.Size([3, 48, 48, 48])
        # w size is  torch.Size([3, 110592, 1])
        # x size is  torch.Size([3, 110592, 48])
        # self.mask_token size is  torch.Size([3, 110592, 48])

        #print ('use mask_MIM is ',mask_flag)
        
        x = x * (1. - w) + mask_tokens * w
        
        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        #print ('there are steps numbers ',self.num_layers)
        #print ('before x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('*'*50)
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            #print ('**'*50)
            #print('layer number is ',i)
            #print ('before X size is ', x.size())
            #print (' i ',i)
            #print (' i ',x)
            x= layer(x)

            #x_out is the one 
            
            
            #print ('after x size is ', x.size())
            #print ('after x_out1  size is ', x_out.size())

            if i==2:
                x_feature=x
            #if i in self.out_indices:
            #    norm_layer = getattr(self, f'norm{i}')
            #    x_out = norm_layer(x_out)
            #    print ('x_out2 size ',x_out.size())
                
            #    out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                
            #    outs.append(out)
        #print ('info: bottle net X size is ', x.size())

        # at this point, x_feature is the image patchs of the 3rd block of swin, 
        # x is the output of the 4rd block of swin, we will work with x_feature 
        # 

        x4_last=x # the feature for image reconstruction
        x3_feature=x_feature # the feature for computing the CLS token 

        B_size = x3_feature.shape[0]
        

        

        
        return x4_last#x3_feature, x3_feature_w_cls_1


    def forward_w_att(self, x,mask,mask_flag):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())  # b,1,96,96,96
        x,x_ful_size = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())  # b,48,48,48,48
        #print ('after patch embd x_ful_size size is  ',x_ful_size.size())  # b,48,48,48,48
        assert mask is not None
        #print ('self.mask_token size ',self.mask_token.size())
        B, L, _ = x.shape
        _,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        #print ('mask size is ',mask.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #temperary comments for debug
        #print ('x size is ',x.size())
        #print ('self.mask_token size is ',mask_tokens.size())


        # before patch embd x size is  torch.Size([3, 1, 96, 96, 96])
        # after patch embd x size is   torch.Size([3, 110592, 48])
        # after patch embd x_ful_size size is   torch.Size([3, 48, 48, 48, 48])
        # mask size is  torch.Size([3, 48, 48, 48])
        # w size is  torch.Size([3, 110592, 1])
        # x size is  torch.Size([3, 110592, 48])
        # self.mask_token size is  torch.Size([3, 110592, 48])

        #print ('use mask_MIM is ',mask_flag)
        if mask_flag:
            x = x * (1. - w) + mask_tokens * w
            #print ('info: Use MIM')
        #else:
        #    print ('info: Not use MIM')

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        #print ('there are steps numbers ',self.num_layers)
        #print ('before x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('*'*50)
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            #print ('**'*50)
            #print('layer number is ',i)
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)

            #x_out is the one 
            
            
            #print ('after x size is ', x.size())
            #print ('after x_out1  size is ', x_out.size())

            if i==2:
                x_feature=x
            #if i in self.out_indices:
            #    norm_layer = getattr(self, f'norm{i}')
            #    x_out = norm_layer(x_out)
            #    print ('x_out2 size ',x_out.size())
                
            #    out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                
            #    outs.append(out)
        #print ('info: bottle net X size is ', x.size())

        # at this point, x_feature is the image patchs of the 3rd block of swin, 
        # x is the output of the 4rd block of swin, we will work with x_feature 
        # 

        x4_last=x # the feature for image reconstruction
        x3_feature=x_feature # the feature for computing the CLS token 

        B_size = x3_feature.shape[0]
        

        cls_tokens = self.cls_token.expand(B_size, -1, -1)  

        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens,atten_map = blk(x3_feature,cls_tokens)

        #warning: cls_token size is  torch.Size([16, 1, 384])
        #warning: x3_feature size is  torch.Size([16, 512, 384])
        #warning: x3_feature_w_cls size is  torch.Size([16, 513, 384])
        #warning: x3_feature_w_cls[:, 0] size is  torch.Size([16, 384])


        #print ('warning: cls_token size is ',cls_tokens.size()) #torch.Size([16, 1, 384])

        #print ('warning: x3_feature size is ',x3_feature.size())# torch.Size([16, 512, 384])

        #x3_feature_w_cls = torch.cat((cls_tokens, x3_feature), dim=1)

        #print ('warning: x3_feature_w_cls size is ',x3_feature_w_cls.size()) # torch.Size([16, 513, 384])
                
        #x3_feature_w_cls = self.norm_CaiT(x3_feature_w_cls)

        #x3_feature_w_cls_1=x3_feature_w_cls[:, 0] # torch.Size([16, 384])

        #print ('warning: x3_feature_w_cls[:, 0] size is ',x3_feature_w_cls_1.size())

        #x3_feature_w_cls_1 = self.head_Cait(x3_feature_w_cls_1)
        #return x3_feature,x4_last,x3_feature_w_cls_1

        # x3_feature for patch loss,  cls_token for CLS_loss, x4_last for image recons loss
        return x3_feature, cls_tokens, x4_last,atten_map#x3_feature, x3_feature_w_cls_1

    def get_atten_map(self,x):

        """Forward function."""
        
        x,x_ful_size = self.patch_embed(x)
        #
        B, L, _ = x.shape
        _,_,Wh,Ww,Wt=x_ful_size.shape
        

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        #print ('there are steps numbers ',self.num_layers)
        #print ('before x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('*'*50)
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            #print ('**'*50)
            #print('layer number is ',i)
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)

            #x_out is the one 
            
            
            #print ('after x size is ', x.size())
            #print ('after x_out1  size is ', x_out.size())

            if i==2:
                x_feature=x
            

        x4_last=x 
        x3_feature=x_feature

        B_size = x3_feature.shape[0]
        

        cls_tokens = self.cls_token.expand(B_size, -1, -1)  

        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens,atten_map = blk(x3_feature,cls_tokens)

        #print ('atten_map size ',atten_map.size())
        return atten_map


class Trans_SMIT_pre_train_cls_rec_Student(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=1,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = False,
        ibot_head_share: bool = True,

    ) -> None:
       
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        
        self.use_MIM_mask=True
        
        self.transformer = SwinTransformerV2_Cls_MIM(patch_size=config.patch_size, 
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        

        

        cls_norm=partial(nn.LayerNorm, eps=1e-5)

        self.norm_cls = cls_norm(768)
        self.norm_patch = cls_norm(384)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # for reconstruction loss use
        self.encoder_stride=32
        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x_in, mask):
        
        #print (x_in_all.size())
        #x_in=x_in_all[0]
        #mask=x_in_all[1]
        x_feature,x_last = self.transformer(x_in,mask,self.use_MIM_mask)
        #print ('info: x_last size ',x_last.size())

        #print ('info: x_feature size ',x_feature.size())

        x_region = self.norm_patch(x_feature)  # B L C
        x_last= self.norm_cls(x_last) 
        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        
        x_cls = torch.flatten(x_cls, 1)
        
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)

        #compute reconstruction 
        x_rec1=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec2 = self.decoder1(x_rec1)
        x_rec= rearrange(x_rec2, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 


       
        mask_intp = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask_intp).sum() / (mask_intp.sum() + 1e-5) 

        
        #print ('x_region_all size ',x_region_all.size())
        return x_region_all ,loss,x_rec

class Trans_SMIT_pre_train_cls_patch_rec_Student_CaiT_All_3_Loss(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=1,
        feature_size: int = 48,  #This is used for miccai paper 
        #feature_size: int = 96,   # This one now try to enlarge the feature size 

        hidden_size: int = 768, #This is used for miccai paper 
        #hidden_size: int = 1536, # This one now try to enlarge the feature size 
        
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = False,
        ibot_head_share: bool = True,

    ) -> None:
       
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        
        self.use_MIM_mask=True
        
        self.transformer = SwinTransformer_Unetr_Mask_In_Seperate_only_feature_out_CaiT_All_3_loss(patch_size=config.patch_size, 
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           Cait_layer=config.Cait_layer
                                           )
        

        

        cls_norm=partial(nn.LayerNorm, eps=1e-5)

        self.norm_cls = cls_norm(768) # used to miccai
        #self.norm_cls = cls_norm(1536) # used to enlarge the network
        
        self.norm_patch = cls_norm(384) # used to miccai
        #self.norm_patch = cls_norm(768) # used to enlarge the network

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # for reconstruction loss use
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1) # used to miccai
        #self.decoder1 = nn.Conv3d(1536,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1) # used to enlarge the network

        
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x_in, mask):
        
        #print (x_in_all.size())
        #x_in=x_in_all[0]
        #mask=x_in_all[1]
        x_last = self.transformer(x_in,mask,self.use_MIM_mask)

        'Rember here the features are not normalized so normalization is needed'
        'x_feature for patch loss and classfication loss'
        'x_CaiT for [CLS] TOken to mask the image'
        'x_last for image reconstruction loss'

        #print ('info: x_last size ',x_last.size()) # ([16, 64, 768])

        #print ('info: x_feature size ',x_feature.size()) # ([16, 512, 384])

        #print ('info: x_CaiT ',x_CaiT.size()) # x_CaiT  torch.Size([16, 384])

        #info: x_last size  torch.Size([16, 64, 768])
        #info: x_feature size  torch.Size([16, 512, 384])
        #info: x_CaiT  torch.Size([16, 384])

       

        #compute reconstruction 
        x_rec1=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec2 = self.decoder1(x_rec1)
        x_rec= rearrange(x_rec2, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 


       
        mask_intp = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        x_in=x_in.float()
        x_rec=x_rec.float()
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask_intp).sum() / (mask_intp.sum() + 1e-5) 

        loss=loss.half()
        #print ('x_region_all size ',x_region_all.size())
        return loss,x_rec


    def forward_w_Att_model(self, x_in, mask):
        
        #print (x_in_all.size())
        #x_in=x_in_all[0]
        #mask=x_in_all[1]
        

        x_feature,x_CaiT,x_last,att_map = self.transformer(x_in,mask,self.use_MIM_mask)
        #print ('info: x_last size ',x_last.size()) # ([16, 64, 768])

        #print ('info: x_feature size ',x_feature.size()) # ([16, 512, 384])

        #print ('info: x_CaiT ',x_CaiT.size()) # x_CaiT  torch.Size([16, 384])

        #info: x_last size  torch.Size([16, 64, 768])
        #info: x_feature size  torch.Size([16, 512, 384])
        #info: x_CaiT  torch.Size([16, 384])

        x_region = self.norm_patch(x_feature)  # B L C
        x_last= self.norm_cls(x_last) 
        x_CaiT=self.norm_patch(x_CaiT)  
        #print ('after all transformer x size is ',x_region.size())
        #x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        
        #x_cls = torch.flatten(x_cls, 1)
        #print ('warning: x_cls size ',x_cls.size())
        #print ('warning: x_cls.unsqueeze(1) size ',x_cls.unsqueeze(1).size())
        #print ('warning: x_CaiT.unsqueeze(1) size ',x_CaiT.unsqueeze(1).size())
        #.unsqueeze(1)
        x_region_all=torch.cat([x_CaiT,x_region], dim=1)

        #compute reconstruction 
        x_rec1=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec2 = self.decoder1(x_rec1)
        x_rec= rearrange(x_rec2, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 


       
        mask_intp = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask_intp).sum() / (mask_intp.sum() + 1e-5) 

        
        #print ('x_region_all size ',x_region_all.size())
        return x_region_all ,loss,x_rec,att_map


class SwinTransformerV2_MIM(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('error: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            print (' info: dim at i is ', i)
            print (' info: dim at i is ', in_dim)
            print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue

        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x_in,mask):
        x_for_rec = self.forward_features(x_in,mask)
        #print ('x size is ',x_for_rec.size())

        x_for_rec=self.proj_feat(x_for_rec, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x_for_rec.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x_for_rec)
        
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 

        #print ('x_rec size ',x_rec.shape)
        
        
        #mask = mask.repeat_interleave(self.pt_size, 1).repeat_interleave(self.pt_size, 2).repeat_interleave(self.pt_size, 3).unsqueeze(1).contiguous()
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
       

        x_in=x_in.float()
        x_rec=x_rec.float()
        mask=mask.float()

        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)

        loss=loss.half()


        return loss, x_rec
        #x = self.forward_head(x)
        #return x




class SwinTransformerV2_Cls_MIM(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('error: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            #print (' info: dim at i is ', i)
            #print (' info: dim at i is ', in_dim)
            #print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward_features_for_SMIT(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

        #x = self.patch_embed(x)

        #print ('info: after patch_embed x size ',x_reshape.shape)

       

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        x_final = self.layers(x_reshape)

        x=x_reshape

        layer_ct=0
        for cur_stage in self.layers:
            
            x= cur_stage(x)

            if layer_ct==2:
                x_feature=x

            layer_ct=layer_ct+1
            
        #print ('info: after all layer x size ',x.shape)
        x_final = self.norm(x_final)
        return x_feature,x_final
    
    def forward(self, x_in,mask):

        #print ('info x in size is ',x_in.size())
        #print ('info mask size is ',mask.size())


        x_region,x_for_rec = self.forward_features_for_SMIT(x_in,mask)
        #print ('x size is ',x_for_rec.size())

        x_for_rec=self.proj_feat(x_for_rec, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x_for_rec.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x_for_rec)
        
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 



        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls = torch.flatten(x_cls, 1)
        x_region_all=torch.cat([x_cls.unsqueeze(1),x_region], dim=1)

        return x_region_all ,0,x_rec
    
        #return loss, x_rec
        #x = self.forward_head(x)
        #return x

        #below is for SMIT use 
        ''''
        x_region = self.norm_patch(x_feature)  # B L C
        x_last= self.norm_cls(x_last) 
        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        
        x_cls = torch.flatten(x_cls, 1)
        
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)

        #compute reconstruction 
        x_rec1=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec2 = self.decoder1(x_rec1)
        x_rec= rearrange(x_rec2, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 


       
        mask_intp = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask_intp).sum() / (mask_intp.sum() + 1e-5) 

        
        #print ('x_region_all size ',x_region_all.size())
        return x_region_all ,loss,x_rec
        '''


class SwinTransformerV2_Cls_MIM_no_Patch(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('error: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            #print (' info: dim at i is ', i)
            #print (' info: dim at i is ', in_dim)
            #print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward_features_for_SMIT(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

        #x = self.patch_embed(x)

        #print ('info: after patch_embed x size ',x_reshape.shape)

       

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        x_final = self.layers(x_reshape)

        x=x_reshape

        layer_ct=0
        for cur_stage in self.layers:
            
            x= cur_stage(x)

            if layer_ct==2:
                x_feature=x

            layer_ct=layer_ct+1
            
        #print ('info: after all layer x size ',x.shape)
        x_final = self.norm(x_final)
        return x_feature,x_final
    
    def forward(self, x_in,mask):

        #print ('info x in size is ',x_in.size())
        #print ('info mask size is ',mask.size())


        x_region,x_for_rec = self.forward_features_for_SMIT(x_in,mask)
        #print ('x size is ',x_for_rec.size())

        x_for_rec=self.proj_feat(x_for_rec, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x_for_rec.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x_for_rec)
        
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 



        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls = torch.flatten(x_cls, 1)
        x_cls=x_cls.unsqueeze(1)
        #x_region_all=torch.cat([x_cls.unsqueeze(1),x_region], dim=1)

        return x_cls ,0,x_rec
    
        #return loss, x_rec
        #x = self.forward_head(x)
        #return x

        #below is for SMIT use 
        ''''
        x_region = self.norm_patch(x_feature)  # B L C
        x_last= self.norm_cls(x_last) 
        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        
        x_cls = torch.flatten(x_cls, 1)
        
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)

        #compute reconstruction 
        x_rec1=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec2 = self.decoder1(x_rec1)
        x_rec= rearrange(x_rec2, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 


       
        mask_intp = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask_intp).sum() / (mask_intp.sum() + 1e-5) 

        
        #print ('x_region_all size ',x_region_all.size())
        return x_region_all ,loss,x_rec
        '''

class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        #print('self.num_heads ',self.num_heads)

        #print('dim ',dim)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x ):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls, attn 


class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    
    def forward(self, x, x_cls):
        #
        #u = torch.cat((x_cls,x),dim=1)
        
        
        #x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        
        u = torch.cat((x_cls,x),dim=1)
        
        u1=self.norm1(u)

        u1, atten_map=self.attn(u1)
        
        x_cls = x_cls + self.drop_path(self.gamma_1 * u1)
        
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls,atten_map


class SwinTransformerV2_Cls_MIM_smart(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('error: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            print (' info: dim at i is ', i)
            print (' info: dim at i is ', in_dim)
            print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
        self.pt_size=self.encoder_stride
        self.feat_size= [int(img_size/self.encoder_stride),int(img_size/self.encoder_stride),int(img_size/self.encoder_stride)]

        Cait_layer=2
        qk_scale=None 
        depth_token_only=Cait_layer
        # add for SMART use 
        num_heads_CaiT=12
        embed_dim_CaiT=768  #this is for miccai used
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim_CaiT))
        self.blocks_token_only = nn.ModuleList([
             LayerScale_Block_CA(
                dim=embed_dim_CaiT, num_heads=num_heads_CaiT, mlp_ratio=4.0, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=nn.GELU,Attention_block=Class_Attention,
                Mlp_block= Mlp,init_values=1e-4)
            for i in range(depth_token_only)])
        
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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward_features_for_SMIT(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

        #x = self.patch_embed(x)

        #print ('info: after patch_embed x size ',x_reshape.shape)

       

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        x_final = self.layers(x_reshape)

        x=x_reshape

        layer_ct=0
        for cur_stage in self.layers:
            
            x= cur_stage(x)

            if layer_ct==2:
                x_feature=x

            layer_ct=layer_ct+1
            
        #print ('info: after all layer x size ',x.shape)
        x_final = self.norm(x_final)


        #x4_last=x # the feature for image reconstruction
        x3_feature=x_feature # the feature for computing the CLS token 

        B_size = x3_feature.shape[0]
        cls_tokens = self.cls_token.expand(B_size, -1, -1)  

        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens,atten_map = blk(x3_feature,cls_tokens)

        return x_feature,x_final ,cls_tokens,atten_map 
    
    def forward(self, x_in,mask):

        #print ('info x in size is ',x_in.size())
        #print ('info mask size is ',mask.size())

        x_region,x_for_rec, x_CaiT, att_map = self.forward_features_for_SMIT(x_in,mask)
        #print ('x size is ',x_for_rec.size())

        x_for_rec=self.proj_feat(x_for_rec, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x_for_rec.size())
        
        #z = self.encoder(x, mask)
        
        #get x_rec image 
        x_rec = self.decoder1(x_for_rec)
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 




        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls = torch.flatten(x_cls, 1)

        x_region_all=torch.cat([x_cls.unsqueeze(1), x_CaiT, x_region], dim=1)
        #x_region_all=torch.cat([x_cls.unsqueeze(1),x_region], dim=1)



        return x_region_all , att_map , x_rec
    
        

class SwinTransformerV2_MIM_SMIT(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('warning: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            #print (' info: dim at i is ', i)
            #print (' info: dim at i is ', in_dim)
            #print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue

        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
        self.pt_size=self.encoder_stride
        self.feat_size= [int(img_size/self.encoder_stride),int(img_size/self.encoder_stride),int(img_size/self.encoder_stride)]

        #Other heads for CLS and Patch 

        cls_norm=partial(nn.LayerNorm, eps=1e-5)

        self.norm_cls = cls_norm(self.num_features)
        n_norm_feature= int(self.num_features/2)
        self.norm_patch = cls_norm(n_norm_feature)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x


    def forward_features_for_SMIT(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        x_final = self.layers(x_reshape)

        x=x_reshape

        layer_ct=0
        for cur_stage in self.layers:
            
            x= cur_stage(x)

            if layer_ct==2:
                x_feature=x

            layer_ct=layer_ct+1
            
        #print ('info: after all layer x size ',x.shape)
        x_final = self.norm(x_final)
        return x_feature,x_final
    
    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x_in,mask):
        
        ## below are the forward for SMIT_V1

        
    
        #Here get the output of the last feature of swin Transformer
        #x_for_rec = self.forward_features(x_in,mask)

        x_feature,x_last=self.forward_features_for_SMIT(x_in,mask)


        x_for_rec=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec = self.decoder1(x_for_rec)
        
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 
        
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
       

        #x_in=x_in.float()
        #x_rec=x_rec.float()
        #mask=mask.float()

        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)

        #loss=loss.half()


        #print ('x_feature size ',x_feature.shape)
        #print ('x_last size ',x_last.shape)

        x_region = self.norm_patch(x_feature)  # B L C


        x_last= self.norm_cls(x_last) 


        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls = torch.flatten(x_cls, 1)
        x_region_all=torch.cat([x_cls.unsqueeze(1),x_region], dim=1)
        #compute reconstruction 

        


        #return loss, x_rec
        #print('x_region_all is ',x_region_all)
        att_map=x_rec
        return x_region_all , loss, x_rec,att_map
        #x = self.forward_head(x)
        #return x
        


class SwinTransformerV2_MIM_SMIT_Half(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('warning: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            #print (' info: dim at i is ', i)
            #print (' info: dim at i is ', in_dim)
            #print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue

        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
        self.pt_size=self.encoder_stride
        self.feat_size= [int(img_size/self.encoder_stride),int(img_size/self.encoder_stride),int(img_size/self.encoder_stride)]

        #Other heads for CLS and Patch 

        cls_norm=partial(nn.LayerNorm, eps=1e-5)

        self.norm_cls = cls_norm(self.num_features)
        n_norm_feature= int(self.num_features/2)
        self.norm_patch = cls_norm(n_norm_feature)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x


    def forward_features_for_SMIT(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        x_final = self.layers(x_reshape)

        x=x_reshape

        layer_ct=0
        for cur_stage in self.layers:
            
            x= cur_stage(x)

            if layer_ct==2:
                x_feature=x

            layer_ct=layer_ct+1
            
        #print ('info: after all layer x size ',x.shape)
        x_final = self.norm(x_final)
        return x_feature,x_final
    
    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x_in,mask):
        
        ## below are the forward for SMIT_V1

        
    
        #Here get the output of the last feature of swin Transformer
        #x_for_rec = self.forward_features(x_in,mask)

        x_feature,x_last=self.forward_features_for_SMIT(x_in,mask)


        x_for_rec=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec = self.decoder1(x_for_rec)
        
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 
        
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
       

        x_in=x_in.float()
        x_rec=x_rec.float()
        mask=mask.float()

        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)

        #loss=loss.half()


        #print ('x_feature size ',x_feature.shape)
        #print ('x_last size ',x_last.shape)

        x_region = self.norm_patch(x_feature)  # B L C


        x_last= self.norm_cls(x_last) 


        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls = torch.flatten(x_cls, 1)
        #x_region_all=torch.cat([x_cls.unsqueeze(1),x_region], dim=1)

        x_region_all=torch.cat([x_cls.unsqueeze(1),x_cls.unsqueeze(1),x_region], dim=1)

        #compute reconstruction 

        


        #return loss, x_rec
        #print('x_region_all is ',x_region_all)

        att_map=x_rec
        return x_region_all , loss, x_rec,att_map 
        #x = self.forward_head(x)
        #return x

        #return x_region_all ,loss,x_rec,att_map



class SwinTransformerV2_MIM_SMIT_Half_multiply_mean_mask_rec(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('warning: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            #print (' info: dim at i is ', i)
            #print (' info: dim at i is ', in_dim)
            #print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue

        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
        self.pt_size=self.encoder_stride
        self.feat_size= [int(img_size/self.encoder_stride),int(img_size/self.encoder_stride),int(img_size/self.encoder_stride)]

        #Other heads for CLS and Patch 

        cls_norm=partial(nn.LayerNorm, eps=1e-5)

        self.norm_cls = cls_norm(self.num_features)
        n_norm_feature= int(self.num_features/2)
        self.norm_patch = cls_norm(n_norm_feature)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x


    def forward_features_for_SMIT(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        x_final = self.layers(x_reshape)

        x=x_reshape

        layer_ct=0
        for cur_stage in self.layers:
            
            x= cur_stage(x)

            if layer_ct==2:
                x_feature=x

            layer_ct=layer_ct+1
            
        #print ('info: after all layer x size ',x.shape)
        x_final = self.norm(x_final)
        return x_feature,x_final
    
    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x_in,mask):
        
        ## below are the forward for SMIT_V1

        
    
        #Here get the output of the last feature of swin Transformer
        #x_for_rec = self.forward_features(x_in,mask)

        x_feature,x_last=self.forward_features_for_SMIT(x_in,mask)


        x_for_rec=self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec = self.decoder1(x_for_rec)
        
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 
        
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
       

        x_in=x_in.float()
        x_rec=x_rec.float()
        mask=mask.float()

        #loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        loss_recon = F.l1_loss(x_in*mask, x_rec*mask, reduction='mean')

        

        #loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)

        #loss=loss.half()


        #print ('x_feature size ',x_feature.shape)
        #print ('x_last size ',x_last.shape)

        #x_region = self.norm_patch(x_feature)  # B L C
        x_region=x_feature 

        #x_last= self.norm_cls(x_last) 


        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls = torch.flatten(x_cls, 1)
        #x_region_all=torch.cat([x_cls.unsqueeze(1),x_region], dim=1)

        x_region_all=torch.cat([x_cls.unsqueeze(1),x_cls.unsqueeze(1),x_region], dim=1)

        #compute reconstruction 

        


        #return loss, x_rec
        #print('x_region_all is ',x_region_all)

        att_map=x_rec
        return x_region_all , loss_recon, x_rec,att_map 
        #x = self.forward_head(x)
        #return x

        #return x_region_all ,loss,x_rec,att_map

class SwinTransformerV2_MIM_w_Seg_DSC_Loss(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    
        self.dice_loss = DiceCELoss(to_onehot_y=True,
                           softmax=True,
                           squared_pred=True,
                           smooth_nr=0.0,
                           smooth_dr=1e-6)
        print ('warning: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            #print (' info: dim at i is ', i)
            #print (' info: dim at i is ', in_dim)
            #print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue

        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.decoderSeg = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 47, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x


    
    def forward_features_No_Mask(self, x_in):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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


        
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x_in,mask,x_seg):
        x_for_rec = self.forward_features(x_in,mask)
        #x_for_seg = self.forward_features_No_Mask(x_in.detach())
        #print ('x size is ',x_for_rec.size())

        x_for_rec=self.proj_feat(x_for_rec, self.hidden_size, self.feat_size)
        #x_for_seg=self.proj_feat(x_for_seg, self.hidden_size, self.feat_size)

        #print ('info: after proj x_for_rec size ',x_for_rec.size())
        #print ('info: after proj x_for_seg size ',x_for_seg.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x_for_rec)
        #print ('info: after decoder1 size ',x_rec.size()) # 3, 32768, 4, 4, 4

        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 

        #x_rec_seg = self.decoderSeg(x_for_seg)
        x_rec_seg = self.decoderSeg(x_for_rec)
        #print ('info: after decoderSeg size ',x_rec_seg.size()) # [3, 1540096, 4, 4, 4])


        x_rec_seg= rearrange(x_rec_seg, 'b (s1 s2 s3 cn) h w t -> b cn (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size,cn=47) 
        
        print ('info: x_rec_seg size ',x_rec_seg.shape)
        
        
        #mask = mask.repeat_interleave(self.pt_size, 1).repeat_interleave(self.pt_size, 2).repeat_interleave(self.pt_size, 3).unsqueeze(1).contiguous()
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
       

        x_in=x_in.float()
        x_rec=x_rec.float()
        x_rec_seg=x_rec_seg.float()
        x_seg=x_seg.float()
        #x_rec_seg=F.argmax(x_rec_seg,axis=1)
        #x_rec_seg=torch.argmax(x_rec_seg, dim=1)
        #x_rec_seg=x_rec_seg.float()
        mask=mask.float()

        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        #loss_seg_recon = F.l1_loss(x_seg, x_rec_seg)

        print ('info: x_rec_seg size ',x_rec_seg.shape)
        print ('info: x_seg size ',x_seg.shape)

        loss_seg_recon=self.dice_loss(x_rec_seg,x_seg)
        print ('loss_seg_recon is ',loss_seg_recon.item())

        loss_img_rec=(loss_recon * mask).sum() / (mask.sum() + 1e-5)

        #loss = loss_img_rec + 0.5*loss_seg_recon
        loss = loss_seg_recon

        loss=loss.half()


        return loss,loss_img_rec, loss_seg_recon, x_rec,x_rec_seg
        #x = self.forward_head(x)
        #return x


class SwinTransformerV2_MIM_w_Seg(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 128,
            patch_size: int = 2,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 48,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0,0,0),
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
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.feature_info = []
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))    

        print ('error: self.num_features size is ',self.num_features)
        if not isinstance(embed_dim, (tuple, list)):
            #embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
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
            print (' info: dim at i is ', i)
            print (' info: dim at i is ', in_dim)
            print (' info: embed_dim dim is ', embed_dim)


            #V2 use 
            layers += [SwinTransformerV2Stage(
                dim=int(embed_dim[0] * 2 ** i),#in_dim,#out_dim,#in_dim,#,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_embed.grid_size[0] // scale,
                    self.patch_embed.grid_size[1] // scale,
                    self.patch_embed.grid_size[2] // scale,),
                    

                depth=depths[i],
                downsample= True,#i > 0,
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

            #V1 use 
            # build layers
            """ qk_scale=None
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim[0] * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=(window_size,window_size,window_size),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i_layer],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                    use_checkpoint=False,
                                    pat_merg_rf=2,)
                self.layers.append(layer)  """
                
            self.feature_info += [dict(num_chs=out_dim, reduction=8 * scale, module=f'layers.{i}')]

        #only V2 use
        self.layers = nn.Sequential(*layers)

        #only V1 use 
        #self.layers = nn.Sequential(*self.layers)

        self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        #Add by Jue

        self.encoder_stride=32
        #self.encoder_stride=patch_size
        print ('info patch_size size ',patch_size)
        self.decoder1 = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.decoderSeg = nn.Conv3d(embed_dim[-1]*2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.hidden_size=embed_dim[-1]*2
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

    def mask_model(self, x, mask):
        #print ('info: x size is ',x.shape)
        #print ('info: mask size is ',mask.shape)
        #x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        _,_,Wh,Ww,Wt=x.shape
        x=x.flatten(2).transpose(1, 2)
        #print ('x size is ',x.shape)
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.masked_embed.expand(B, L, -1)
        #print ('mask_tokens size is ',mask_tokens.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #print (mask_tokens.shape)
        #print (x.shape)

        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x
    
    def forward_features(self, x_in,mask):
        B, nc, w, h, t = x_in.shape
        #print ('info: x_in size ',x_in.shape) # 2,1,96,96,96
        x_reshape = self.patch_embed(x_in)

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

        x_reshape = x_reshape * (1 - w) + mask_tokens * w

        
        #print ((x==22).nonzero())

        


        #x_3 = x_3.view(B, Wh, Ww, Wt,C)
        

            

        #For V2 use only
        #x = self.layers(x_reshape)
        #print (x_reshape)
        x = self.layers(x_reshape)
        #print (x)
        #print ('info: after all layer x size ',x.shape)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x_in,mask,x_seg):
        x_for_rec = self.forward_features(x_in,mask)
        #print ('x size is ',x_for_rec.size())

        x_for_rec=self.proj_feat(x_for_rec, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x_for_rec.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x_for_rec)
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 

        x_rec_seg = self.decoderSeg(x_for_rec)
        x_rec_seg= rearrange(x_rec_seg, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.pt_size,s2=self.pt_size,s3=self.pt_size) 
        
        #print ('x_rec size ',x_rec.shape)
        
        
        #mask = mask.repeat_interleave(self.pt_size, 1).repeat_interleave(self.pt_size, 2).repeat_interleave(self.pt_size, 3).unsqueeze(1).contiguous()
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
       

        x_in=x_in.float()
        x_rec=x_rec.float()
        x_rec_seg=x_rec_seg.float()
        mask=mask.float()

        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        loss_seg_recon = F.l1_loss(x_seg, x_rec_seg)

        loss_img_rec=(loss_recon * mask).sum() / (mask.sum() + 1e-5)

        loss = loss_img_rec + 0.5*loss_seg_recon
        
        loss_img_rec=loss_img_rec.half()
        loss_seg_recon=loss_seg_recon.half()
        loss=loss.half()


        return loss,loss_img_rec, loss_seg_recon, x_rec,x_rec_seg
        #x = self.forward_head(x)
        #return x

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



@register_model 
def swinv2_3D_tiny_window4_96(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model

#Swin-T: C = 96, layer numbers = {2, 2, 6, 2}   num_heads = (3, 6, 12, 24) Batch size of 9 best fit
#Swin-S: C = 96, layer numbers ={2, 2, 18, 2}    num_heads =  (3, 6, 12, 24) Batch size of 7 best fit
#Swin-B: C = 128, layer numbers ={2, 2, 18, 2}  num_heads = (4, 8, 16, 32) Batch size of 5 best fit
#Not investigate
#Swin-L: C = 192, layer numbers ={2, 2, 18, 2}  (6, 12, 24, 48)  Batch size of 2 best fit

@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_Swin_T(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,6,2), num_heads=(3,6,12,24),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model

@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_Swin_S(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,18,2), num_heads=(3,6,12,24),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_Swin_B(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM(
       img_size=128,window_size=4, embed_dim=128,patch_size=2, depths=(2,2,18,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_Swin_L(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM(
       img_size=128,window_size=4, embed_dim=192,patch_size=2, depths=(2,2,18,2), num_heads=(6,12,24,48),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_96(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model

@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_Swin_S_Change_Head(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,18,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model

@register_model 
def swinv2_Cls_MIM_No_Patch_3D_tiny_window4_Swin_T(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_no_Patch(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,6,2), num_heads=(3,6,12,24),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model

@register_model 
def swinv2_Cls_MIM_No_Patch_3D_tiny_window4_Swin_S(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_no_Patch(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,18,2), num_heads=(3,6,12,24),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model 
def swinv2_Cls_MIM_No_Patch_3D_tiny_window4_Swin_B(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_no_Patch(
       img_size=128,window_size=4, embed_dim=128,patch_size=2, depths=(2,2,18,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_Swin_L(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_no_Patch(
       img_size=128,window_size=4, embed_dim=192,patch_size=2, depths=(2,2,18,2), num_heads=(6,12,24,48),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model 
def swinv2_Cls_MIM_No_Patch_3D_tiny_window4_96(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_no_Patch(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model

@register_model 
def swinv2_Cls_MIM_No_Patch_3D_tiny_window4_Swin_S_Change_Head(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_no_Patch(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,18,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model



#

@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_128(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM(
       img_size=128,window_size=4, embed_dim=128,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_96_SMART(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_smart(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model

@register_model 
def swinv2_Cls_MIM_3D_tiny_window4_128_SMART(pretrained=False, **kwargs):

    model = SwinTransformerV2_Cls_MIM_smart(
       img_size=128,window_size=4, embed_dim=128,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_SMIT_96(pretrained=False, **kwargs):
    model = SwinTransformerV2_MIM_SMIT(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model

@register_model 
def swinv2_3D_SMIT_96_2_2_12_2(pretrained=False, **kwargs):
    model = SwinTransformerV2_MIM_SMIT(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,12,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_SMIT_96_2_2_12_2_half(pretrained=False, **kwargs):
    model = SwinTransformerV2_MIM_SMIT_Half(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,12,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_SMIT_96_2_2_12_2_half_multiply_mean_mask_rec(pretrained=False, **kwargs):
    model = SwinTransformerV2_MIM_SMIT_Half_multiply_mean_mask_rec(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,12,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model



@register_model 
def swinv2_3D_tiny_window4_96_w_Seg(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM_w_Seg(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,12,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
       #img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,4,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_tiny_window4_128_w_Seg(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM_w_Seg(
       img_size=128,window_size=4, embed_dim=128,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
       #img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,4,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_tiny_window4_48_w_Seg_2_2_20_2(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM_w_Seg(
       img_size=128,window_size=4, embed_dim=48,patch_size=2, depths=(2,2,20,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
       
    #model.default_cfg = _cfg()

    
    return model

@register_model 
def swinv2_3D_tiny_window4_96_w_Seg_2_2_20_2(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM_w_Seg(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,20,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
       
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_tiny_window4_128_w_Seg_2_2_20_2(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM_w_Seg(
       img_size=128,window_size=4, embed_dim=128,patch_size=2, depths=(2,2,20,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
       
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_tiny_window4_192_w_Seg(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM_w_Seg(
       img_size=128,window_size=4, embed_dim=192,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
       #img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,4,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model

@register_model 
def swinv2_3D_tiny_window4_96_w_Seg_DSC_Loss(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM_w_Seg_DSC_Loss(
       img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
       #img_size=128,window_size=4, embed_dim=96,patch_size=2, depths=(2,2,4,2), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_tiny_window4_dim_128(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM(
       img_size=128,window_size=4, embed_dim=128,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model

@register_model 
def swinv2_3D_tiny_window4_dim_160(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM(
       img_size=128,window_size=4, embed_dim=160,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_tiny_window4_dim_192(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM(
       img_size=128,window_size=4, embed_dim=192,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


@register_model 
def swinv2_3D_tiny_window4_96_dim256(pretrained=False, **kwargs):

    model = SwinTransformerV2_MIM(
       img_size=128,window_size=4, embed_dim=256,patch_size=2, depths=(2,2,40,4), num_heads=(4,8,16,32),qkv_bias=True, **kwargs)
    #model.default_cfg = _cfg()

    
    return model


