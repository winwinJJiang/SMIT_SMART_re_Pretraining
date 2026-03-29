'''
TransMorph model

Paper:
Chen, J., Du, Y., He, Y., Segars, P. W., Li, Y., & Frey, E. C. (2021). 
TransMorph: Transformer for Unsupervised Medical Image Registration. arXiv preprint.

Swin-Transformer code was retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original Swin-Transformer paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.

Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''


from functools import partial
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import models.configs_TransMorph as configs
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock#,UnetrUpOnlyBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
import math
import torch.nn.functional as F
from einops import rearrange
from ibot_model import head

from monai.networks.nets import SwinTransformer_Mask_In_Monai
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from typing import Optional, Sequence, Tuple, Type, Union

class PixelShuffle3D(nn.Module):
    """
    # 三维PixelShuffle模块
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: tensor的放大倍数
        """
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


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

class WindowAttention(nn.Module):
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
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

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
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

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

        x = self.norm(x)
        x = self.reduction(x)

        return x

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
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
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

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

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
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


class PatchEmbed(nn.Module):
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
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x

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

class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)

class SwinTransformer(nn.Module):
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
                 pat_merg_rf=2,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
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
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())
        x = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())

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
        #print ('x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
            #print ('after X size is ', x.size())
            
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class SwinTransformer_Unetr(nn.Module):
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
                 pat_merg_rf=2,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
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
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())
        x = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())

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
        #print ('x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('bottle net X size is ', x.size())
            
        return x,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr, self).train(mode)
        self._freeze_stages()



class SwinTransformer_Unetr_Seperate(nn.Module):
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
                 pat_merg_rf=2,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
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
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())
        x = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())

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
        #print ('x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)

            if i==2:
                x_feature=x
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('bottle net X size is ', x.size())
            
        return x,x_feature,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_Seperate, self).train(mode)
        self._freeze_stages()

class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Reconstruction_Task(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        #mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='mean')

        loss = loss_recon#(loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss
        
class SwinTransformer_Unetr_No_Last_Sample(nn.Module):
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
                 pat_merg_rf=2,):
        super().__init__()
        #embed_dim=96
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
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
                                downsample=PatchMerging if (i_layer < self.num_layers-1) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())
        x = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())

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
        #print ('x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('bottle net X size is ', x.size())
            
        return x,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_No_Last_Sample, self).train(mode)
        self._freeze_stages()


class SwinTransformer_Unetr_Mask_In(nn.Module):
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
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x,mask):
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
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                #print ('before X size is ', x.size())
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('info: bottle net X size is ', x.size())
            
        return x,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_Mask_In, self).train(mode)
        self._freeze_stages()


class SwinTransformer_Unetr_Mask_In_Seperate(nn.Module):
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
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x,mask):
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
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            #print ('**'*50)
            #print('layer number is ',i)
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            
            
            #print ('after x size is ', x.size())

            if i==2:
                x_feature=x
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                
                outs.append(out)
        #print ('info: bottle net X size is ', x.size())
            
        return x,x_feature,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_Mask_In_Seperate, self).train(mode)
        self._freeze_stages()


class SwinTransformer_Unetr_Mask_In_2(nn.Module):
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
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x,mask):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())  # b,1,96,96,96
        x,x_ful_size = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())  # b,48,48,48,48
        #assert mask is not None
        #print ('self.mask_token size ',self.mask_token.size())
        B, L, _ = x.shape
        _,_,Wh,Ww,Wt=x_ful_size.shape
        
        #print ('mask size is ',mask.size())
        
        #print ('w size is ',w.size())
        #temperary comments for debug
        #print ('x size is ',x.size())
        #print ('self.mask_token size is ',mask_tokens.size())
        if mask is not None:
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
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
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                #print ('before X size is ', x.size())
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('info: bottle net X size is ', x.size())
            
        return x,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_Mask_In_2, self).train(mode)
        self._freeze_stages()

class SwinTransformer_Unetr_Mask_In_No_Last_downsample(nn.Module):
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
                 pat_merg_rf=2,):
        super().__init__()
        
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        #embed_dim=96
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
                                downsample=PatchMerging if (i_layer < self.num_layers-1) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

    def forward(self, x,mask):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())  # b,1,96,96,96
        x,x_ful_size = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())  # b,48,48,48,48
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
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                #print ('before X size is ', x.size())
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('info: bottle net X size is ', x.size())
            
        return x,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_Mask_In_No_Last_downsample, self).train(mode)
        self._freeze_stages()

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)

class Conv3d_only(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        

        super(Conv3d_only, self).__init__(conv)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class TransMorph(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(TransMorph, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
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
        self.up0 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(1, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(1, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.out = Conv3d_only(in_channels=16, out_channels=14, kernel_size=3, padding=1,stride=1)
        
        #self.reg_head = RegistrationHead(
        #    in_channels=config.reg_head_chan,
        #    out_channels=3,
        #    kernel_size=3,
        #)
        #self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        source = x[:, 0:1, :, :]
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)  # 4,1,48,48,48
            #print ('x_s1 size is ',x_s1.size()) # 
            f4 = self.c1(x_s1) # torch.Size([4, 48, 48, 48, 48])
            #print ('f4 size is ',f4.size())
            f5 = self.c2(x_s0) # torch.Size([4, 16, 96, 96, 96])

            #print ('f5 size is ',f5.size())
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)
        #print ('out_feats size is ',len(out_feats))
        #print (out_feats[0].size())
        #print (out_feats[1].size())
        #print (out_feats[2].size())
        #print (out_feats[3].size())
        #
        #'''''
        #torch.Size([4, 48, 24, 24, 24])  out_features[0]
        #torch.Size([4, 96, 12, 12, 12])  out_features[1]
        #torch.Size([4, 192, 6, 6, 6])    out_features[2]
        #torch.Size([4, 384, 3, 3, 3])    out_features[3]
        #''''

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        # start upsampling
        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)

        x=self.out(x)
        return x#out, flow



class TransMorph_Unetr(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=14,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore
        #self.up1=nn.ConvTranspose3d(384, 192, 2, stride=2)
        #self.up2=nn.ConvTranspose3d(192, 96, 2, stride=2)
        #self.up3=nn.ConvTranspose3d(96, 48, 2, stride=2)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):

        x, out_feats = self.transformer(x)

        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        logits = self.out(dec_upsample)

        return logits


class TransMorph_Unetr_msk_in(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=14,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In_2(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore
        #self.up1=nn.ConvTranspose3d(384, 192, 2, stride=2)
        #self.up2=nn.ConvTranspose3d(192, 96, 2, stride=2)
        #self.up3=nn.ConvTranspose3d(96, 48, 2, stride=2)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def train_forward(self, x_in,mask):

        x, out_feats = self.transformer(x_in,mask)

        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        logits = self.out(dec_upsample)

        return logits

    def forward(self, x_in):

        x, out_feats = self.transformer(x_in,None)

        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        logits = self.out(dec_upsample)

        return logits

class TransMorph_Unetr_pre_train_all(nn.Module):
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
        res_block: bool = True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore
        #self.up1=nn.ConvTranspose3d(384, 192, 2, stride=2)
        #self.up2=nn.ConvTranspose3d(192, 96, 2, stride=2)
        #self.up3=nn.ConvTranspose3d(96, 48, 2, stride=2)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):

        x, out_feats = self.transformer(x)

        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        logits = self.out(dec_upsample)

        return logits




class TransMorph_Swin_SSIM_pre_train(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore
        self.in_chans=1
        #self.up1=nn.ConvTranspose3d(384, 192, 2, stride=2)
        #self.up2=nn.ConvTranspose3d(192, 96, 2, stride=2)
        #self.up3=nn.ConvTranspose3d(96, 48, 2, stride=2)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):

        x, out_feats = self.transformer(x_in,mask)

        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        x_rec = self.out(dec_upsample)

        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()


        # for masked reconstruction
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans   # for masked _loss

        # for all reconstruction
        #loss = F.l1_loss(x_in, x_rec, reduction='mean')

        
        return x_rec,loss


class TransMorph_Swin_SSIM_pre_train_simple_Dec_5_Trans3d(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        
        

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore
        self.in_chans=1
        
        half_new_patch_size = (2, 2, 2)
        self.up1 = nn.ConvTranspose3d(hidden_size, hidden_size//2, kernel_size=half_new_patch_size, stride=half_new_patch_size)   #768->384,6,6,6
        self.up2 = nn.ConvTranspose3d(hidden_size//2, hidden_size//4, kernel_size=half_new_patch_size, stride=half_new_patch_size) #384->192,12,12,12
        self.up3 = nn.ConvTranspose3d(hidden_size//4, hidden_size//8, kernel_size=half_new_patch_size, stride=half_new_patch_size) #192->96,24,24,24
        self.up4 = nn.ConvTranspose3d(hidden_size//8, hidden_size//16, kernel_size=half_new_patch_size, stride=half_new_patch_size) #96->48,48,48,48
        self.up5 = nn.ConvTranspose3d(hidden_size//16, hidden_size//32, kernel_size=half_new_patch_size, stride=half_new_patch_size) #48->1,96,96,96
        self.conv3d_out=nn.Conv3d(hidden_size//32,1,kernel_size=1)


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):

        x, out_feats = self.transformer(x_in,mask)

        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after projection size',x.size())
        
        x=self.up1(x)
        #print ('after up1 size',x.size())

        x=self.up2(x)
        #print ('after up2 size',x.size())

        x=self.up3(x)
        #print ('after up3 size',x.size())

        x=self.up4(x)
        #print ('after up4 size',x.size())

        x=self.up5(x)
        #print ('after up5 size',x.size())



        
        x_rec = self.conv3d_out(x)
        #print ('x_rec size',x_rec.size())

        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()

        # for masked reconstruction
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans   # for masked _loss

        # for all reconstruction
        #loss = F.l1_loss(x_in, x_rec, reduction='mean')

        
        return x_rec,loss


class TransMorph_Swin_SSIM_pre_train_simple_Dec_V2(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        self.norm = nn.LayerNorm(hidden_size)
        new_patch_size = (4, 4, 4)
        half_new_patch_size = (2, 2, 2)
        self.conv3d_transpose = nn.ConvTranspose3d(hidden_size, hidden_size//4, kernel_size=new_patch_size, stride=new_patch_size)
        #nn.Conv3d(768,out_channels=self.encoder_stride  ** 2, kernel_size=1)
        self.conv3d_1=nn.Conv3d(hidden_size//4,hidden_size//4,kernel_size=1)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(hidden_size//4, hidden_size//16, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_2=nn.Conv3d(hidden_size//16,config.patch_size*config.patch_size*config.patch_size,kernel_size=1)
        self.conv3d_transpose_2 = nn.ConvTranspose3d(in_channels=config.patch_size*config.patch_size*config.patch_size, out_channels=1, kernel_size=half_new_patch_size, stride=half_new_patch_size )
        self.in_chans =1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):

        x, out_feats = self.transformer(x_in,mask)

        x = self.norm(x)
        
        x=self.proj_feat(x, self.hidden_size, self.feat_size)

        """x = x.transpose(1, 2)
        cuberoot = round(math.pow(x.size()[2], 1 / 3))
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0], x_shape[1], cuberoot, cuberoot, cuberoot]) """
        #print ('x1 projected size', x.size())
        x = self.conv3d_transpose(x)
        #print ('x1 transpose 1 size', x.size())
        #x=self.conv3d_1(x)
        #print ('x1 transpose 2 size', x.size())
        x = self.conv3d_transpose_1(x)
        #print ('x1 transpose 3 size', x.size())
        x=self.conv3d_2(x)
        #print ('x1 transpose 4 size', x.size())

        x_rec = self.conv3d_transpose_2(x)
        #print ('x1 transpose 5 size', x.size())
        #print ('x_rec projected size', x_rec.size())
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()


        # for masked reconstruction
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans   # for masked _loss

        # for all reconstruction
        #loss = F.l1_loss(x_in, x_rec, reduction='mean')

        
        return x_rec,loss

class TransMorph_Swin_SSIM_pre_train_simple_Dec(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        self.norm = nn.LayerNorm(hidden_size)
        new_patch_size = (4, 4, 4)
        half_new_patch_size = (2, 2, 2)
        self.conv3d_transpose = nn.ConvTranspose3d(hidden_size, hidden_size//2, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(hidden_size//2, 16, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_2 = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=half_new_patch_size, stride=half_new_patch_size )
        self.in_chans =1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):

        x, out_feats = self.transformer(x_in,mask)

        #x = self.norm(x)
        #x=self.proj_feat(x, self.hidden_size, self.feat_size)

        x = x.transpose(1, 2)
        cuberoot = round(math.pow(x.size()[2], 1 / 3))
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0], x_shape[1], cuberoot, cuberoot, cuberoot])

        x = self.conv3d_transpose(x)
        #print ('x1 projected size', x.size())
        x = self.conv3d_transpose_1(x)
        #print ('x2 projected size', x.size())
        x_rec = self.conv3d_transpose_2(x)
        #print ('x_rec projected size', x_rec.size())
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()


        # for masked reconstruction
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans   # for masked _loss

        # for all reconstruction
        #loss = F.l1_loss(x_in, x_rec, reduction='mean')

        
        return x_rec,loss


class TransMorph_Swin_SSIM_pre_train_linear(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32
        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride  ** 2, kernel_size=1)
        #self.decoder2 = nn.Conv3d(in_channels=self.encoder_stride ** 1,out_channels=self.encoder_stride ** 2 * 1, kernel_size=1)
        self.decoder3 = nn.Conv3d(in_channels=self.encoder_stride ** 2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.out=PixelShuffle3D(self.encoder_stride)
        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec1 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec=self.out(x_rec3)
        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_Tanh(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=5*self.encoder_stride  ** 2, kernel_size=1)
        #self.decoder2 = nn.Conv3d(in_channels=self.encoder_stride ** 1,out_channels=self.encoder_stride ** 2 * 1, kernel_size=1)
        self.decoder3 = nn.Conv3d(in_channels=5*self.encoder_stride ** 2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.out=nn.Tanh()
        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec1 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        x_rec=self.out(x_rec)
        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss


class TransMorph_Swin_SSIM_pre_train_linear_rearrange(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=5*self.encoder_stride  ** 2, kernel_size=1)
        #self.decoder2 = nn.Conv3d(in_channels=self.encoder_stride ** 1,out_channels=self.encoder_stride ** 2 * 1, kernel_size=1)
        self.decoder3 = nn.Conv3d(in_channels=5*self.encoder_stride ** 2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        #self.out=PixelShuffle3D(self.encoder_stride)
        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec1 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_3_layer(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=3*self.encoder_stride  ** 2, kernel_size=1)
        self.decoder2 = nn.Conv3d(in_channels=3*self.encoder_stride ** 2,out_channels=12*self.encoder_stride ** 2, kernel_size=1)
        self.decoder3 = nn.Conv3d(in_channels=12*self.encoder_stride ** 2,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.out=PixelShuffle3D(self.encoder_stride)
        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec1 = self.decoder1(x)
        x_rec2= self.decoder2(x_rec1)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        x_rec3 = self.decoder3(x_rec2)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss

class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_discriminator(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.encoder_stride=16
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//self.encoder_stride,config.img_size[1]//self.encoder_stride,config.img_size[2]//self.encoder_stride)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In_No_Last_downsample(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.encoder_stride,s2=self.encoder_stride,s3=self.encoder_stride) 

        return x_rec#,loss


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_5_down_sample_discriminator(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.encoder_stride=32
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//self.encoder_stride,config.img_size[1]//self.encoder_stride,config.img_size[2]//self.encoder_stride)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.encoder_stride,s2=self.encoder_stride,s3=self.encoder_stride) 

        return x_rec#,loss


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_pseodu_label_1_view(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.decoder_view1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,view1,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)

        x_view1_rec = self.decoder_view1(x)
       

        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        x_view1_rec= rearrange(x_view1_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss_recon_view1 = F.l1_loss(view1, x_view1_rec, reduction='none')
        

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        loss_view1 = (loss_recon_view1 * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        
        return x_rec,x_view1_rec,  loss, loss_view1


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_pseodu_label_1_view_Seg(nn.Module):
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
        res_block: bool = True, #False

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1_img_rec = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

       

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        
        set_out_channel=200
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=set_out_channel)  # type: ignore

        

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,view1,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        # for the decoder image reconstruction loss
        x_rec3 = self.decoder1_img_rec(x)
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        # for the pseudo-label segmentation loss 
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]


        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        pseudo_seg = self.out(dec_upsample)


        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

       
        

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

       
        
        return x_rec,  loss, pseudo_seg


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_pseodu_label(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.decoder_view1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.decoder_view2 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        self.decoder_view3 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,view1,view2,view3,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)

        x_view1_rec = self.decoder_view1(x)
        x_view2_rec = self.decoder_view2(x)
        x_view3_rec = self.decoder_view3(x)

        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        x_view1_rec= rearrange(x_view1_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        x_view2_rec= rearrange(x_view2_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        x_view3_rec= rearrange(x_view3_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss_recon_view1 = F.l1_loss(view1, x_view1_rec, reduction='none')
        loss_recon_view2 = F.l1_loss(view2, x_view2_rec, reduction='none')
        loss_recon_view3 = F.l1_loss(view3, x_view3_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        loss_view1 = (loss_recon_view1 * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        loss_view2 = (loss_recon_view2 * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        loss_view3 = (loss_recon_view3 * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,x_view1_rec,x_view2_rec,x_view3_rec,   loss, loss_view1,loss_view2,loss_view3



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_pseodu_label_cat(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.decoder_view1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 3, kernel_size=1)
        

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,view1,view2,view3,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)

        x_view1_rec = self.decoder_view1(x)
       

        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        x_view1_rec= rearrange(x_view1_rec, 'b (c s1 s2 s3) h w t -> b c (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32,c=3) 
        

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss_recon_view1 = F.l1_loss(torch.cat((view1,view2,view3),1), x_view1_rec, reduction='none')
        

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        loss_view1 = (loss_recon_view1 * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        
        return x_rec,x_view1_rec,  loss, loss_view1
MERGING_MODE = {"merging": PatchMerging}

class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Monai(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.transformer = SwinTransformer_Mask_In_Monai(in_chans=in_channels,
                                                        embed_dim=feature_size,
                                                        window_size=window_size,
                                                        patch_size=patch_size,
                                                        depths=depths,
                                                        num_heads=num_heads,
                                                        mlp_ratio=4.0,
                                                        qkv_bias=True,
                                                        drop_rate=drop_rate,
                                                        attn_drop_rate=attn_drop_rate,
                                                        drop_path_rate=dropout_path_rate,
                                                        norm_layer=nn.LayerNorm,
                                                        use_checkpoint=use_checkpoint,
                                                        spatial_dims=spatial_dims,
                                                        downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
                                                    )
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, _,_,_,_ = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss

    def get_hidden_embd(self, x_in):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        

        return x

    
    def get_token_feature(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x_token=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        
        #mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        #loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        #loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_token,mask



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_No_mask(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        self.feat_size=(6,6,6)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss

    def get_hidden_embd(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        

        return x

    
    def get_token_feature(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x_token=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        
        #mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        #loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        #loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_token,mask


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_transpose(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)

        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        x_rec3 = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (s1 h) (s2 w) (s3 t)', s1=32,s2=32,s3=32) 

        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 

        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss
        
class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_and_Dec_No_Res(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=8 # the scale of the linear

        self.decoder_linear_1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 96, kernel_size=1)
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        x_linear = self.decoder_linear_1(x)
        #print ('x_linear size',x_linear.size())
        x_linear= rearrange(x_linear, 'b (s1 s2 s3 c) h w t -> b c (h s1) (w s2) (t s3)', s1=8,s2=8,s3=8,c=96) 
        #finish the linear then start dec_no_res
        #print ('x_linear size',x_linear.size())
        
        #enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        #enc3 = out_feats[-2]
        #enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        #dec4 = self.decoder5(x, enc4)
        #dec3 = self.decoder4(dec4, enc3)
        #dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(x_linear, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        x_rec = self.out(dec_upsample)


        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss

class TransMorph_Unetr_pre_train_all_No_Dec_Res_Contrast(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore

        #self.up1=nn.ConvTranspose3d(384, 192, 2, stride=2)
        #self.up2=nn.ConvTranspose3d(192, 96, 2, stride=2)
        #self.up3=nn.ConvTranspose3d(96, 48, 2, stride=2)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):

        x, out_feats = self.transformer(x)
        x_feature=x
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        logits = self.out(dec_upsample)

        return logits,x_feature

class TransMorph_Swin_SSIM_pre_train_Dec_No_Res_and_linear_rearrange_1_layer(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.encoder_stride=4 # the scale of the linear

        self.decoder_linear_1 = nn.Conv3d(96,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('after encoder x size ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        #x_linear = self.decoder_linear_1(x)
        #print ('x_linear size',x_linear.size())
        #x_linear= rearrange(x_linear, 'b (s1 s2 s3 c) h w t -> b c (h s1) (w s2) (t s3)', s1=8,s2=8,s3=8,c=96) 
        #finish the linear then start dec_no_res
        #print ('x_linear size',x_linear.size())
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        #enc1 = out_feats[-4]   
        x_linear = self.decoder_linear_1(enc2)
        x_rec= rearrange(x_linear, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.encoder_stride,s2=self.encoder_stride,s3=self.encoder_stride) 



        #print ('x_rec size after encoder is ',x_rec.size())  # 4,256,3,3,3
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return x_rec,loss

class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Teacher(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_No_Last_Sample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1

        self.head=head.iBOTHead(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            shared_head='True',
        )

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('after encoder x size ',x.size())
        
        #print ('after proj x size ',x.size())
        
        x_out=self.head(x)
        #x=self.proj_feat(x, self.hidden_size, self.feat_size)
        

        return x_out


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Student(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_Mask_In_No_Last_downsample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
        self.head=head.iBOTHead(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head='True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())
        x_token=self.head(x)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Teacher_w_cls_token(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_No_Last_Sample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1

        self.head=head.iBOTHead_w_Cls_Token(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            shared_head='True',
        )

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('after encoder x size ',x.size())
        
        #print ('after proj x size ',x.size())
        
        x_out=self.head(x)
        #x=self.proj_feat(x, self.hidden_size, self.feat_size)
        

        return x_out


class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Student_w_cls_token(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_Mask_In_No_Last_downsample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
        self.head=head.iBOTHead_w_Cls_Token(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head='True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())
        x_token=self.head(x)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Teacher_w_cls_token_AvgPool(nn.Module):
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
        ibot_head_share: bool =True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_No_Last_Sample(patch_size=config.patch_size, 
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
        
        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        
        self.norm = cls_norm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        self.head=head.iBOTHead_w_Cls_Token(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        

        return x_token




class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Teacher_w_cls_token_AvgPool_Seperate(nn.Module):
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
        ibot_head_share: bool =True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        #self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,

        #self.transformer = SwinTransformer_Unetr_No_Last_Sample(patch_size=config.patch_size,  

        self.transformer = SwinTransformer_Unetr_Seperate(patch_size=config.patch_size, 
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
        
        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        
        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        self.head=head.iBOTHead_w_Cls_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        _,x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        

        return x_token





class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Teacher_w_cls_token_AvgPool_Seperate_No_share_head(nn.Module):
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
        ibot_head_share: bool =True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        #self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,

        #self.transformer = SwinTransformer_Unetr_No_Last_Sample(patch_size=config.patch_size,  

        self.transformer = SwinTransformer_Unetr_Seperate(patch_size=config.patch_size, 
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
        
        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        
        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_group=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

        self.head_cls=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        _,x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C
        x_region_token=self.head_group(x_region)

        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        
        x_cls_token=self.head_cls(x_cls)
        #print ('x_cls_token size is ',x_cls_token.size())

        #print ('x_region size',x_region.shape)
        #x_token=torch.cat([x_cls_token, x_region_token], dim=1)
        #print ('INFO: x_token size',x_token.shape)

        x_token=x_cls_token,x_region_token




        return x_token

class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Student_w_cls_token_AvgPool(nn.Module):
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
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_Mask_In_No_Last_downsample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head=head.iBOTHead_w_Cls_Token(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec


    def get_token(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Student_w_cls_token_AvgPool_Seperate(nn.Module):
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
        '''
        TransMorph Model
        '''
        
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
        
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_Mask_In_Seperate(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head=head.iBOTHead_w_Cls_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, x_feature, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x_feature)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec


    def get_token(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Student_w_cls_token_AvgPool_Seperate_No_share_head(nn.Module):
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
        '''
        TransMorph Model
        '''
        
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
        
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_Mask_In_Seperate(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_group=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

        self.head_cls=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

        

    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, x_feature, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x_feature)  # B L C
        #print ('x_region_token before size',x_region.shape)
        x_region_token=self.head_group(x_region)
        #print ('x_region_token after size',x_region_token.shape)

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        
        x_cls_token=self.head_cls(x_cls)
        #print ('x_cls_token size is ',x_cls_token.size())

        #print ('x_region size',x_region.shape)
        #x_token=torch.cat([x_cls_token, x_region_token], dim=1)
        #print ('INFO: x_token size',x_token.shape)

        x_token=x_cls_token,x_region_token


        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec


    def get_token(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec

class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Student_w_cls_token_AvgPool_No_Share_Head(nn.Module):
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
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        hidden_size=384
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        print('warning INFO config.embed_dim, ',config.embed_dim)
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_Mask_In_No_Last_downsample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(384,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_cls=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
        self.head_group=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        #print ('x shape ')
        x_region = self.norm(x)  # B L C
        #print ('x_region_token before size',x_region.shape)
        x_region_token=self.head_group(x_region)
        #print ('x_region_token after size',x_region_token.shape)

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        
        x_cls_token=self.head_cls(x_cls)
        #print ('x_cls_token size is ',x_cls_token.size())

        #print ('x_region size',x_region.shape)
        #x_token=torch.cat([x_cls_token, x_region_token], dim=1)
        #print ('INFO: x_token size',x_token.shape)

        x_token=x_cls_token,x_region_token

        #x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Student_w_cls_token_AvgPool_No_Patch_Token(nn.Module):
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
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        #hidden_size=384
        self.hidden_size = hidden_size
        #self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        print('warning INFO config.embed_dim, ',config.embed_dim)
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_cls=head.iBOTHead_for_one_Token(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
        

    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())
        x_region = self.norm(x)  # B L C
        

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        
        x_cls_token=self.head_cls(x_cls)
        #print ('x_cls_token size is ',x_cls_token.size())

        #print ('x_region size',x_region.shape)
        #x_token=torch.cat([x_cls_token, x_region_token], dim=1)
        #print ('INFO: x_token size',x_token.shape)

        #x_token=x_cls_token

        #x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_cls_token,x_rec



class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Teacher_w_cls_token_AvgPool_No_Patch_Token(nn.Module):
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
        '''
        TransMorph Model
        '''
        #hidden_size=384
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        #self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_cls=head.iBOTHead_for_one_Token(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
        

    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())
        x_region = self.norm(x)  # B L C
        

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        x_cls_token=self.head_cls(x_cls)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        


        return x_cls_token

    def get_patch_token(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C
       


        return x_region

class TransMorph_Swin_SSIM_pre_train_linear_rearrange_1_layer_Teacher_w_cls_token_AvgPool_No_Share_Head(nn.Module):
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
        '''
        TransMorph Model
        '''
        hidden_size=384
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//16,config.img_size[1]//16,config.img_size[2]//16)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        #self.transformer = SwinTransformer_Unetr_Mask_In(patch_size=config.patch_size,
        self.transformer = SwinTransformer_Unetr_No_Last_Sample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(384,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_cls=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
        self.head_group=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C
        #print ('x_region_token before size',x_region.shape)
        x_region_token=self.head_group(x_region)
        #print ('x_region_token after size',x_region_token.shape)

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        
        x_cls_token=self.head_cls(x_cls)
        #print ('x_cls_token size is ',x_cls_token.size())

        #print ('x_region size',x_region.shape)
        #x_token=torch.cat([x_cls_token, x_region_token], dim=1)
        #print ('INFO: x_token size',x_token.shape)

        x_token=x_cls_token,x_region_token


        return x_token

    def get_patch_token(self, x_in):
        

        x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C
       


        return x_region


class Ibot_VIT_Small_Student_w_cls_token_AvgPool(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        
        self.transformer = SwinTransformer_Unetr_Mask_In_No_Last_downsample(patch_size=config.patch_size, 
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
        

        #below is the decoder from UnetR
        self.encoder_stride=16

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        self.head=head.iBOTHead_w_Cls_Token(
            768,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head='True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec

class TransMorph_Unetr_pre_train_all_No_Dec_Res(nn.Module):
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

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR

        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  # type: ignore

        #self.up1=nn.ConvTranspose3d(384, 192, 2, stride=2)
        #self.up2=nn.ConvTranspose3d(192, 96, 2, stride=2)
        #self.up3=nn.ConvTranspose3d(96, 48, 2, stride=2)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):

        x, out_feats = self.transformer(x)

        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        logits = self.out(dec_upsample)

        return logits

class Swin_enc_pre_train(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=14,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim

        #This SwinTransformer_Unetr returns the hidden features
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
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
        

        #below is the decoder from UnetR
        self.norm = nn.LayerNorm(hidden_size)

        new_patch_size = (4, 4, 4)
        half_new_patch_size = (2, 2, 2)
        self.conv3d_transpose = nn.ConvTranspose3d(hidden_size, hidden_size//2, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(hidden_size//2, 16, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_2 = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=half_new_patch_size, stride=half_new_patch_size )

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):

        x, out_feats = self.transformer(x)

        x = self.norm(x)
        x = x.transpose(1, 2)
        cuberoot = round(math.pow(x.size()[2], 1 / 3))
        x_shape = x.size()
        #print (x_shape)
        x = torch.reshape(x, [x_shape[0], x_shape[1], cuberoot, cuberoot, cuberoot])
        #print (x.size())
        x = self.conv3d_transpose(x)
        #print (x.size())
        x = self.conv3d_transpose_1(x)
        rec = self.conv3d_transpose_2(x)
        #print (rec.size())

        return rec

class TransMorphML(nn.Module):
    def __init__(self, config):
        '''
        Multi-resolution TransMorph
        '''
        super(TransMorphML, self).__init__()
        self.feat_visualize = config.feat_visualize
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
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
                                           pat_merg_rf=config.pat_merg_rf)
        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2,
                                use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan, skip_channels=config.reg_head_chan,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.reg_head_s1 = RegistrationHead(in_channels=config.reg_head_chan, out_channels=3, kernel_size=3,)
        self.reg_head_s2 = RegistrationHead(in_channels=embed_dim // 2, out_channels=3,kernel_size=3, )
        self.reg_head_s3 = RegistrationHead(in_channels=embed_dim, out_channels=3, kernel_size=3, )

    def forward(self, x):
        source = x[:, 0:1, :, :]
        x_s0 = x.clone()
        x_s1 = self.avg_pool(x)
        out_feats = self.transformer(x)  # (B, n_patch, hidden)
        f = out_feats[-2]  # torch.cat([out_m[-2], out_t[-2]], dim=1)#self.c1(out[-2])
        x = self.up1(out_feats[-1], f)
        f = out_feats[-3]  # torch.cat([out_m[-3], out_t[-3]], dim=1)#out[-3]#self.c2(out[-3])
        x = self.up2(x, f)
        flow_s3 = self.reg_head_s3(x)
        flow_s3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)(flow_s3)

        f = self.c1(x_s1)
        x = self.up3(x, f)
        flow_s2 = self.reg_head_s2(x)
        flow_s2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)(flow_s2)

        f = self.c2(x_s0)
        x = self.up4(x, f)
        flow_s1 = self.reg_head_s1(x)
        out_s3 = self.spatial_trans(source, flow_s3)
        out_s2 = self.spatial_trans(out_s3, flow_s2)
        out_s1 = self.spatial_trans(out_s2, flow_s1)
        flow = flow_s1 + flow_s2 + flow_s3
        out = self.spatial_trans(source, flow)
        outs = [out, out_s1, out_s2, out_s3]
        if self.feat_visualize:
            return outs, flow
        return outs, flow


CONFIGS = {
    'TransMorph': configs.get_3DTransMorph_config(),
    'TransMorph-No-Conv-Skip': configs.get_3DTransMorphNoConvSkip_config(),
    'TransMorph-No-Trans-Skip': configs.get_3DTransMorphNoTransSkip_config(),
    'TransMorph-No-Skip': configs.get_3DTransMorphNoSkip_config(),
    'TransMorph-Lrn': configs.get_3DTransMorphLrn_config(),
    'TransMorph-Sin': configs.get_3DTransMorphSin_config(),
    'TransMorph-Large': configs.get_3DTransMorphLarge_config(),
    'TransMorph-Small': configs.get_3DTransMorphSmall_config(),
    'TransMorph-Small_Unetr': configs.get_3DTransMorphSmall_config_Unetr(), 
    #'TransMorph-Small_Unetr_No_Res': configs.get_3DTransMorphSmall_config_Unetr(), 
    'TransMorph-Small_Unetr_pre_train': configs.get_3DTransMorphSmall_config_Unetr(), 
    'TransMorph-Small_SSIM_pre_train': configs.get_3DTransMorphSmall_config_Unetr(), 
    'TransMorph-Tiny': configs.get_3DTransMorphTiny_config(),
    'TransMorph-Small_SSIM_pre_train':configs.get_3DTransMorphSmall_config_Unetr_patch8(), 
    'TransMorph-Large_SSIM_pre_train':configs.get_3DTransMorphLarge_config_Unetr(), 
    'TransMorph-Large_SSIM_pre_train_128_middle':configs.get_3DTransMorphMiddle_config_Unetr_128(), 
    'TransMorph-Large_SSIM_pre_train_128':configs.get_3DTransMorphLarge_config_Unetr_128(), 
    'TransMorph-Large_SSIM_pre_train_128_middle_brats_config_large':configs.get_3DTransMorphMiddle_config_Unetr_128_brain_brats_large(),  
    'TransMorph-Large_SSIM_pre_train_128_middle_brats_config':configs.get_3DTransMorphMiddle_config_Unetr_128_brain_brats(),  

}


"below for the Discriminator"
from torch.autograd import Variable
def define_D_3D(input_nc=1, ndf=64, which_model_netD='n_layers',
             n_layers_D=5, norm='instance', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator3D(input_nc, ndf, n_layers=4, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator3D(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
        #netD.cuda()
    init_weights(netD, init_type=init_type)
    return netD


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator3D, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        norm_layer=nn.InstanceNorm3d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        #a=self.model(input)
        #print ('discrminator size',a.size())
        return self.model(input)



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def get_norm_layer_cbin(layer_type='cbin', num_con=0):
    if layer_type == 'cbbn':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        c_norm_layer = functools.partial(CBBNorm2d, affine=True, num_con=num_con)
    elif layer_type == 'cbin':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=False, num_con=num_con)
    elif layer_type == 'adain':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(AdaINorm2d, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

import functools
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        #print (m)
        init.normal_(m.weight.data, 0.0, 0.02)
    #elif classname.find('Deconv') != -1:
    #    init.normal_(m.weight.data, 0.0, 0.02) 
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

from torch.nn import init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

        
class Multi_Scale_Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    # Multi-scale discriminator architecture fk.......
    def __init__(self, input_dim=1,numer_scale=3):
        super(Multi_Scale_Discriminator, self).__init__()

        #FIRST_DIM: 64
        #NORM: none
        #ACTIVATION: lrelu
        #N_LAYER: 4
        #GAN_TYPE: lsgan
        #NUM_SCALES: 3
        #PAD_TYPE: mirror

        self.n_layer = 4#params['N_LAYER']
        self.gan_type = 'lsgan'#params['GAN_TYPE']
        self.dim = 64#params['FIRST_DIM']
        self.norm = 'none'#params['NORM']
        self.activ = 'lrelu'#params['ACTIVATION']
        self.num_scales = numer_scale#params['NUM_SCALES']
        self.pad_type = 'mirror'#params['PAD_TYPE']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1,1,1], count_include_pad=False)

        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        single_d=NLayerDiscriminator3D(input_nc=1,n_layers=3)
        return single_d

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 0.9)**2)
                #loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss/self.num_scales

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                #loss += torch.mean((out0 - 1)**2) # LSGAN
                loss += torch.mean((out0 - 0.9)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss/self.num_scales