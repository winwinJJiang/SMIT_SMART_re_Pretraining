# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from einops import rearrange
import torch.nn.functional as F
__all__ = ["ViTAutoEnc"]




class ViTAutoEnc_Mask_In(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = ViTAutoEnc(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), output will be same size as of input
            >>> net = ViTAutoEnc(in_channels=3, patch_size=(16,16,16), img_size=(128,128,128), pos_embed='conv')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        if spatial_dims == 2:
            raise ValueError("Not implemented for 2 dimensions, please try 3")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.embed_dim=768
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))      
        

    def forward(self, x,mask):
        
        
        # before patch embd x size is  torch.Size([3, 1, 96, 96, 96])
        # after patch embd x size is   torch.Size([3, 110592, 48])
        # after patch embd x_ful_size size is   torch.Size([3, 48, 48, 48, 48])
        # mask size is  torch.Size([3, 48, 48, 48])
        # w size is  torch.Size([3, 110592, 1])
        # x size is  torch.Size([3, 110592, 48])
        # self.mask_token size is  torch.Size([3, 110592, 48])

        x = self.patch_embedding(x)

        # [3,216,768]  [3,1728,768]
        
        #print (x.shape)
        
        assert mask is not None
        #print ('self.mask_token size ',self.mask_token.size())
        B, L, _ = x.shape
        #_,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        #print ('mask size is ',mask.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        #print ('w size is ',w.size())
        #temperary comments for debug
        #print ('x size is ',x.size())
        #print ('self.mask_token size is ',mask_tokens.size())
        #print ('x before size is ',x.size())
        #x = x.transpose(1, 2)
        #print ('x size is ',x.size())
        
        #x = x * (1. - w) + mask_tokens * w
        x = x * (1. - w) #+ mask_tokens * w

        #x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        #Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        
        return x , hidden_states_out


class ViTAutoEnc(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = ViTAutoEnc(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), output will be same size as of input
            >>> net = ViTAutoEnc(in_channels=3, patch_size=(16,16,16), img_size=(128,128,128), pos_embed='conv')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        if spatial_dims == 2:
            raise ValueError("Not implemented for 2 dimensions, please try 3")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        new_patch_size = (4, 4, 4)
        self.conv3d_transpose = nn.ConvTranspose3d(hidden_size, 16, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(
            in_channels=16, out_channels=1, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x_in):
        x = self.patch_embedding(x_in)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        cuberoot = round(math.pow(x.size()[2], 1 / 3))
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0], x_shape[1], cuberoot, cuberoot, cuberoot])
        x = self.conv3d_transpose(x)
        x_rec = self.conv3d_transpose_1(x)

        # calculate the losses
        loss_recon = F.l1_loss(x_in, x_rec)
        return x_rec, loss_recon



from typing import Tuple, Union
class Vit_pre_train_linear_rearrange_1_layer(nn.Module):
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
        
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim

        patch_size_=16
        self.patch_size=patch_size_
        self.feat_size=(config.img_size[0]//patch_size_,config.img_size[1]//patch_size_,config.img_size[2]//patch_size_)
        self.scal_=int(96/patch_size_)
        self.ViTEncoder = ViTAutoEnc_Mask_In(in_channels=1,
                                            img_size=(96, 96, 96),
                                            patch_size=(patch_size_, patch_size_, patch_size_),
                                            pos_embed='conv',
                                            hidden_size=768,
                                            mlp_dim=3072,
                                           )
        
        #below is the decoder from UnetR
        self.encoder_stride=self.patch_size#self.scal_

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.ViTEncoder(x_in,mask)
        #print ('after vit x size is ',x.shape)
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size is ',x.shape)
        
        


        x_rec3 = self.decoder1(x)
        #print ('x_rec3 size ',x_rec3.shape)

        #Transpose
        x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (s1 h) (s2 w) (s3 t)', s1=self.patch_size,s2=self.patch_size,s3=self.patch_size)  # This maybe wrong?
        #Original Used
        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=self.patch_size,s2=self.patch_size,s3=self.patch_size)   #new run setting seems correct
        #print ('x_rec size ',x_rec.shape)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
        
        #loss_recon= F.l1_loss(x_in, x_rec)
        #loss=loss_recon

        return x_rec,loss

class Vit_pre_train_linear_rearrange_2_dec_layer(nn.Module):
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
        self.feat_size=(config.img_size[0]//8,config.img_size[1]//8,config.img_size[2]//8)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim


        self.ViTEncoder = ViTAutoEnc_Mask_In(in_channels=1,
                                            img_size=(96, 96, 96),
                                            patch_size=(8, 8, 8),
                                            pos_embed='conv',
                                            hidden_size=768,
                                            mlp_dim=3072,
                                           )
        
        #below is the decoder from UnetR
        self.encoder_stride=8

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        new_patch_size_2 = (2, 2, 2)
        new_patch_size_4 = (4, 4, 4)
        self.conv3d_transpose = nn.ConvTranspose3d(hidden_size, 16, kernel_size=new_patch_size_4, stride=new_patch_size_4)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(
            in_channels=16, out_channels=1, kernel_size=new_patch_size_2, stride=new_patch_size_2
        )

        self.in_chans=1
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, out_feats = self.ViTEncoder(x_in,mask)
        #print ('after vit x size is ',x.shape)
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size is ',x.shape)
        
        
        x_rec3 = self.conv3d_transpose(x)
        x_rec = self.conv3d_transpose_1(x_rec3)

        #x_rec3 = self.decoder1(x)
        #print ('x_rec3 size ',x_rec3.shape)
        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (s1 h) (s2 w) (s3 t)', s1=8,s2=8,s3=8) 
        #x_rec= rearrange(x_rec3, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=8,s2=8,s3=8) 
        #print ('x_rec size ',x_rec.shape)
        
        mask = mask.repeat_interleave(8, 1).repeat_interleave(8, 2).repeat_interleave(8, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)

        return x_rec,loss

class ViTAutoEnc_Mask(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = ViTAutoEnc(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), output will be same size as of input
            >>> net = ViTAutoEnc(in_channels=3, patch_size=(16,16,16), img_size=(128,128,128), pos_embed='conv')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        if spatial_dims == 2:
            raise ValueError("Not implemented for 2 dimensions, please try 3")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        new_patch_size = (4, 4, 4)
        self.conv3d_transpose = nn.ConvTranspose3d(hidden_size, 16, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(
            in_channels=16, out_channels=1, kernel_size=new_patch_size, stride=new_patch_size
        )

        
    def forward(self, x):
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        cuberoot = round(math.pow(x.size()[2], 1 / 3))
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0], x_shape[1], cuberoot, cuberoot, cuberoot])
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x, hidden_states_out