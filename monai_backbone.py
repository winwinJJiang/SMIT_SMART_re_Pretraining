"""
MONAI SwinUNETR V2 backbone wrapper for SMIT pretraining.
Replaces TransMorph's SwinTransformer with MONAI's SwinTransformer (use_v2=True),
while providing the same forward interface:
    returns: (x_feature, cls_tokens, x_last, att_map)

This ensures the SMIT pretraining losses (patch distillation, CLS distillation, 
image reconstruction) work identically, but on VoCo's architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Tuple, Union

from monai.networks.nets.swin_unetr import SwinTransformer as MonaiSwinTransformer
from timm.models.layers import trunc_normal_, DropPath

# Import CaiT components from TransMorph (they are architecture-independent)
from swin_models.TransMorph import Class_Attention, LayerScale_Block_CA, Mlp


class MonaiSwinBackbone(nn.Module):
    """
    MONAI SwinTransformer backbone adapted for SMIT pretraining.
    
    Wraps MONAI's SwinTransformer (same as VoCo) and adds:
    1. mask_token for MIM (masked image modeling)
    2. cls_token + CaiT cross-attention blocks for [CLS] distillation
    3. Returns (x_feature, cls_tokens, x_last, att_map) matching TransMorph interface
    
    Architecture: depths=(2,2,2,2), num_heads=(3,6,12,24), window_size=7, embed_dim=48
    This matches VoCo's MONAI SwinUNETR exactly.
    """
    
    def __init__(
        self,
        img_size=(96, 96, 96),
        patch_size=2,
        in_chans=1,
        embed_dim=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_v2=True,
        use_checkpoint=False,
        Cait_layer=2,
        feature_extract_stage=2,  # which stage to extract x_feature from (0-indexed)
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.feature_extract_stage = feature_extract_stage
        self.use_MIM_mask = True
        
        # MONAI SwinTransformer (same as VoCo)
        self.swin = MonaiSwinTransformer(
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_v2=use_v2,
            use_checkpoint=use_checkpoint,
        )
        
        # Mask token for MIM
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=0.02)
        
        # CaiT [CLS] token and cross-attention blocks
        # embed_dim at stage 2 output = embed_dim * 2^2 * 2 = embed_dim * 8
        # But actually after stage 2 downsample, dim = embed_dim * 2^3 = embed_dim * 8
        # x_feature is stage 2 OUTPUT (before downsample) = embed_dim * 2^2 = embed_dim * 4
        # Wait - need to check what stage the original code uses
        # Original: x_feature = x at i==2 (after stage 2 blocks, before stage 2 downsample)
        # Stage 2 dim = embed_dim * 2^2 = embed_dim * 4
        # But in MONAI, stages are layers1-4, and outputs are x0_out through x4_out
        # x0_out: embed_dim, x1_out: embed_dim*2, x2_out: embed_dim*4, x3_out: embed_dim*8
        # x4_out: embed_dim*16 (after final downsample)
        
        # For CaiT, original uses embed_dim * 2*2*2 = embed_dim * 8
        # This is the dim AFTER stage 2's PatchMerging (= stage 3's input dim)
        # Actually looking at original code: embed_dim_CaiT = embed_dim*2*2*2 = embed_dim*8
        # And x_feature = x at i==2 which is AFTER stage 2 downsample
        # So x_feature has dim = embed_dim * 2^3 = embed_dim * 8
        
        # Let me re-check: in original TransMorph forward:
        # for i in range(self.num_layers):
        #     x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
        #     if i==2: x_feature = x  # x is the DOWNSAMPLED output
        # So x_feature has dim = embed_dim * 2^(2+1) = embed_dim * 8 (after stage 2 downsample)
        
        # x4_last = x (after stage 3) has dim = embed_dim * 2^(3+1) = embed_dim * 16
        
        self.cait_embed_dim = embed_dim * 8  # dim of x_feature (after stage 2 downsample)
        self.bottleneck_dim = embed_dim * 16  # dim of x_last (after stage 3 downsample)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cait_embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        
        self.blocks_token_only = nn.ModuleList([
            LayerScale_Block_CA(
                dim=self.cait_embed_dim,
                num_heads=num_heads[feature_extract_stage],  # use stage 2's num_heads
                mlp_ratio=4.0,
                qkv_bias=qkv_bias,
                qk_scale=None,
            )
            for _ in range(Cait_layer)
        ])
    
    def forward(self, x_in, mask, mask_flag):
        """
        Args:
            x_in: input image [B, C, D, H, W]
            mask: binary mask [B, D', H', W'] at patch resolution
            mask_flag: whether to apply MIM masking
            
        Returns:
            x_feature: stage 2 features for patch distillation [B, N, C]
            cls_tokens: CaiT [CLS] token [B, 1, C]
            x_last: bottleneck features for reconstruction [B, N', C']
            att_map: attention map from CaiT cross-attention
        """
        # Get patch embeddings from MONAI's swin
        # MONAI SwinTransformer.forward returns list of stage outputs
        # But we need to intercept after patch_embed for masking
        
        # Step 1: Patch embedding
        x = self.swin.patch_embed(x_in)
        # x shape: [B, C, D', H', W'] where D'=D/patch_size etc.
        
        B = x.shape[0]
        
        # Step 2: Apply MIM mask
        if mask is not None and mask_flag:
            # Reshape to [B, N, C] for masking
            if len(x.shape) == 5:
                _, C, D, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
                mask_tokens = self.mask_token.expand(B, x_flat.shape[1], -1)
                w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
                x_flat = x_flat * (1.0 - w) + mask_tokens * w
                x = x_flat.transpose(1, 2).view(B, C, D, H, W)
        
        # Step 3: Run through MONAI SwinTransformer stages
        # We need to manually run through stages to extract intermediate features
        # MONAI uses layers1, layers2, layers3, layers4 (and layers1c etc for v2)
        
        x0 = x
        if hasattr(self.swin, 'layers1c') and len(self.swin.layers1c) > 0:
            x0 = self.swin.layers1c[0](x0.contiguous())
        x0_out = self.swin.layers1[0](x0.contiguous())
        
        if hasattr(self.swin, 'layers2c') and len(self.swin.layers2c) > 0:
            x0_out = self.swin.layers2c[0](x0_out.contiguous())
        x1_out = self.swin.layers2[0](x0_out.contiguous())
        
        if hasattr(self.swin, 'layers3c') and len(self.swin.layers3c) > 0:
            x1_out = self.swin.layers3c[0](x1_out.contiguous())
        x2_out = self.swin.layers3[0](x1_out.contiguous())
        
        if hasattr(self.swin, 'layers4c') and len(self.swin.layers4c) > 0:
            x2_out = self.swin.layers4c[0](x2_out.contiguous())
        x3_out = self.swin.layers4[0](x2_out.contiguous())
        
        # x2_out is after stage 3 (layers3) = x_feature equivalent
        # x3_out is after stage 4 (layers4) = x_last equivalent
        
        # Convert to [B, N, C] format
        # MONAI outputs are in [B, C, D, H, W] format
        if len(x2_out.shape) == 5:
            x_feature = rearrange(x2_out, 'b c d h w -> b (d h w) c')
        else:
            x_feature = x2_out
            
        if len(x3_out.shape) == 5:
            x_last = rearrange(x3_out, 'b c d h w -> b (d h w) c')
        else:
            x_last = x3_out
        
        # Step 4: CaiT cross-attention for [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        att_map = None
        for blk in self.blocks_token_only:
            cls_tokens, att_map = blk(x_feature, cls_tokens)
        
        return x_feature, cls_tokens, x_last, att_map


class SMIT_MONAI_Pretrain_Model(nn.Module):
    """
    Full SMIT pretraining model using MONAI SwinTransformer backbone.
    Drop-in replacement for Trans_SMIT_pre_train_cls_patch_rec_Student_CaiT_All_3_Loss.
    
    Provides identical forward interface:
        forward(x_in, mask) -> (x_region_all, rec_loss, x_rec, att_map)
    """
    
    def __init__(
        self,
        config,
        out_channels=1,
        Cait_layer=2,
    ):
        super().__init__()
        
        img_size = config.img_size
        embed_dim = config.embed_dim
        
        self.use_MIM_mask = True
        
        # Use MONAI backbone
        self.transformer = MonaiSwinBackbone(
            img_size=img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size if hasattr(config, 'window_size') else (7, 7, 7),
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            use_v2=True,
            use_checkpoint=config.use_checkpoint,
            Cait_layer=Cait_layer,
        )
        
        # Dims matching original
        self.hidden_size = embed_dim * 16  # bottleneck dim
        self.feat_size = (img_size[0] // 32, img_size[1] // 32, img_size[2] // 32)
        
        cait_dim = embed_dim * 8  # x_feature dim
        
        # Normalization layers (matching original)
        cls_norm = partial(nn.LayerNorm, eps=1e-5)
        self.norm_cls = cls_norm(self.hidden_size)
        self.norm_patch = cls_norm(cait_dim)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Reconstruction decoder
        self.encoder_stride = 32
        self.decoder1 = nn.Conv3d(
            self.hidden_size,
            out_channels=self.encoder_stride ** 3 * 1,
            kernel_size=1,
        )
    
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def forward(self, x_in, mask):
        """
        Args:
            x_in: input image [B, 1, D, H, W]
            mask: binary mask [B, D', H', W']
            
        Returns:
            x_region_all: concatenated [cls, CaiT_cls, patch_features] for distillation
            loss: reconstruction loss (scalar)
            x_rec: reconstructed image
            att_map: attention map from CaiT
        """
        x_feature, x_CaiT, x_last, att_map = self.transformer(
            x_in, mask, self.use_MIM_mask
        )
        
        # Normalize
        x_region = self.norm_patch(x_feature)  # [B, N, C]
        x_last = self.norm_cls(x_last)         # [B, N', C']
        x_CaiT = self.norm_patch(x_CaiT)      # [B, 1, C]
        
        # Global [CLS] via avg pooling
        x_cls = self.avgpool(x_region.transpose(1, 2))  # [B, C, 1]
        x_cls = torch.flatten(x_cls, 1)                  # [B, C]
        
        # Concatenate: [avg_cls, CaiT_cls, patch_tokens]
        x_region_all = torch.cat([x_cls.unsqueeze(1), x_CaiT, x_region], dim=1)
        
        # Reconstruction
        x_rec1 = self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec2 = self.decoder1(x_rec1)
        x_rec = rearrange(
            x_rec2,
            'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)',
            s1=32, s2=32, s3=32,
        )
        
        # Masked reconstruction loss
        mask_intp = (
            mask.repeat_interleave(2, 1)
            .repeat_interleave(2, 2)
            .repeat_interleave(2, 3)
            .unsqueeze(1)
            .contiguous()
        )
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        loss = (loss_recon * mask_intp).sum() / (mask_intp.sum() + 1e-5)
        
        return x_region_all, loss, x_rec, att_map
    
    def forward_w_Att_model(self, x_in, mask):
        """Same as forward but without avg_cls in output (for attention model)."""
        x_feature, x_CaiT, x_last, att_map = self.transformer(
            x_in, mask, self.use_MIM_mask
        )
        
        x_region = self.norm_patch(x_feature)
        x_last = self.norm_cls(x_last)
        x_CaiT = self.norm_patch(x_CaiT)
        
        x_region_all = torch.cat([x_CaiT, x_region], dim=1)
        
        x_rec1 = self.proj_feat(x_last, self.hidden_size, self.feat_size)
        x_rec2 = self.decoder1(x_rec1)
        x_rec = rearrange(
            x_rec2,
            'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)',
            s1=32, s2=32, s3=32,
        )
        
        mask_intp = (
            mask.repeat_interleave(2, 1)
            .repeat_interleave(2, 2)
            .repeat_interleave(2, 3)
            .unsqueeze(1)
            .contiguous()
        )
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')
        loss = (loss_recon * mask_intp).sum() / (mask_intp.sum() + 1e-5)
        
        return x_region_all, loss, x_rec, att_map
