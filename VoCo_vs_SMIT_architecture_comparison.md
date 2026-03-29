# VoCo (MONAI SwinUNETR) vs SMIT (TransMorph) Architecture Comparison

## Overview

This document provides an exhaustive comparison between the VoCo model (MONAI SwinUNETR with `use_v2=True`) and the SMIT model (TransMorph-based Swin Transformer), based on line-by-line source code analysis.

**Source files analyzed:**
- **SMIT**: `/lila/data/deasy/Eric_Data/Transformer_related/UNETR/BTCV/models/TransMorph.py`
  - Config: `get_3DTransMorphMiddle_config_Unetr_96_bias_True` in `configs_TransMorph.py`
  - Model class: `TransMorph_Unetr_CT_Lung_Tumor_Batch_Norm_Correction_Official_No_Unused_Parameters` (line 1734)
  - Encoder class: `SwinTransformer_Unetr` (line 969)
- **VoCo**: `/lila/.../miniconda310_nnUnet/envs/voco/lib/python3.9/site-packages/monai/networks/nets/swin_unetr.py`
  - Model class: `SwinUNETR`
  - Encoder class: `SwinTransformer`

---

## 1. High-Level Architecture Config

| Parameter | SMIT (TransMorph) | VoCo (MONAI SwinUNETR) | Same? |
|-----------|-------------------|----------------------|-------|
| `embed_dim` | 48 | 48 | ✅ |
| `depths` | **(2, 2, 8, 2)** = 14 blocks | **(2, 2, 2, 2)** = 8 blocks | ❌ |
| `num_heads` | **(4, 4, 8, 16)** | **(3, 6, 12, 24)** | ❌ |
| `window_size` | **(4, 4, 4)** | **(7, 7, 7)** | ❌ |
| `mlp_ratio` | 4 | 4 | ✅ |
| `patch_size` | 2 | 2 | ✅ |
| `drop_path_rate` | 0.1 | 0.0 (VoCo default) | ❌ |
| `qkv_bias` | True | True (False default) | ✅ (when configured) |
| `pat_merg_rf` / reduce_factor | 4 (custom param) | hardcoded 2×dim | ✅ (same effect: 8C→2C) |
| `Total params` | **~64.7M** | **~72.8M** | ❌ |
| `use_v2` conv blocks | ❌ Not available | ✅ UnetrBasicBlock before each stage | ❌ |

### Head dimension per stage

| Stage | SMIT (dim / heads) | VoCo (dim / heads) |
|-------|-------------------|-------------------|
| 0 (dim=48) | 48/4 = **12** | 48/3 = **16** |
| 1 (dim=96) | 96/4 = **24** | 96/6 = **16** |
| 2 (dim=192) | 192/8 = **24** | 192/12 = **16** |
| 3 (dim=384) | 384/16 = **24** | 384/24 = **16** |

VoCo uses uniform head_dim=16 across all stages. SMIT has variable head_dim (12→24).

### Window attention receptive field

| | SMIT (w=4) | VoCo (w=7) |
|---|---|---|
| Tokens per window | 4³ = **64** | 7³ = **343** |
| Receptive field ratio | 1× | **5.4×** |
| Attention matrix size | 64×64 | 343×343 |
| FLOPs per window | Lower | **Higher** |

---

## 2. WindowAttention Class

### 2.1 QKV Computation
```python
# SMIT (TransMorph.py, line 309)
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

# VoCo (swin_unetr.py, line 508)
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
```
**Result: IDENTICAL**

### 2.2 Attention Scale
```python
# SMIT (line 287) — supports custom qk_scale override
self.scale = qk_scale or head_dim ** -0.5

# VoCo (line 464) — no override
self.scale = head_dim**-0.5
```
**Result: FUNCTIONALLY EQUIVALENT** (SMIT defaults to same value, override rarely used)

### 2.3 Relative Position Bias Table
```python
# SMIT (lines 290-291)
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2*Wh-1) * (2*Ww-1) * (2*Wt-1), num_heads))

# VoCo (lines 468-472)
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2*Wh-1) * (2*Ww-1) * (2*Wt-1), num_heads))
```
**Result: IDENTICAL** implementation. But different VALUES due to different window_size:
- SMIT w=4: table shape = (2×4-1)³ × heads = **343 × heads**
- VoCo w=7: table shape = (2×7-1)³ × heads = **2197 × heads**

This means **pretrained weights are NOT directly transferable** between window sizes for this parameter.

### 2.4 Relative Position Index Computation
```python
# SMIT (lines 294-307)
coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # deprecated syntax
...
relative_position_index = relative_coords.sum(-1)
self.register_buffer("relative_position_index", relative_position_index)

# VoCo (lines 474-506)
mesh_args = torch.meshgrid.__kwdefaults__
if mesh_args is not None:
    coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
else:
    coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
...
relative_position_index = relative_coords.sum(-1)
self.register_buffer("relative_position_index", relative_position_index)
```
**Result: FUNCTIONALLY EQUIVALENT** — same math, VoCo handles newer PyTorch `indexing` kwarg.

### 2.5 Attention Forward Pass
```python
# SMIT (lines 316-341)
q = q * self.scale
attn = (q @ k.transpose(-2, -1))
relative_position_bias = self.relative_position_bias_table[
    self.relative_position_index.view(-1)].view(N, N, -1)
attn = attn + relative_position_bias.unsqueeze(0)
...
attn = self.attn_drop(attn)
x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

# VoCo (lines 515-538)
q = q * self.scale
attn = q @ k.transpose(-2, -1)
relative_position_bias = self.relative_position_bias_table[
    self.relative_position_index.clone()[:n, :n].reshape(-1)].reshape(n, n, -1)
attn = attn + relative_position_bias.unsqueeze(0)
...
attn = self.attn_drop(attn).to(v.dtype)  # <-- dtype conversion
x = (attn @ v).transpose(1, 2).reshape(b, n, c)
```
**Key differences:**
1. **Position index access**: SMIT uses `.view(-1)` (static), VoCo uses `.clone()[:n, :n].reshape(-1)` (dynamic, supports variable window sizes at inference)
2. **Dtype conversion**: VoCo adds `.to(v.dtype)` after dropout for mixed-precision stability

**Result: FUNCTIONALLY DIFFERENT** — VoCo is more robust for dynamic inference

### 2.6 Initialization
```python
# Both use identical initialization:
trunc_normal_(self.relative_position_bias_table, std=0.02)
```
**Result: IDENTICAL**

---

## 3. SwinTransformerBlock

### 3.1 Architecture Pattern
Both use **pre-norm** architecture:
```
Input → Norm1 → WindowAttention → DropPath → Add(residual)
      → Norm2 → MLP → DropPath → Add(residual)
```
**Result: IDENTICAL** pattern

### 3.2 MLP Implementation

```python
# SMIT (TransMorph.py, lines 37-53) — custom Mlp class
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)        # layer name: fc1
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)        # layer name: fc2
        x = self.drop(x)
        return x

# VoCo (swin_unetr.py, line 25) — MONAI MLPBlock
from monai.networks.blocks import MLPBlock as Mlp
# Uses layer names: linear1, linear2
# Constructor: Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")
```

**Key differences:**
1. **Parameter naming**: `fc1`/`fc2` vs `linear1`/`linear2`
   - This means **pretrained weights need key remapping**: `fc1→linear1`, `fc2→linear2`
2. **Constructor interface**: Different parameter names
3. **Dropout mode**: VoCo uses `dropout_mode="swin"`

**Result: MATHEMATICALLY EQUIVALENT**, but weight keys differ (requires remapping for weight transfer)

### 3.3 Window Shift Computation
```python
# SMIT
shift_size = (window_size[0]//2, window_size[1]//2, window_size[2]//2)
# w=4: shift = (2, 2, 2)

# VoCo
shift_size = tuple(i // 2 for i in window_size)
# w=7: shift = (3, 3, 3)
```
**Result: SAME LOGIC**, different values due to window_size

### 3.4 Padding for Non-Divisible Sizes
```python
# Both implementations:
pad_r = (window_size[0] - H % window_size[0]) % window_size[0]
# ... same for other dimensions
x = F.pad(x, ...)
```
**Result: IDENTICAL** logic

### 3.5 Gradient Checkpointing
```python
# SMIT
x = checkpoint.checkpoint(blk, x, attn_mask)  # use_reentrant=True (default)

# VoCo
x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
```
**Result: FUNCTIONALLY EQUIVALENT**, VoCo uses more efficient `use_reentrant=False`

---

## 4. PatchMerging (Downsampling Between Stages)

### 4.1 Concatenation Order — **CRITICAL DIFFERENCE**
```python
# SMIT (TransMorph.py, lines 474-482) — explicit spatial ordering
x0 = x[:, 0::2, 0::2, 0::2, :]  # (0,0,0)
x1 = x[:, 1::2, 0::2, 0::2, :]  # (1,0,0)
x2 = x[:, 0::2, 1::2, 0::2, :]  # (0,1,0)
x3 = x[:, 0::2, 0::2, 1::2, :]  # (0,0,1)
x4 = x[:, 1::2, 1::2, 0::2, :]  # (1,1,0)
x5 = x[:, 0::2, 1::2, 1::2, :]  # (0,1,1)
x6 = x[:, 1::2, 0::2, 1::2, :]  # (1,0,1)
x7 = x[:, 1::2, 1::2, 1::2, :]  # (1,1,1)
x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
# Order: (000),(100),(010),(001),(110),(011),(101),(111)

# VoCo (swin_unetr.py PatchMergingV2, lines 739-741) — itertools.product ordering
x = torch.cat(
    [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
)
# itertools.product produces: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
# Order: (000),(001),(010),(011),(100),(101),(110),(111)
```

**Mapping between orderings:**
```
SMIT position → VoCo position
x0 (000) → position 0 (same)
x1 (100) → position 4
x2 (010) → position 2
x3 (001) → position 1
x4 (110) → position 6
x5 (011) → position 3
x6 (101) → position 5
x7 (111) → position 7 (same)
```

**Result: FUNDAMENTALLY DIFFERENT**
- The Linear reduction layer receives features in different channel ordering
- **Pretrained weights for the `reduction` Linear layer are NOT directly transferable**
- Same weights applied to different feature arrangements will produce different outputs
- This is a **structural incompatibility** that cannot be resolved by key remapping alone

### 4.2 Linear Reduction
```python
# SMIT (line 453) — parameterized by reduce_factor
self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
# With reduce_factor=4: Linear(8*dim, 2*dim) → 8C → 2C

# VoCo (line 728) — hardcoded
self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
# Linear(8*dim, 2*dim) → 8C → 2C
```
**Result: SAME OUTPUT DIMENSION** (both produce 2×dim), but different internal structure due to concatenation order above.

### 4.3 Norm Placement
```python
# Both:
x = self.norm(x)      # LayerNorm on 8*dim BEFORE reduction
x = self.reduction(x)  # Linear 8*dim → 2*dim
```
**Result: IDENTICAL** placement

### 4.4 Channel Evolution Through Stages
Both produce identical feature map shapes:

| Stage | Spatial | Channels |
|-------|---------|----------|
| 0 | 48×48×48 | 48 |
| 1 | 24×24×24 | 96 |
| 2 | 12×12×12 | 192 |
| 3 | 6×6×6 | 384 |
| Bottleneck | 3×3×3 | 768 |

**Result: IDENTICAL** feature shapes

---

## 5. PatchEmbedding

### 5.1 Implementation
```python
# SMIT (TransMorph.py, line 615)
self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

# VoCo — uses MONAI PatchEmbed
from monai.networks.blocks import PatchEmbed
# Also uses Conv3d with same kernel_size=stride=patch_size
```
**Result: FUNCTIONALLY IDENTICAL**

### 5.2 Post-Embedding Normalization
```python
# Both apply optional LayerNorm after patch embedding when patch_norm=True
```
**Result: IDENTICAL**

---

## 6. BasicLayer (Stage Container)

### 6.1 Block Construction
```python
# Both create SwinTransformerBlock list with alternating shift_size:
# Even blocks: shift_size = (0, 0, 0)
# Odd blocks: shift_size = (ws//2, ws//2, ws//2)
```
**Result: IDENTICAL** logic

### 6.2 Attention Mask Computation
```python
# SMIT — computes mask in BasicLayer.forward() every forward pass
# VoCo — uses compute_mask() function, also computed per forward

# Both use same formula:
img_mask[:, h, w, t, :] = cnt  # 27 regions for 3D
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
```
**Result: FUNCTIONALLY EQUIVALENT**

### 6.3 Dynamic Window Size (VoCo only)
```python
# VoCo has get_window_size() function that reduces window_size
# when feature map is smaller than window_size
window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
```
SMIT does not have this — relies on padding instead.

**Result: DIFFERENT** edge case handling, but both work correctly

---

## 7. Full Model: Encoder + Decoder

### 7.1 use_v2 Extra Conv Blocks — **VoCo ONLY**

VoCo adds a `UnetrBasicBlock` (3×3×3 conv + InstanceNorm + residual) **before** each Swin stage:

```python
# VoCo SwinTransformer forward (lines 1065-1080):
if self.use_v2:
    x0 = self.layers1c[0](x0.contiguous())  # Extra conv block
x1 = self.layers1[0](x0.contiguous())       # Swin transformer blocks

if self.use_v2:
    x1 = self.layers2c[0](x1.contiguous())  # Extra conv block
x2 = self.layers2[0](x1.contiguous())       # Swin transformer blocks
# ... same for stages 3 and 4
```

Each `layerc` is:
```python
UnetrBasicBlock(
    spatial_dims=3,
    in_channels=embed_dim * 2**i_layer,
    out_channels=embed_dim * 2**i_layer,
    kernel_size=3, stride=1,
    norm_name="instance",
    res_block=True,
)
```

**SMIT has NO equivalent.** This adds:
- Local feature refinement before each transformer stage
- Additional parameters (~8M extra)
- Inductive bias toward local features

**Result: MAJOR STRUCTURAL DIFFERENCE**

### 7.2 Skip Connections
```python
# SMIT (lines 2206-2231)
enc0 = self.encoder1(x_in)           # Raw input → feature_size
enc1 = self.encoder2(enc11)          # Stage 0 output
enc2 = self.encoder3(enc22)          # Stage 1 output
enc3 = self.encoder4(enc33)          # Stage 2 output
dec4 = self.encoder10(x)             # Bottleneck
dec3 = self.decoder5(dec4, enc44)    # Stage 3 output as skip
dec2 = self.decoder4(dec3, enc3)
dec1 = self.decoder3(dec2, enc2)
dec0 = self.decoder2(dec1, enc1)
out = self.decoder1(dec0, enc0)

# VoCo (lines 322-337) — IDENTICAL pattern
enc0 = self.encoder1(x_in)
enc1 = self.encoder2(hidden_states_out[0])
enc2 = self.encoder3(hidden_states_out[1])
enc3 = self.encoder4(hidden_states_out[2])
dec4 = self.encoder10(hidden_states_out[4])
dec3 = self.decoder5(dec4, hidden_states_out[3])
dec2 = self.decoder4(dec3, enc3)
dec1 = self.decoder3(dec2, enc2)
dec0 = self.decoder2(dec1, enc1)
out = self.decoder1(dec0, enc0)
```
**Result: IDENTICAL** skip connection pattern

### 7.3 Encoder Processing Blocks
```python
# SMIT — uses custom UnetrBasicBlock_No_DownSampling for encoder2-4
self.encoder2 = UnetrBasicBlock_No_DownSampling(...)  # Custom class
self.encoder3 = UnetrBasicBlock_No_DownSampling(...)
self.encoder4 = UnetrBasicBlock_No_DownSampling(...)

# VoCo — uses standard MONAI UnetrBasicBlock
self.encoder2 = UnetrBasicBlock(...)  # Standard MONAI
self.encoder3 = UnetrBasicBlock(...)
self.encoder4 = UnetrBasicBlock(...)
```
`UnetrBasicBlock_No_DownSampling` uses `UnetResBlock_No_Downsampleing` internally, which has `stride=stride` in conv1 but ensures no spatial downsampling.

**Result: LIKELY EQUIVALENT** but implementation details may differ slightly

### 7.4 Decoder Blocks
```python
# Both use standard MONAI UnetrUpBlock for all 5 decoder levels
self.decoder5 = UnetrUpBlock(...)  # 16*fs → 8*fs
self.decoder4 = UnetrUpBlock(...)  # 8*fs → 4*fs
self.decoder3 = UnetrUpBlock(...)  # 4*fs → 2*fs
self.decoder2 = UnetrUpBlock(...)  # 2*fs → fs
self.decoder1 = UnetrUpBlock(...)  # fs → fs
self.out = UnetOutBlock(...)       # fs → out_channels
```
**Result: IDENTICAL**

---

## 8. Position Embeddings

### 8.1 Absolute Position Embedding (APE)
```python
# SMIT — supports APE (disabled by default: config.ape=False)
if self.ape:
    self.absolute_pos_embed = nn.Parameter(...)
    # Added to features after patch embedding

# VoCo — NO APE support
```
**Result: DIFFERENT** (SMIT has option, VoCo doesn't). Both currently use relative-only.

### 8.2 Sinusoidal Position Embedding (SPE)
```python
# SMIT — supports SPE (disabled: config.spe=False)
elif self.spe:
    self.pos_embd = SinPositionalEncoding3D(embed_dim)

# VoCo — NO SPE support
```
**Result: DIFFERENT** (SMIT has option, VoCo doesn't). Currently disabled in SMIT config.

---

## 9. Weight Initialization

```python
# Both use:
trunc_normal_(linear.weight, std=0.02)
nn.init.constant_(linear.bias, 0)
nn.init.constant_(layernorm.bias, 0)
nn.init.constant_(layernorm.weight, 1.0)
```
**Result: IDENTICAL**

---

## 10. Weight Transfer Compatibility

### Key Mapping: SMIT → VoCo (MONAI SwinUNETR)
```
module.backbone.transformer.patch_embed.X    → swinViT.patch_embed.X
module.backbone.transformer.layers.0.X       → swinViT.layers1.0.X
module.backbone.transformer.layers.1.X       → swinViT.layers2.0.X
module.backbone.transformer.layers.2.X       → swinViT.layers3.0.X
module.backbone.transformer.layers.3.X       → swinViT.layers4.0.X
*.mlp.fc1.*                                  → *.mlp.linear1.*
*.mlp.fc2.*                                  → *.mlp.linear2.*
```

### What transfers successfully (same window_size):
- ✅ Patch embedding weights (Conv3d)
- ✅ All attention weights (qkv, proj)
- ✅ All MLP weights (after fc→linear renaming)
- ✅ All LayerNorm weights
- ✅ All PatchMerging norm weights
- ✅ Relative position bias table (if same window_size)
- **Total: 210 / 251 keys matched** (w=4 to w=4)

### What does NOT transfer:
- ❌ PatchMerging reduction weights (different concat order → different learned mapping)
- ❌ Relative position bias table (if different window_size: 343 vs 2197)
- ❌ use_v2 conv blocks (layers1c/2c/3c/4c) — SMIT has no equivalent
- ❌ Decoder weights — not in pretrained encoder checkpoint
- ❌ Encoder processing blocks (encoder2-10) — not in pretrained checkpoint

### Critical: PatchMerging Weight Incompatibility
Even when key names match, the PatchMerging `reduction` Linear layer weights are **semantically incompatible** due to different concatenation order. The same weight matrix applied to SMIT's `[x0,x1,x2,x3,x4,x5,x6,x7]` ordering vs VoCo's `[x0,x3,x2,x5,x1,x6,x4,x7]` ordering will produce different results.

**Quantification:** There are 4 PatchMerging layers (one per stage). Each has:
- Stage 0: Linear(384, 96) = 36,864 params
- Stage 1: Linear(768, 192) = 147,456 params
- Stage 2: Linear(1536, 384) = 589,824 params
- Stage 3: Linear(3072, 768) = 2,359,296 params
- **Total affected: ~3.1M params** out of ~64.7M (4.8%)

---

## 11. Summary of All Differences

### Configurable Differences (can be matched)
| Difference | Impact | Can Match? |
|-----------|--------|-----------|
| depths: (2,2,8,2) vs (2,2,2,2) | High | ✅ Config change |
| num_heads: (4,4,8,16) vs (3,6,12,24) | High | ✅ Config change |
| window_size: 4 vs 7 | High | ✅ Monkey-patch |
| drop_path_rate: 0.1 vs 0.0 | Medium | ✅ Config change |
| qkv_bias default | Low | ✅ Config change |

### Structural Differences (cannot be matched without code changes)
| Difference | Impact | Can Match? |
|-----------|--------|-----------|
| **PatchMerging concat order** | **High** | ❌ Different code |
| **use_v2 extra conv blocks** | **High** | ❌ VoCo-only feature |
| MLP class (fc1/fc2 vs linear1/linear2) | Low | ✅ Key remapping |
| WindowAttention dynamic indexing | Low | N/A |
| Encoder block class | Low | Likely equivalent |
| APE/SPE support | Low | Both disabled |

### Implications for Fair Comparison

To do a **truly fair SSL method comparison** (SMIT pretraining vs VoCo pretraining), you need **one of the following**:

**Option A: Pretrain SMIT on VoCo's architecture (MONAI SwinUNETR)**
- Use MONAI SwinUNETR with depths=(2,2,2,2), heads=(3,6,12,24), window=7
- Apply SMIT's pretraining losses (MIM + self-distillation)
- This eliminates ALL architectural differences
- PatchMerging order, use_v2 blocks, MLP naming — all become identical
- **Recommended approach**

**Option B: Implement VoCo pretraining on TransMorph architecture**
- Use TransMorph with depths=(2,2,8,2), heads=(4,4,8,16), window=4
- Apply VoCo's contrastive pretraining loss
- Keeps your architecture, changes only the SSL method
- Still has PatchMerging order difference vs published VoCo results

**Option C: Report results with architectural caveats**
- Run both models as-is with their published configs
- Document all differences in the paper
- Acknowledge that performance differences may be due to architecture, not just SSL method

---

## 12. References

1. **Swin Transformer V1**: Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
2. **Swin Transformer V2**: Liu, Z., et al. "Swin Transformer V2: Scaling Up Capacity and Resolution." CVPR 2022.
3. **SwinUNETR**: Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." MICCAI BrainLes 2022.
4. **VoCo**: Wu, L., Zhuang, J., & Chen, H. "VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis." CVPR 2024.
5. **SMIT**: Jiang, J., & Veeraraghavan, H. "Self-distilled Masked Image Transformer for Medical Image Segmentation." (SMIT paper)
6. **TransMorph**: Chen, J., et al. "TransMorph: Transformer for Unsupervised Medical Image Registration." Medical Image Analysis 2022.
7. **nnU-Net Revisited**: Isensee, F., et al. "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation." MICCAI 2024.
8. **MONAI**: MONAI Consortium. "MONAI: Medical Open Network for AI." https://monai.io/

---

## Appendix: Current Experiment Matrix

| # | Machine | GPU | Model Code | Architecture Config | Pretrained | wandb name |
|---|---------|-----|-----------|-------------------|------------|------------|
| 1 | lc01 | 0 | MONAI SwinUNETR | (2,2,2,2) w=7 h=(3,6,12,24) | VoCo SSL | VoCo_pretrained |
| 2 | lc01 | 1 | MONAI SwinUNETR | (2,2,2,2) w=7 h=(3,6,12,24) | None | VoCo_scratch |
| 3 | lc01 | 2 | TransMorph | (2,2,2,2) w=7 h=(3,6,12,24) pat_merg_rf=4 | None | SMIT_VoCo_arch_scratch |
| 4 | lc01 | 3 | TransMorph | (2,2,8,2) w=4 h=(4,4,8,16) pat_merg_rf=4 | None | SMIT_scratch |
| 5 | lc05 | 0 | MONAI SwinUNETR | (2,2,8,2) w=7 h=(4,4,8,16) | SMIT SSL (182/210 matched) | MONAI_SMIT_w7_pretrained |
| 6 | lc05 | 1 | MONAI SwinUNETR | (2,2,8,2) w=7 h=(4,4,8,16) | None | MONAI_SMIT_w7_scratch |
| 7 | lc05 | 2 | MONAI SwinUNETR | (2,2,8,2) w=4 h=(4,4,8,16) | SMIT SSL (210/210 matched) | MONAI_SMIT_w4_pretrained |
| 8 | lc05 | 3 | MONAI SwinUNETR | (2,2,8,2) w=4 h=(4,4,8,16) | None | MONAI_SMIT_w4_scratch |

### Key Comparisons:
- **1 vs 2**: VoCo SSL pretraining value
- **2 vs 3**: MONAI vs TransMorph code difference (same VoCo config)
- **4 vs 8**: TransMorph vs MONAI code difference (same SMIT config, w=4)
- **7 vs 8**: SMIT SSL pretraining value (on MONAI architecture, w=4, perfect weight match)
- **5 vs 6**: SMIT SSL pretraining value (on MONAI architecture, w=7, partial weight match)
- **6 vs 2**: SMIT depths (2,2,8,2) vs VoCo depths (2,2,2,2) on MONAI
- **6 vs 8**: Window size 7 vs 4 effect on MONAI with SMIT config
