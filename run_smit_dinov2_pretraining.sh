#!/bin/bash
# SMIT pretraining with ALL DINOv2 improvements
# SK centering + KoLeo + FlashAttention + high drop path + untied heads + larger out_dim

# Architecture (VoCo)
embed_dim=48
depths="2 2 2 2"
num_heads="3 6 12 24"
window_size=7

# Input
img_size3D_x=96
img_size3D_y=96
img_size3D_z=96

# Training
epoch=500
batch_size_per_gpu=7
lr=0.0008
use_fp16=1
pred_start_epoch=400

# === DINOv2 improvements ===
use_dinov2_loss=1       # SK centering for CLS + patch
use_koleo=1             # KoLeo regularizer
koleo_weight=0.1        # KoLeo weight
rec_weight=10.0         # Reconstruction weight (reduced from 60)
use_flash_attn=1        # FlashAttention via SDPA
drop_path_rate=0.3      # Higher stochastic depth (was 0.1)
out_dim=65536           # Larger prototype vocabulary (was 8192)
shared_head_teacher=0   # Untied teacher heads (was 1)

# SMIT-specific (disabled for baseline)
use_att_mask=0
teacher_mask_ratio=0

# Data
data_path="/data1/lia5/Jue/Data/CT/"
data_txt_list="previous_14k_data.txt"

# Output
output="snapshots/SMIT_MONAI_VoCo_arch_96_dinov2_full"

hst_name=$(hostname)

export PATH=/data1/lia5/Jue/envs/smit_pretrain/bin:$PATH
export WANDB_API_KEY=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

torchrun --nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr=$hst_name \
--master_port=12547 main_smit_pretraining_dinov2.py \
--use_monai_backbone=1 \
--monai_embed_dim=$embed_dim \
--monai_depths $depths \
--monai_num_heads $num_heads \
--monai_window_size=$window_size \
--use_dinov2_loss=$use_dinov2_loss \
--use_koleo=$use_koleo \
--koleo_weight=$koleo_weight \
--rec_weight_override=$rec_weight \
--use_flash_attn=$use_flash_attn \
--out_dim=$out_dim \
--patch_out_dim=$out_dim \
--shared_head_teacher=$shared_head_teacher \
--interval=2 \
--arch="model_small" \
--data_path=$data_path \
--list_path3D=$data_txt_list \
--output_dir=$output \
--batch_size_per_gpu=$batch_size_per_gpu \
--epochs=$epoch \
--teacher_mask_ratio=$teacher_mask_ratio \
--resume=0 \
--use_3D_resize=1 \
--use_att_mask=$use_att_mask \
--Patch_Token_at_3_block=1 \
--teacher_student_same_mask=0 \
--mask_pred_ratio=0.75 \
--Cait_layer=2 \
--saveckp_freq=50 \
--warmup_epochs=40 \
--lr=$lr \
--num_workers=12 \
--momentum_teacher=0.996 \
--img_rec_weight=60.0 \
--clip_grad=0.3 \
--intensity_Aug=1 \
--image_distil=0 \
--img_dis_weight=0 \
--image_distil_start_epoch=-100 \
--teacher_rec=0 \
--att_threshold=0.1 \
--flip_prob=0.5 \
--use_fp16=$use_fp16 \
--img_size3D_x=$img_size3D_x \
--img_size3D_y=$img_size3D_y \
--img_size3D_z=$img_size3D_z \
--pred_start_epoch=$pred_start_epoch \
--wandb_project=SMIT_pretraining \
--wandb_run=SMIT_MONAI_dinov2_full_96
