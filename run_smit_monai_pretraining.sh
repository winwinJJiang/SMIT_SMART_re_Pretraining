#!/bin/bash
# SMIT pretraining with MONAI SwinUNETR V2 backbone
# Configurable: img_size, depths, num_heads, window_size, embed_dim

# ====== Architecture (VoCo default) ======
embed_dim=48
depths="2 2 2 2"
num_heads="3 6 12 24"
window_size=7

# ====== Input size (change as needed: 96 or 128) ======
img_size3D_x=96
img_size3D_y=96
img_size3D_z=96

# ====== Training ======
epoch=500
batch_size_per_gpu=7
lr=0.0008
teacher_mask_ratio=0.3
img_rec_weight=60.0
use_fp16=1
pred_start_epoch=400

# ====== Data ======
data_path="/data1/lia5/Jue/Data/CT/"
data_txt_list="previous_14k_data.txt"

# ====== Output ======
output="snapshots/SMIT_MONAI_VoCo_arch_${img_size3D_x}_pretraining"

hst_name=$(hostname)

torchrun --nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr=$hst_name \
--master_port=12546 main_smit_pretraining.py \
--use_monai_backbone=1 \
--monai_embed_dim=$embed_dim \
--monai_depths $depths \
--monai_num_heads $num_heads \
--monai_window_size=$window_size \
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
--use_att_mask=1 \
--Patch_Token_at_3_block=1 \
--teacher_student_same_mask=0 \
--mask_pred_ratio=0.75 \
--Cait_layer=2 \
--saveckp_freq=50 \
--warmup_epochs=40 \
--lr=$lr \
--num_workers=12 \
--momentum_teacher=0.996 \
--img_rec_weight=$img_rec_weight \
--clip_grad=0.3 \
--intensity_Aug=1 \
--image_distil=0 \
--img_dis_weight=0 \
--image_distil_start_epoch=-100 \
--teacher_rec=0 \
--att_threshold=0.1 \
--flip_prob=0.2 \
--use_fp16=$use_fp16 \
--img_size3D_x=$img_size3D_x \
--img_size3D_y=$img_size3D_y \
--img_size3D_z=$img_size3D_z \
--pred_start_epoch=$pred_start_epoch
