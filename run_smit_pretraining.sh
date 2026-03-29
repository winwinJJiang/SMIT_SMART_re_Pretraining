lr=1e-3
b2=0.98
eps=1e-6
dpr=0.1
ls=0.0

#Below are the Swin configrations

#Swin-T: C = 96, layer numbers = {2, 2, 6, 2}   num_heads = (3, 6, 12, 24)
#Swin-S: C = 96, layer numbers ={2, 2, 18, 2}    num_heads =  (3, 6, 12, 24)
#Swin-B: C = 128, layer numbers ={2, 2, 18, 2}  num_heads = (4, 8, 16, 32)
#Swin-L: C = 192, layer numbers ={2, 2, 18, 2}  (6, 12, 24, 48)
batch_size_per_gpu=7 #Swin_S
#batch_size_per_gpu=5 #Swin_B


#CT data only

output='snapshots/Swin_SMART_14K_CT_20241030_Swin_B_192_192_'
data_path='/data1/lia5/Jue/Data/CT/'
data_txt_list='previous_14k_data.txt'
epoch=500
teacher_mask_ratio=0.3
use_3D_resize=1
use_att_mask=1
img_rec_weight=60.0
saveckp_freq=50
use_fp16=1
pred_start_epoch=400
## input size and batch size 
img_size3D_x=192
img_size3D_y=192
img_size3D_z=64
batch_size_per_gpu=5


hst_name=$(hostname)
#hst_name=iscb024.mskcc.org # iscb[024, 025,027,028]

torchrun -m torch.distributed.launch --nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
--master_addr=$hst_name \
--master_port=12546 main_swin_ASM_Teacher_feature_distil_img_recl_128_CaiT_AttMask_2_loss.py \
--interval=2 \
--arch='model_small' \
--data_path=$data_path \
--list_path3D=$data_txt_list \
--output_dir=$output \
--batch_size_per_gpu=$batch_size_per_gpu \
--epochs=$epoch \
--teacher_mask_ratio=$teacher_mask_ratio \
--resume=0 \
--use_3D_resize=$use_3D_resize \
--use_att_mask=$use_att_mask \
--Patch_Token_at_3_block=1 \
--teacher_student_same_mask=0 \
--mask_pred_ratio=0.75 \
--Cait_layer=2 \
--saveckp_freq=200 \
--warmup_epochs=40 \
--lr=0.0008 \
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
--pred_start_epoch=$pred_start_epoch \