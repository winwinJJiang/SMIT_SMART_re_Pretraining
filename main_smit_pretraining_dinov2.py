import matplotlib
matplotlib.use('Agg')
try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False
#import UnetsegLSTM
#matplotlib.use('pdf')

import matplotlib.pyplot as plt

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
import models.MiT as MiTs
#from models.MiT import Head
from models.MiT import iBOTHead_w_Rec_2_Loss_Att as iBOTHead #as Head 
#from ibot_models.head import iBOTHead as Head 
from swin_models.TransMorph import CONFIGS as CONFIGS_TM
import swin_models.TransMorph as TransMorph
from monai_backbone_fixed import SMIT_MONAI_Pretrain_Model  # Fix: use fixed version
from dinov2_loss.dino_clstoken_loss import DINOLoss
from dinov2_loss.ibot_patch_loss import iBOTPatchLoss
from dinov2_loss.koleo_loss_fixed import KoLeoLoss
#import swin_models.TransMorphV2 as TransMorph
#import swin_models.TransMorphV2_Bake as TransMorph



from att_mask.attmask import AttMask,AttMask_Debug
from data_loader3D_mask_ASM_Teacher_fixed import Dataset3D,Dataset3D_No_Intensity_Aug,Dataset3D_Jue_Custmzed,Dataset3D_Jue_Custmzed_CT_and_MRI,Dataset3D_Jue_Custmzed_CT_and_MRI_Not_Square,Dataset3D_Jue_Custmzed_CT_Not_Square  # Fix #13
#from data_loader2D import Dataset2D
import nibabel as nib
fig = plt.figure()
ax = fig.add_subplot(211)


def get_args_parser():
    parser = argparse.ArgumentParser('UniMiSS', add_help=False)

    # Model parameters
    parser.add_argument('--interval', default=2, type=int)
    parser.add_argument('--arch', default='model_small', type=str, choices=['model_tiny', 'model_small'],
                        help="""Name of architecture to train.""")

    parser.add_argument("--img_size3D_x", type=int, default=128,help="Size of the 3D image in [x, y, z] format")
    parser.add_argument("--img_size3D_y", type=int,default=128, help="Size of the 3D image in [x, y, z] format")
    parser.add_argument("--img_size3D_z", type=int, default=128,help="Size of the 3D image in [x, y, z] format")

    parser.add_argument('--img_size2D', default=224, type=int,
                        help="""Size in pixels of input square 2D patches.""")
    parser.add_argument('--img_size3D', default=[128,128,128], type=int,
                        help="""Size in pixels of input square 3D patches.""")

    #parser.add_argument('--out_dim', default=65536, type=int,
    #                    help="""Dimensionality of the UniMiSS head output.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the UniMiSS head.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                        help="""Base EMA parameter for teacher update.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help="""Final value (after linear warmup) of the teacher temperature.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--pred_start_epoch', default=30, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--Cait_layer', default=2, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")    
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")    
    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False,
                        help="""Whether or not to use half precision for training.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=0.3,
                        help="""Maximal parameter gradient norm if using gradient clipping.""")
    parser.add_argument('--teacher_mask_ratio', type=float, default=0.3,
                        help="""Maximal parameter gradient norm if using gradient clipping.""")
    parser.add_argument('--batch_size_per_gpu', default=4, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs during which we keep the output layer fixed. 
                        Typically doing so during the first epoch helps training. 
                        Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--resume', default=0, type=int,
                        help="""Number of epochs during which we keep the output layer fixed. 
                        Typically doing so during the first epoch helps training. 
                        Try increasing this value if the loss does not decrease.""")        
    parser.add_argument("--lr", default=0.0008, type=float,
                        help="""Learning rate at the end of linear warmup (highest LR used during training).""")
    parser.add_argument("--att_threshold", default=0.1, type=float,
                        help="""Learning rate at the end of linear warmup (highest LR used during training).""")              
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--image_distil", default=0, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")    
    parser.add_argument("--image_distil_start_epoch", default=0, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")          
    parser.add_argument("--teacher_rec", default=0, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")       
    parser.add_argument("--teacher_student_same_mask", default=0, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")                                
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="""Target LR at the end of optimization.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer.""")
    parser.add_argument('--pre_rec_path', default=None, type=str, help="""Type of optimizer.""")
    parser.add_argument('--att_masking_prob', type=float, default=1.0, help=""""Perform token masking 
                        based on attention with specific probability, works only for --pred_shape attmask_high, attmask_hint, attmask_low""")  
    parser.add_argument('--att_pred_shape', default='attmask_hint', type=str, help="""Shape of partial prediction. 
                        Select between attmask_high, attmask_hint, attmask_low, rand or block""")   
    parser.add_argument('--show_max', type=float, default=0.1, help="The top salient tokens from which a random sample will be revealed")    
    parser.add_argument('--mask_pred_ratio', type=float, default=0.7, help="The top salient tokens from which a random sample will be revealed")   
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--intensity_Aug', default=1, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--use_3D_resize', default=1, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--use_monai_backbone', default=0, type=int,
                        help='Use MONAI SwinUNETR backbone (VoCo architecture) instead of TransMorph')
    parser.add_argument('--monai_embed_dim', default=48, type=int, help='MONAI backbone embed dim')
    parser.add_argument('--monai_depths', nargs=4, type=int, default=[2,2,2,2], help='MONAI backbone depths')
    parser.add_argument('--monai_num_heads', nargs=4, type=int, default=[3,6,12,24], help='MONAI backbone num_heads')
    parser.add_argument('--monai_window_size', default=7, type=int, help='MONAI backbone window size')
    parser.add_argument('--use_flash_attn', default=0, type=int, help='Use FlashAttention via PyTorch SDPA')
    parser.add_argument('--use_dinov2_loss', default=0, type=int, help='Use DINOv2-style SK losses instead of EMA center')
    parser.add_argument('--use_koleo', default=0, type=int, help='Add KoLeo regularizer')
    parser.add_argument('--koleo_weight', default=0.1, type=float, help='KoLeo loss weight')
    parser.add_argument('--rec_weight_override', default=0.0, type=float, help='Override img_rec_weight (0=use original)')
    parser.add_argument('--wandb_project', default='SMIT_pretraining', type=str, help='wandb project')
    parser.add_argument('--wandb_run', default='', type=str, help='wandb run name')
    parser.add_argument('--use_att_mask', default=1, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--flip_prob', default=0.2, type=float, help=""""See 
        `--image_rec weight`""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")    
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--Patch_Token_at_3_block', default=1, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    # Others
    parser.add_argument('--img_rec_weight', default=10.0, type=float, help=""""See 
        `--image_rec weight`""")
    parser.add_argument('--img_dis_weight', default=1.0, type=float, help=""""See 
        `--image_rec weight`""")
    parser.add_argument('--data_path', default='../data/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--list_path2D', default='2D_images.txt', type=str)
    parser.add_argument('--list_path3D', default='3D_images.txt', type=str)
    parser.add_argument('--output_dir', default="snapshots", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=200, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_func(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    #args.data_path='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'

    print (args.teacher_mask_ratio)
    args.img_size3D=[args.img_size3D_x,args.img_size3D_y,args.img_size3D_z]
    use_CT_only=True

    if use_CT_only:
        #CT only 
        #train_set3D = Dataset3D_Jue_Custmzed(args.data_path, args.list_path3D,args.teacher_mask_ratio, crop_size_3D=args.img_size3D,use_intencty_Aug=args.intensity_Aug,used_3D_resize=args.use_3D_resize)
        #if args.intensity_Aug==1:
        #    train_set3D = Dataset3D(args.data_path, args.list_path3D,args.teacher_mask_ratio, crop_size_3D=args.img_size3D)
        #else:
        #    train_set3D = Dataset3D_No_Intensity_Aug(args.data_path, args.list_path3D,args.teacher_mask_ratio, crop_size_3D=args.img_size3D)
        # Below use cusmized input size 

        
        #CT  with customised input 
        train_set3D = Dataset3D_Jue_Custmzed_CT_Not_Square(args.data_path, args.list_path3D,0, crop_size_3D=args.img_size3D,use_intencty_Aug=0,used_3D_resize=1)


    else:
        #CT and MRI

        train_set3D = Dataset3D_Jue_Custmzed_CT_and_MRI_Not_Square(args.data_path, args.list_path3D,args.teacher_mask_ratio, crop_size_3D=args.img_size3D,use_intencty_Aug=args.intensity_Aug,used_3D_resize=args.use_3D_resize)


    data_loader3D = torch.utils.data.DataLoader(
        train_set3D,
        sampler=torch.utils.data.DistributedSampler(train_set3D, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Data loaded: there are {len(train_set3D)} images.")

    # ============ building student and teacher networks ... ============
    

    # === FlashAttention via SDPA ===
    if args.use_flash_attn:
        import monai.networks.nets.swin_unetr as swin_mod
        _orig_wa_forward = swin_mod.WindowAttention.forward
        def _flash_wa_forward(self, x, mask):
            b, n, c = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # Use PyTorch SDPA (auto-selects FlashAttention when possible)
            attn_mask_sdpa = None
            if mask is not None:
                nw = mask.shape[0]
                attn_mask_sdpa = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(b, self.num_heads, n, n)
            # Add relative position bias
            rel_pos_bias = self.relative_position_bias_table[
                self.relative_position_index.clone()[:n, :n].reshape(-1)
            ].reshape(n, n, -1).permute(2, 0, 1).contiguous().unsqueeze(0)
            if attn_mask_sdpa is not None:
                attn_mask_sdpa = attn_mask_sdpa + rel_pos_bias
            else:
                attn_mask_sdpa = rel_pos_bias.expand(b, -1, -1, -1)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask_sdpa.to(q.dtype),
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(b, n, c)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        swin_mod.WindowAttention.forward = _flash_wa_forward
        print("FlashAttention enabled via SDPA monkey-patch")

    if args.use_monai_backbone:
        # === MONAI SwinUNETR V2 backbone (VoCo architecture) ===
        import ml_collections
        config = ml_collections.ConfigDict()
        config.patch_size = 2
        config.in_chans = 1
        config.embed_dim = args.monai_embed_dim
        config.depths = tuple(args.monai_depths)
        config.num_heads = tuple(args.monai_num_heads)
        config.window_size = tuple([args.monai_window_size] * 3)
        config.mlp_ratio = 4
        config.qkv_bias = True
        config.drop_rate = 0
        config.drop_path_rate = 0.1
        config.use_checkpoint = False
        config.img_size = args.img_size3D
        
        print(f'Using MONAI SwinUNETR V2 backbone')
        print(f'  embed_dim={config.embed_dim}')
        print(f'  depths={config.depths}')
        print(f'  num_heads={config.num_heads}')
        print(f'  window_size={config.window_size}')
        print(f'  img_size={config.img_size}')
        
        student = SMIT_MONAI_Pretrain_Model(config, Cait_layer=args.Cait_layer)
        student.use_MIM_mask = True
        
        config.drop_path_rate = 0.0
        teacher = SMIT_MONAI_Pretrain_Model(config, Cait_layer=args.Cait_layer)
        teacher.use_MIM_mask = True
        if args.teacher_mask_ratio == 0:
            teacher.use_MIM_mask = False
        
        embed_dim = config.embed_dim * 2 * 2 * 2  # CaiT dim = embed_dim * 8
    else:
        # === Original TransMorph backbone ===
        config = CONFIGS_TM['TransMorph-Large_SSIM_pre_train_128_middle_bias_True_Swin_B']
        config.img_size = args.img_size3D
        config.drop_path_rate = 0.1
        config.Cait_layer = args.Cait_layer
        
        if args.Patch_Token_at_3_block == 1:
            student = TransMorph.Trans_SMIT_pre_train_cls_patch_rec_Student_CaiT_All_3_Loss(config)
            print('info: loading model with patch token at block 3')
        else:
            student = TransMorph.Trans_SMIT_pre_train_cls_patch_rec_Student_CaiT_All_3_Loss_Patch_Token_At_4_Block(config)
            print('info: loading model with patch token at block 4')
        student.use_MIM_mask = True
        
        config.drop_path_rate = 0.0
        config.Cait_layer = args.Cait_layer
        if args.Patch_Token_at_3_block == 1:
            teacher = TransMorph.Trans_SMIT_pre_train_cls_patch_rec_Student_CaiT_All_3_Loss(config)
        else:
            teacher = TransMorph.Trans_SMIT_pre_train_cls_patch_rec_Student_CaiT_All_3_Loss_Patch_Token_At_4_Block(config)
        teacher.use_MIM_mask = True
        if args.teacher_mask_ratio == 0:
            teacher.use_MIM_mask = False
        
        embed_dim = config.embed_dim * 2 * 2 * 2 
    
    print (student)
    pytorch_total_params = sum(p.numel() for p in student.transformer.parameters() if p.requires_grad)
    print('info: warning: Total parameters in transformer count', pytorch_total_params)
    
    student = utils.Module_Mask_Atten_Wrapper(student,iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))

    #for n, p in student.named_parameters():
    #    if p.grad is None:
    #        print(f'{n} has no grad')
    
    #print(student)
    teacher = utils.Module_Mask_Atten_Wrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu],find_unused_parameters=True)
        #teacher = nn.DataParallel(teacher,device_ids=[args.local_rank])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    

    # DDP wrapper...
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu],find_unused_parameters=True)
    #student = nn.DataParallel(student,device_ids=[args.local_rank])


    # teacher and student start with the same weights (only on fresh training)
    # Fix #2: skip if resuming to preserve EMA teacher state
    if args.resume == 0:
        teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both Swin based Transformer.")

    # ============ preparing loss ... ============
    """ trainloss = TrainLoss(
        args.out_dim,
        2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda() """

    # ============ preparing loss ... ============
    if args.use_dinov2_loss:
        # DINOv2-style losses with Sinkhorn-Knopp centering
        dino_loss = DINOLoss(
            out_dim=args.out_dim,
            student_temp=0.1,
            center_momentum=0.9,
        ).cuda()
        ibot_loss = iBOTPatchLoss(
            out_dim=args.out_dim if (args.shared_head or args.shared_head_teacher) else args.patch_out_dim,
            student_temp=0.1,
            center_momentum=0.9,
        ).cuda()
        koleo_loss_fn = KoLeoLoss() if args.use_koleo else None
        trainloss = None  # not used in DINOv2 mode
        print(f'Using DINOv2-style losses: DINO_SK + iBOT_SK + KoLeo={args.use_koleo} (weight={args.koleo_weight})')
    else:
        # Original iBOT loss with EMA centering
        koleo_loss_fn = KoLeoLoss() if args.use_koleo else None

    if not args.use_dinov2_loss:
        same_dim = args.shared_head or args.shared_head_teacher
        trainloss = iBOTLoss(
            args.out_dim,
            args.out_dim if same_dim else args.patch_out_dim,
            #args.global_crops_number,
            #args.local_crops_number,
            2,
            args.warmup_teacher_temp,
            args.teacher_temp,

        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,

        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with UniMiSS
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        #args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.lr,
        args.min_lr,
        args.epochs, len(data_loader3D),  # Fix #1: use len(data_loader) not len(dataset)
        warmup_epochs=args.warmup_epochs,  # Fix: use --warmup_epochs arg
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader3D),  # Fix #1
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader3D))  # Fix #1
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    #to_restore = {"epoch": 0}

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        #os.path.join(args.output_dir, "checkpoint0220.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss=trainloss,
    )

    start_epoch = to_restore["epoch"]

    # Swin_only_rec_epoch500_plot_patch128_10k_data
    """ if args.pre_rec_path is not None:
        #print ('start to resume from ',args.pre_rec_path)
        utils.resume_from_checkpoint(
        os.path.join('./snapshots',args.pre_rec_path, "checkpoint.pth"),
        run_variables=None,
        student=student,
        teacher=teacher,
        #optimizer=optimizer,
        #fp16_scaler=fp16_scaler,
        #loss=trainloss,
        ) """
    # teacher and student start with the same weights (only on fresh training)
    # Fix #2: skip if resuming to preserve EMA teacher state
    if args.resume == 0:
        teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both Swin based Transformer.")

    if args.resume ==1:
        resume_path='/lila/data/deasy/Eric_Data/Transformer_related/UniMiss_3D/snapshots/New_epoch800_10k_data_patch_128_teacher_mask_ratio_0.7_Intensity_Aug_1_batch_size_8_wCaiT_layer_2_AttMask_0_Resize3D_1_2loss/'
        #resume_path='/lila/data/deasy/Eric_Data/Transformer_related/UniMiss_3D/snapshots/New_epoch800_10k_data_patch_128_teacher_mask_ratio_0.7_Intensity_Aug_1_batch_size_8_wCaiT_layer_2_AttMask_1_AttMask_ratio_0.7_Resize3D_1_2loss_stoped/'

        
        resume_path='/lila/data/deasy/Eric_Data/Transformer_related/UniMiss_3D/snapshots/New_epoch400_6k_data_patch_128_teacher_mask_ratio_0.7_Intensity_Aug_1_batch_size_4_wCaiT_layer_2_AttMask_1_AttMask_ratio_0.5_Resize3D_1_2loss/'
        resume_path='/lila/data/deasy/Eric_Data/Transformer_related/UniMiss_3D/snapshots/New_epoch400_10k_data_patch_128_teacher_mask_ratio_0.7_Intensity_Aug_1_batch_size_8_wCaiT_layer_2_AttMask_1_Resize3D_1_2loss_timeout/'
        
        resume_path='/lila/data/deasy/Eric_Data/Transformer_related/UniMiss_3D/snapshots/New_epoch400_6k_data_IMG_REC_0_patch_128_teacher_mask_ratio_0.7_Intensity_Aug_1_Patch_Token_at_3_block_1_batch_size_8_wCaiT_layer_2_AttMask_1_AttMask_ratio_0.7_Resize3D_1_2loss/'
        resume_path='/lila/data/deasy/Eric_Data/Transformer_related/UniMiss_3D/snapshots/Teacher_MaskRatio_Exp_epoch800_10k_data_patch_128_teacher_mask_ratio_0.5_Intensity_Aug_1_Patch_Token_at_3_block_1_batch_size_8_wCaiT_layer_2_AttMask_1_AttMask_ratio_0.7_Resize3D_1_2loss/'
        
        resume_path='/lila/data/deasy/Eric_Data/Transformer_related/UniMiss_3D/snapshots/Large_2_2_12_2_epoch500_10k_data_patch_128_SMIT_lc10_teacher_mask_ratio_0.0_Intensity_Aug_0_Patch_Token_at_3_block_1_batch_size_8_wCaiT_layer_2_AttMask_0_AttMask_ratio_0.7_Resize3D_1_img_rec_weight_60.0_2loss/'
        
        utils.restart_from_checkpoint(
        os.path.join(resume_path, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss=trainloss,
        ) 

    start_epoch = to_restore["epoch"]

    interval = args.interval
    loss_all=[]
    lr_all=[]

    loss_cls=[]
    loss_patch=[]
    loss_rec=[]
    loss_teacher_rec=[]
    # wandb init
    if HAS_WANDB and args.wandb_run and utils.is_main_process():
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        print('wandb initialized:', args.wandb_project, args.wandb_run)

    for epoch in range(start_epoch, args.epochs):

        

        train_several_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader3D, epoch, epoch+1,
                        optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, '3D', args,loss_all,lr_all,loss_cls,loss_patch,loss_rec,loss_teacher_rec)
        
        if epoch %  10 ==0:
            print ('info: writing to folder: ',args.output_dir)




def train_several_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader, start_epoch, end_epoch,
                        optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, modal_type, args,loss_all,lr_all,loss_cls,loss_patch,loss_rec,loss_teacher_rec):

    start_time = time.time()

    
    print("Starting " + modal_type + " SMIT training !")
    for epoch in range(start_epoch, end_epoch):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of UniMiSS ... ============
    
        # Here only for 3D 
        #for name, param in student.module.named_parameters():  
        #    if '3D' in name:  
        #        param.requires_grad = True
        # Fix #4: removed DDP re-wrapping (done once before training loop)
        train_stats = train_one_epoch3D(student, teacher, teacher_without_ddp, trainloss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss': trainloss.state_dict() if trainloss is not None else {'dino': dino_loss.state_dict(), 'ibot': ibot_loss.state_dict()},
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f: f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training ' + modal_type + ' time {}'.format(total_time_str))

    loss_all.append(train_stats['loss_volume'])
    loss_cls.append(train_stats['loss_cls'])
    loss_patch.append(train_stats['loss_patch'])

    lr_all.append(train_stats['lr3D'])
    loss_rec.append(train_stats['loss_rec'])
    loss_teacher_rec.append(train_stats['loss_teacher_rec'])
    # wandb logging
    if HAS_WANDB and args.wandb_run and utils.is_main_process() and wandb.run:
        wandb.log({
            'loss_volume': train_stats['loss_volume'],
            'loss_cls': train_stats['loss_cls'],
            'loss_patch': train_stats['loss_patch'],
            'loss_rec': train_stats['loss_rec'],
            'loss_teacher_rec': train_stats['loss_teacher_rec'],
            'loss_koleo': train_stats.get('loss_koleo', 0),
            'lr': train_stats['lr3D'],
            'epoch': epoch,
        })
    if args.local_rank == 0:
        show_img=True
        if show_img:
            plt.figure(1, figsize=(8, 8))
            plt.subplot(2, 3, 1)
            #print('info: loss_volume ',loss_all)
            plt.plot(loss_all)
            plt.grid()
            plt.title('Training Loss')

            plt.subplot(2, 3, 2)
            plt.plot(loss_cls)
            plt.grid()
            plt.title('CLS_loss')

            plt.subplot(2, 3, 3)
            plt.plot(loss_patch)
            plt.grid()
            plt.title('Patch_loss')

            plt.subplot(2, 3, 4)
            plt.plot(loss_rec)
            plt.grid()
            plt.title('Img_Rec_Loss')

            plt.subplot(2, 3, 5)
            plt.plot(loss_teacher_rec)
            plt.grid()
            plt.title('Teacher_Img_Rec_Loss')


            plt.subplot(2, 3, 6)
            plt.plot(lr_all)
            plt.grid()
            plt.title('Learning Rate')


            plt.savefig(os.path.join(args.output_dir, 'train_loss_plots.png'))
            plt.close(1)





def train_one_epoch3D(student, teacher, teacher_without_ddp, trainloss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # Fix #3: dict-based EMA param alignment (guarantees correct name matching)
    student_params = dict(student.module.named_parameters())
    teacher_params = dict(teacher_without_ddp.named_parameters())
    names_common = sorted(set(student_params.keys()) & set(teacher_params.keys()))
    assert len(names_common) > 0, 'No common parameters between student and teacher!'
    params_q = [student_params[n] for n in names_common]
    params_k = [teacher_params[n] for n in names_common]
    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # Fix #1: use len(data_loader) not len(dataset)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move 3D images to gpu
        #print ('warning: data lens of one batch is ', len(subjects_batch) )

        #print (' warning image view 1 size: ',subjects_batch[0].size())
        #print (' warning image view 2 size : ',subjects_batch[1].size())
        #print (' warning mask view 1 size : ',subjects_batch[2].size())
        #print (' warning mask view 2 size : ',subjects_batch[3].size())

        images = [im.cuda(non_blocking=True) for im in subjects_batch[0:2]] # all the images
        masks = [im.cuda(non_blocking=True) for im in subjects_batch[2:4]] # all the masks 
        mask_token= [im.cuda(non_blocking=True) for im in subjects_batch[4:6]] # all the mask tokens
        masks_teacher= [im.cuda(non_blocking=True) for im in subjects_batch[6:8]] # all the mask for teachers

        # Fix #8: removed unused image_slices computation

        # teacher and student forward passes + compute 3D loss
        # Fix #7: single autocast context
        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None):
            with torch.no_grad():
                #if args.teacher_student_same_mask==1:
                #    teacher_output_all = teacher(images[:2],masks)
                
                #else:

                teacher_output_all = teacher(images[:2],masks_teacher)
            teacher_output=teacher_output_all[0:2]

            teacher_rec_loss=teacher_output_all[2]*args.img_rec_weight

            #print ('teacher_output_all is ',len(teacher_output_all))
            #print ('teacher_output_all 0 is ',teacher_output_all[0].size())
            #print ('teacher_output_all 1 is ',teacher_output_all[1].size())
            #print ('teacher_output_all 2 is ',teacher_output_all[2])
            #print ('teacher_output_all 3 is ',teacher_output_all[3].size())

            teach_rec_img=teacher_output_all[3]
            #print ('image_slices ',image_slices.size())
            #teacher_output_slices = teacher(image_slices) # slices
            #teacher_output_slices = teacher_output_slices.view(images[0].size()[0] * 2, images[0].size()[2], -1).mean(1)

            #print (' len (images) ',len(images))
            #print (' len (masks) ',len(masks))
            #print (' len (images_and_masks) ',len(images_and_masks))

            #all_data=dist.all_gather(images,mask=masks)
            #student.use_MIM_mask=True 

            use_att_mask=args.use_att_mask
            teacher_attention =teacher_output_all[4]
            #nh = teacher_attention.shape[1] # number of head # here is 6
            #bc_sz=teacher_attention.shape[0]
            #print('nh is ',nh)
            # we keep only the  output patch attention and not show for the cls
            # 0 is for the cls attnetion to other position
            #print('atten map before reshape',teacher_attention.shape) #[3,900]
            #teacher_attention = teacher_attention[:, :, 0, 1:].reshape(nh, -1)
            #print('atten map after reshape',teacher_attention.shape) #[3,900]
            #teacher_attention =np.average(teacher_attention, axis=0)

            teacher_attention = teacher_attention[:, :, 0, 1:].mean(1).detach().clone()
            img_patch_size=16

            sc_x=int(args.img_size3D_x/img_patch_size)
            sc_y=int(args.img_size3D_y/img_patch_size)
            sc_z=int(args.img_size3D_z/img_patch_size)

            if use_att_mask:
                
                #print('atten map before reshape',teacher_attention.shape) #[3,900]
                
                

                #attentions_mean =torch.mean(attentions, 0)
                #attentions = attentions.reshape(nh,  sc, sc,sc).float()
                #attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=img_patch_size, mode="nearest")[0].cpu().numpy()
                #print('atten map after resampling',attentions.shape) #[3,900]


                # generate the mean attention map of different heads

                

                #Then produce the masks according teacher_attention 

                # Get mean [CLS] token attention
                #cls_attention = teacher_attention[:, :, 0, 1:].mean(1).detach().clone()

                # Get AttMask. cls_attention should be in shape (batch_size, number_of_tokens)
                masks = AttMask(teacher_attention,
                                args.att_masking_prob,
                                args.att_pred_shape,
                                args.mask_pred_ratio,#,data_loader.dataset.get_pred_ratio(), # For each sample in the batch we perform the same masking ratio
                                args.show_max*args.mask_pred_ratio,
                                args.show_max
                                )
                mask_sum=torch.sum(masks)

                
                
                bc_sz=masks.shape[0]

                check_debug=False 
                if check_debug:
                    for bc_id in range(0,bc_sz):

                        cur_mask=masks[bc_id]

                        if torch.max(cur_mask)==0:
                            print (' !!!!!!!!!!!!!!!!!!! cur_mask failed: ', max(cur_mask))
                            masks = AttMask_Debug(teacher_attention,
                                    args.att_masking_prob,
                                    args.att_pred_shape,
                                    args.mask_pred_ratio,#,data_loader.dataset.get_pred_ratio(), # For each sample in the batch we perform the same masking ratio
                                    args.show_max*args.mask_pred_ratio,
                                    args.show_max
                                    )
                            assert torch.max(cur_mask)>0
                mask_token = masks.reshape(bc_sz,  sc_x, sc_y,sc_z).float()

                computed_mask_ratio=mask_sum/(bc_sz*sc_x*sc_y*sc_z)

                # Fix #5: stay on GPU, avoid CPU/NumPy roundtrip
                masks = nn.functional.interpolate(mask_token.unsqueeze(1), scale_factor=8, mode="nearest").squeeze(1)
                masks = masks.chunk(2)
                mask_token = mask_token.chunk(2)
                #print ('computed msk size is ',masks.shape)
                #attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=img_patch_size, mode="nearest")[0].cpu().numpy()
                #masks = [mask.reshape(-1, 128//img_patch_size, 128//img_patch_size,128//img_patch_size) 
                #            for mask in masks.chunk(args.global_crops_number, 0)]
                        
            # x1, x2 ,x_rec,x_rec_img,x_att
            student_output_all = student(images,masks) # volume
            student_output=student_output_all[0:2]
            student_rec_img=student_output_all[3]

            rec_w = args.rec_weight_override if args.rec_weight_override > 0 else args.img_rec_weight
            image_rec_loss=student_output_all[2]*rec_w
            #print ('image_rec loss ',image_rec_loss.item())
            #print ('info: student_output cls size ',student_output[0].size())   # 20,8192
            #print ('info: student_output patch size ',student_output[1].size()) # 20,216,8192

            #print ('info: teacher_output cls size ',teacher_output[0].size()) # 20,8192
            #print ('info: teacher_output patch size ',teacher_output[1].size()) # 20,216, 8192

            # get local cls for student 
            #student.use_MIM_mask=False 
            #student_local_cls = student(images,masks)[0] # volume
            student_local_cls=None
            #print ('info: student local CLS size ',student_local_cls.size()) # 20,216, 8192
            #student_output = student(images) # volume
            #student_output_slices = student(image_slices) # slices
            #student_output_slices = student_output_slices.view(images[0].size()[0] * 2, images[0].size()[2], -1).mean(1)
            student.use_MIM_mask=True 
            #all_loss = trainloss(student_output, teacher_output, epoch)  # volume loss only

            if args.use_dinov2_loss:
                # DINOv2-style loss computation
                student_cls, student_patch = student_output
                teacher_cls, teacher_patch = teacher_output

                # DINO CLS loss with SK centering
                # student_cls: [2*B, out_dim], teacher_cls: [2*B, out_dim]
                student_cls_list = student_cls.chunk(2)  # list of [B, out_dim]
                teacher_cls_stacked = torch.stack(teacher_cls.chunk(2))  # [2, B, out_dim]
                temp_cls = args.teacher_temp
                loss_cls = dino_loss(student_cls_list, teacher_cls_stacked, temp_cls)

                # iBOT patch loss with SK centering
                student_patch_chunks = student_patch.chunk(2)
                teacher_patch_chunks = teacher_patch.chunk(2)
                temp_patch = args.teacher_patch_temp
                loss_patch = torch.tensor(0.0, device=student_patch.device)
                n_patch_terms = 0
                for q in range(2):
                    # Compute patch loss on same-view pairs (masked prediction)
                    mask_flat = mask_token[q].flatten(-3, -1).bool().cuda()
                    if mask_flat.sum() > 0:
                        loss_patch = loss_patch + ibot_loss(
                            student_patch_chunks[q], teacher_patch_chunks[q],
                            mask_flat, temp_patch,
                        )
                        n_patch_terms += 1
                if n_patch_terms > 0:
                    loss_patch = loss_patch / n_patch_terms

                volume_loss = loss_cls + loss_patch

                # KoLeo regularizer on student CLS embeddings
                koleo_val = torch.tensor(0.0, device=student_cls.device)
                if args.use_koleo and koleo_loss_fn is not None:
                    # Use student CLS features before projection head
                    koleo_val = koleo_loss_fn(student_cls) * args.koleo_weight
                    volume_loss = volume_loss + koleo_val

            else:
                all_loss = trainloss(student_output, teacher_output, student_local_cls, mask_token, epoch)

                volume_loss = all_loss.pop('loss')
                loss_cls = all_loss.pop('cls')
                loss_patch = all_loss.pop('patch')

                # KoLeo (also available in original mode)
                koleo_val = torch.tensor(0.0, device=loss_cls.device)
                if args.use_koleo and koleo_loss_fn is not None:
                    student_cls_feat, _ = student_output
                    koleo_val = koleo_loss_fn(student_cls_feat) * args.koleo_weight
                    volume_loss = volume_loss + koleo_val

            # start to compuate the teacher-student image distillization loss 
           
            
            imge_rec_diss_loss=0

            if args.image_distil and epoch > args.image_distil_start_epoch:
                imge_rec_diss_loss = F.l1_loss(teach_rec_img.detach(), student_rec_img, reduction='mean')

                imge_rec_diss_loss=imge_rec_diss_loss* args.img_dis_weight     
                
            if args.image_distil and epoch > args.image_distil_start_epoch:
                loss = volume_loss +image_rec_loss +imge_rec_diss_loss#+ slices_volume_loss + slices_loss + volume_slices_loss
            elif args.teacher_rec:
                loss = volume_loss +image_rec_loss+teacher_rec_loss
            else:
                loss = volume_loss +image_rec_loss

        show_image=True 
        if it %100 ==0 and show_image:
            
            if args.local_rank==0:
                img_show=images[0]
                img_show=img_show[0]
                
                img_show=img_show[0]
                #print (' img_show size ',img_show.shape)
                val_img_save=img_show.float()#.cuda()=
                val_img_save=val_img_save.data.cpu().numpy()
                val_img_save=np.squeeze(val_img_save)
                val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
                pred_sv_name_img=args.output_dir+'/train_debug_View1.nii.gz'
                nib.save(val_img_save, pred_sv_name_img)

                img_show=images[1]
                img_show=img_show[0]
                
                img_show=img_show[0]
                #print (' img_show size ',img_show.shape)
                val_img_save=img_show.float()#.cuda()=
                val_img_save=val_img_save.data.cpu().numpy()
                val_img_save=np.squeeze(val_img_save)
                val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
                pred_sv_name_img=args.output_dir+'/train_debug_View2.nii.gz'
                nib.save(val_img_save, pred_sv_name_img)

                #print ('student_rec_img size ',student_rec_img.shape)
                #img_show=student_rec_img[1]
                img_show=student_rec_img[0]
                
                img_show=img_show[0]
                #print (' img_show size ',img_show.shape)
                val_img_save=img_show.float()#.cuda()=
                val_img_save=val_img_save.data.cpu().numpy()
                val_img_save=np.squeeze(val_img_save)
                val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
                pred_sv_name_img=args.output_dir+'/train_debug_View1_Rec.nii.gz'
                nib.save(val_img_save, pred_sv_name_img)
                
                
                teacher_attention_show=teacher_attention[0]
                teacher_attention_show=teacher_attention_show.reshape(1, sc_x, sc_y,sc_z).float()
                #print ('info: teacher_attention_show size ',teacher_attention_show.shape)
                teacher_attention_show=nn.functional.interpolate(teacher_attention_show.unsqueeze(0), scale_factor=16, mode="nearest")#.cpu().numpy()

                #print ('teacher_attention_show size ',teacher_attention_show.shape)

                teacher_attention_show=teacher_attention_show.cpu().numpy()


                val_img_save=teacher_attention_show#.float()#.cuda()=
                #val_img_save=th_attn[hd_id]#.float()#.cuda()=

                #val_img_save=val_img_save.data.cpu().numpy()
                val_img_save=np.squeeze(val_img_save)
                val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
                pred_sv_name_img=args.output_dir+'/Atten_Map_Head_Mean.nii.gz'
                nib.save(val_img_save, pred_sv_name_img)

                if use_att_mask:
                    mask_token_show=mask_token[0]
                    #print ('info: masks_show size ',mask_token_show.shape)
                    mask_token_show=mask_token_show[0]
                    mask_token_show=mask_token_show.unsqueeze(0)

                    mask_sum=torch.sum(mask_token_show)

                    masks_show=nn.functional.interpolate(mask_token_show.unsqueeze(0), scale_factor=16, mode="nearest").cpu().numpy()

                    #print ('masks_show size ',masks_show.shape)
                    

                    val_img_save=masks_show#.float()#.cuda()=
                    #val_img_save=th_attn[hd_id]#.float()#.cuda()=

                    #val_img_save=val_img_save.data.cpu().numpy()
                    val_img_save=np.squeeze(val_img_save)

                    #cc=np.sum(val_img_save)
                    #cur_mk_r=cc/(128*128*128.)
                    #print ('max cur_mk_r ',np.max(val_img_save))
                    #print('cur_mk_r ',cur_mk_r)
                    val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
                    pred_sv_name_img=args.output_dir+'/Att_Mask.nii.gz'
                    nib.save(val_img_save, pred_sv_name_img)




        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))  # Fix #6: removed force=True
            sys.exit(1)

        # student update

        #torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher

        #print ('student parameters ', student.module.parameters())


        #print ('teacher_without_ddp parameters ', teacher_without_ddp.module.parameters())

         # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss_volume=volume_loss.item())
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_patch=loss_patch.item())
        metric_logger.update(loss_rec=image_rec_loss.item())
        if args.use_koleo:
            metric_logger.update(loss_koleo=koleo_val.item())
        metric_logger.update(loss_teacher_rec=teacher_rec_loss.item())
        if args.image_distil:
            metric_logger.update(loss_img_dis=imge_rec_diss_loss.item())
        
        metric_logger.update(lr3D=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd3D=optimizer.param_groups[0]["weight_decay"])

        
                

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ncrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        #self.ngcrops = ngcrops
        #self.nlcrops = nlcrops
        self.ncrops = ncrops #ngcrops + nlcrops
        #self.register_buffer("center", torch.zeros(1, out_dim)) # for class
        self.register_buffer("center", torch.zeros(1, 2, out_dim)) # for class
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim)) # for patch
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output # get student output for 
        teacher_cls, teacher_patch = teacher_output # get teacher output distLearning
        
        #print ('student_cls size ',student_cls.shape)
        #print ('student_patch size ',student_patch.shape)
        #if student_local_cls is not None:
        #    student_cls = torch.cat([student_cls, student_local_cls]) # get the local student cls
        #student_cls=student_local_cls
        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)

        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ncrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]

        #print ('teacher_cls size ',teacher_cls.shape)
        #print ('self.center size ',self.center.shape)

        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ncrops) # maybe 2?

        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ncrops) # maybe 2 ?


        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q: #compute the patch loss 
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    #mask = student_mask[v].flatten(-2, -1)
                    mask=student_mask[v].flatten(-3,-1).cuda()
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else: # compute the CLS loss

                    #print ('student_cls_c[v] size ',student_cls_c[v].shape)
                    #print ('teacher_cls_c[q] size ',teacher_cls_c[q].shape)

                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)

                    #loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        #print ('info: self.center size is ',self.center.size())
        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)


class TrainLoss_cls_only(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                #if v == iq:
                    # we skip cases where student and teacher operate on the same view but here can do the feature distillization 
                    #loss= torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)+torch.sum(-q * F.log_softmax(student_out[iq], dim=-1), dim=-1)
                #    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser('UniMiSS', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.teacher_student_same_mask==0:
        trainning_flag='teacher_mask_ratio_'+str(args.teacher_mask_ratio)+'_Intensity_Aug_'+str(args.intensity_Aug)+'_Patch_Token_at_3_block_'+str(args.Patch_Token_at_3_block)+'_batch_size_'+str(args.batch_size_per_gpu)+'_wCaiT_layer_'+str(args.Cait_layer)+'_AttMask_'+str(args.use_att_mask)+'_AttMask_ratio_'+str(args.mask_pred_ratio)+'_Resize3D_'+str(args.use_3D_resize)+'_img_rec_weight_'+str(args.img_rec_weight)+'_2loss/'
    else:
        trainning_flag='teacher_mask_ratio_'+str(args.teacher_mask_ratio)+'_Intensity_Aug_'+str(args.intensity_Aug)+'_Patch_Token_at_3_block_'+str(args.Patch_Token_at_3_block)+'_batch_size_'+str(args.batch_size_per_gpu)+'_wCaiT_layer_'+str(args.Cait_layer)+'_AttMask_'+str(args.use_att_mask)+'_AttMask_ratio_'+str(args.mask_pred_ratio)+'_Resize3D_'+str(args.use_3D_resize)+'_teacher_student_same_mask_'+str(args.teacher_student_same_mask)+'_2loss/'
    
        
    args.output_dir=args.output_dir+trainning_flag

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #os.environ[
    #    "TORCH_DISTRIBUTED_DEBUG"
    #] = "DETAIL" 
    print (args)

    #current_file_path = os.path.abspath(sys.argv[0])

    # 获取当前运行脚本的完整路径
    current_file_path = os.path.abspath(sys.argv[0])
    target_folder = args.output_dir
    updated_file_path = os.path.join(target_folder, os.path.basename(current_file_path))

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)

    # 读取当前文件内容
    with open(current_file_path, 'r') as file:
        content = file.readlines()

    # 在文件头部增加运行时传入的参数信息
    arguments = ' '.join(sys.argv[1:])
    content.insert(0, f'# Runtime arguments: {arguments}\n')

    # 将更新后的内容写入新文件
    with open(updated_file_path, 'w') as updated_file:
        updated_file.writelines(content)


    shutil.copy(current_file_path, args.output_dir)
    train_func(args)
