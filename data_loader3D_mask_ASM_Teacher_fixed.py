import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import os
import math
from torch.utils.data import Dataset


from batchgenerators.transforms import Compose
#from batchgenerators.transforms.compose import Compose

from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform



class MaskGenerator:
    def __init__(self, input_size=96, mask_patch_size=8, model_patch_size=2, mask_ratio=0.6):
        self.input_size = input_size  # input image
        self.mask_patch_size = mask_patch_size 
        self.model_patch_size = model_patch_size # image patch size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size # 12 # 6    #  6 for ssim
        self.scale = self.mask_patch_size // self.model_patch_size # 4 # 8 # # 8 for siim
        
        self.token_count = self.rand_size ** 3  # 12*12*12 # # 6*6*6  # 6*6 for simim
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio)) # 27*0.6  # 36*0.6
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count] # 27~ 0.6
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1 # efficient mask
        
        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size)) # 3*3*3 
        token_mask=mask
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        
        #return token_mask, mask
        return token_mask, mask #mask 


class MaskGenerator_list:
    def __init__(self, input_size=[128,128,48], mask_patch_size=8, model_patch_size=2, mask_ratio=0.6):
        self.input_size = input_size  # input image
        self.mask_patch_size = mask_patch_size 
        self.model_patch_size = model_patch_size # image patch size
        self.mask_ratio = mask_ratio
        
        xx=self.input_size[0]
        yy=self.input_size[1]
        zz=self.input_size[2]
        
        assert xx % self.mask_patch_size == 0
        assert yy % self.mask_patch_size == 0
        assert zz % self.mask_patch_size == 0

        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size_x = self.input_size[0] // self.mask_patch_size # 12 # 6    #  6 for ssim
        self.rand_size_y = self.input_size[1] // self.mask_patch_size # 12 # 6    #  6 for ssim
        self.rand_size_z = self.input_size[2] // self.mask_patch_size # 12 # 6    #  6 for ssim

        self.scale = self.mask_patch_size // self.model_patch_size # 4 # 8 # # 8 for siim
        
        self.token_count = self.rand_size_x * self.rand_size_y * self.rand_size_z# 12*12*12 # # 6*6*6  # 6*6 for simim
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio)) # 27*0.6  # 36*0.6
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count] # 27~ 0.6
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1 # efficient mask
        
        mask = mask.reshape((self.rand_size_x, self.rand_size_y, self.rand_size_z)) # 3*3*3 
        token_mask=mask
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        
        #return token_mask, mask
        return token_mask, mask #mask 
    
class Dataset3D_Jue_Custmzed_96_Mask_8(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_128_Mask_8(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_128_Mask_16(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=16,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=16,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
class Dataset3D_Jue_Custmzed_96_Mask_16(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=16,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=16,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_Size_Patch_Size(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=patch_size,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=patch_size,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        #print ('mask1_all size ',mask1_all[0].shape)
        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=patch_size,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=patch_size,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        #print ('mask1_all size ',mask1_all[0].shape)
        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        #img.append(mask2)

        img.append(mask1_token)
        #img.append(mask2_token)

        img.append(mask1_teacher)
        #img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug




class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Not_Square(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_h, self.crop3D_w, self.crop3D_d = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=patch_size,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=patch_size,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_h, img_w, img_d = image.shape
        
        #print('----------> info image ori size ',image.shape)
        scaler_h = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)
        
        #print ('info: img_h is ',img_h)
        #print ('info: self.crop3D_h is ',self.crop3D_h)
        #print ('info: scale_h is ',scale_h)

        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        #image_crop = image[:, d0:d1, h0: h1, w0: w1]

        image_crop = image[:, h0: h1, w0: w1, d0:d1]

        if 'CT/' in img_name:
            image_crop = self.truncate(image_crop)
        #image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ()
        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)


        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0
            

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        #image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_crop = image[:,  h0: h1, w0: w1,d0: d1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        #val_img_save=image_crop[0]#.float()#.cuda()=
        #val_img_save=val_img_save.data.cpu().numpy()
        #val_img_save=np.squeeze(val_img_save)
        #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
        #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize.nii.gz'
        #nib.save(val_img_save, pred_sv_name_img)

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            
            #print ('info -----------> size before resize ',image_crop.shape)
            image_crop = cv2.resize(image_crop[0], (self.crop3D_w, self.crop3D_h), interpolation=cv2.INTER_LINEAR)

            #print ('info -----------> size after resize ',image_crop.shape)
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=val_img_save.data.cpu().numpy()
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_1st_resize.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop.transpose(2,0,1)

            
            

            #only resize [d,h]
            image_crop = cv2.resize(image_crop, (self.crop3D_h, self.crop3D_d), interpolation=cv2.INTER_LINEAR)





            
            image_crop = image_crop.transpose(1,2,0)
            
            #print ('error 4 image_crop size is ',image_crop.shape) # (48, 48, 256)

            
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_2st_resize_final.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop[np.newaxis, :]

            

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_h - img.shape[0])
        cols_missing = math.ceil(self.crop3D_w - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img



    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        #image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            #print('info image_crop_ori size is ',image_crop_ori.shape)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        #print ('mask1_all size ',mask1_all[0].shape)
        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        #img.append(mask2)

        img.append(mask1_token)
        #img.append(mask2_token)

        img.append(mask1_teacher)
        #img.append(mask2_teacher)


        
            
        return img




class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Not_Square_less_Aug(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        random.shuffle(self.img_ids)
        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_h, self.crop3D_w, self.crop3D_d = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=patch_size,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=patch_size,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_h, img_w, img_d = image.shape
        
        #print('----------> info image ori size ',image.shape)
        scaler_h = np.random.uniform(1.0, 1.8) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 1.8)  #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 1.8)  #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)
        
        #print ('info: img_h is ',img_h)
        #print ('info: self.crop3D_h is ',self.crop3D_h)
        #print ('info: scale_h is ',scale_h)

        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        #image_crop = image[:, d0:d1, h0: h1, w0: w1]

        image_crop = image[:, h0: h1, w0: w1, d0:d1]

        if 'CT/' in img_name:
            image_crop = self.truncate(image_crop)
        

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ()
        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)


        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0
            

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        #image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_crop = image[:,  h0: h1, w0: w1,d0: d1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        #val_img_save=image_crop[0]#.float()#.cuda()=
        #val_img_save=val_img_save.data.cpu().numpy()
        #val_img_save=np.squeeze(val_img_save)
        #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
        #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize.nii.gz'
        #nib.save(val_img_save, pred_sv_name_img)

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            
            #print ('info -----------> size before resize ',image_crop.shape)
            image_crop = cv2.resize(image_crop[0], (self.crop3D_w, self.crop3D_h), interpolation=cv2.INTER_LINEAR)

            #print ('info -----------> size after resize ',image_crop.shape)
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=val_img_save.data.cpu().numpy()
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_1st_resize.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop.transpose(2,0,1)

            
            

            #only resize [d,h]
            image_crop = cv2.resize(image_crop, (self.crop3D_h, self.crop3D_d), interpolation=cv2.INTER_LINEAR)





            
            image_crop = image_crop.transpose(1,2,0)
            
            #print ('error 4 image_crop size is ',image_crop.shape) # (48, 48, 256)

            
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_2st_resize_final.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop[np.newaxis, :]

            

        return image_crop

    def pad_image(self, img,img_name):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_h - img.shape[0])
        cols_missing = math.ceil(self.crop3D_w - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        if 'CT/' in img_name:
            padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant',constant_values=-2000)
        else:
            padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant',constant_values=0)
        return padded_img



    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image,name)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        #image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            #print('info image_crop_ori size is ',image_crop_ori.shape)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        #print ('mask1_all size ',mask1_all[0].shape)
        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        #img.append(mask2)

        img.append(mask1_token)
        #img.append(mask2_token)

        img.append(mask1_teacher)
        #img.append(mask2_teacher)


        
            
        return img
    
class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Swin(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750

        min_HU = -500
        max_HU = 500

        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))


        mask1=mask1_all[1]

        
        img.append(mask1)
        #img.append(mask2)




        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
    

class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Swin_CT_MRI_2(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750

        min_HU = -500
        max_HU = 500

        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]

        if 'CT/' in img_name:
            image_crop = self.truncate(image_crop)

        #image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))


        mask1=mask1_all[1]

        
        img.append(mask1)
        #img.append(mask2)




        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug

class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Swin_CT_and_MRI(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        if 'CT/' in img_name:
            image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"] #
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))


        mask1=mask1_all[1]

        
        img.append(mask1)
        #img.append(mask2)




        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Swin_more_crop(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.5, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.5, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.5, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))


        mask1=mask1_all[1]

        
        img.append(mask1)
        #img.append(mask2)




        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug



class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Swin_more_crop_MRI(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.5, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.5, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.5, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]

        #print ('info: **************************************** img_name is ', name)
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))


        mask1=mask1_all[1]

        
        img.append(mask1)
        #img.append(mask2)




        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Swin_w_Seg(Dataset):
    def __init__(self, root, seg_root,list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.seg_root=seg_root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,image_seg):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_seg_crop= image_seg[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)
        image_seg_crop = self.truncate(image_seg_crop)

        return image_crop,image_seg_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image,image_seg, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_seg_crop = image_seg[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
            image_seg_crop = image_seg_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
            image_seg_crop = image_seg_crop[:, :, ::-1]

        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]
            image_seg_crop = image_seg_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                image_seg_crop = cv2.resize(image_seg_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(2,0,1)

                # only resize [d,w]
                image_seg_crop = cv2.resize(image_seg_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_NEAREST)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)
                image_seg_crop = cv2.resize(image_seg_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
                image_seg_crop = cv2.resize(image_seg_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(0,2,1)

            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                image_seg_crop = cv2.resize(image_seg_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_NEAREST)

                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)

                image_seg_crop = cv2.resize(image_seg_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(0,2,1)

            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]
            image_seg_crop = image_seg_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop,image_seg_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            imgnii_seg= nib.load(self.seg_root +datafiles["img"])
            image = imageNII.get_fdata()
            image_seg=imgnii_seg.get_fdata()
            
            #print('seg name ',self.seg_root +datafiles["img"])
            #print('img name ',self.root +datafiles["img"])
            #print (image.shape)
        except:

            image= np.zeros((128,128,128))
            image_seg=np.zeros((128,128,128))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)
        image_seg=self.pad_image(image_seg)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]
        image_seg = image_seg[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        image_seg = image_seg.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori,image_seg_crop_ori = self.crop_scale0_w_depth(image,image_seg)
            image_crop1,image_seg_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori,image_seg_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_seg_crop1=image_seg_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(image_seg_crop1))

            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_seg_crop1))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))


        mask1=mask1_all[1]

        
        img.append(mask1)
        #img.append(mask2)




        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug




class Dataset3D_Jue_Custmzed_Size_Patch_Size_no_Teacher_Swin_w_Seg_MRI_Brain_Mets(Dataset):
    def __init__(self, root, seg_root,list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=16):
        
        #print ('info: root ',root)
        self.root = root
        self.seg_root=seg_root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=patch_size,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,MRI):
        # truncate
        #min_HU = -500
        #max_HU = 500
        #CT[np.where(CT <= min_HU)] = min_HU
        #CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        #CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        #return CT

        # 1) 强度截断 (使用 1% 和 99% 分位数)
        low = np.percentile(MRI, 1)
        high = np.percentile(MRI, 99)
        MRI = np.clip(MRI, low, high)

        # 2) Z-score normalization
        mean = MRI.mean()
        std = MRI.std()
        if std < 1e-8:
            MRI = MRI - mean   # 避免除以零
        else:
            MRI = (MRI - mean) / std

        return MRI

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,image_seg):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_seg_crop= image_seg[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)
        image_seg_crop = self.truncate(image_seg_crop)

        return image_crop,image_seg_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image,image_seg, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_seg_crop = image_seg[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
            image_seg_crop = image_seg_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
            image_seg_crop = image_seg_crop[:, :, ::-1]

        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]
            image_seg_crop = image_seg_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                image_seg_crop = cv2.resize(image_seg_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(2,0,1)

                # only resize [d,w]
                image_seg_crop = cv2.resize(image_seg_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_NEAREST)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)
                image_seg_crop = cv2.resize(image_seg_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
                image_seg_crop = cv2.resize(image_seg_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(0,2,1)

            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                image_seg_crop = cv2.resize(image_seg_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_NEAREST)

                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)

                image_seg_crop = cv2.resize(image_seg_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_NEAREST)
                image_seg_crop = image_seg_crop.transpose(0,2,1)

            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]
            image_seg_crop = image_seg_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop,image_seg_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:

            #print ('self.root + datafiles["img"] ',self.root + datafiles["img"])

            #print ('seg path ',self.seg_root +datafiles["img"].replace('t1c_stripped.nii.gz','groupedLabelMap_aparc.DKTatlas+aseg.mapped.nii.gz'))

            imageNII = nib.load(self.root + datafiles["img"])

            imgnii_seg= nib.load(self.seg_root +datafiles["img"].replace('t1c_stripped.nii.gz','groupedLabelMap_aparc.DKTatlas+aseg.mapped.nii.gz'))
            image = imageNII.get_fdata()
            image_seg=imgnii_seg.get_fdata()
            
            #print('seg name ',self.seg_root +datafiles["img"])
            #print('img name ',self.root +datafiles["img"])
            #print (image.shape)
        except:

            image= np.zeros((128,128,128))
            image_seg=np.zeros((128,128,128))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)
        image_seg=self.pad_image(image_seg)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]
        image_seg = image_seg[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        image_seg = image_seg.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori,image_seg_crop_ori = self.crop_scale0_w_depth(image,image_seg)
            image_crop1,image_seg_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori,image_seg_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_seg_crop1=image_seg_crop1.transpose((0, 2, 3, 1))
        #image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(image_seg_crop1))

            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_seg_crop1))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))


        mask1=mask1_all[1]

        
        img.append(mask1)
        #img.append(mask2)




        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug

class Dataset3D_Jue_Custmzed_96_Mask_8_No_Crop(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = 1#np.random.uniform(1, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = 1#np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        
        image_crop_ori = self.crop_scale0_w_depth(image)
        

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop_ori=image_crop_ori.transpose((0, 2, 3, 1))
        

        mask1_all= self.mask_generator()
        

        mask1_all_teacher=self.mask_generator_Teacher()
        
        
        
        img.append(torch.from_numpy(image_crop_ori.astype(np.float32)))
        

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        

        mask1_teacher=mask1_all_teacher[1]
        

        img.append(mask1)
        

        img.append(mask1_token)
        

        img.append(mask1_teacher)
        


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_96_Mask_16_No_Crop(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=16,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = 1#np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = 1#np.random.uniform(1, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = 1#np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        
        image_crop_ori = self.crop_scale0_w_depth(image)
        

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop_ori=image_crop_ori.transpose((0, 2, 3, 1))
        

        mask1_all= self.mask_generator()
        

        mask1_all_teacher=self.mask_generator_Teacher()
        
        
        
        img.append(torch.from_numpy(image_crop_ori.astype(np.float32)))
        

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        

        mask1_teacher=mask1_all_teacher[1]
        

        img.append(mask1)
        

        img.append(mask1_token)
        

        img.append(mask1_teacher)
        


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug

class Dataset3D_Jue_Custmzed_96_Mask_8_Heavily_Data_Aug(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=0.0,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug



class Dataset3D_Jue_Custmzed_128_Mask_8_Heavily_Data_Aug(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=0.0,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=8,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_96_Mask_8_Heavily_Data_Aug_Patch_16(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=16,#,
            mask_ratio=0.0,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=16,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file

        try:
            imageNII = nib.load(self.root + datafiles["img"])

            image = imageNII.get_fdata()

            #print (image.shape)
        except:

            image= np.zeros((256,256,130))
            print('info: ERROR: Warning: error ***************************** at ',datafiles["name"])

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
    
class Dataset3D(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop


    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img

    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


            
        return img


class Dataset3D_Debug(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0_w_depth(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        #image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))

        

        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        image_crop4=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop3))
        img_debug.append(torch.from_numpy(image_crop4))
            
        return img_debug

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug



class Dataset3D_Jue_Custmzed(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug



class Dataset3D_Jue_Custmzed_CT_and_MRI(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]

        if 'CT/' in img_name:
            image_crop = self.truncate(image_crop)
            image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
    


class Dataset3D_Jue_Custmzed_CT_and_MRI_Not_Square(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_h, self.crop3D_w, self.crop3D_d = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_h, img_w, img_d = image.shape
        
        #print('----------> info image ori size ',image.shape)
        scaler_h = np.random.uniform(1.4, 1.8)  #Heavy  1.0, 2.2 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)
        
        #print ('info: img_h is ',img_h)
        #print ('info: self.crop3D_h is ',self.crop3D_h)
        #print ('info: scale_h is ',scale_h)

        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        #image_crop = image[:, d0:d1, h0: h1, w0: w1]

        image_crop = image[:, h0: h1, w0: w1, d0:d1]

        if 'CT/' in img_name:
            image_crop = self.truncate(image_crop)
            image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2) #0.5 1.5 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)


        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0
            

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        #image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_crop = image[:,  h0: h1, w0: w1,d0: d1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        #val_img_save=image_crop[0]#.float()#.cuda()=
        #val_img_save=val_img_save.data.cpu().numpy()
        #val_img_save=np.squeeze(val_img_save)
        #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
        #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize.nii.gz'
        #nib.save(val_img_save, pred_sv_name_img)

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            
            #print ('info -----------> size before resize ',image_crop.shape)
            image_crop = cv2.resize(image_crop[0], (self.crop3D_w, self.crop3D_h), interpolation=cv2.INTER_LINEAR)

            #print ('info -----------> size after resize ',image_crop.shape)
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=val_img_save.data.cpu().numpy()
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_1st_resize.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop.transpose(2,0,1)

            
            

            #only resize [d,h]
            image_crop = cv2.resize(image_crop, (self.crop3D_h, self.crop3D_d), interpolation=cv2.INTER_LINEAR)





            
            image_crop = image_crop.transpose(1,2,0)
            
            #print ('error 4 image_crop size is ',image_crop.shape) # (48, 48, 256)

            
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_2st_resize_final.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop[np.newaxis, :]

            

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_h - img.shape[0])
        cols_missing = math.ceil(self.crop3D_w - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print ('------------> image size ',image.shape) #(260, 190, 310)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' --------------> padded image size ',image.shape) #(260, 192, 310)

        image = image[np.newaxis, :]
        
        #print (' before transposed image size ',image.shape)
        #[b,x,y,z]
        
        #image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            #print('before image_crop_ori size is ',image.shape)
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            #print('info image_crop_ori size is ',image_crop_ori.shape)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #print('info after image_crop_ori size is ',image_crop1.shape)
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        
        #print ('view 1 image before size ',image_crop1.shape)
        #image_crop1=image_crop1.transpose((0, 2, 1, 3))
        #image_crop2=image_crop2.transpose((0, 2, 1, 3))

        #print ('view 1 image after size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]
        #print('mask1 size ',mask1.shape)
        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
    

class Dataset3D_Jue_Custmzed_CT_MRI_Not_Square(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_h, self.crop3D_w, self.crop3D_d = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_h, img_w, img_d = image.shape
        
        #print('----------> info image ori size ',image.shape)
        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)
        
        #print ('info: img_h is ',img_h)
        #print ('info: self.crop3D_h is ',self.crop3D_h)
        #print ('info: scale_h is ',scale_h)

        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        #image_crop = image[:, d0:d1, h0: h1, w0: w1]

        image_crop = image[:, h0: h1, w0: w1, d0:d1]

        if 'CT/' in img_name:
            image_crop = self.truncate(image_crop)
            image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)


        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0
            

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        #image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_crop = image[:,  h0: h1, w0: w1,d0: d1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        #val_img_save=image_crop[0]#.float()#.cuda()=
        #val_img_save=val_img_save.data.cpu().numpy()
        #val_img_save=np.squeeze(val_img_save)
        #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
        #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize.nii.gz'
        #nib.save(val_img_save, pred_sv_name_img)

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            
            #print ('info -----------> size before resize ',image_crop.shape)
            image_crop = cv2.resize(image_crop[0], (self.crop3D_w, self.crop3D_h), interpolation=cv2.INTER_LINEAR)

            #print ('info -----------> size after resize ',image_crop.shape)
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=val_img_save.data.cpu().numpy()
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_1st_resize.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop.transpose(2,0,1)

            
            

            #only resize [d,h]
            image_crop = cv2.resize(image_crop, (self.crop3D_h, self.crop3D_d), interpolation=cv2.INTER_LINEAR)





            
            image_crop = image_crop.transpose(1,2,0)
            
            #print ('error 4 image_crop size is ',image_crop.shape) # (48, 48, 256)

            
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_2st_resize_final.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop[np.newaxis, :]

            

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_h - img.shape[0])
        cols_missing = math.ceil(self.crop3D_w - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print ('------------> image size ',image.shape) #(260, 190, 310)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' --------------> padded image size ',image.shape) #(260, 192, 310)

        image = image[np.newaxis, :]
        
        #print (' before transposed image size ',image.shape)
        #[b,x,y,z]
        
        #image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            #print('before image_crop_ori size is ',image.shape)
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            #print('info image_crop_ori size is ',image_crop_ori.shape)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #print('info after image_crop_ori size is ',image_crop1.shape)
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        
        #print ('view 1 image before size ',image_crop1.shape)
        #image_crop1=image_crop1.transpose((0, 2, 1, 3))
        #image_crop2=image_crop2.transpose((0, 2, 1, 3))

        #print ('view 1 image after size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]
        #print('mask1 size ',mask1.shape)
        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug

class Dataset3D_Jue_Custmzed_CT_Not_Square(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_h, self.crop3D_w, self.crop3D_d = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_h, img_w, img_d = image.shape
        
        #print('----------> info image ori size ',image.shape)
        scaler_h = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)
        
        #print ('info: img_h is ',img_h)
        #print ('info: self.crop3D_h is ',self.crop3D_h)
        #print ('info: scale_h is ',scale_h)

        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        #image_crop = image[:, d0:d1, h0: h1, w0: w1]

        image_crop = image[:, h0: h1, w0: w1, d0:d1]

        
        image_crop = self.truncate(image_crop)
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)


        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0
            

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        #image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_crop = image[:,  h0: h1, w0: w1,d0: d1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        #val_img_save=image_crop[0]#.float()#.cuda()=
        #val_img_save=val_img_save.data.cpu().numpy()
        #val_img_save=np.squeeze(val_img_save)
        #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
        #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize.nii.gz'
        #nib.save(val_img_save, pred_sv_name_img)

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            
            #print ('info -----------> size before resize ',image_crop.shape)
            image_crop = cv2.resize(image_crop[0], (self.crop3D_w, self.crop3D_h), interpolation=cv2.INTER_LINEAR)

            #print ('info -----------> size after resize ',image_crop.shape)
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=val_img_save.data.cpu().numpy()
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_1st_resize.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop.transpose(2,0,1)

            
            

            #only resize [d,h]
            image_crop = cv2.resize(image_crop, (self.crop3D_h, self.crop3D_d), interpolation=cv2.INTER_LINEAR)





            
            image_crop = image_crop.transpose(1,2,0)
            
            #print ('error 4 image_crop size is ',image_crop.shape) # (48, 48, 256)

            
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_2st_resize_final.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop[np.newaxis, :]

            

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_h - img.shape[0])
        cols_missing = math.ceil(self.crop3D_w - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print ('------------> image size ',image.shape) #(260, 190, 310)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' --------------> padded image size ',image.shape) #(260, 192, 310)

        image = image[np.newaxis, :]
        
        #print (' before transposed image size ',image.shape)
        #[b,x,y,z]
        
        #image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            #print('before image_crop_ori size is ',image.shape)
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            #print('info image_crop_ori size is ',image_crop_ori.shape)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #print('info after image_crop_ori size is ',image_crop1.shape)
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        
        #print ('view 1 image before size ',image_crop1.shape)
        #image_crop1=image_crop1.transpose((0, 2, 1, 3))
        #image_crop2=image_crop2.transpose((0, 2, 1, 3))

        #print ('view 1 image after size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]
        #print('mask1 size ',mask1.shape)
        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug



class Dataset3D_Jue_Custmzed_CT_Not_Square_Brain_Mets(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_h, self.crop3D_w, self.crop3D_d = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_h, img_w, img_d = image.shape
        
        #print('----------> info image ori size ',image.shape)
        scaler_h = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)
        
        #print ('info: img_h is ',img_h)
        #print ('info: self.crop3D_h is ',self.crop3D_h)
        #print ('info: scale_h is ',scale_h)

        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        #image_crop = image[:, d0:d1, h0: h1, w0: w1]

        image_crop = image[:, h0: h1, w0: w1, d0:d1]

        
        image_crop = self.truncate(image_crop)
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)


        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0
            

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        #image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_crop = image[:,  h0: h1, w0: w1,d0: d1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        #val_img_save=image_crop[0]#.float()#.cuda()=
        #val_img_save=val_img_save.data.cpu().numpy()
        #val_img_save=np.squeeze(val_img_save)
        #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
        #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize.nii.gz'
        #nib.save(val_img_save, pred_sv_name_img)

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            
            #print ('info -----------> size before resize ',image_crop.shape)
            image_crop = cv2.resize(image_crop[0], (self.crop3D_w, self.crop3D_h), interpolation=cv2.INTER_LINEAR)

            #print ('info -----------> size after resize ',image_crop.shape)
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=val_img_save.data.cpu().numpy()
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_1st_resize.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop.transpose(2,0,1)

            
            

            #only resize [d,h]
            image_crop = cv2.resize(image_crop, (self.crop3D_h, self.crop3D_d), interpolation=cv2.INTER_LINEAR)





            
            image_crop = image_crop.transpose(1,2,0)
            
            #print ('error 4 image_crop size is ',image_crop.shape) # (48, 48, 256)

            
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_2st_resize_final.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop[np.newaxis, :]

            

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_h - img.shape[0])
        cols_missing = math.ceil(self.crop3D_w - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print ('------------> image size ',image.shape) #(260, 190, 310)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' --------------> padded image size ',image.shape) #(260, 192, 310)

        image = image[np.newaxis, :]
        
        #print (' before transposed image size ',image.shape)
        #[b,x,y,z]
        
        #image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            #print('before image_crop_ori size is ',image.shape)
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            #print('info image_crop_ori size is ',image_crop_ori.shape)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #print('info after image_crop_ori size is ',image_crop1.shape)
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        
        #print ('view 1 image before size ',image_crop1.shape)
        #image_crop1=image_crop1.transpose((0, 2, 1, 3))
        #image_crop2=image_crop2.transpose((0, 2, 1, 3))

        #print ('view 1 image after size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]
        #print('mask1 size ',mask1.shape)
        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug

class Dataset3D_Jue_Custmzed_No_Teacher_CT_Not_Square(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1,patch_size=2):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_h, self.crop3D_w, self.crop3D_d = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=patch_size,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator_list(
            input_size=[self.crop3D_h, self.crop3D_w, self.crop3D_d],
            mask_patch_size=16,#
            model_patch_size=patch_size,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate1(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT
    
    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_h, img_w, img_d = image.shape
        
        #print('----------> info image ori size ',image.shape)
        scaler_h = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.0, 2.2) #(1.0, 2.2) (1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)
        
        #print ('info: img_h is ',img_h)
        #print ('info: self.crop3D_h is ',self.crop3D_h)
        #print ('info: scale_h is ',scale_h)

        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        #image_crop = image[:, d0:d1, h0: h1, w0: w1]

        image_crop = image[:, h0: h1, w0: w1, d0:d1]

        
        image_crop = self.truncate(image_crop)
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _,  img_h, img_w,img_d = image.shape

        
        

        scaler_h = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.5, 1.5) #(0.5, 1.5) #(0.8, 1.2) 
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)


        if img_h !=scale_h:
            h0 = random.randint(0, img_h - scale_h)
        else:
            h0=0
        
        if img_w !=scale_w:
            w0 = random.randint(0, img_w - scale_w)
        else:
            w0=0 
        if img_d !=scale_d:
            d0 = random.randint(0, img_d - scale_d)
        else:
            d0=0
            

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        #image_crop = image[:, d0: d1, h0: h1, w0: w1]
        image_crop = image[:,  h0: h1, w0: w1,d0: d1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        #val_img_save=image_crop[0]#.float()#.cuda()=
        #val_img_save=val_img_save.data.cpu().numpy()
        #val_img_save=np.squeeze(val_img_save)
        #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
        #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize.nii.gz'
        #nib.save(val_img_save, pred_sv_name_img)

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            
            #print ('info -----------> size before resize ',image_crop.shape)
            image_crop = cv2.resize(image_crop[0], (self.crop3D_w, self.crop3D_h), interpolation=cv2.INTER_LINEAR)

            #print ('info -----------> size after resize ',image_crop.shape)
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=val_img_save.data.cpu().numpy()
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_1st_resize.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop.transpose(2,0,1)

            
            

            #only resize [d,h]
            image_crop = cv2.resize(image_crop, (self.crop3D_h, self.crop3D_d), interpolation=cv2.INTER_LINEAR)





            
            image_crop = image_crop.transpose(1,2,0)
            
            #print ('error 4 image_crop size is ',image_crop.shape) # (48, 48, 256)

            
            #val_img_save=image_crop#.float()#.cuda()=
            #val_img_save=np.squeeze(val_img_save)
            #val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))    
            #pred_sv_name_img='/data1/lia5/Jue/Transformer/UniMiss_3D/snapshots/Swin_MIM_CT_55k_only_MIM_Swin_S_tep/train_debug_View_before_ersize_2st_resize_final.nii.gz'
            #nib.save(val_img_save, pred_sv_name_img)

            image_crop = image_crop[np.newaxis, :]

            

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_h - img.shape[0])
        cols_missing = math.ceil(self.crop3D_w - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print ('------------> image size ',image.shape) #(260, 190, 310)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' --------------> padded image size ',image.shape) #(260, 192, 310)

        image = image[np.newaxis, :]
        
        #print (' before transposed image size ',image.shape)
        #[b,x,y,z]
        
        #image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            #print('before image_crop_ori size is ',image.shape)
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            #print('info image_crop_ori size is ',image_crop_ori.shape)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            #print('info after image_crop_ori size is ',image_crop1.shape)
            #image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            #image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        
        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        #print ('mask1_all size ',mask1_all[0].shape)
        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            #img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            #img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        #img.append(mask2)

        img.append(mask1_token)
        #img.append(mask2_token)

        img.append(mask1_teacher)
        #img.append(mask2_teacher)


        
            
        return img

    
class Dataset3D_Jue_Custmzed_MRI(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.75,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -750
        max_HU = 1750
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT-min_HU)/(max_HU-min_HU)
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image,img_name):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]


        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image,name)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D_Jue_Custmzed_Stage_2_use(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio

        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
    

class Dataset3D_Jue_Custmzed_Stage_1_use(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio

        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=4,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=4,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
        


class Dataset3D_Jue_Custmzed_Stage_4_use(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio

        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=32,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=32,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug
    

class Dataset3D_Jue_Custmzed_Stage_1_use(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2,use_intencty_Aug=0,used_3D_resize=1):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob  # Fix #13: was hardcoded to 0
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.use_intencty_Aug=use_intencty_Aug
        self.used_3D_resize=used_3D_resize
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio

        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=4,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=4,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale0_w_depth(self, image):
        _, img_d, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scaler_d = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)
        scale_d = int(self.crop3D_d * scaler_d)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)
        d0 = random.randint(0, img_d - scale_d)

        h1 = h0 + scale_h
        w1 = w0 + scale_w
        d1 = d0 + scale_d

        image_crop = image[:, d0:d1, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop

    def crop_scale_mirror_golbal_w_depth(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        
        

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        
        scaler_d = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_d * scaler_d) >= img_d):
            scaler_d = 1.

        


        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]

        #print ('scaler_d ',scaler_d)
        #print ('scaler_w ',scaler_w)
        #print ('scaler_h ',scaler_h)

        
        selection_int=np.random.randint(1,4)
        
        #if selection_int==1:
        #    scaler_d=1
        #elif selection_int==2:
        #    scaler_h=1
        #elif selection_int==3:
        #    scaler_w=1 
        #else:
            #raise NotImplementedError

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            
            if selection_int==1:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                # only resize [d,w]
                image_crop = cv2.resize(image_crop, (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                
            elif selection_int==2:
                # only resize [:,w,h]
                image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(2,0,1)

                

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            elif selection_int==3:
                

                # only resize [d,w]
                image_crop = cv2.resize(image_crop[0], (self.crop3D_d, self.crop3D_h), interpolation=cv2.INTER_LINEAR)
                #image_crop = image_crop.transpose(3,1,2)

                #only resize [d,h]
                image_crop = cv2.resize(image_crop.transpose(0,2,1), (self.crop3D_d, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
                image_crop = image_crop.transpose(0,2,1)
            else:

                raise NotImplementedError

            image_crop = image_crop[np.newaxis, :]

            #print ('!!! image_crop size after',image_crop.shape)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img


    def pad_image_512(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(512 - img.shape[0])
        cols_missing = math.ceil(512 - img.shape[1])
        dept_missing = math.ceil(500 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image_200(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(200 - img.shape[0])
        cols_missing = math.ceil(200 - img.shape[1])
        dept_missing = math.ceil(190 - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 

        
        #self.used_3D_resize=used_3D_resize

        if self.used_3D_resize==1:
            image_crop_ori = self.crop_scale0_w_depth(image)
            image_crop1 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal_w_depth(image_crop_ori, axes=(0, 1, 2))


        else:
            image_crop_ori = self.crop_scale0(image)
            image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
            image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 

        

        

        

        image_crop1=image_crop1.transpose((0, 2, 3, 1))
        image_crop2=image_crop2.transpose((0, 2, 3, 1))
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        if self.use_intencty_Aug==1:
            data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
            data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

            # data augmentation for the two views 
            data_dict1 = self.tr_transforms3D_global0(**data_dict1)
            data_dict2 = self.tr_transforms3D_global1(**data_dict2)

            img.append(torch.from_numpy(data_dict1['image']))
            img.append(torch.from_numpy(data_dict2['image']))
        else:
            img.append(torch.from_numpy(image_crop1.astype(np.float32)))
            img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


        
            
        return img

    def __getitem___Not_Use(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]

        iamge_ori=image
        #print ('iamge_ori size ',iamge_ori.shape)
        
        #Pad image 
        #image = self.pad_image(image)
        #print (' padded image size ',image.shape)

        #print ('!!! iamge_ori size ',iamge_ori.shape)
        #image=self.pad_image_512(iamge_ori)

        #iamge_ori2=image  # here no problem 
        #print (' padded image size ',image.shape) #512,512,500

        image = image[np.newaxis, :]

        #print (' image[np.newaxis, :] image size ',image.shape)

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))  # 0,1,2,3
        #[b,z,y,x]
        #print ('  image.transpose((0, 3, 1, 2)) image size ',image.shape)  # 1,500,512,512
        
        iamge_ori=image
        
        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        
        image_crop3=image_crop1.transpose((0, 2, 3, 1))
        #image_crop3=image_crop1
        
        
        #print ('!!! image_crop1 size ',image_crop3.shape)
        #iamge_ori=self.pad_image_200(image_crop3[0])
        #print ('!!! padded image_crop1/iamge_ori size ',iamge_ori.shape)
        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)

        img.append(torch.from_numpy(iamge_ori))
        img.append(torch.from_numpy(image_crop_ori))
        img.append(torch.from_numpy(image_crop1))
        img.append(torch.from_numpy(image_crop2))


        img_debug=[]
        img_debug.append(torch.from_numpy(image_crop1))
            
        return img_debug#_debug


class Dataset3D(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2):
        
        #print ('info: root ',root)
        self.root = root
        self.flip_prob=flip_prob
        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop


    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img

    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]

        #[b,x,y,z]
        image = image.transpose((0, 3, 1, 2))
        #[b,z,y,x]

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(data_dict1['image']))
        img.append(torch.from_numpy(data_dict2['image']))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


            
        return img



class Dataset3D_No_Intensity_Aug(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal',flip_prob=0.2):
        
        #print ('info: root ',root)
        self.root = root

        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        self.flip_prob=flip_prob
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        #self.tr_transforms3D_global0 = get_train_transform3D_global0()
        #self.tr_transforms3D_global1 = get_train_transform3D_global1()

        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < self.flip_prob:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        return image_crop


    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img

    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]
        image = image.transpose((0, 3, 1, 2))

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        #data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        #data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        image_crop1 = image_crop1.transpose((0, 3, 1, 2))
        image_crop2 = image_crop2.transpose((0, 3, 1, 2))

        img.append(torch.from_numpy(image_crop1.astype(np.float32)))
        img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


            
        return img



class Dataset3D_No_Intensity_Aug_96(Dataset):
    def __init__(self, root, list_path, teacher_mask_ratio,crop_size_3D=(64, 256, 256), data_type='3D_Modal'):
        
        #print ('info: root ',root)
        self.root = root

        #self.root='/lila/data/deasy/Eric_Data/EVA_clinic_data/Extracted_all_data/'
        self.list_path = self.root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()
        self.teacher_mask_ratio=teacher_mask_ratio
        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        #self.tr_transforms3D_global0 = get_train_transform3D_global0()
        #self.tr_transforms3D_global1 = get_train_transform3D_global1()

        self.mask_generator = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )

        self.mask_generator_Teacher = MaskGenerator(
            input_size=crop_size_3D[0],
            mask_patch_size=8,#
            model_patch_size=2,#,
            mask_ratio=self.teacher_mask_ratio,
        )

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def truncate(self,CT):
        # truncate
        min_HU = -500
        max_HU = 500
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU

        #CT=CT+500
        CT=(CT+500)/1000.
        # CT = CT - 158.58
        #CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        
        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.crop3D_d * scaler_d)
        scale_h = int(self.crop3D_h * scaler_h)
        scale_w = int(self.crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d) # random crop one views 
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, ::-1]


        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1,2,0), (self.crop3D_h, self.crop3D_w), interpolation=cv2.INTER_LINEAR)

            image_crop = image_crop[np.newaxis, :].transpose(0,3,1,2)

        # here NEED to transfor it back 
        #image_crop=image_crop.transpose(0,3,1,2)
        return image_crop


    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(128 - img.shape[0])
        cols_missing = math.ceil(128 - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img

    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])

        image = imageNII.get_fdata()

        #print (' image size ',image.shape)
        name = datafiles["name"]
        #Pad image 
        image = self.pad_image(image)

        #print (' padded image size ',image.shape)

        image = image[np.newaxis, :]
        image = image.transpose((0, 3, 1, 2))

        #print (' transposed image size ',image.shape)
        img = []

        # first crop the image and scale it 
        image_crop_ori = self.crop_scale0(image)

        #print (' 1st crop padded image size ',image_crop_ori.shape)

        # Global patches # 2 global views 
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        #print ('view 1 image size ',image_crop1.shape)
        #print ('view 2 image size ',image_crop2.shape)

        mask1_all= self.mask_generator()
        mask2_all= self.mask_generator()

        mask1_all_teacher=self.mask_generator_Teacher()
        mask2_all_teacher=self.mask_generator_Teacher()
        
        #data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        #data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # data augmentation for the two views 
        #data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        #data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(torch.from_numpy(image_crop1.astype(np.float32)))
        img.append(torch.from_numpy(image_crop2.astype(np.float32)))

        mask1_token=mask1_all[0]
        mask1=mask1_all[1]

        #print ('info: mask1_token size ',mask1_token.shape)
        #print ('info: mask1 size ',mask1.shape)

        mask2_token=mask2_all[0]
        mask2=mask2_all[1]

        mask1_teacher=mask1_all_teacher[1]
        mask2_teacher=mask2_all_teacher[1]

        img.append(mask1)
        img.append(mask2)

        img.append(mask1_token)
        img.append(mask2_token)

        img.append(mask1_teacher)
        img.append(mask2_teacher)


            
        return img


def get_train_transform3D_global0():

    tr_transforms = []
    # add gaussion noise 
    tr_transforms.append(GaussianNoiseTransform(data_key="image"))
    # gaussion blur it 
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.1, 2.), different_sigma_per_channel=True, p_per_channel=0.8, p_per_sample=0.8, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.8, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.4, True, p_per_sample=0.8, p_per_channel=0.8, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True, p_per_sample=0.8, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_train_transform3D_global1():

    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(data_key="image"))
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.1, 2.), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.8, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=1.0, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.4, True, p_per_sample=0.5, p_per_channel=0.8, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True, p_per_sample=1.0, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms
