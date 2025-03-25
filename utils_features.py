import os
import io
import sys
import glob
import random
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import staintools
import openslide

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms as pth_transforms
from transformers import AutoImageProcessor, ViTModel
import timm
import vision_transformer as vits
from vision_transformer import vit_base
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import login, hf_hub_download

sys.path.append('/scratch_tmp/users/k21066795/RandStainNA')
from randstainna import RandStainNA

sys.path.append('/scratch_tmp/users/k21066795')
from EXAONEPath.vision_transformer import VisionTransformer
from EXAONEPath.macenko import macenko_normalizer

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def Reinhard(img_arr):
    standard_img = "/scratch_tmp/users/k21066795/NBT-Classifier/data/he.jpg"
    target = staintools.read_image(standard_img)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target)
    #img = staintools.read_image(img_path)
    img_to_transform = staintools.LuminosityStandardizer.standardize(img_arr)
    img_transformed = normalizer.transform(img_to_transform)
    return img_transformed
    


def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = pth_transforms.Compose([pth_transforms.ToTensor(), 
                                           pth_transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val




def get_model(model_name, device):
    if model_name == "resnet50":
        resnet50 = models.resnet50(pretrained=True)
        # Remove the final fully connected layer
        model = torch.nn.Sequential(*list(resnet50.children())[:-1])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        transform = pth_transforms.Compose([
            pth_transforms.Resize(256),                
            pth_transforms.CenterCrop(224),            
            pth_transforms.ToTensor(),                 
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif model_name == "conch":
        model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", 
                                                 hf_auth_token="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")
    
    elif model_name == "DINO_BRCA":
        # https://github.com/Richarizardd/Self-Supervised-ViT-Path.git
        # model configuration
        arch = 'vit_small'
        image_size=(256,256)
        pretrained_weights = '/scratch/users/k21066795/prj_normal/ckpts/vits_tcga_brca_dino.pt'
        checkpoint_key = 'teacher'
        # get model
        model = vits.__dict__[arch](patch_size=16, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False

        transform = pth_transforms.Compose([
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif model_name == "HIPT256":
        # https://github.com/mahmoodlab/HIPT.git
        pretrained_weights = '/scratch/users/k21066795/prj_normal/ckpts/vit256_small_dino.pth'
        checkpoint_key = 'teacher'
        arch = 'vit_small'
        image_size=(256,256)
        model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
        for p in model256.parameters():
            p.requires_grad = False
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        model = model256

        transform = pth_transforms.Compose([
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    elif model_name == "iBOT":
        # https://github.com/owkin/HistoSSLscaling.git
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

        image_size = (224, 224)
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        normalize = pth_transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std)
        
        transform = pth_transforms.Compose([
            pth_transforms.Resize((256,256)),
            pth_transforms.CenterCrop((224, 224)),
            pth_transforms.ToTensor(),
            normalize])

    elif model_name == "UNI":
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        # transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        # local_dir = "/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/ckpts/UNI"
        # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        # model = timm.create_model(
        #     "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        # )
        # model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        transform = pth_transforms.Compose(
            [pth_transforms.Resize((256,256)),
             pth_transforms.CenterCrop((224, 224)),
             # pth_transforms.Resize(224),
             pth_transforms.ToTensor(),
             pth_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    elif model_name == "gigapath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    
        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(256, interpolation=pth_transforms.InterpolationMode.BICUBIC),
                pth_transforms.CenterCrop(224),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    # elif model_name == "ctranspath":
    #     # https://github.com/Xiyue-Wang/TransPath/blob/main/get_features_CTransPath.py 
    #     model = ctranspath()
    #     model.head = nn.Identity()
    #     td = torch.load(r'/scratch/users/k21066795/prj_normal/awesome_normal_breast/ckpts/ctranspath.pth')
    #     model.load_state_dict(td['model'], strict=True)

    #     mean = (0.485, 0.456, 0.406)
    #     std = (0.229, 0.224, 0.225)
    #     transform = pth_transforms.Compose([
    #         pth_transforms.Resize(224),
    #         pth_transforms.ToTensor(),
    #         pth_transforms.Normalize(mean = mean, std = std)])

    elif model_name == "EXAONEPath":
        hf_token = "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"
        model = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath", use_auth_token=hf_token)
        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                pth_transforms.CenterCrop(224),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
    model.eval()
    model.to(device)
    
    return model, transform




class Dataset_fromWSI(Dataset):    
    def __init__(self, bag_df, WSIs, stainFunc=None, transforms_eval=eval_transforms()):
        self.csv = bag_df
        self.WSIs = WSIs
        self.transforms = transforms_eval   
        self.stainFunc = stainFunc

    def __getwsi__(self, wsi_id):
        wsi_pt = glob.glob(f"{self.WSIs}/*/{wsi_id}*.*")
        if not wsi_pt:
            wsi_pt = glob.glob(f"{self.WSIs}/*/*/{wsi_id}*.*")
        if not wsi_pt:
            raise FileNotFoundError(f"WSI file for {wsi_id} not found in {self.WSIs} or subdirectories.")
        self.wsi = openslide.OpenSlide(wsi_pt[0])
    
    def __getxy__(self, patch_id):
        grid_x, grid_y, patch_size = patch_id.split("_")[-3:]
        grid_x, grid_y, patch_size = int(grid_x),int(grid_y),int(patch_size)
        return grid_x, grid_y, patch_size

    def __getitem__(self, index):
        patch_id = self.csv.iloc[index]['patch_id']
        grid_x, grid_y, patch_size = self.__getxy__(patch_id)
        wsi_id = self.csv.iloc[index]['wsi_id']
        self.__getwsi__(wsi_id)
        patch_im = self.wsi.read_region((grid_x*patch_size, grid_y*patch_size), 0, (patch_size, patch_size)).convert("RGB")
        
        if self.stainFunc == 'Reinhard': 
            patch_im = Image.fromarray(Reinhard(np.array(patch_im)))
       
        elif self.stainFunc == 'RandStainNA':
            augmentor = RandStainNA(
                yaml_file = '/scratch_tmp/users/k21066795/RandStainNA/CRC_LAB_randomTrue_n0.yaml',
                std_hyper = 0.0,
                distribution = 'normal',
                probability = 1.0,
                is_train = True) # is_train:True——> img is RGB format
            patch_im = Image.fromarray(augmentor(patch_im))
            
        elif self.stainFunc== "Macenko":
            normalizer = macenko_normalizer(target_path = '/scratch_tmp/users/k21066795/EXAONEPath/macenko_target/target_TCGA-55-A48X_coords_[19440  9824]_[4096 4096].png')
            patch_im = normalizer(patch_im)
        
        elif self.stainFunc == 'raw':
            patch_im = Image.fromarray(np.array(patch_im))
            
        return self.transforms(patch_im), self.csv.iloc[index]['patch_id']
    
    def __len__(self):
        return self.csv.shape[0]

    

class Dataset_frompatch(Dataset):
    def __init__(self, patch_df, stainFunc, transforms_eval):
        self.csv = patch_df.copy()
        self.stainFunc = stainFunc
        self.transforms = transforms_eval
        
    def __getitem__(self, index):
        img_pt = self.csv.iloc[index]['file_path']
        patch_im = Image.open(img_pt).convert('RGB')
        
        if self.stainFunc == 'Reinhard': 
            patch_im = Image.fromarray(Reinhard(np.array(patch_im)))
        elif self.stainFunc == 'RandStainNA':
            augmentor = RandStainNA(
                yaml_file = '/scratch_tmp/users/k21066795/RandStainNA/CRC_LAB_randomTrue_n0.yaml',
                std_hyper = 0.0,
                distribution = 'normal',
                probability = 1.0,
                is_train = True) # is_train:True——> img is RGB format
            patch_im = Image.fromarray(augmentor(patch_im))
        elif self.stainFunc== "Macenko":
            normalizer = macenko_normalizer(target_path = '/scratch_tmp/users/k21066795/EXAONEPath/macenko_target/target_TCGA-55-A48X_coords_[19440  9824]_[4096 4096].png')
            patch_im = normalizer(patch_im)
        elif self.stainFunc == 'raw':
            patch_im = Image.fromarray(np.array(patch_im))
            
        return self.transforms(patch_im), self.csv.iloc[index]['patch_id']
        
    def __len__(self):
        return self.csv.shape[0]



# class Dataset_fromROI(Dataset):    
#     def __init__(self, ROIs, patch_size=512, transforms_eval=None):
#         self.ROIs = ROIs
#         self.patch_size = patch_size
#         self.transforms = transforms_eval if transforms_eval else transforms.Compose([transforms.ToTensor()])
#         self.roi_files = glob.glob(f"{self.ROIs}/*/*/*.png")  # Adjust the pattern based on your ROI files
                
#     def __getitem__(self, index):
#         roi_path = self.roi_files[index]
#         roi = np.array(Image.open(roi_path))
        
#         h, w = roi.shape[:2]
#         patches = []
#         for y in range(0, h, self.patch_size):
#             for x in range(0, w, self.patch_size):
#                 if x + self.patch_size <= w and y + self.patch_size <= h:
#                     patch = roi[y : y + self.patch_size, x : x + self.patch_size]
#                     patch_im_resized = Image.fromarray(patch).resize((256, 256))  # Resize patch to 256x256
#                     patch_im_norm = self.transforms(patch_im_resized)
#                     patches.append(patch_im_norm)
        
#         if not patches:
#             return torch.zeros(3, 256, 256), roi_path
        
#         return random.choice(patches), roi_path
    
#     def __len__(self):
#         return len(self.roi_files)


