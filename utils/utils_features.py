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

import timm
import utils.vision_transformer as vits
from utils.vision_transformer import vit_base

from transformers import AutoModel
from transformers import AutoImageProcessor, ViTModel
# from huggingface_hub import login, hf_hub_download
# login()


sys.path.append('/app/RandStainNA')
from randstainna import RandStainNA

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def Reinhard(img_arr):
    standard_img = "/app/BreastAgeNet/data/he.jpg"
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
    custom_path = '/app/BreastAgeNet/weights' 
    os.makedirs(custom_path, exist_ok=True)
    
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
    
    elif model_name == "phikon":
        model = ViTModel.from_pretrained("owkin/phikon", cache_dir=custom_path, add_pooling_layer=False)
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
        # local_dir = f"{custom_path}/UNI"
        # os.makedirs(local_dir, exist_ok=True) 
        # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(custom_path, 'UNI', "pytorch_model.bin"), map_location="cpu"), strict=True)
        transform = pth_transforms.Compose(
            [pth_transforms.Resize((256,256)),
             pth_transforms.CenterCrop((224, 224)),
             pth_transforms.ToTensor(),
             pth_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    elif model_name == "gigapath":
        # local_dir = f"{custom_path}/gigapath"
        # os.makedirs(local_dir, exist_ok=True)  
        # hf_hub_download("prov-gigapath/prov-gigapath", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
        model.load_state_dict(torch.load(os.path.join(custom_path, 'gigapath', "pytorch_model.bin"), map_location="cpu"), strict=True)
        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(256, interpolation=pth_transforms.InterpolationMode.BICUBIC),
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
        wsi_pt = glob.glob(f"{self.WSIs}/**/{wsi_id}*.*", recursive=True)
        if not wsi_pt:
            raise FileNotFoundError(f"WSI file for '{wsi_id}' not found in '{self.WSIs}' or subdirectories.")
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
        
        if self.stainFunc == 'reinhard': 
            patch_im = Image.fromarray(Reinhard(np.array(patch_im)))
       
        elif self.stainFunc == 'augmentation':
            augmentor = RandStainNA(
                yaml_file = '/app/RandStainNA/CRC_LAB_randomTrue_n0.yaml',
                std_hyper = 0.0,
                distribution = 'normal',
                probability = 1.0,
                is_train = True) # is_train:True——> img is RGB format
            patch_im = Image.fromarray(augmentor(patch_im))
            

        elif self.stainFunc == 'raw':
            patch_im = Image.fromarray(np.array(patch_im))
            
        return self.transforms(patch_im), self.csv.iloc[index]['patch_id']
    
    def __len__(self):
        return self.csv.shape[0]




class Dataset_frompatch(Dataset):
    def __init__(self, patch_df, stainFunc, transforms_eval):
        self.csv = patch_df.copy()
        self.stainFunc = stainFunc
        print(f'stain normalsation: {self.stainFunc}')
        self.transforms = transforms_eval
        
    def __getitem__(self, index):
        img_pt = self.csv.iloc[index]['file_path']
        patch_im = Image.open(img_pt).convert('RGB')
        
        if self.stainFunc == 'reinhard': 
            patch_im = Image.fromarray(Reinhard(np.array(patch_im)))
            
        elif self.stainFunc == 'augmentation':
            augmentor = RandStainNA(
                yaml_file = '/app/RandStainNA/CRC_LAB_randomTrue_n0.yaml',
                std_hyper = 0.0,
                distribution = 'normal',
                probability = 1.0,
                is_train = True) # is_train:True——> img is RGB format
            patch_im = Image.fromarray(augmentor(patch_im))

        elif self.stainFunc == 'raw':
            patch_im = Image.fromarray(np.array(patch_im))
            
        return self.transforms(patch_im), self.csv.iloc[index]['patch_id']
        
    def __len__(self):
        return self.csv.shape[0]



def extract_features(model, bag_dataset, batch_size, num_workers, device, fname):
    bag_dataloader = torch.utils.data.DataLoader(bag_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    print(f"Total number of batches: {len(bag_dataloader)}")
    
    embeddings, labels = [], []
    for batch, target in tqdm(bag_dataloader, desc="Extracting features"):
        with torch.no_grad():
            batch = batch.to(device)
            try:
                output = model(batch)
                embeddings.append(output.detach().cpu().numpy())
            except Exception as e:
                try:
                    features = model.encode_image(batch, proj_contrast=False, normalize=False)
                    embeddings.append(features.detach().cpu().numpy())
                except Exception as e:
                    output = model(batch, output_hidden_states=True)
                    _embeddings = output.hidden_states[-1][:, 0, :].detach().cpu().numpy()
                    embeddings.append(_embeddings)
            labels.extend(target)
        
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels).squeeze()
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    with h5py.File(fname, mode='w') as f:
        f.create_dataset(name="embeddings", shape=embeddings.shape, dtype=np.float32, data=embeddings)
        labels = [i.encode("utf-8") for i in labels]  # Ensure labels are byte-encoded
        dt = h5py.string_dtype(encoding='utf-8', length=None)
        f.create_dataset(name="patch_id", shape=[len(labels)], dtype=dt, data=labels)
    print(f"Saved to {fname}")



