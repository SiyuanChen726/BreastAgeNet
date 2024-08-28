import os
import io
import glob
import h5py
from tqdm import tqdm
import random
import numpy as np
import pandas as pd

import umap
import umap.plot
import staintools
import openslide
from PIL import Image
import matplotlib.pyplot as plt


import sys
sys.path.append('/scratch/users/k21066795/prj_normal/RandStainNA')
from randstainna import RandStainNA


import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torchvision.models as models
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import vision_transformer as vits
from transformers import AutoImageProcessor, ViTModel
import timm
from huggingface_hub import login, hf_hub_download
# from ctran import ctranspath

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def Reinhard(img_arr):
    standard_img = "/scratch/users/k21066795/prj_normal/he_shg_synth_workflow/thumbnails/he.jpg"
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
            pth_transforms.CenterCrop(image_size),
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

    elif model_name == "ctranspath":
        # https://github.com/Xiyue-Wang/TransPath/blob/main/get_features_CTransPath.py 
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'/scratch/users/k21066795/prj_normal/awesome_normal_breast/ckpts/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = pth_transforms.Compose([
            pth_transforms.Resize(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean = mean, std = std)])
        
    model.eval()
    model.to(device)
    return model, transform



class CSVDataset_WSI(Dataset):    
    # get patches from df
    def __init__(self, bag_df, WSIs, stainFunc=None, transforms_eval=eval_transforms()):
        self.csv = bag_df
        self.WSIs = WSIs
        self.transforms = transforms_eval   
        self.stainFunc = stainFunc

    def __getwsi__(self, wsi_id):
        wsi_pt = glob.glob(f"{self.WSIs}/{wsi_id}*.*")[0]
        self.wsi = openslide.OpenSlide(wsi_pt)

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
                yaml_file = '/scratch/users/k21066795/prj_normal/RandStainNA/CRC_LAB_randomTrue_n0.yaml',
                std_hyper = 0.0,
                distribution = 'normal',
                probability = 1.0,
                is_train = True) # is_train:True——> img is RGB format
            patch_im = Image.fromarray(augmentor(patch_im))
        else:
            patch_im = Image.fromarray(patch_im)
            
        return self.transforms(patch_im), self.csv.iloc[index]['patch_id']
    
    def __len__(self):
        return self.csv.shape[0]



def extract_features(bag_df, WSIs, model, stainFunc, transform, device, fname, batch_size, num_workers):
    
    bag_dataset = CSVDataset_WSI(bag_df, WSIs, stainFunc=stainFunc, transforms_eval=transform)
    
    unique_items, counts = np.unique(list(bag_df['cls']), return_counts=True)
    print(f"------{unique_items} {counts} patches------")
    
    bag_dataloader = torch.utils.data.DataLoader(bag_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    print(len(bag_dataloader))
    
    embeddings, labels = [], []
    for batch, target in tqdm(bag_dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            try:
                embeddings.append(model(batch).detach().cpu().numpy())
            except:
                output = model(batch, output_hidden_states=True)
                _embeddings = output.hidden_states[-1][:, 0, :].detach().cpu().numpy()
                embeddings.append(_embeddings)
            labels.extend(target)
        
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels).squeeze()
    print(f"embeddings shape is {embeddings.shape}")
    print(f"labels shape is {labels.shape}")
    
    with h5py.File(fname, mode='w') as f:
        f.create_dataset(name="embeddings", shape=embeddings.shape, dtype=np.float32, data=embeddings)
        labels = [i.encode("utf-8") for i in labels]
        dt = h5py.string_dtype(encoding='utf-8', length=None)
        f.create_dataset(name="patch_id", shape=[len(labels)], dtype=dt, data=labels)
    
    print(f"{fname} saved!")




def labelingFeature(feature_pt, meta_label):
    slide_folder = os.path.split(feature_pt)[0]
    csv_pt = glob.glob(f"{slide_folder}/*_patch_merge.csv")[0]
    # print(feature_pt, csv_pt)
    
    with h5py.File(feature_pt, "r") as f:
        embedding = np.array(f["embeddings"])
        img_id = np.array(f["patch_id"])
        
    img_id = [i.decode("utf-8") for i in img_id]
    df = pd.read_csv(csv_pt) 
    df.index = df["patch_id"]
    label = np.array(df.loc[img_id, meta_label].values)
    
    return embedding, label





def ensemble_feas_labels(model_name, rootdir, meta_label, stainFunc):
    feature_files = glob.glob(f"{rootdir}/*/*_bagFeature_*{model_name}_{stainFunc}.h5")
    counter = 0
    for feature_pt in feature_files:
        try:
            if counter==0:
                embeddings, labels = labelingFeature(feature_pt, meta_label)
            else:
                embedding, label = labelingFeature(feature_pt, meta_label)
                embeddings = np.concatenate((embeddings, embedding), axis=0)
                labels = np.concatenate((labels, label), axis=0)
            counter += 1
        except:
            continue
    return embeddings, labels



def create_UMAP(embeddings, labels, sample_num, n=50, d=0.2, save_pt=None):
    
    fig = plt.figure(figsize=(10, 10), dpi=100)
    
    if len(embeddings.shape) != 2:
        embeddings = np.squeeze(embeddings)
    random_index = random.sample(range(0, embeddings.shape[0]), sample_num)
    embeddings, labels = embeddings[random_index], labels[random_index]
    
    mapper = umap.UMAP(n_neighbors=n, min_dist=d, random_state=42).fit(embeddings)
    umap.plot.points(mapper, labels=labels, width=600, height=600)
    
    plt.tight_layout()
    
    if save_pt is not None:
        plt.savefig(save_pt)
        print(f"{save_pt} saved!")
        image = Image.open(save_pt)
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        
    return image




def sampeled_UMP(model_name, meta_label, sample_num, rootdir, stainFunc, n=50, d=0.2, save_pt=None):
    embeddings, labels = ensemble_feas_labels(model_name, rootdir, meta_label, stainFunc)
    
    print(embeddings.shape)
    
    random_index = random.sample(range(0, embeddings.shape[0]), sample_num)
    img = create_UMAP(embeddings[random_index], labels[random_index], n, d, save_pt)
    
    return img



# class CSVDataset_patch(Dataset): 
#     # get patches from patches saved in a folder
#     def __init__(self, patch_dir, patch_df, transforms_eval=eval_transforms()):
#         self.patch_dir = patch_dir
#         self.csv = patch_df
#         self.csv.reset_index(inplace=True)
#         self.transforms = transforms_eval   
        
#     def __getitem__(self, index):
#         img_pt = glob.glob(f"{self.patch_dir}/{self.csv.iloc[index]['patch_id']}*")[0]
#         img = Image.open(img_pt)
#         return self.transforms(img), self.csv.iloc[index]['patch_id']
    
#     def __len__(self):
#         return self.csv.shape[0]


    
# class CSVDataset_ROI(Dataset):    
#     # get patches from df
#     def __init__(self, csv_pt, ROIs, patch_size, transforms_eval=eval_transforms()):
#         self.ROIs = ROIs
#         self.csv = pd.read_csv(csv_pt)
#         self.patch_size = patch_size
#         self.transforms = transforms_eval   

#     def __getroi__(self, roi_id):
#         roi_pt = glob.glob(f"{self.ROIs}/*/*/{roi_id}*")
#         roi = np.array(Image.open(roi_pt[0]))
#         return roi
        
#     def __getitem__(self, index):
#         roi_id = self.csv.iloc[index]['roi_id']
#         roi = self.__getroi__(roi_id)
#         patch_id = self.csv.iloc[index]['patch_id']
#         x,y = patch_id.split("_")[-2:]
#         x, y = int(x), int(y)
#         patch_im = roi[x : (x+self.patch_size), y : (y+self.patch_size)]
#         patch_im_norm = Image.fromarray(Reinhard(np.array(patch_im)))
        
#         return self.transforms(patch_im_norm), patch_id
    
#     def __len__(self):
#         return self.csv.shape[0]




# class CSVDataset_wsi(Dataset):    
#     # get patches from df
#     def __init__(self, patch_df, wsi, transforms_eval=eval_transforms()):
#         self.csv = patch_df
#         self.wsi = wsi
#         self.transforms = transforms_eval   

#     def __getxy__(self, patch_id):
#         grid_x, grid_y, patch_size = patch_id.split("_")[-3:]
#         grid_x, grid_y, patch_size = int(grid_x),int(grid_y),int(patch_size)
#         return grid_x, grid_y, patch_size

#     def __getitem__(self, index):
#         patch_id = self.csv.iloc[index]['patch_id']
#         grid_x, grid_y, patch_size = self.__getxy__(patch_id)
   
#         patch_im = self.wsi.read_region((grid_x*patch_size, grid_y*patch_size), 0, (patch_size, patch_size)).convert("RGB")
#         patch_im_norm = Image.fromarray(Reinhard(np.array(patch_im)))
#         return self.transforms(patch_im_norm), self.csv.iloc[index]['patch_id']
    
#     def __len__(self):
#         return self.csv.shape[0]




# def save_wsi_embeddings(model_name, patch_df, wsi, fname):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, transform = get_model(model_name, device)    
    
#     dataset = CSVDataset_wsi(patch_df, wsi, transforms_eval=transform)
    
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False, num_workers=4)

#     print(f"device: {device}")
#     print(f"Using {model_name} to extract embeddings of {len(dataset)} patches")

#     embeddings, labels = [], []
#     for batch, target in tqdm(dataloader):
#         with torch.no_grad():
#             batch = batch.to(device)
#             try:
#                 embeddings.append(model(batch).detach().cpu().numpy())
#             except:
#                 output = model(batch, output_hidden_states=True)
#                 _embeddings = output.hidden_states[-1][:, 0, :].detach().cpu().numpy()
#                 embeddings.append(_embeddings)
#             labels.extend(target)
        
#     embeddings = np.vstack(embeddings)
#     labels = np.vstack(labels).squeeze()
#     with h5py.File(fname, mode='w') as f:
#         f.create_dataset(name="embeddings", shape=embeddings.shape, dtype=np.float32, data=embeddings)
#         labels = [i.encode("utf-8") for i in labels]
#         dt = h5py.string_dtype(encoding='utf-8', length=None)
#         f.create_dataset(name="patch_id", shape=[len(labels)], dtype=dt, data=labels)
#     print(f"{fname} saved!")
    