import os
import glob
import h5py
import numpy as np
import pandas as pd
import random
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
print("Is CUDA available:", torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())

from utils_features import get_model, extract_features

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="/scratch/users/k21066795/prj_normal/BreastAgeNet_MIL/configs/", config_name="featureExtraction")
def run(config : DictConfig) -> None:
    
    WSIs = '/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/WSIs/' + config.WSIs
    rootdir = '/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/RootDir/' + config.RootDir
    cls = config.cls
    model_name = config.model_name
    batch_size = config.batch_size
    num_workers = config.num_workers
    stainFunc = config.stainFunc


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device: {device}")
    
    model, transform = get_model(model_name, device)
    print(f"model_name: {model_name}")

    wsinames = os.listdir(rootdir)
    random.shuffle(wsinames)
    for wsiname in wsinames:
        bag_folder = os.path.join(rootdir, wsiname)
        fname = f"{bag_folder}/{os.path.basename(bag_folder)}_bagFeature_{model_name}_{stainFunc}.h5"
        print(f"------extracting {model_name} features from {stainFunc} patches for {wsiname}------")
        
        if not os.path.exists(fname):
            try:
                bag_csv = glob.glob(f"{bag_folder}/*_patch.csv")[0]
                bag_df = pd.read_csv(bag_csv)
                
                if len(bag_df) > 0:
                    print(f"------extracting {model_name} features for {wsiname}------")
                    bag_df = bag_df.loc[bag_df['cls'].isin(cls)]
                    extract_features(bag_df, WSIs, model, stainFunc, transform, device, fname, batch_size, num_workers)
            except:
                print(f"failed to extract {model_name} features for {wsiname}")
                continue
        else:
            print(f"{fname} exists!")





if __name__ == "__main__":
    run()  


