import os
import glob
import h5py
import numpy as np
import pandas as pd
import random
import torch
import traceback
import logging
import argparse
torch.multiprocessing.set_sharing_strategy('file_system')
from utils_features import *



# Setup argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction script for BreastAgeNet")

    # Add arguments
    parser.add_argument('--WSIs', type=str, required=True, help="Path to the WSIs directory")
    parser.add_argument('--FEATURES', type=str, required=True, help="Path to the FEATURES directory")
    parser.add_argument('--model_name', type=str, required=True, choices=["resnet50", "iBOT", "UNI", "gigapath"], help="Model name for feature extraction")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--stainFunc', type=str, default='Reinhard', choices=["reinhard", "augmentation"], help="Staining function to use (e.g., 'Reinhard')")
    parser.add_argument('--log_file', type=str, default='feature_extraction.log', help="Path to the log file")

    return parser.parse_args()



# Feature extraction function
def extract_features_for_wsi(wsiname, FEATURES, WSIs, model, stainFunc, transform, device, batch_size, num_workers):
    try:
        bag_folder = os.path.join(FEATURES, wsiname)
        fname = f"{bag_folder}/{wsiname}_bagFeature_{model_name}_{stainFunc}.h5"

        if not os.path.exists(fname):
            bag_csv = glob.glob(f"{bag_folder}/*_patch.csv")[0]
            bag_df = pd.read_csv(bag_csv)

            if len(bag_df) > 0:
                # bag_df = bag_df.groupby('cls', as_index=False).apply(lambda x: x.sample(n=5000, replace=True)) # avoid getting too many patches
                # bag_df = bag_df.reset_index(drop=True).drop_duplicates()
                print(f"------ Extracting {model_name} features for {wsiname} with {stainFunc} ------")
                logging.info(f"Extracting {model_name} features for {wsiname} with {stainFunc}")
                extract_features(bag_df, WSIs, model, stainFunc, transform, device, fname, batch_size, num_workers)
            else:
                logging.warning(f"No patches found for {wsiname}. Skipping extraction.")
        else:
            logging.info(f"{fname} already exists!")

    except Exception as e:
        logging.error(f"Failed to extract {model_name} features for {wsiname}. Error: {str(e)}")
        logging.error(traceback.format_exc())

        
        
        
# Parse arguments
args = parse_args()

# Setup logger
logging.basicConfig(filename=args.log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Started feature extraction process.')

# Assign arguments to variables
WSIs = args.WSIs
FEATURES = args.FEATURES
model_name = args.model_name
batch_size = args.batch_size
num_workers = args.num_workers
stainFunc = args.stainFunc

# Setup CUDA device and model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device: {device}")

# Initialize model and transformation
model, transform = get_model(model_name, device)
print(f"Model: {model_name}")


# Iterate over WSIs and extract features
wsinames = os.listdir(FEATURES)
for wsiname in wsinames:
    extract_features_for_wsi(wsiname, FEATURES, WSIs, model, stainFunc, transform, device, batch_size, num_workers)

logging.info('Feature extraction process completed.')


