import h5py
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import torch
import torch.optim as optim
from fastai.vision.all import *
from utils_train import *
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Train BreastAgeNet for ordinal classification of ageing status of NBT")
    
    # Hyperparameters
    parser.add_argument('--model_name', type=str, default='UNI', choices=['UNI', 'gigapath', 'iBOT', 'resnet50'],
                        help='Model for feature extraction (default: UNI)')
    parser.add_argument('--TC_epi', type=float, default=0.9, choices=[0.0, 0.9],
                        help='Threshold for epithelium patches (default: 0.0)')
    parser.add_argument('--bag_size', type=int, default=150, choices=[50, 100, 150, 250, 350, 500], 
                        help='Bag size (default: 150)')
    parser.add_argument('--attention', type=str, default='MultiHeadAttention', 
                        choices=['MultiHeadAttention', 'Attention', 'GatedAttention'], 
                        help='Attention mechanism to use (default: MultiHeadAttention)')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--n_classes', type=int, default=3, help='Number of output classes (default: 3)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate (default: 0.00001)')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay (default: 0.00005)')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (default: 10)')
    parser.add_argument('--minEpochTrain', type=int, default=0, help='Minimum epoch for training (default: 0)')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum epochs for training (default: 200)')
    
    # Paths
    parser.add_argument('--FEATURES', type=str, default='path/save/FEATURES', help='Path to FEATURES directory')
    parser.add_argument('--CLINIC', type=str, default='path/save/clinicData_all.csv', help='Path to clinical data CSV')
    parser.add_argument('--RESULTS', type=str, default='path/save/RESULTS', help='Directory to save results')
    
    # Number of splits for cross-validation
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for Stratified K-Fold (default: 5)')
    
    args = parser.parse_args()
    return args


  
    

# Parse command-line arguments
args = parse_args()

model_name = args.model_name
TC_epi = args.TC_epi
bag_size = args.bag_size
attention = args.attention
n_heads = args.n_heads
n_classes = args.n_classes
batch_size = args.batch_size
lr = args.lr
weight_decay = args.weight_decay
patience = args.patience
minEpochTrain = args.minEpochTrain
max_epochs = args.max_epochs
FEATURES = args.FEATURES
CLINIC = args.CLINIC
RESULTS = args.RESULTS
n_splits = args.n_splits  


# Load and process clinical data
wsi_df = pd.read_csv(CLINIC)
wsi_df = wsi_df.loc[wsi_df["cohort"].isin(["SGK", "EPFL"]), :].copy()
wsi_df = add_ageGroup(wsi_df)
valid_wsi = []
valid_patches = []
for wsi_id in wsi_df.wsi_id.values:
    try:
        # Read features from HDF5 file
        fea_pt = f"{FEATURES}/*/{wsi_id}/{wsi_id}_bagFeature_{model_name}_reinhard.h5"
        with h5py.File(fea_pt, "r") as file:
            bag = np.array(file["embeddings"])
            bag = np.squeeze(bag)
            img_id = np.array(file["patch_id"])
        img_id = [i.decode("utf-8") for i in img_id]
        bag_df = pd.DataFrame(bag)
        bag_df.index = img_id

        # Read corresponding CSV file with patch info
        csv_pt =  os.path.join(Path(fea_pt).parent, Path(fea_pt).stem.split('_bagFeature_')[0]+'_patch.csv')
        df = pd.read_csv(csv_pt)

        # Filter patches based on TC_epi threshold
        valid_id = list(df['patch_id'][df['TC_epi'] > TC_epi])
        valid_id = list(set(valid_id) & set(bag_df.index))
        valid_patches.extend(valid_id)

        if valid_id:
            valid_wsi.extend([wsi_id] * len(valid_id))

    except Exception as e:
        print(f"Error processing WSI {wsi_id}: {e}")


# Filter WSI IDs that have at least 5 patches
a, b = np.unique(valid_wsi, return_counts=True)
filtered_indices = b >= 5
valid_ids = a[filtered_indices]
valid_patches = [patch_id for patch_id in valid_patches if parse_wsi_id(patch_id) in valid_ids]
print(len(np.unique(valid_wsi)))
print(len(valid_patches))
df = wsi_df.copy()
df = df.loc[df["wsi_id"].isin(valid_ids), :]
print(np.unique(df["age_group"], return_counts=True))


# Data split for K-Fold cross-validation
patientID = df["patient_id"].values
truelabels = df["age_group"].values
kf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
kf.get_n_splits(patientID, truelabels)


# Cross-validation training
for foldcounter, (train_index, test_index) in enumerate(kf.split(patientID, truelabels)):
    fold_pt = f"{RESULTS}/epi{TC_epi}_{bag_size}_{model_name}_{attention}_fold{foldcounter}_train.csv"
    train_df, val_df, test_df = split_data(df, FEATURES, patientID, truelabels, train_index, test_index, fold_pt)

    # dataloaders
    dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                   get_x = ColReader('h5df'),
                   get_y = ColReader('age_group'),
                   splitter = ColSplitter('is_valid'),
                   item_tfms = MILBagTransform(train_df.h5df, bag_size, valid_patches=valid_patches))

    dls = dblock.dataloaders(train_df, bs = batch_size)
    trainLoaders = dls.train
    valLoaders = dls.valid
    # for testing, print shapes of data
    (patch_ids, embeds), labels = next(iter(trainLoaders))
    patch_ids = np.array(patch_ids)  
    patch_ids = np.transpose(patch_ids)  
    print(patch_ids.shape, embeds.shape, labels.shape)  # (16, 150) torch.Size([16, 150, 1536]) torch.Size([16])

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model
    n_feats = get_dim_input(model_name)
    model = BreastAgeNet(n_feats, attention, n_classes, n_heads=8, n_latent=512, embed_attn=False).to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    value_counts = train_df.loc[(train_df["is_valid"] == False) & (train_df["age_group"] != 0), "age_group"].value_counts()
    reversed_weights = 1.0 / (value_counts / value_counts.sum())  # Inverse of frequency-based weights
    reversed_weights = reversed_weights / reversed_weights.sum()  # Normalize the weights to sum to 1
    weights = torch.tensor(reversed_weights.values, dtype=torch.float32).to(device)  # Convert to tensor and move to device
    print(weights)
    criterion = OrdinalCrossEntropyLoss(n_classes=n_classes, weights=weights)
    criterion.to(device)  # Move to GPU if applicable

    # trainer
    ckpt_name = f'{RESULTS}/epi{TC_epi}_{bag_size}_{model_name}_{attention}_fold{foldcounter}_bestModel.pt'
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
        model, trainLoaders, valLoaders, 
        optimizer, criterion, 
        patience, minEpochTrain, max_epochs, ckpt_name)

    # testing in the test set
    testLoader = dls.test_dl(test_df)
    predicted_ranks = test_model(model, testLoader)
    MAE = np.abs(test_df["age_group"].values - predicted_ranks).mean()

    # plot MAE and loss curves for train/val sets
    save_pt = f"{RESULTS}/epi{TC_epi}_{bag_size}_{model_name}_{attention}_fold{foldcounter}_trainvalCurves_MAE{MAE}.png"
    plot_training(train_loss_history, val_loss_history, train_acc_history, val_acc_history)























