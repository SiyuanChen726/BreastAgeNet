'''
modified from 
https://github.com/KatherLab/HIA/blob/main/models/model_Attmil.py
https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
'''

import os
from pathlib import Path
from typing import Tuple, Optional, List, Union
import h5py
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import torch.optim as optim
from torch.nn import CrossEntropyLoss
from fastai.vision.all import *
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def add_ageGroup(df):
    df['age_group'] = 3
    df.loc[df['age'] < 35, 'age_group'] = 0
    df.loc[(df['age'] >= 35) & (df['age'] < 45), 'age_group'] = 1
    df.loc[(df['age'] >= 45) & (df['age'] < 55), 'age_group'] = 2
    return df



def get_data(config): 
    clinic_df = pd.read_csv(config["clinic_path"])
    clinic_df = add_ageGroup(clinic_df)
    
    wsi_ids = []
    for wsi_id in clinic_df["wsi_id"]:
        # Look for files that match the pattern
        file = list(glob.glob(f'{config["FEATURES"]}/{wsi_id}*/*{config["model_name"]}*{config["stainFunc"]}.h5'))
        if file:  # Check if file list is not empty
            wsi_ids.append(wsi_id)
    
    clinic_df = clinic_df[clinic_df["wsi_id"].isin(wsi_ids)]
    print("Filtered DataFrame length:", len(clinic_df))
    print("Unique age groups and counts:", np.unique(clinic_df["age_group"], return_counts=True))
    
    valid_wsi = []
    valid_patches = []
    for wsi_id in clinic_df.wsi_id.values:
        fea_pt = glob.glob(f'{config["FEATURES"]}/{wsi_id}*/*{config["model_name"]}*{config["stainFunc"]}.h5')[0] 
        with h5py.File(fea_pt, "r") as file:
            bag = np.array(file["embeddings"])
            bag = np.squeeze(bag)
            img_id = np.array(file["patch_id"])
        img_id = [i.decode("utf-8") for i in img_id]
        bag_df = pd.DataFrame(bag)
        bag_df.index = img_id
    
        csv_pt = fea_pt.split('_bagFeature_')[0]+ '_patch.csv'
        df = pd.read_csv(csv_pt)
    
        valid_id = list(df['patch_id'][df['TC_epi'] > config["TC_epi"]])
        valid_id = list(set(valid_id) & set(bag_df.index))
        valid_patches.extend(valid_id)
        if valid_id:
            valid_wsi.extend([wsi_id] * len(valid_id))
    
    print(len(np.unique(valid_wsi)))
    print(len(valid_patches))
    a, b = np.unique(valid_wsi, return_counts=True)
    # sorted_zip = sorted(zip(b, a))
    # a = [i[1] for i in sorted_zip]
    # b = [i[0] for i in sorted_zip]
    # plt.plot(b , marker='o', markersize=1)
    # plt.title('Line Plot of Values')
    # plt.xticks(rotation=90, fontsize=1)
    # plt.show()
    # print(b)
    filtered_a = [i for i, count in zip(a, b) if count >= 5]
    print(len(filtered_a))
    
    clinic_df = clinic_df[clinic_df["patient_id"].isin(filtered_a)]
    print("Filtered DataFrame length:", len(clinic_df))
    print("Unique age groups and counts:", np.unique(clinic_df["age_group"], return_counts=True))
    
    print(len(valid_patches))
    valid_patches = [i for i in valid_patches if i.split("_")[0] in filtered_a]
    print(len(valid_patches))

    return clinic_df, valid_patches
    


def split_data_per_fold(clinic_df, patientID, truelabels, train_index, test_index, FEATURES, model_name, stainFunc):
    print('preparing train/test patients, 0.8/0.2')
    test_patients = patientID[test_index]
    test_data = clinic_df[clinic_df['patient_id'].isin(test_patients)]
    test_data.reset_index(inplace=True, drop=True)
    feaBag_pts = [glob.glob(f"{FEATURES}/{patient_id}*/*{model_name}*{stainFunc}.h5")[0] for patient_id in list(test_data['patient_id'])]
    test_data = test_data.copy()
    test_data['h5df'] = [Path(i) for i in feaBag_pts]
        
    print('preparing train/val patients, 0.8/0.2')
    train_patients = patientID[train_index]
    train_data = clinic_df[clinic_df['patient_id'].isin(train_patients)]
    train_data.reset_index(inplace=True, drop=True)
    
    val_data = train_data.groupby('age_group', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    feaBag_pts = [glob.glob(f"{FEATURES}/{patient_id}*/*{model_name}*{stainFunc}.h5")[0] for patient_id in list(val_data['patient_id'])]
    val_data['h5df'] = [Path(i) for i in feaBag_pts]
    val_data['is_valid'] = True
        
    train_data = train_data.copy()
    train_data = train_data.drop(train_data.loc[train_data['patient_id'].isin(list(val_data['patient_id']))].index).reset_index() # epithelium
    feaBag_pts = [glob.glob(f"{FEATURES}/{patient_id}*/*{model_name}*{stainFunc}.h5")[0] for patient_id in list(train_data['patient_id'])]
    train_data['h5df'] = [Path(i) for i in feaBag_pts]
    train_data['is_valid'] = False

    return train_data, val_data, test_data




# prepare input bag containing {bag_size} patches, with augmentation or not
class MILBagTransform(Transform):
    def __init__(self, h5df_files, bag_size=250, valid_patches=None):
        self.h5df_files = h5df_files
        self.bag_size = bag_size
        self.valid_patches = valid_patches
        self.items = {f: self._draw(f) for f in h5df_files}

    def encodes(self, f: Path):
        """Returns the bag of patches for a given file path."""
        return self.items.get(f, self._draw(f))

    def _sample_with_distinct_or_duplicate(self, population, k):
        """Sample k items from population, ensuring diversity in selection."""
        if len(population) >= k:
            return random.sample(population, k)  # Sample without replacement
        else:
            sampled_items = random.sample(population, len(population))  # Sample all unique items
            remaining_samples = random.choices(population, k=k - len(sampled_items))  # Fill with duplicates
            return sampled_items + remaining_samples

    def _draw(self, f: Path):
        """Draw a bag of patches from a given WSI."""
        # Handle augmentation if required
        if "augmentation" in f.name:
            chance = random.random() > 0.8
            if chance:
                f = f.replace("augmentation", "raw")

        with h5py.File(f, "r") as file:
            # Load embeddings and patch ids
            bag = np.array(file["embeddings"])
            img_id = np.array(file["patch_id"])

        # Decode patch ids if they are in bytes format
        img_id = [i.decode("utf-8") if isinstance(i, bytes) else i for i in img_id]
        
        bag_df = pd.DataFrame(bag)
        bag_df.index = img_id

        if self.valid_patches is not None:
            patch_ids = np.intersect1d(self.valid_patches, list(bag_df.index))
        else:
            patch_ids = list(bag_df.index)

        patch_ids = list(np.unique(patch_ids))
        sampled_items = self._sample_with_distinct_or_duplicate(patch_ids, self.bag_size)

        bag_df = bag_df.loc[sampled_items, :]
        bag_df = np.squeeze(np.array(bag_df))
        bag_df = torch.from_numpy(bag_df)
        
        return bag_df



class MILBagTransform(Transform):
    def __init__(self, h5df_files, bag_size=250, valid_patches=None):
        self.h5df_files = h5df_files
        self.bag_size = bag_size
        self.valid_patches = valid_patches
        self.items = {f: self._draw(f) for f in h5df_files}
        
    def encodes(self, f: Path):
        patch_ids, bag_df = self.items.get(f, self._draw(f))
        return patch_ids, bag_df  # Return both patch_ids and embeddings (bag_df)
        
    def _sample_with_distinct_or_duplicate(self, population, k):
        if len(population) >= k:
            return random.sample(population, k)  # Sample without replacement
        else:
            sampled_items = random.sample(population, len(population))
            remaining_samples = random.choices(population, k=k-len(sampled_items))
            return sampled_items + remaining_samples
            
    def _draw(self, f: Path):
        with h5py.File(f, "r") as file:
            bag = np.array(file["embeddings"])
            img_id = np.array(file["patch_id"])
        img_id = [i.decode("utf-8") if isinstance(i, bytes) else i for i in img_id]
        bag_df = pd.DataFrame(bag)
        bag_df.index = img_id
        
        if self.valid_patches is not None:
            patch_ids = np.intersect1d(self.valid_patches, list(bag_df.index))
        else:
            patch_ids = list(bag_df.index)
            
        patch_ids = list(np.unique(patch_ids))
        sampled_items = self._sample_with_distinct_or_duplicate(patch_ids, self.bag_size)
        bag_df = bag_df.loc[sampled_items, :]
        bag_df = np.squeeze(np.array(bag_df))
        bag_df = torch.from_numpy(bag_df)
        
        return sampled_items, bag_df
        


def Attention(n_in, n_latent: Optional[int] = None) -> nn.Module:
    n_latent = n_latent or (n_in +1) // 2
    return nn.Sequential(
        nn.Linear(n_in, n_latent),
        nn.Tanh(),
        nn.Linear(n_latent, 1)
    )



class GatedAttention(nn.Module):
    def __init__(self, n_in, n_latent: Optional[int] = None) -> None:
        super().__init__()
        n_latent = n_latent or (n_in + 1) // 2
        self.fc1 = nn.Linear(n_in, n_latent)
        self.gate = nn.Linear(n_in, n_latent)
        self.fc2 = nn.Linear(n_latent, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(h)) * torch.sigmoid(self.gate(h)))




class MultiHeadAttention(nn.Module):
    def __init__(self, n_in, n_heads, n_latent: Optional[int] = None, dropout: float = 0.125) -> None:
        super().__init__()
        self.n_heads = n_heads
        n_latent = n_latent or n_in
        self.d_k = n_latent // n_heads
        self.q_proj = nn.Linear(n_in, n_latent)
        self.k_proj = nn.Linear(n_in, n_latent)
        self.fc = nn.Linear(n_heads, 1)
        self.dropout = nn.Dropout(dropout)
        self.split_heads = lambda x: x.view(x.shape[0], x.shape[1], self.n_heads, self.d_k).transpose(1, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(x)  # Project Q
        K = self.k_proj(x)  # Project K
        # Q = torch.tanh(self.q_proj(x))  # Project Q
        # K = torch.sigmoid(self.k_proj(x))  # Project K
        Q = self.split_heads(Q)  # [batch_size, n_heads, bag_size, d_k]
        K = self.split_heads(K)  # [batch_size, n_heads, bag_size, d_k]
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # Scaled dot-product with temperature
        attention_output = attention_weights.mean(dim=-2)  # Aggregate across bag_size
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)  # Flatten heads
        attention_output = self.dropout(attention_output)  
        output = self.fc(attention_output)  
        return output



def get_input_dim(model_name):
    if model_name == 'UNI':
        input_dim = 1024
    elif model_name == 'iBOT':
        input_dim = 768
    elif model_name == 'gigapath':
        input_dim = 1536
    elif model_name == 'resnet50':
        input_dim = 2048
    return input_dim




class BreastAgeNet(nn.Module):
    def __init__(self, n_feats, attention, n_classes, n_heads=8, n_latent=512, attn_dropout=0.5, temperature=0.5, embed_attn=False):
        super(BreastAgeNet, self).__init__()
        self.n_classes = n_classes
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, n_latent),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        if attention == 'MultiHeadAttention':
            self.attentions = nn.ModuleList([MultiHeadAttention(n_latent, n_heads, dropout = attn_dropout) for _ in range(n_classes)])
        elif attention == 'Attention':
            self.attentions = nn.ModuleList([Attention(n_latent) for _ in range(n_classes)])
        elif attention == 'GatedAttention':
            self.attentions = nn.ModuleList([GatedAttention(n_latent) for _ in range(n_classes)])
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_latent, n_latent // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(n_latent // 2, 1)
            ) for _ in range(n_classes)
        ])
        self.embed_attn = embed_attn
        self.temperature = temperature  
    
    def forward(self, bags):
        bags = bags.to(next(self.parameters()).device)  
        embeddings = self.encoder(bags) 
        logits = []
        attns = []
        for i in range(self.n_classes):
            attention_scores = self.attentions[i](embeddings) 
            attention_scores = attention_scores / self.temperature
            attns.append(attention_scores)
            
            A = F.softmax(attention_scores, dim=1)            
            M = torch.bmm(A.transpose(1, 2), embeddings)  
            M = M.squeeze(1) 
            logit = self.classifiers[i](M)  # [batch_size, 1]
            logits.append(logit)
        
        logits = torch.cat(logits, dim=1)  # [batch_size, n_classes]
        attns = torch.cat(attns, dim=2)  # [batch_size, bag_size, n_classes]

        if self.embed_attn:
            return logits, embeddings, attns  # logits: [batch_size, n_classes], embeddings: [batch_size, bag_size, n_latent], attns: [batch_size, bag_size, n_classes]
        else:
            return logits  
            


def get_sample_weights(train_df):
    class_counts = train_df["age_group"].value_counts().sort_index()
    total = len(train_df)
    counts = [class_counts.get(i, 0) for i in range(len(class_counts))]
    
    a, b, c, d = counts if len(counts) == 4 else [0]*4
    return {
        "0": a / total,
        "1": b / total,
        "2": c / total,
        "3": d / total
    }



def get_sample_weights(train_df):
    class_counts = train_df["age_group"].value_counts().sort_index()
    counts = [class_counts.get(i, 0) for i in range(4)]
    counts = [max(c, 1) for c in counts]  # Ensure no division by zero
    inverse_freqs = [1.0 / c for c in counts]
    total_weight = sum(inverse_freqs)
    normalized_weights = [w / total_weight for w in inverse_freqs]
    return [normalized_weights[i] for i in range(4)]



class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes, weights=None):
        super().__init__()
        self.num_classes = n_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.weights = weights
        self.ordinals = torch.arange(n_classes)
        
    def forward(self, logits, labels):
        device = logits.device
        labels_expanded = labels.unsqueeze(1).expand(-1, self.num_classes)
        ordinal_labels = (labels_expanded > self.ordinals.to(device)).float()
        loss = self.bce(logits, ordinal_labels)
        if self.weights is not None:
            sample_weights = torch.tensor([self.weights[l.item()] for l in labels], device=device).unsqueeze(1)
            loss = loss * sample_weights
        return loss.mean()

        

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta  # Threshold for improvement
        self.counter = 0  # To count the number of epochs without improvement
        self.best_score = None  # Best validation loss seen so far
        self.early_stop = False  # Flag to indicate whether training should stop

    def __call__(self, epoch, val_loss, model, ckpt_name):
        if self.best_score is None:
            # Initialize best_score with the first validation loss
            self.best_score = val_loss
            torch.save(model.state_dict(), ckpt_name)
        elif val_loss < self.best_score - self.delta:
            # If the current loss is better than the previous best by at least delta
            self.best_score = val_loss
            torch.save(model.state_dict(), ckpt_name)  # Save the best model
            self.counter = 0  # Reset the counter if there's improvement
        else:
            self.counter += 1  # Increment the counter if no improvement
            if self.counter >= self.patience:
                # If we have not seen improvement for `patience` epochs, stop training
                self.early_stop = True



def run_one_step(model, inputs, labels, criterion, optimizer, phase='train'):
    optimizer.zero_grad()  
    logits = model(inputs) 
    loss = criterion(logits, labels)  
    
    if phase == 'train':
        loss.backward()  
        optimizer.step()  
    return logits, loss



def compute_mae(logits, labels):    
    probs = torch.sigmoid(logits) 
    binary_predictions = (probs > 0.5).int() 
    predicted_ranks = binary_predictions.sum(dim=1) 
    return torch.abs(labels - predicted_ranks).float().mean()



def run_one_epoch(model, trainLoaders, valLoaders, criterion, optimizer, train_loss_history, train_acc_history, val_loss_history, val_acc_history, early_stopping, epoch, ckpt_name):
    # Training phase
    print('Training phase\n')
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_mae = 0.0

    for (_, inputs), labels in tqdm(trainLoaders): 
        logits, loss = run_one_step(model, inputs, labels, criterion, optimizer, phase='train')
        mae = compute_mae(logits, labels)  # Calculate Mean Absolute Error
        running_loss += loss.item()
        running_mae += mae.item()

    epoch_loss = running_loss / len(trainLoaders)
    train_loss_history.append(epoch_loss)
    epoch_acc = running_mae / len(trainLoaders)
    train_acc_history.append(epoch_acc)

    print(f'\nTraining Loss: {np.round(epoch_loss, 4)}  MAE: {np.round(epoch_acc, 4)}')

    # Validation phase
    if valLoaders:
        print('Validation phase\n')
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        running_mae = 0.0

        with torch.no_grad():  # Disable gradient calculation for validation
            for (_, inputs), labels in tqdm(valLoaders):
                logits, loss = run_one_step(model, inputs, labels, criterion, optimizer, phase='val')
                mae = compute_mae(logits, labels)
                running_loss += loss.item()
                running_mae += mae.item()

        val_loss = running_loss / len(valLoaders)
        val_loss_history.append(val_loss)
        val_acc = running_mae / len(valLoaders)
        val_acc_history.append(val_acc)

        print(f'Validation Loss: {np.round(val_loss, 4)}   MAE: {np.round(val_acc, 4)}')

        # Check for early stopping
        early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
        if early_stopping.early_stop:
            print(f'Early stopping: validation loss did not improve for {early_stopping.patience} epochs!')
            return True  # Exit early if stopping condition is met

    return False  # Continue if early stopping was not triggered





def plot_training(train_loss_history, val_loss_history, train_acc_history, val_acc_history, save_pt=None):
    epochs = range(1, len(train_loss_history) + 1) 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(epochs, train_loss_history, label='Training Loss', color='blue')
    ax1.plot(epochs, val_loss_history, label='Validation Loss', color='orange')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()
    
    ax2.plot(epochs, train_acc_history, label='Training Accuracy', color='green')
    ax2.plot(epochs, val_acc_history, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    if save_pt is not None:
        plt.savefig(save_pt)
    plt.show()


    
    
def test_model(model, dataloaders):
    phase = 'test'
    model.eval()

    predicted_ranks = []
    for item in tqdm(dataloaders):
        _, inputs = item[0]
        with torch.set_grad_enabled(phase=='train'):
            logits = model(inputs)
            probs = torch.sigmoid(logits)  # Shape: [batch_size, n_classes]
            binary_predictions = (probs > 0.5).int()  # Shape: [batch_size, n_classes]
            ranks = binary_predictions.sum(dim=1)  # Shape: [batch_size]
            predicted_ranks = predicted_ranks + ranks.tolist()
    
    return predicted_ranks





def train_CV(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    clinic_df, valid_patches = get_data(config)
    
    patientID = clinic_df["patient_id"].values
    truelabels = clinic_df["age_group"].values
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    kf.get_n_splits(patientID, truelabels)
    
    for foldcounter, (train_index, test_index) in enumerate(kf.split(patientID, truelabels)):
        ckpt_name = f'{config["resFolder"]}/BreastAgeNet_epi{config["TC_epi"]}_{config["model_name"]}_{config["bag_size"]}_{config["attention"]}_fold{foldcounter}_best.pt'
        train_df, val_df, test_df = split_data_per_fold(clinic_df, patientID, truelabels, train_index, test_index, config["FEATURES"], config["model_name"], config["stainFunc"])
        
        dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                           get_x = ColReader('h5df'),
                           get_y = ColReader('age_group'),
                           splitter = ColSplitter('is_valid'),
                           item_tfms = MILBagTransform(train_df.h5df, config["bag_size"], valid_patches=valid_patches))
    
        dls = dblock.dataloaders(train_df, bs = config["batch_size"])
        trainLoaders = dls.train
        valLoaders = dls.valid
        
        (patch_ids, embeds), labels = next(iter(trainLoaders))
        patch_ids = np.array(patch_ids)  
        patch_ids = np.transpose(patch_ids)  
        print(patch_ids.shape, embeds.shape, labels.shape)
    
        input_dim = get_input_dim(config["model_name"])
        model = BreastAgeNet(input_dim, config["attention"], config["n_classes"], config["n_heads"], config["n_latent"], config["attn_dropout"], config["attn_temp"], embed_attn=False).to(device)
        criterion = OrdinalCrossEntropyLoss(n_classes=config["n_classes"], weights=get_sample_weights(train_df))
        criterion.to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
        early_stopping = EarlyStopping(patience=config["patience"], stop_epoch=config["max_epochs"], verbose=True)
    
        since = time.time()
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        
        for epoch in range(config["max_epochs"]):
            print(f'Epoch {epoch}/{config["max_epochs"]-1}\n')
            early_stop = run_one_epoch(
                model, trainLoaders, valLoaders, criterion, optimizer, 
                train_loss_history, train_acc_history, val_loss_history, 
                val_acc_history, early_stopping, epoch, ckpt_name
            )
            if early_stop:
                break
    
        time_elapsed = time.time() - since
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
        plot_training(train_loss_history, val_loss_history, train_acc_history, val_acc_history, save_pt=ckpt_name.replace(".pt", ".png"))
        
        testLoader = dls.test_dl(test_df)
        predicted_ranks = test_model(model, testLoader)
        MAE = np.abs(test_df["age_group"].values - predicted_ranks).mean()
        print(f"Testing MAE is: {MAE}")




def train_full(config):
    ckpt_name = f'{config["resFolder"]}/BreastAgeNet_epi{config["TC_epi"]}_{config["model_name"]}_{config["bag_size"]}_{config["attention"]}_full_best.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    train_df, valid_patches = get_data(config)
    feaBag_pts = [glob.glob(f'{config["FEATURES"]}/{patient_id}*/*{config["model_name"]}*{config["stainFunc"]}.h5')[0] for patient_id in list(train_df['patient_id'])]
    train_df['h5df'] = [Path(i) for i in feaBag_pts]
    train_ids, valid_ids = train_test_split(train_df['patient_id'].unique(), test_size=0.1, random_state=42)
    train_df['is_valid'] = train_df['patient_id'].isin(valid_ids)
    dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                       get_x = ColReader('h5df'),
                       get_y = ColReader('age_group'),
                       splitter = ColSplitter('is_valid'),
                       item_tfms = MILBagTransform(train_df.h5df, config["bag_size"], valid_patches=valid_patches))
    dls = dblock.dataloaders(train_df, bs = config["batch_size"])
    trainLoaders = dls.train
    valLoaders = dls.valid

    (patch_ids, embeds), labels = next(iter(trainLoaders))
    patch_ids = np.array(patch_ids)  
    patch_ids = np.transpose(patch_ids)  
    print(patch_ids.shape, embeds.shape, labels.shape)

    input_dim = get_input_dim(config["model_name"])
    model = BreastAgeNet(input_dim, config["attention"], config["n_classes"], config["n_heads"], config["n_latent"], config["attn_dropout"], config["attn_temp"], embed_attn=False).to(device)
    criterion = OrdinalCrossEntropyLoss(n_classes=config["n_classes"], weights=get_sample_weights(train_df))
    criterion.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
    early_stopping = EarlyStopping(patience=config["patience"])

    since = time.time()
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(config["max_epochs"]):
        print(f'Epoch {epoch}/{config["max_epochs"]-1}\n')
        early_stop = run_one_epoch(
            model, trainLoaders, valLoaders, criterion, optimizer, 
            train_loss_history, train_acc_history, val_loss_history, 
            val_acc_history, early_stopping, epoch, ckpt_name
        )
        if early_stop:
            break

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    plot_training(train_loss_history, val_loss_history, train_acc_history, val_acc_history, save_pt=ckpt_name.replace(".pt", ".png"))




def get_averaged_outputs(outputs):
    # Convert 'branch_0', 'branch_1', 'branch_2' to numeric (if necessary)
    outputs['branch_0'] = pd.to_numeric(outputs['branch_0'], errors='coerce')
    outputs['branch_1'] = pd.to_numeric(outputs['branch_1'], errors='coerce')
    outputs['branch_2'] = pd.to_numeric(outputs['branch_2'], errors='coerce')
    
    # Group by 'wsi_id' and calculate the mean for each branch across repeats
    averaged_data = outputs.groupby('wsi_id')[['branch_0', 'branch_1', 'branch_2']].mean().reset_index()
    averaged_data = pd.merge(outputs.loc[:, ['wsi_id', 'patient_id','cohort', 'source', 'age', 'age_group']], averaged_data, on='wsi_id').drop_duplicates()
    
    # Compute the sigmoid function manually (equivalent to expit)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Apply sigmoid to the averaged values
    averaged_data['sigmoid_0'] = sigmoid(averaged_data['branch_0'])  # Sigmoid for branch 0
    averaged_data['sigmoid_1'] = sigmoid(averaged_data['branch_1'])  # Sigmoid for branch 1
    averaged_data['sigmoid_2'] = sigmoid(averaged_data['branch_2'])  # Sigmoid for branch 2
    
    # Convert sigmoid values to binary (0 or 1)
    averaged_data['binary_0'] = (averaged_data['sigmoid_0'] >= 0.5).astype(int)
    averaged_data['binary_1'] = (averaged_data['sigmoid_1'] >= 0.5).astype(int)
    averaged_data['binary_2'] = (averaged_data['sigmoid_2'] >= 0.5).astype(int)
    
    # Sum the binary predictions to get the final prediction
    averaged_data['final_prediction'] = averaged_data[['binary_0', 'binary_1', 'binary_2']].sum(axis=1)

    return averaged_data


