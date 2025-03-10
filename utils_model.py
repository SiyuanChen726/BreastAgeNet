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
from sklearn.metrics import confusion_matrix

import torch.optim as optim
from torch.nn import CrossEntropyLoss
from fastai.vision.all import *
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def add_ageGroup(df):
    if 'age_group' in df and not df['age_group'].isin([0, 1, 2, 3]).all():
        df['age_group'] = 3
        df.loc[df['age'] < 35, 'age_group'] = 0
        df.loc[(df['age'] >= 35) & (df['age'] < 45), 'age_group'] = 1
        df.loc[(df['age'] >= 45) & (df['age'] < 55), 'age_group'] = 2
    return df



def parse_wsi_id(patch_id):
    if "K" in patch_id: # SGK
        wsi_id = patch_id.split("_")[0]
    elif " HE" in patch_id: # NKI
        wsi_id = patch_id.split(" HE")[0]
    elif "_HE" in patch_id: # NKI
        wsi_id = patch_id.split("_HE")[0]
    elif "_FPE_" in patch_id: # KHP
        wsi_id = "_".join(patch_id.split("_")[:3])
    else: # BCI
        wsi_id = "_".join(patch_id.split("_")[:-7])
    return wsi_id



def get_data(config): 
    clinic_df = pd.read_csv(config["clinic_path"])
    clinic_df = add_ageGroup(clinic_df)
    h5_dict = {"wsi_id": [], "h5df": []} 
    for wsi_id in clinic_df["wsi_id"]:
        file = glob.glob(f'{config["FEATURES"]}/{wsi_id}*/*{config["model_name"]}*{config["stainFunc"]}.h5')
        if not file:
            file = glob.glob(f'{config["FEATURES"]}/*/{wsi_id}*/*{config["model_name"]}*{config["stainFunc"]}.h5')
        if file:
            h5_dict["wsi_id"].append(wsi_id)
            h5_dict["h5df"].append(file[0])  

    h5_df = pd.DataFrame(h5_dict)  
    clinic_df = clinic_df.merge(h5_df, on="wsi_id", how="right")  
    print("Filtered DataFrame length:", len(clinic_df))
    print("Unique age groups and counts:", np.unique(clinic_df["age_group"], return_counts=True))

    valid_wsi = []
    valid_patches = []
    for fea_pt in clinic_df["h5df"]: 
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
            wsi_id = parse_wsi_id(valid_id[0])
            valid_wsi.extend([wsi_id] * len(valid_id))
    
    print(len(np.unique(valid_wsi)))
    print(len(valid_patches))
    a, b = np.unique(valid_wsi, return_counts=True)

    filtered_a = [i for i, count in zip(a, b) if count >= 5]
    print(len(filtered_a))

    clinic_df = clinic_df[clinic_df["wsi_id"].isin(filtered_a)]
    clinic_df["h5df"] = [Path(i) for i in list(clinic_df["h5df"])]
    print("Filtered DataFrame length:", len(clinic_df))
    print("Unique age groups and counts:", np.unique(clinic_df["age_group"], return_counts=True))

    print(len(valid_patches))
    valid_patches = [i for i in valid_patches if parse_wsi_id(i) in filtered_a]
    print(len(valid_patches))

    return clinic_df, valid_patches




def split_data_per_fold(clinic_df, patientID, truelabels, train_index, test_index, stainFunc):
    print('Preparing train/test patients, 80/20 split')
    test_patients = patientID[test_index]
    test_data = clinic_df[clinic_df['patient_id'].isin(test_patients)].copy()
    test_data.reset_index(inplace=True, drop=True)
    
    print('Preparing train/val patients, 80/20 split within training data')
    train_patients = patientID[train_index]
    train_data = clinic_df[clinic_df['patient_id'].isin(train_patients)].copy()
    train_data.reset_index(inplace=True, drop=True)
    
    val_data = train_data.groupby('age_group', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    val_data['is_valid'] = True

    train_data = train_data.loc[~train_data['patient_id'].isin(val_data['patient_id'])].copy()
    train_data.reset_index(inplace=True, drop=True)
    train_data['is_valid'] = False
    
    if stainFunc == "augmentation":
        val_data['h5df'] = val_data['h5df'].apply(lambda x: Path(str(x).replace('augmentation', 'reinhard')))
        test_data['h5df'] = test_data['h5df'].apply(lambda x: Path(str(x).replace('augmentation', 'reinhard')))

    print("Age Group Distribution in Training Set:")
    train_age_group_counts = train_data['age_group'].value_counts()
    print(train_age_group_counts)
    
    print("\nAge Group Distribution in Validation Set:")
    valid_age_group_counts = val_data['age_group'].value_counts()
    print(valid_age_group_counts)

    print("\nAge Group Distribution in Test Set:")
    test_age_group_counts = test_data['age_group'].value_counts()
    print(test_age_group_counts)

    return train_data, val_data, test_data




class MILBagTransform(Transform):
    def __init__(self, h5df_files, bag_size=250, aug_prob=0.5, valid_patches=None):
        self.h5df_files = h5df_files
        self.bag_size = bag_size
        self.aug_prob = aug_prob
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
        if "augmentation" in str(f):  
            chance = random.random() < self.aug_prob
            if chance:
                f = str(f).replace("augmentation", "reinhard")  # Now replacing in the string
                f = Path(f)  # Convert it back to a Path object if needed

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

        Q = self.split_heads(Q)  # [batch_size, n_heads, bag_size, d_k]
        K = self.split_heads(K)  # [batch_size, n_heads, bag_size, d_k]
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # Scaled dot-product with temperature
        attention_output = attention_weights.mean(dim=-2)  # Aggregate across bag_size
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)  # Flatten heads
        attention_output = self.dropout(attention_output)  
        output = self.fc(attention_output)  
        return output




class MultiHeadAttention_v2(nn.Module):
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
        Q = torch.tanh(self.q_proj(x))  # Project Q
        K = torch.sigmoid(self.k_proj(x))  # Project K
        
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
        elif attention == 'MultiHeadAttention_v2':
            self.attentions = nn.ModuleList([MultiHeadAttention_v2(n_latent, n_heads, dropout = attn_dropout) for _ in range(n_classes)])
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
    counts = [max(c, 1) for c in counts]  
    inverse_freqs = [1.0 / c for c in counts]
    total_weight = sum(inverse_freqs)
    normalized_weights = [w / total_weight for w in inverse_freqs]
    return [normalized_weights[i] for i in range(4)]




# class OrdinalCrossEntropyLoss(nn.Module):
#     def __init__(self, n_classes, weights=None):
#         super().__init__()
#         self.num_classes = n_classes
#         self.bce = nn.BCEWithLogitsLoss(reduction="none")
#         self.weights = weights
#         self.ordinals = torch.arange(n_classes)
        
#     def forward(self, logits, labels):
#         device = logits.device
#         labels_expanded = labels.unsqueeze(1).expand(-1, self.num_classes)
#         ordinal_labels = (labels_expanded > self.ordinals.to(device)).float()
        
#         loss = self.bce(logits, ordinal_labels)
#         if self.weights is not None:
#             sample_weights = torch.tensor([self.weights[l.item()] for l in labels], device=device).unsqueeze(1)
#             loss = loss * sample_weights
#         return loss.mean()



class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes, weights=None):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = n_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to handle weights manually
        self.weights = weights

    def forward(self, logits, labels):
        labels_expanded = labels.unsqueeze(1).repeat(1, self.num_classes)  # [batch_size, n_classes]
        binary_labels = (labels_expanded > torch.arange(self.num_classes).to(labels.device)).float()
        loss = self.bce(logits, binary_labels)

        if self.weights is not None:
            loss = loss * self.weights
            return loss.mean()   
        else:
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
    
    ax2.plot(epochs, train_acc_history, label='Training MAE', color='green')
    ax2.plot(epochs, val_acc_history, label='Validation MAE', color='red')
    ax2.set_title('Training and Validation MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
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
        
        train_df, val_df, test_df = split_data_per_fold(clinic_df, patientID, truelabels, train_index, test_index, config["stainFunc"])
        dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                           get_x = ColReader('h5df'),
                           get_y = ColReader('age_group'),
                           splitter = ColSplitter('is_valid'),
                           item_tfms = MILBagTransform(train_df.h5df, config["bag_size"], config["aug_prob"], valid_patches=valid_patches))
    
        dls = dblock.dataloaders(train_df, bs = config["batch_size"])
        trainLoaders = dls.train
        valLoaders = dls.valid
        (patch_ids, embeds), labels = next(iter(trainLoaders))
        patch_ids = np.array(patch_ids)  
        patch_ids = np.transpose(patch_ids)  
        print(patch_ids.shape, embeds.shape, labels.shape)

        input_dim = get_input_dim(config["model_name"])
        model = BreastAgeNet(input_dim, config["attention"], config["n_classes"], config["n_heads"], config["n_latent"], config["attn_dropout"], config["attn_temp"], embed_attn=False).to(device)
        criterion = OrdinalCrossEntropyLoss(n_classes=config["n_classes"], weights=None)
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
        plot_training(train_loss_history, val_loss_history, train_acc_history, val_acc_history, save_pt=ckpt_name.replace(".pt", ".png"))
        time_elapsed = time.time() - since
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        testLoader = dls.test_dl(test_df)
        predicted_ranks = test_model(model, testLoader)
        MAE = np.abs(test_df["age_group"].values - predicted_ranks).mean()
        results_file = f'{config["resFolder"]}/BreastAgeNet_cv_best_MAE.txt'
        with open(results_file, "a") as file:
            file.write("Configuration:\n")
            json.dump(config, file, indent=4)  # Write the dictionary as formatted JSON
            file.write("\n")
            file.write(f"Testing MAE: {MAE}\n")
            
        print(f"Testing MAE is: {MAE}")



def train_full(config):
    ckpt_name = f'{config["resFolder"]}/BreastAgeNet_full_best_{config["name"]}.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    train_df, valid_patches = get_data(config)
    train_ids, valid_ids = train_test_split(
        train_df['patient_id'].unique(), 
        test_size=0.1, 
        random_state=42, 
        stratify=train_df['age_group']  # Stratifying by 'age_group'
    )
    train_df['is_valid'] = train_df['patient_id'].isin(valid_ids)
    if config["stainFunc"] == "augmentation":
        train_df.loc[train_df['is_valid'], 'h5df'] = (
            train_df.loc[train_df['is_valid'], 'h5df']
            .apply(lambda x: Path(str(x).replace('augmentation', 'reinhard')))
        )
    train_age_group_counts = train_df[~train_df['is_valid']]['age_group'].value_counts()
    valid_age_group_counts = train_df[train_df['is_valid']]['age_group'].value_counts()
    print("Age Group Distribution in Training Set:")
    print(train_age_group_counts)
    print("\nAge Group Distribution in Validation Set:")
    print(valid_age_group_counts)

    
    dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                       get_x = ColReader('h5df'),
                       get_y = ColReader('age_group'),
                       splitter = ColSplitter('is_valid'),
                       item_tfms = MILBagTransform(train_df.h5df, config["bag_size"], config["aug_prob"], valid_patches=valid_patches))
    dls = dblock.dataloaders(train_df, bs = config["batch_size"])
    trainLoaders = dls.train
    valLoaders = dls.valid
    (patch_ids, embeds), labels = next(iter(trainLoaders))
    patch_ids = np.array(patch_ids)  
    patch_ids = np.transpose(patch_ids)  
    print(patch_ids.shape, embeds.shape, labels.shape)


    input_dim = get_input_dim(config["model_name"])
    model = BreastAgeNet(input_dim, config["attention"], config["n_classes"], config["n_heads"], config["n_latent"], config["attn_dropout"], config["attn_temp"], embed_attn=False).to(device)
    # criterion = OrdinalCrossEntropyLoss(n_classes=config["n_classes"], weights=get_sample_weights(train_df))
    criterion = OrdinalCrossEntropyLoss(n_classes=config["n_classes"], weights=None)
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
    

    input_dim = get_input_dim(config["model_name"])
    model = BreastAgeNet(input_dim, config["attention"], config["n_classes"], config["n_heads"], 
                         config["n_latent"], config["attn_dropout"], config["attn_temp"], embed_attn=True)
    model.load_state_dict(torch.load(ckpt_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    
    clinic_path = "/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/temp/clinic_df_KHP_RM_reinhard.csv"
    n_iteration=10
    stainFunc="reinhard"
    repeats = inference(model, clinic_path, n_iteration, stainFunc, config)
    output_df = get_averaged_outputs(repeats)
    output_df = add_ageGroup(output_df)
    y_true = output_df["age_group"].values
    y_pred = output_df["final_prediction"].values
    cm = confusion_matrix(y_true, y_pred)
    cm



def inference(model, clinic_path, n_iteration, stainFunc, config):
    config["clinic_path"] = clinic_path
    config["stainFunc"] = stainFunc
    infer_df, valid_patches = get_data(config)
    infer_df['h5df'] = [Path(i) for i in infer_df["h5df"]]
    test_dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                           get_x = ColReader('h5df'),
                           get_y = ColReader('age_group'),
                           item_tfms = MILBagTransform(infer_df.h5df, config["bag_size"], config["aug_prob"], valid_patches=valid_patches))
    test_dls = test_dblock.dataloaders(infer_df, bs=4, shuffle=False)
    dataloaders = test_dls.test_dl(infer_df, with_labels=True)
    phase = 'test'
    model.eval()
    repeats = pd.DataFrame()  # Initialize empty DataFrame
    for _ in range(n_iteration):
        for (patch_ids, inputs), labels in tqdm(dataloaders):
            patch_ids = np.array(patch_ids)  # (150, 7)
            patch_ids = np.transpose(patch_ids)  # Now shape is (7, 150)
            patch_ids = patch_ids.flatten()  # Flattened to shape (1050,)
            with torch.set_grad_enabled(phase == 'train'):
                logits, embeddings, attentions = model(inputs)
                attentions = attentions.view(-1, attentions.shape[-1])  # Flatten attentions: [X, 3]
                embeddings = embeddings.view(-1, embeddings.shape[-1])  # Flatten embeddings: [X, 512]
                batch_size = inputs.size(0)
                bag_size = attentions.size(0) // batch_size  # Calculate bag_size
                logits = logits.repeat(bag_size, 1)  # Repeat logits across the bag size: [X, 3]
            combined_data = np.column_stack((patch_ids, embeddings.cpu().numpy(), logits.cpu().numpy(), attentions.cpu().numpy()))  # Convert to numpy
            dfi = pd.DataFrame(combined_data, columns=['patch_id'] + [f'embedding_{i}' for i in range(embeddings.shape[1])] + [f'branch_{i}' for i in range(logits.shape[1])] + [f'attention_{i}' for i in range(attentions.shape[1])])
            repeats = pd.concat([repeats, dfi], axis=0)  # Append new data to the main dataframe
    repeats = repeats.drop_duplicates().copy()
    repeats["wsi_id"] = [parse_wsi_id(i) for i in list(repeats["patch_id"])]
    repeats = pd.merge(repeats, infer_df, on="wsi_id")
    return repeats



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



def load_BreastAgeNet(config):
    ckpt_name = f'{config["resFolder"]}/BreastAgeNet_full_best_{config["name"]}.pt'
    print(ckpt_name)
    
    input_dim = get_input_dim(config["model_name"])
    model = BreastAgeNet(input_dim, config["attention"], config["n_classes"], config["n_heads"], config["n_latent"], config["attn_dropout"], config["attn_temp"], embed_attn=True)
    model.load_state_dict(torch.load(ckpt_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    return model
