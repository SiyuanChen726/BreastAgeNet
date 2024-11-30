'''
modified from 
https://github.com/KatherLab/HIA/blob/main/models/model_Attmil.py
https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
'''

import os
from pathlib import Path
from typing import Tuple, Any, Optional
import h5py
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch.optim as optim
from torch.nn import CrossEntropyLoss
from fastai.vision.all import *
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "CPU")




def add_ageGroup(df):
    """
    Adds an 'age_group' column to the DataFrame based on the 'age' column.

    Age groups are assigned as follows:
    - 0: Age < 35
    - 1: Age 35-45
    - 2: Age 45-55
    - 3: Age >= 55

    Parameters:
    - df (pandas.DataFrame): DataFrame containing an 'age' column.

    Returns:
    - pandas.DataFrame: Original DataFrame with an added 'age_group' column.
    """
    # Assign default group for age >= 55
    df['age_group'] = 3
    
    # Apply age group categorization with non-overlapping conditions
    df.loc[df['age'] < 35, 'age_group'] = 0
    df.loc[(df['age'] >= 35) & (df['age'] < 45), 'age_group'] = 1
    df.loc[(df['age'] >= 45) & (df['age'] < 55), 'age_group'] = 2

    return df






# find .h5 files containing features extracted by pre-trained image encoders 
def get_h5df(wsi_id, rootdir, model_name, train):
    if train:
        h5df = glob.glob(f"{rootdir}/{wsi_id}*/*{model_name}*augmentation.h5")[0]
    else:
        h5df = glob.glob(f"{rootdir}/{wsi_id}*/*{model_name}*raw.h5")[0]
    return h5df

       



def feaBag_pt(patient_id, rootdir, model_name, train):
    if train:
        # feature_list = glob.glob(f"{rootdir}/{patient_id}*/*{model_name}*reinhard.h5")
        feature_list = glob.glob(f"{rootdir}/{patient_id}*/*{model_name}*augmentation.h5")
    else:
        feature_list = glob.glob(f"{rootdir}/{patient_id}*/*{model_name}*raw.h5")
    index = random.randint(0, len(feature_list)-1)
    bag_pt = feature_list[index]
    return bag_pt




def split_data(clinic_df, rootdir, patientID, yTrue, train_index, test_index, fold_pt):
    model_name = os.path.basename(fold_pt).split("_")[0]
    
    print('preparing train/test patients, 0.8/0.2')
    # test data
    test_patients = patientID[test_index]
    test_data = clinic_df[clinic_df['patient_id'].isin(test_patients)]
    test_data.reset_index(inplace=True, drop=True)
    test_data.to_csv(fold_pt.replace('train', 'test'), index = False)
    print(f"{fold_pt.replace('train', 'test')} saved!")
    
    feaBag_pts = [feaBag_pt(patient_id, rootdir, model_name, train=False) for patient_id in list(test_data['patient_id'])]
    test_data = test_data.copy()
    test_data['feaBag_pt'] = [Path(i) for i in feaBag_pts]
    
    print('preparing train/val patients, 0.8/0.2')
    train_patients = patientID[train_index]
    train_data = clinic_df[clinic_df['patient_id'].isin(train_patients)]
    train_data.reset_index(inplace=True, drop=True)
    
    val_data = train_data.groupby('age_group', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    feaBag_pts = [feaBag_pt(patient_id, rootdir, model_name, train=False) for patient_id in list(val_data['patient_id'])]
    val_data['feaBag_pt'] = [Path(i) for i in feaBag_pts]
    val_data['is_valid'] = True
    
    train_data = train_data.copy()
    train_data = train_data.drop(train_data.loc[train_data['patient_id'].isin(list(val_data['patient_id']))].index).reset_index() # epithelium
    feaBag_pts = [feaBag_pt(patient_id, rootdir, model_name, train=True) for patient_id in list(train_data['patient_id'])]
    train_data['feaBag_pt'] = [Path(i) for i in feaBag_pts]
    train_data['is_valid'] = False
    
    train_data = pd.concat([train_data, val_data], axis=0)
    train_data.to_csv(fold_pt, index = False)
    print(f'{fold_pt} saved!')
 
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



######### image encoders
def get_dim_input(model_name):
    if model_name == 'UNI':
        dim_input = 1024
    elif model_name == 'iBOT':
        dim_input = 768
    elif model_name == 'gigapath':
        dim_input = 1536
    elif model_name == 'resnet50':
        dim_input = 2048
    return dim_input



######### attention modules
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
    def __init__(self, n_in, n_heads, n_latent: Optional[int] = None, dropout: float = 0.1) -> None:
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

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # Scaled dot-product
        attention_weights = F.softmax(attention_scores, dim=-1)  # Softmax to get weights
        attention_weights = self.dropout(attention_weights)  # Apply dropout for regularization
        
        # Aggregate attention weights
        attention_output = attention_weights.mean(dim=-2)  # Aggregate across bag_size
       
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)  # Flatten heads
      
        output = self.fc(attention_output)  # Final linear layer for output

        return output




class BreastAgeNet(nn.Module):
    def __init__(self, n_feats, attention, n_classes, n_heads=8, n_latent=512, embed_attn=False, temperature=None):
        super(BreastAgeNet, self).__init__()
        self.n_classes = n_classes
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, n_latent),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Choose the attention mechanism for each class
        if attention == 'MultiHeadAttention':
            self.attentions = nn.ModuleList([MultiHeadAttention(n_latent, n_heads) for _ in range(n_classes)])
        elif attention == 'Attention':
            self.attentions = nn.ModuleList([Attention(n_latent) for _ in range(n_classes)])
        elif attention == 'GatedAttention':
            self.attentions = nn.ModuleList([GatedAttention(n_latent) for _ in range(n_classes)])
        
        # Bag classifiers for each class
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
        bags = bags.to(next(self.parameters()).device)  # Ensure bags are on the correct device
        embeddings = self.encoder(bags)  # [batch_size, bag_size, n_latent]

        logits = []
        attns = []
        for i in range(self.n_classes):
            # Get attention scores for the current class
            attention_scores = self.attentions[i](embeddings)  # [batch_size, bag_size, 1]
            
            if self.temperature is not None:  # Adjust based on experimentation
                attention_scores = attention_scores / temperature

            attns.append(attention_scores)
            A = F.softmax(attention_scores, dim=1)            
            # Apply weighted sum for each instance in the bag
            M = torch.bmm(A.transpose(1, 2), embeddings)  # [batch_size, 1, n_latent]
            M = M.squeeze(1)  # Ensure we only remove the bag_size dimension
            
            # Pass through classifier to get logits for this class
            logit = self.classifiers[i](M)  # [batch_size, 1]
            logits.append(logit)
        
        # Stack logits along a new dimension for consistent output shape
        logits = torch.cat(logits, dim=1)  # [batch_size, n_classes]

        # Concatenate attention scores across classes along dim=2
        attns = torch.cat(attns, dim=2)  # [batch_size, bag_size, n_classes]

        # Return embeddings and attention weights if requested
        if self.embed_attn:
            return logits, embeddings, attns  # logits: [batch_size, n_classes], embeddings: [batch_size, bag_size, n_latent], attns: [batch_size, bag_size, n_classes]
        else:
            return logits  # [batch_size, n_classes]


            

#### Loss function
class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes, weights=None):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = n_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to handle weights manually
        self.weights = weights

    def forward(self, logits, labels):
        # Expand labels to create binary labels for each class threshold
        labels_expanded = labels.unsqueeze(1).repeat(1, self.num_classes)  # [batch_size, n_classes]
        
        # Create binary labels: [0,0,0], [1,0,0], [1,1,0], [1,1,1], etc.
        binary_labels = (labels_expanded > torch.arange(self.num_classes).to(labels.device)).float()
        
        # Compute the BCE loss for each class (ignoring the reduction)
        loss = self.bce(logits, binary_labels)

        # If weights are provided, apply them (per sample, per class)
        if self.weights is not None:
            loss = loss * self.weights
            return loss.mean()   # Apply weights based on true class
        else:
            # Return the average loss
            return loss.mean()  # Averaging over the batch




class EarlyStopping:
    def __init__(self, patience = 20, stop_epoch = 50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss    



        

#### train on one step, validate on one epoch deciding whether early stopping
def train_model(model, trainLoaders, valLoaders, optimizer, criterion,
                patience, minEpochTrain, max_epochs, ckpt_name):
    since = time.time()
    
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    early_stopping = EarlyStopping(patience=patience, stop_epoch=minEpochTrain, verbose=True)

    for epoch in range(max_epochs):
        print(f"{epoch}/{max_epochs-1} epoch")
        print('\nTraining\n')
        phase = 'train'
        model.train()
        run_loss = 0.0
        run_MAE = 0.0
        
        for inputs,labels in tqdm(trainLoaders): # inputs = (emeddings, patch lens)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = model(inputs)
                loss = criterion(logits, labels)  # BCEWithLogitsLoss expects logits as inputs
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                run_loss += loss.sum().item()  # Sum across the batch and accumulate
                # Compute MAE
                probs = torch.sigmoid(logits)  # Shape: [batch_size, n_classes]
                binary_predictions = (probs > 0.5).int()  # Shape: [batch_size, n_classes]
                predicted_ranks = binary_predictions.sum(dim=1)  # Shape: [batch_size]
                true_ranks = labels.sum(dim=1)
                run_MAE += torch.abs(true_ranks - predicted_ranks).float().mean()
      
        epoch_loss = run_loss / len(trainLoaders.dataset)
        epoch_acc = run_MAE.item() / len(trainLoaders.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        print(f'\n {phase} Loss: {np.round(epoch_loss, 4)}  MAE: {np.round(epoch_acc, 4)}')

        if valLoaders:
            print('Validation....\n')
            phase = 'val'
            model.eval()
            
            run_loss = 0.0
            run_MAE = 0.0
            for inputs, labels in tqdm(valLoaders):
                with torch.set_grad_enabled(phase=='train'):
                    logits = model(inputs)
                    loss = criterion(logits, labels) 
                    run_loss += loss.sum().item()  # Sum across the batch and accumulate
                    # Compute MAE
                    probs = torch.sigmoid(logits)  # Shape: [batch_size, n_classes]
                    binary_predictions = (probs > 0.5).int()  # Shape: [batch_size, n_classes]
                    predicted_ranks = binary_predictions.sum(dim=1)  # Shape: [batch_size]
                    true_ranks = labels.sum(dim=1)
                    run_MAE += torch.abs(true_ranks - predicted_ranks).float().mean()

            val_loss = run_loss / len(valLoaders.dataset)
            val_acc = run_MAE.item() / len(valLoaders.dataset)
            val_loss_history.append(val_loss)     
            val_acc_history.append(val_acc) 
            print(f'\n {phase} Loss: {np.round(val_loss, 4)}   MAE: {np.round(val_acc, 4)}')
            
            early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
            if early_stopping.early_stop:
                print('-'*30)
                print(f'Training stop, the validation loss did not drop for {patience} epochs!!!')
                print('-'*30)
                break
            print('-' * 30)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history




def plot_training(train_loss_history, val_loss_history, train_acc_history, val_acc_history, save_pt=None):
    
    epochs = range(1, len(train_loss_history) + 1) 
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Loss
    ax1.plot(epochs, train_loss_history, label='Training Loss', color='blue')
    ax1.plot(epochs, val_loss_history, label='Validation Loss', color='orange')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()
    
    # Plot Accuracy
    ax2.plot(epochs, train_acc_history, label='Training Accuracy', color='green')
    ax2.plot(epochs, val_acc_history, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()
    
    # Show plots
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






# from scipy.special import expit  # For sigmoid function

# # Step 1: Convert 'branch_0', 'branch_1', 'branch_2' to numeric (if necessary)
# outputs_WSI['branch_0'] = pd.to_numeric(outputs_WSI['branch_0'], errors='coerce')
# outputs_WSI['branch_1'] = pd.to_numeric(outputs_WSI['branch_1'], errors='coerce')
# outputs_WSI['branch_2'] = pd.to_numeric(outputs_WSI['branch_2'], errors='coerce')

# # Step 2: Group by 'wsi_id' and calculate the mean for each branch across repeats
# averaged_data = outputs_WSI.groupby('wsi_id')[['branch_0', 'branch_1', 'branch_2']].mean().reset_index()

# # Step 3: Apply sigmoid to the averaged values
# averaged_data['sigmoid_0'] = expit(averaged_data['branch_0'])  # Sigmoid for branch 0
# averaged_data['sigmoid_1'] = expit(averaged_data['branch_1'])  # Sigmoid for branch 1
# averaged_data['sigmoid_2'] = expit(averaged_data['branch_2'])  # Sigmoid for branch 2

# # Step 4: Convert sigmoid values to binary (0 or 1)
# averaged_data['binary_0'] = (averaged_data['sigmoid_0'] >= 0.5).astype(int)
# averaged_data['binary_1'] = (averaged_data['sigmoid_1'] >= 0.5).astype(int)
# averaged_data['binary_2'] = (averaged_data['sigmoid_2'] >= 0.5).astype(int)

# # Step 5: Sum the binary predictions to get the final prediction
# averaged_data['final_prediction'] = averaged_data[['binary_0', 'binary_1', 'binary_2']].sum(axis=1)

# # Display the results (only final prediction for simplicity)
# print(averaged_data[['wsi_id', 'final_prediction']])

