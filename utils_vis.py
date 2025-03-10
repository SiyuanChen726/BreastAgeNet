import os
from pathlib import Path
import sys
import glob
import json
import random
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union

from PIL import Image, ImageDraw
import openslide
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import seaborn as sns
from skimage import morphology
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.stats import mode
from shapely.geometry import Point, Polygon
import geopandas as gpd

import torch
from torch.utils.data import Dataset, DataLoader


def branch_ROC(df, branch = 0, class_name = ">35y", save_pt=None):
    df["branch0_truth"] = (df["age"] > 35).astype(int)
    df["branch1_truth"] = (df["age"] > 45).astype(int)
    df["branch2_truth"] = (df["age"] > 55).astype(int)
    classes = [0, 1]
    
    y_true = label_binarize(df[f'branch{branch}_truth'], classes=classes)
    y_pred = df.loc[:, [f'sigmoid_{branch}']].values  
    
    plt.figure(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)  
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('One-vs-Rest ROC Curves for Each Class', fontsize=12)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    if savefig is not None:
        plt.savefig(save_pt, format = 'pdf')
    plt.show()



def plot_cm(y_true, y_pred, fontsize=16):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true),
                annot_kws={"size": fontsize})  # Adjust font size for annotations
    
    plt.xlabel('Predicted', fontsize=fontsize)
    plt.ylabel('True', fontsize=fontsize)
    plt.title('Confusion Matrix', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()




def plot_cm_norm(y_true, y_pred, fontsize=16):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true),
                annot_kws={"size": fontsize})  # Adjust font size for annotations
    
    plt.xlabel('Predicted', fontsize=fontsize)
    plt.ylabel('True', fontsize=fontsize)
    plt.title('Normalized Confusion Matrix', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()


def barplot_multiple_WSIs(output_df, save_pt):
    output_df["patient_id"] = output_df["patient_id"].fillna('').astype(str)
    output_df_sorted = output_df.sort_values(by='age', ascending=True)
    
    pivot_df = output_df_sorted.groupby(['patient_id', 'final_prediction']).size().unstack(fill_value=0)
    patient_ids_sorted = output_df_sorted['patient_id'].drop_duplicates().values
    pivot_df = pivot_df.loc[patient_ids_sorted]
    ages = output_df_sorted.drop_duplicates(subset='patient_id')['age'].values
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    custom_colors = ["#262262", "#87ACC5", "#00A261", "#FFF200"] * (len(pivot_df.columns) // 4 + 1)
    custom_colors = custom_colors[:len(pivot_df.columns)]
    pivot_df.plot(kind='bar', stacked=True, color=custom_colors, ax=ax1, width=0.9)

    ax1.set_xlabel('Patient ID (Ordered by Age)', fontsize=12)
    ax1.set_ylabel('Count of Predicted Ranks', fontsize=12)
    ax1.set_xticklabels(patient_ids_sorted, fontsize=10, rotation=90)
    ax1.legend(title="Predicted Ranks", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_title('Stacked Predicted Ranks by Patient ID (Ordered by Age)', fontsize=14)
    plt.tight_layout()
    
    if save_pt is not None:
        plt.savefig(save_pt, format='pdf')
    
    plt.show()



def get_tsne_df(repeats):
    tsne_df = repeats.copy()
    
    embedding_columns = [col for col in tsne_df.columns if col.startswith('embedding_')]
    tsne = TSNE(n_components=2, random_state=42)
    projections = tsne.fit_transform(tsne_df.loc[:, embedding_columns])
    
    tsne_df["tsne1"] = -projections[:,0]
    tsne_df["tsne2"] = -projections[:,1]
    return tsne_df


def plot_RidgePlot(tsne_df, label="age_group", save_pt=None):
    pal = sns.cubehelix_palette(4, rot=-.25, light=.7)
    g = sns.FacetGrid(tsne_df, row=label, hue=label, aspect=15, height=1, palette=pal)

    g.map(sns.kdeplot, "tsne1",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "tsne1", clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, "tsne1")
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    fig = plt.gcf()  # Get the current figure
    fig.set_size_inches(5, 5)  # Set the height and width of the figure (in inches)
    if save_pt is not None:
        plt.savefig(save_pt, format="pdf", dpi=300)
    plt.show()




def plot_2D_DensityPlot(tsne_df, label="age_group", max_categories=4, save_pt=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.set(style="white")
    axes = axes.flatten()
    for i in range(max_categories):
        category = np.unique(tsne_df[label])[i]
        subset = tsne_df[tsne_df[label] == category]

        sns.kdeplot(
            x=subset["tsne1"],
            y=subset["tsne2"],
            fill=True,
            alpha=0.5,
            ax=axes[i],
            cmap="viridis"
        )

        axes[i].set_title(f"Density Plot for {category}")
        axes[i].set_xlabel("t-SNE 1")
        axes[i].set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.suptitle("t-SNE Density Plots by Category", y=1.02)
    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()





def plot_tsne(tsne_df, color='age_group', cluster_colors=None, vmin=None, vmax=None, figsize=(10, 8), point_size=3, alpha=0.5, save_pt=None):
    # cluster_colors = {0: '#8da0cb', 3: '#66c2a5', 2: '#b3b3b3', 1: '#ffd92f'}
    # cluster_colors = {0: '#8da0cb', 3: '#66c2a5', 2: '#b3b3b3', 1: '#ffd92f',
    #                   4: '#A6C8E1', 5: '#FEF39E', 6: '#E5E5E5', 7: '#B1DED6'}
    fea_df = tsne_df.copy() 
    fea_df[color] = pd.to_numeric(fea_df[color], errors='coerce')
    fea_df = fea_df.dropna(subset=[color])
    if vmin is None:
        vmin = fea_df[color].min()  # or a specific value, e.g., 0
    if vmax is None:
        vmax = fea_df[color].max()  # or a specific value, e.g., 1

    if color == "age_group":
        cmap = plt.get_cmap('viridis', fea_df['age_group'].nunique())  # 'viridis' for age group
    elif "attention" in color:
        cmap = "viridis"
    else:
        if cluster_colors is None:
            unique_clusters = fea_df[color].unique()
            cmap_set1 = plt.get_cmap("Set1")
            cluster_colors = {cluster: cmap_set1(i % cmap_set1.N) for i, cluster in enumerate(unique_clusters)}
        fea_df['cluster_color'] = fea_df[color].map(cluster_colors)
        cmap = None

    plt.figure(figsize=figsize)
    if cmap is not None:
        scatter = plt.scatter(
            fea_df["tsne1"], 
            fea_df["tsne2"], 
            c=fea_df[color], 
            cmap=cmap, 
            alpha=alpha, 
            s=point_size,
            vmin=vmin,  # Set minimum color scale value
            vmax=vmax   # Set maximum color scale value
        )
        plt.colorbar(scatter, label=color)
    else:
        scatter = plt.scatter(
            fea_df["tsne1"], 
            fea_df["tsne2"], 
            c=fea_df['cluster_color'],  # Use the custom cluster colors
            alpha=alpha, 
            s=point_size
        )
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in cluster_colors.values()]
        labels = list(cluster_colors.keys())
        plt.legend(handles, labels, title=color, loc="upper right")
    
    plt.title(f't-SNE Visualization Colored by {color}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.grid(False)
    plt.show()




def highlight_pattern_in_tsne(tsne_df, label='Cluster', cmap="coolwarm", 
                              max_size=50, min_size=10, 
                              max_alpha=1.0, min_alpha=0.2, 
                              save_pt=None):
    fea_df = tsne_df.copy()
    unique_clusters = sorted(fea_df[label].unique())
    num_clusters = len(unique_clusters)
    grid_size = int(np.ceil(np.sqrt(num_clusters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))
    axes = axes.flatten()
    sns.set_style("whitegrid")
    color_palette = sns.color_palette(cmap, as_cmap=False, n_colors=num_clusters)
    for i, cluster in enumerate(unique_clusters):
        ax = axes[i]
        fea_df["color"] = fea_df[label].apply(lambda x: 1 if x == cluster else 0)
        fea_df["size"] = fea_df[label].apply(lambda x: max_size if x == cluster else min_size)
        fea_df["alpha"] = fea_df[label].apply(lambda x: max_alpha if x == cluster else min_alpha)

        scatter = ax.scatter(
            fea_df["tsne1"], fea_df["tsne2"], 
            c=fea_df["color"], 
            cmap=cmap, 
            alpha=fea_df["alpha"], 
            s=fea_df["size"],
            edgecolors="k", linewidth=0.3
        )

        ax.set_title(f"Highlighting Cluster {cluster}", fontsize=12)
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)

        if i >= len(unique_clusters):
            fig.delaxes(ax)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    legend = fig.legend(
        handles, ["Background", "Highlighted"], 
        loc='lower center', 
        ncol=2, fontsize=12
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_pt:
        plt.savefig(save_pt, bbox_inches="tight", dpi=300)
    plt.show()

    


def Cluster_AgeGroups_heatmap(tsne_df, save_pt=None):
    proportions = tsne_df.groupby(['age_group', 'Cluster']).size().unstack(fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)  # Normalize to 100%
    proportions = proportions.reset_index().melt(id_vars="age_group", value_vars=proportions.columns, var_name="Cluster", value_name="proportion")
    
    heatmap_data = proportions.pivot(index="age_group", columns="Cluster", values="proportion")
    desired_order = ['P0', 'P1', 'P2', 'P3']
    available_columns = [col for col in desired_order if col in heatmap_data.columns]
    if len(available_columns) == len(desired_order):
        # Reorder the columns
        heatmap_data = heatmap_data[available_columns]
    else:
        print("Mismatch in columns. Available columns:", heatmap_data.columns)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Proportion'}, fmt='.2f',
                annot_kws={'size': 16})
    plt.title("Heatmap of Cluster Proportions by Age Group", fontsize=12)
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel("Age Group", fontsize=12)
    plt.tight_layout()
    if save_pt is not None:
        plt.savefig(fname=save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()




def Cluster_AgeGroups_barplot(tsne_df, save_pt):
    proportions = tsne_df.groupby(['age_group', 'Cluster']).size().unstack(fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)  # Normalize to 100%
    proportions = proportions.reset_index().melt(id_vars="age_group", value_vars=proportions.columns, 
                                                 var_name="Cluster", value_name="proportion")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=proportions, x='age_group', y='proportion', hue='Cluster', palette='Set2')
    plt.title("Proportional Barplot of Cluster Proportions by Age Group", fontsize=14)
    plt.xlabel("Age Group", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_pt is not None:
        plt.savefig(save_pt, format="pdf", dpi=300)
    plt.show()




def plot_Attentions_violinplot(tsne_df, save_pt=None):
    attention_columns = ['attention_0', 'attention_1', 'attention_2']
    attention_df = tsne_df[['Cluster'] + attention_columns]
    attention_df = attention_df.melt(id_vars="Cluster", value_vars=attention_columns, var_name="Attention Type", value_name="Attention Value")
    cluster_order = ['P0', 'P1', 'P2', 'P3']
    cluster_colors = {'P0': '#A6C8E1', 'P1': '#B1DED6', 'P2': '#FEF39E', 'P3': '#E5E5E5'}
    
    g = sns.FacetGrid(attention_df, col="Attention Type", hue="Cluster", height=5, aspect=1.2, sharey=False)
    g.map(sns.violinplot, "Cluster", "Attention Value", palette=cluster_colors, inner=None, scale="area", width=0.5, order=cluster_order)
    g.map(sns.boxplot, "Cluster", "Attention Value", palette=cluster_colors, showfliers=True, width=0.2, 
          flierprops=dict(marker='o', color='black', markersize=1.25, alpha=0.5), order=cluster_order)
    g.set_axis_labels("Cluster", "Attention Value")
    g.set_titles("{col_name}")
    g.fig.suptitle("Attention Value Distribution Across Clusters for Each Attention Branch", fontsize=16)
    g.fig.subplots_adjust(top=0.85)
    g.add_legend(title="Cluster")
    plt.tight_layout()
    if save_pt is not None:
        plt.savefig(fname=save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()




def parse_wsi_id(patch_id):
    if " HE" in patch_id: # NKI
        wsi_id = patch_id.split(" HE")[0]
    elif "_FPE_" in patch_id: # KHP
        wsi_id = "_".join(patch_id.split("_")[:3])
    else: # rest
        wsi_id = patch_id.split("_")[0]
    return wsi_id


def get_xy(patch_id):
    try:
        # Split the patch_id to extract coordinates and patch size.
        x, y, patch_size = patch_id.split("_")[-3:]
        x, y, patch_size = int(x), int(y), int(patch_size)
        return x * patch_size, y * patch_size, patch_size
    except ValueError as e:
        raise ValueError(f"Invalid patch_id format: {patch_id}. Expected format '..._x_y_patch_size'.") from e



def paste_HE_on_tsne(tsne_df, WSI_folder='/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/WSIs',
                     cluster_colors=None, max_dim=200, n_samples=500, 
                     image_size=(4000, 3000), random_state=42):
    
    random.seed(random_state)
    random_df = tsne_df.groupby('Cluster').sample(n=n_samples, replace=True, random_state=random_state)
    random_df = random_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    random_df.index = random_df['patch_id']

    tx, ty = random_df["tsne1"], random_df["tsne2"]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width, height = image_size
    full_image = Image.new('RGBA', (width, height))
    
    for patch_id, tsne_x, tsne_y, cluster_id in zip(random_df.index, tx, ty, random_df['Cluster']):
        wsi_id = parse_wsi_id(patch_id)  # Assuming `parse_wsi_id` is defined elsewhere
        wsi_path = glob.glob(f"{WSI_folder}/*/{wsi_id}*.*")
        
        if not wsi_path:
            wsi_path = glob.glob(f"{WSI_folder}/*/*/{wsi_id}*.*")
        
        try:
            # Open the WSI using OpenSlide and extract the patch
            wsi = openslide.OpenSlide(wsi_path[0])
            x, y, patch_size = get_xy(patch_id)  # Assuming `get_xy` is defined elsewhere
            im = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        except Exception as e:
            print(f"Error processing WSI {wsi_id} for patch {patch_id}: {e}")
            continue  # Skip this patch if there's an issue

        tile = Image.fromarray(np.array(im))
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)))

        x_pos = int((width - max_dim) * tsne_x)
        y_pos = int((height - max_dim) * tsne_y)
        border_color = cluster_colors.get(cluster_id, '#000000')  # Default to black if not found

        border_size = 10  # Set the thickness of the border
        tile_with_border = Image.new('RGBA', (tile.width + 2 * border_size, tile.height + 2 * border_size), border_color)
        tile_with_border.paste(tile, (border_size, border_size))  # Paste the original tile inside the border
        full_image.paste(tile_with_border, (x_pos, y_pos), mask=tile_with_border.convert('RGBA'))

    return full_image



def visualize_top_patches(tsne_df, label='Cluster', patch_num=25, WSIs='/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/WSIs', save_pt=None):
    for cluster_id in np.unique(tsne_df[label]):
        patch_ids = tsne_df.loc[tsne_df[label] == cluster_id, "patch_id"].tolist()
        if not patch_ids:
            print(f"[INFO] Skipping cluster {cluster_id}: No patches found.")
            continue

        patch_ids = random.sample(patch_ids, min(patch_num, len(patch_ids)))
        patch_list = []
        for patch_id in patch_ids:
            wsi_id = parse_wsi_id(patch_id)
            wsi_path_list = glob.glob(f"{WSIs}/*/{wsi_id}*.*")
            if not wsi_path_list:
                print(f"[WARNING] No WSI found for {wsi_id}. Skipping patch {patch_id}.")
                continue
            try:
                wsi = openslide.OpenSlide(wsi_path_list[0])
                x, y, patch_size = get_xy(patch_id)
                patch_im = np.array(wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB"))
                patch_list.append(patch_im)
            except Exception as e:
                print(f"[ERROR] Failed to process WSI {wsi_id} for patch {patch_id}: {e}")
                continue
            if len(patch_list) >= patch_num:
                break

        if not patch_list:
            print(f"[INFO] Skipping cluster {cluster_id}: No valid patches extracted.")
            continue

        grid_size = math.ceil(math.sqrt(len(patch_list)))
        sns.set_theme(style='white')
        fig = plt.figure(figsize=(6, 6))
        plt.title(f"Cluster {cluster_id}", fontsize=12)
        plt.axis("off")
        grid = ImageGrid(fig, 111, nrows_ncols=(grid_size, grid_size), axes_pad=0.05)
        for ax, im in zip(grid, patch_list):
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        if save_pt:
            save_path = f"{save_pt}_cluster{cluster_id}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"[INFO] Saved visualization for cluster {cluster_id} at {save_path}")
        plt.show()





class WSIDataset(Dataset):
    def __init__(self, df, bag_size):
        self.data = df.values  # The patch data (features)
        self.patch_ids = df.index  # The patch IDs (from DataFrame index)
        self.bag_size = bag_size  # The bag size (how many patches in each bag)
        self.num_patches = self.data.shape[0]  # Total number of patches

    def __len__(self):
        return int(np.ceil(self.num_patches / self.bag_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.bag_size  # Start index for the bag
        end_idx = min(start_idx + self.bag_size, self.num_patches)  # End index for the bag
        patches = self.data[start_idx:end_idx]  # Get the patches for the current bag
        patch_ids = self.patch_ids[start_idx:end_idx]  # Get the patch IDs for the current bag

        if len(patches) < self.bag_size:
            padding = np.zeros((self.bag_size - len(patches), patches.shape[1]))  # Pad with zeros
            patches = np.vstack([patches, padding])  # Add padding to patches
            patch_ids = np.append(patch_ids, [''] * (self.bag_size - len(patch_ids)))  # Add empty strings for patch IDs

        patch_ids = list(patch_ids)  # Convert to a list (or np.array)
        return patch_ids, torch.tensor(patches, dtype=torch.float32)


def WSI_loader(df, batch_size=256, bag_size=250, shuffle=False):
    dataset = WSIDataset(df, bag_size)  # Create the custom dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # Create DataLoader
    return dataloader


def run_BreastAgeNet_through_WSI(model, wsi_id=None, batch_size=4, bag_size=250, folder=None):
    WSI_info = glob.glob(f'{folder}/{wsi_id}/{wsi_id}_patch.csv')[0]
    print(WSI_info)
    WSI_info = pd.read_csv(WSI_info)
    valid_ids = list(WSI_info['patch_id'][WSI_info['TC_epi'] > 0.9])

    WSI_fea = glob.glob(f'{folder}/{wsi_id}/{wsi_id}_bagFeature_UNI_augmentation.h5')[0]
    print(WSI_fea)
    with h5py.File(WSI_fea, "r") as file:
        bag = np.array(file["embeddings"])
        bag = np.squeeze(bag)
        img_id = np.array(file["patch_id"])
    img_id = [i.decode("utf-8") for i in img_id]
    bag_df = pd.DataFrame(bag)
    bag_df.index = img_id
    bag_df.index = bag_df.index.str.split("_").str[:3].str.join("_") + "_" + bag_df.index.str.split("_").str[-3:].str.join("_")
    bag_df = bag_df.loc[bag_df.index.isin(valid_ids), :]

    phase = 'test'
    model.eval()
    wsiloader = WSI_loader(bag_df, batch_size=batch_size, bag_size=bag_size)
    WSI_df = pd.DataFrame()  # Initialize empty DataFrame
    for patch_ids, inputs in tqdm(wsiloader):
        patch_ids = np.array(patch_ids)
        patch_ids = np.transpose(patch_ids)
        patch_ids = patch_ids.flatten()
        with torch.set_grad_enabled(phase == 'train'):
            logits, embeddings, attentions = model(inputs)
            attentions = attentions.view(-1, attentions.shape[-1])  # Flatten attentions
            embeddings = embeddings.view(-1, embeddings.shape[-1])  # Flatten embeddings
        combined_data = np.column_stack((patch_ids, embeddings.cpu().numpy(), attentions.cpu().numpy())) 
        dfi = pd.DataFrame(combined_data, columns=['patch_id'] + [f'embedding_{i}' for i in range(embeddings.shape[1])] + [f'attention_{i}' for i in range(attentions.shape[1])]) 
        WSI_df = pd.concat([WSI_df, dfi], axis=0) 

    coord_X, coord_Y = [], []
    for patch_id in WSI_df["patch_id"]:
        parts = patch_id.split("_")
        if len(parts) >= 3:
            try:
                coord_X.append(int(parts[-3]))
                coord_Y.append(int(parts[-2]))
            except ValueError:
                coord_X.append(None)
                coord_Y.append(None)
        else:
            coord_X.append(None)
            coord_Y.append(None)
    
    WSI_df["coord_X"], WSI_df["coord_Y"] = coord_X, coord_Y
    WSI_df = WSI_df[WSI_df['coord_X'].notna()].copy()  # Keep only rows where patch_id is not None/NaN
    
    return WSI_df



def apply_kmeans(WSI_df, reference, kmeans_model):

    embedding_columns = [f'embedding_{i}' for i in range(512)]  # Adjust if necessary
    reference_labels = kmeans_model.predict(reference[embedding_columns].values)
    y_train = reference["Cluster"].values  

    label_mapping = {}
    for true_label in np.unique(y_train):
        mask = y_train == true_label
        most_common_label = mode(reference_labels[mask], keepdims=True).mode[0]
        label_mapping[most_common_label] = true_label
    print("Label Mapping:", label_mapping)

    
    WSI_df = WSI_df.drop_duplicates().dropna()
    WSI_df[embedding_columns] = WSI_df[embedding_columns].apply(pd.to_numeric, errors="coerce")
    WSI_df.dropna(subset=embedding_columns, inplace=True)
    assert WSI_df[embedding_columns].isnull().sum().sum() == 0, "New data contains NaN values."

    X_new = WSI_df[embedding_columns].values
    new_labels = kmeans_model.predict(X_new)
    new_labels = np.array([label_mapping.get(label, -1) for label in new_labels])
    WSI_df["Cluster"] = new_labels

    return WSI_df




def add_orig_coords(WSI_df):
    WSI_df = WSI_df.copy()
    split_data = WSI_df["patch_id"].str.split("_", expand=True)
    grid_x = split_data.iloc[:, -3].astype(int)
    grid_y = split_data.iloc[:, -2].astype(int)
    patch_size = split_data.iloc[:, -1].astype(int)
    x_orig = grid_x * patch_size
    y_orig = grid_y * patch_size
    WSI_df.loc[:, "x_orig"] = x_orig
    WSI_df.loc[:, "y_orig"] = y_orig
    WSI_df.loc[:, "patch_size"] = patch_size
    return WSI_df



def draw_wsi_with_clusters(WSI_df, wsi_path=None, cluster_colors=None, level=5, save_pt=None):
    WSI_df = add_orig_coords(WSI_df)
    patch_size = int(np.unique(WSI_df["patch_size"]))
    
    wsi = openslide.OpenSlide(wsi_path)
    level_dimensions = wsi.level_dimensions[level]
    wsi_img = wsi.read_region((0, 0), level, level_dimensions)
    wsi_img = wsi_img.convert("RGBA")  # Convert to RGBA mode for transparency support
    scale_factor = wsi.level_downsamples[level]  # Downsampling factor for the chosen level
    
    draw = ImageDraw.Draw(wsi_img)
    box_size = int(patch_size / scale_factor)  # Scale the box size according to the level
    fill_opacity = 255  # Adjust this to make the filled boxes stronger (0-255, where 255 is fully opaque)
    border_width = 0  # Set width of the black border around each box
    for _, row in WSI_df.iterrows():
        x_orig, y_orig = row['x_orig'], row['y_orig']
        cluster = row['Cluster']
            
        cluster_color = cluster_colors.get(cluster, '#000000')  # Default to black if cluster is not in the map
        rgb_color = mcolors.hex2color(cluster_color)  # Convert hex color to RGB
        rgba_color = tuple(int(c * 255) for c in rgb_color) + (fill_opacity,)  # Convert to RGBA
        x_scaled = x_orig / scale_factor
        y_scaled = y_orig / scale_factor
    
        draw.rectangle([x_scaled - box_size / 2 - border_width, 
                        y_scaled - box_size / 2 - border_width, 
                        x_scaled + box_size / 2 + border_width, 
                        y_scaled + box_size / 2 + border_width], 
                       outline="black", width=border_width)
        draw.rectangle([x_scaled - box_size / 2, 
                        y_scaled - box_size / 2, 
                        x_scaled + box_size / 2, 
                        y_scaled + box_size / 2], 
                       fill=rgba_color, outline="black")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wsi_img)
    ax.axis('off')  # Hide axes
    cmap = mcolors.ListedColormap([cluster_colors[0], cluster_colors[1], cluster_colors[2], cluster_colors[3]])
    norm = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=4)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical', shrink=0.5, fraction=0.02, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels([0, 1, 2, 3])
    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.show()
    return WSI_df, wsi_img


def plot_cluster_proportion_for_a_WSI(WSI_df, save_pt):
    phenotype_counts = WSI_df['Cluster'].value_counts().sort_index()
    ordered_phenotypes = [0, 1, 2, 3]

    for phenotype in ordered_phenotypes:
        if phenotype not in phenotype_counts.index:
            phenotype_counts[phenotype] = 0

    phenotype_counts = phenotype_counts[ordered_phenotypes].sort_index()
    plt.figure(figsize=(3, 3))
    cluster_colors = {0: '#8da0cb', 1: '#66c2a5', 2: '#ffd92f', 3: '#b3b3b3'}
    x_pos = range(len(ordered_phenotypes))

    for i, phenotype in enumerate(ordered_phenotypes):
        plt.bar(x_pos[i], phenotype_counts[phenotype], color=cluster_colors[phenotype], label=phenotype)

    plt.title("Phenotype Counts for WSI")
    plt.ylabel("Count")
    plt.xlabel("Phenotype")
    plt.xticks(x_pos, ['P0', 'P1', 'P2', 'P3'])
    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.grid(False)
    plt.show()



def build_poly(tx: np.ndarray, ty: np.ndarray, bx: np.ndarray, by: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    px = np.vstack((tx, bx, bx, tx)).T
    py = np.vstack((ty, ty, by, by)).T
    return px, py
    


def clusters_json_for_a_WSI(WSI_df, wsi_id, cluster_colors, json_dir=None, require_bounds=False):
    # cluster_colors = {0: '#8da0cb', 1: '#66c2a5', 2: '#ffd92f', 3: '#b3b3b3'}
    tx = np.array(WSI_df["x_orig"]).astype("int")
    ty = np.array(WSI_df["y_orig"]).astype("int")
    bx = np.array(WSI_df["x_orig"] + WSI_df["patch_size"]).astype("int")
    by = np.array(WSI_df["y_orig"] + WSI_df["patch_size"]).astype("int")

    if require_bounds:  # this is meant for the NKI cohort
        bounds_x = int(wsi.properties['openslide.bounds-x'])
        bounds_y = int(wsi.properties['openslide.bounds-y'])
        tx = np.array(tx - bounds_x).astype("int")
        ty = np.array(ty - bounds_y).astype("int")
        bx = np.array(bx - bounds_x).astype("int")
        by = np.array(by - bounds_y).astype("int")

    polys_x, polys_y = build_poly(tx=tx, ty=ty, bx=bx, by=by)
    
    values = list(WSI_df['Cluster'])
    names = list(WSI_df['Cluster'])

    coords = {}
    for i in range(len(polys_x)):
        phenotype = names[i]
        if phenotype in cluster_colors:
            color = cluster_colors[phenotype]
        else:
            color = '#000000'  # Default to black if phenotype is not in cluster_colors

        coords['poly{}'.format(i)] = {
            "coords": np.vstack((polys_x[i], polys_y[i])).tolist(),
            "class": phenotype, 
            "name": phenotype, 
            "color": [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]  # Convert hex to RGB
        }

    json_pt = f"{json_dir}/{wsi_id}_BreastAgeNet_clusters.json"
    with open(json_pt, 'w') as outfile:
        json.dump(coords, outfile)
    print(f"{json_pt} saved!")




def plot_oneline(img_list, caption_list, figure_size, save_pt=None):
    fig, axes = plt.subplots(1, len(img_list), figsize=figure_size)
    
    for index in range(len(img_list)):
        axes[index].imshow(img_list[index])
        axes[index].axis("off") 
        caption_i = caption_list[index]
        if isinstance(caption_i, str):
            axes[index].set_title(f"{caption_i}")
        else:
            axes[index].set_title(f"{np.around(caption_i, 2)}")

    if save_pt is not None:
        plt.savefig(save_pt, pad_inches=0, bbox_inches="tight", dpi=300)
  
    plt.show()


    
    
def plot_multiple(img_list, caption_list, grid_x, grid_y, figure_size, title=None, save_pt=None, cmap=None):
    fig, axes = plt.subplots(grid_x, grid_y, figsize=figure_size)
    if title is not None:
        plt.suptitle(title, fontsize=16)

    counter = 0
    for x in range(grid_x):
        for y in range(grid_y):
            if counter < len(img_list):
                # Apply colormap if provided
                axes[x][y].imshow(img_list[counter], cmap=cmap)
                axes[x][y].axis("off")  # Hide axis for a cleaner image display
                if caption_list is not None and counter < len(caption_list):
                    axes[x][y].set_title(f"{caption_list[counter]}")
                counter += 1
            else:
                axes[x][y].axis("off")  # Hide unused subplots

    if save_pt is not None:
        plt.savefig(save_pt, pad_inches=0, bbox_inches="tight", dpi=300)
    
    plt.show()



def sample_k_patches(wsi, patch_ids, num=25, aug=False):
    img_list = []  # List to store the sampled patches
    Augimg_list = []  # List to store augmented patches if augmentation is applied
    
    if aug:
        augmentor = RandStainNA(
            yaml_file = '../RandStainNA/CRC_LAB_randomTrue_n0.yaml',
            std_hyper = 0.0,
            distribution = 'normal',
            probability = 1.0,
            is_train = True
        )
        # img:is_train:false——>np.array()(cv2.imread()) #BGR
        # img:is_train:True——>PIL.Image #RGB

    random.shuffle(patch_ids)
    lens = min(len(patch_ids), num)
    for i in range(lens):
        patch_id = patch_ids[i]
        x, y, patch_size = get_xy(patch_id)
        
        try:
            im = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            img_list.append(im)  # Append the raw patch to img_list
            
            if aug:  # Apply augmentation if the flag is True
                augmented_im = augmentor(im)
                Augimg_list.append(augmented_im)  # Append the augmented patch
            else:
                Augimg_list.append(im)  # If no augmentation, add the raw patch to Augimg_list

        except Exception as e:
            print(f"Error reading patch {patch_id} at ({x}, {y}): {e}")
            continue  # Skip the current patch if an error occurs

    if not aug:
        return img_list
    return img_list, Augimg_list




def assign_annotation_labels(patch_df, annotation_files):
    def load_annotation(wsi_id):
        geojson_file = f'/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/Manual_anno/KHP_lobule/geojson/{wsi_id}.geojson'
        with open(geojson_file, 'r') as f:
            data = json.load(f)
        
        polygons = []
        labels = []
        
        for feature in data['features']:
            # Extract the label
            label = feature['properties']['classification']['name']
            # Extract the coordinates and create a Polygon
            points = feature['geometry']['coordinates'][0]  # coordinates is a list of coordinates
            polygon = Polygon(points)
            polygons.append(polygon)
            labels.append(label)  # Store the corresponding label
        
        return polygons, labels
    
    # Initialize the list to store the labels for each patch
    labels = []
    
    # Iterate over each patch in the DataFrame
    for _, row in patch_df.iterrows():
        wsi_id = row['wsi_id']
        patch_coords = (row['x_orig'], row['y_orig'])
        
        # Load the annotations for the current wsi_id
        polygons, polygon_labels = load_annotation(wsi_id)
        
        # Check if the patch is inside any polygon
        patch_point = Point(patch_coords)
        assigned_label = 'No_annotation'  # Default label if no polygon is found
        
        for polygon, label in zip(polygons, polygon_labels):
            if polygon.contains(patch_point):  # Check if the patch is inside the polygon
                assigned_label = label  # Assign the label of the polygon
                break
        
        # Append the assigned label to the list
        labels.append(assigned_label)
    
    # Add the 'annotation_label' column to the DataFrame
    patch_df['annotation_label'] = labels
    
    return patch_df



def lobulemask_fromAnnotation(wsi_path=None, anno_pt=None):
    slide = openslide.OpenSlide(wsi_path)
    HE_img = np.array(slide.read_region((0, 0), slide.level_count-1, slide.level_dimensions[slide.level_count-1]).convert("RGB"))
    
    with open(anno_pt, "r") as f:
        shapes = json.load(f)
    
    level = slide.level_count - 1
    scale_factor = 1 / slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]
    
    # Set the background to NaN
    mask = np.full((height, width, 3), np.nan, dtype=np.float32)  # NaN background
    
    # Iterate through the shapes and add colored regions based on class
    for shape in shapes["features"]:
        points = shape["geometry"]["coordinates"][0]
        points = np.array([(p[0], p[1]) for p in points])
        points = points * scale_factor
        points = points.astype(int)
    
        cls = shape["properties"]["classification"]["name"]
        
        if cls == "3":
            color = (33, 102, 172)  # Blue (BGR format)
        elif cls == "2":
            color = (191, 212, 178)  # Light Green (BGR format)
        elif cls == "1":
            color = (178, 24, 43)  # Red (BGR format)
        
        # Draw contours and fill the polygons with the specified colors
        mask = cv2.drawContours(mask, [points], -1, color=color, thickness=1)
        mask = cv2.fillPoly(mask, [points], color=color)
        
    return mask, HE_img





