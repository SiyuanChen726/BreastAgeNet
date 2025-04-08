import os
from pathlib import Path
import sys
import math
import cv2
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
plt.rcParams["font.family"] = "DejaVu Sans"
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib import colors as mcolors
import seaborn as sns
from skimage import morphology
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import pairwise_distances
from itertools import combinations  # Add this import


from scipy.stats import mode
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import zoom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.colors import SymLogNorm


from shapely.geometry import Point, Polygon
import geopandas as gpd

import torch
from torch.utils.data import Dataset, DataLoader





def vis_branch_predictions(df, figsize=(6, 6), save_pt=None):
    fig, axes = plt.subplots(1, 3, figsize=figsize)  # 3 rows, 1 column
    
    heatmap_data1 = df[["sigmoid_0", "sigmoid_1", "sigmoid_2"]]
    sns.heatmap(heatmap_data1, cmap="coolwarm", cbar=True, xticklabels=True, yticklabels=False, linewidths=0.5, linecolor='gray', ax=axes[0])
    axes[0].set_xlabel("Sigmoid Categories")
    axes[0].set_title("Heatmap for \n Sigmoid Predictions", fontsize=10)
    
    heatmap_data2 = df[["binary_0", "binary_1", "binary_2"]]
    sns.heatmap(heatmap_data2, cmap="binary", cbar=False, xticklabels=True, yticklabels=False, linewidths=0.5, linecolor='gray', ax=axes[1])
    axes[1].set_xlabel("Binary Categories")
    axes[1].set_title("Heatmap for \n Binary Predictions", fontsize=10)
    
    heatmap_data3 = df[["age_group"]]
    set1_cmap = ListedColormap(["#262262", "#87ACC5", "#00A261", "#FFF200"])
    sns.heatmap(heatmap_data3, cbar=True, xticklabels=True, yticklabels=False, linewidths=0.5, linecolor='gray', cmap=set1_cmap, cbar_kws={"ticks": [0, 1, 2, 3]}, ax=axes[2])
    axes[2].set_xlabel("Age Group")
    axes[2].set_title("Heatmap for \n Age Group", fontsize=10)
    plt.tight_layout()
    if save_pt:
        plt.savefig(save_pt, bbox_inches='tight', dpi=300)
    plt.show()




def branch_ROC(df, branch=0, class_name=">35y", ax=None, savefig=None, fontsize=12, line_thickness=2):
    df["branch0_truth"] = (df["age"] > 35).astype(int)
    df["branch1_truth"] = (df["age"] > 45).astype(int)
    df["branch2_truth"] = (df["age"] > 55).astype(int)
    classes = [0, 1]
    y_true = label_binarize(df[f'branch{branch}_truth'], classes=classes)
    y_pred = df.loc[:, [f'sigmoid_{branch}']].values  
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    if ax: 
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', linewidth=line_thickness)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)  # Thicker line for random classifier
        ax.set_title(f'Branch {branch}', fontsize=fontsize)
        ax.set_xlabel('False Positive Rate', fontsize=fontsize)
        ax.set_ylabel('True Positive Rate', fontsize=fontsize)
        ax.legend(loc="lower right", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)  # X-axis tick labels
        ax.tick_params(axis='y', labelsize=fontsize)  # Y-axis tick labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    elif savefig: 
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})', linewidth=line_thickness)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=line_thickness)  
        plt.title('One-vs-Rest ROC Curves for Each Class', fontsize=fontsize)
        plt.xlabel('False Positive Rate (FPR)', fontsize=fontsize)
        plt.ylabel('True Positive Rate (TPR)', fontsize=fontsize)
        plt.legend(loc='lower right', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(True)
        plt.savefig(savefig, format='pdf')
        plt.show()





def plot_cm(y_true, y_pred, fontsize=16, save_pt=None, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:  # If no axis is provided, create a new figure
        plt.figure(figsize=(6, 5))
        ax = plt.gca()  # Get current axis for the new figure
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true),
                annot_kws={"size": fontsize+1}, ax=ax)  # Plot on the provided axis
    
    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_ylabel('True', fontsize=fontsize)
    ax.set_title('Confusion Matrix', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Adjust the colorbar ticks size
    cbar = ax.collections[0].colorbar  # Get colorbar object from the heatmap
    cbar.ax.tick_params(labelsize=fontsize-1)  # Adjust font size of colorbar ticks
    
    if save_pt is not None:
        plt.savefig(save_pt, format="pdf", dpi=300, transparent=True)
    if ax is None:  # If the plot was created individually (no ax), show it
        plt.show()



def plot_cm_norm(y_true, y_pred, fontsize=16, save_pt=None, ax=None):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    if ax is None:  # If no axis is provided, create a new figure
        plt.figure(figsize=(6, 5))
        ax = plt.gca()  # Get current axis for the new figure
    
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true),
                annot_kws={"size": fontsize+1}, vmin=0, vmax=1, ax=ax)  # Plot on the provided axis
    
    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_ylabel('True', fontsize=fontsize)
    ax.set_title('Normalized Confusion Matrix', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Adjust the colorbar ticks size
    cbar = ax.collections[0].colorbar  # Get colorbar object from the heatmap
    cbar.ax.tick_params(labelsize=fontsize-1)  # Adjust font size of colorbar ticks
    
    if save_pt is not None:
        plt.savefig(save_pt, format="pdf", dpi=300, transparent=True)
    if ax is None:  # If the plot was created individually (no ax), show it
        plt.show()




def barplot_multiple_WSIs(output_df, save_pt):
    # Prepare data
    output_df["patient_id"] = output_df["patient_id"].fillna('').astype(str)
    output_df_sorted = output_df.sort_values(by='age', ascending=True)
    
    # Create pivot table for stacked bar chart
    pivot_df = output_df_sorted.groupby(['patient_id', 'final_prediction']).size().unstack(fill_value=0)
    patient_ids_sorted = output_df_sorted['patient_id'].drop_duplicates().values
    pivot_df = pivot_df.loc[patient_ids_sorted]
    ages = output_df_sorted.drop_duplicates(subset='patient_id')['age'].values
    
    # Create the figure and axes for side-by-side plots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [0.5, 1]})
    
    # Age plot (now on the top with black color)
    ax1.plot(patient_ids_sorted, ages, color='black', marker='o', linestyle='-', linewidth=1, markersize=4, label='Patient Age')
    
    # Set labels for the age plot
    ax1.set_xlabel('Patient ID (Ordered by Age)', fontsize=10)
    ax1.set_ylabel('Age', fontsize=10, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticklabels(patient_ids_sorted, fontsize=10, rotation=90)
    ax1.set_title('Patient Age by Patient ID (Ordered by Age)', fontsize=10)
    
    # Restrict y-ticks to 0, 35, 45, 55
    ax1.set_yticks([0, 35, 45, 55])
    
    # Add dashed lines at y=35, y=45, and y=55
    ax1.axhline(y=35, color='navy', linestyle='--', linewidth=0.75)
    ax1.axhline(y=45, color='navy', linestyle='--', linewidth=0.75)
    ax1.axhline(y=55, color='navy', linestyle='--', linewidth=0.75)

    # Stacked predicted ranks plot (now at the bottom)
    custom_colors = ["#262262", "#87ACC5", "#00A261", "#FFF200"] * (len(pivot_df.columns) // 4 + 1)
    custom_colors = custom_colors[:len(pivot_df.columns)]
    pivot_df.plot(kind='bar', stacked=True, color=custom_colors, ax=ax2, width=0.9)

    # Set labels and title for the stacked bar chart
    ax2.set_xlabel('Patient ID (Ordered by Age)', fontsize=12)
    ax2.set_ylabel('Count of Predicted Ranks', fontsize=12)
    ax2.set_xticklabels(patient_ids_sorted, fontsize=10, rotation=90)
    ax2.legend(title="Predicted Ranks", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_title('Stacked Predicted Ranks by Patient ID (Ordered by Age)', fontsize=14)

    # Tight layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_pt is not None:
        plt.savefig(save_pt, format='pdf')
    
    # Show the plot
    plt.show()




def compute_patient_level_predictions(df):
    patient_predictions = []
    
    for repetition in range(10):
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id]
            available_wsi_count = patient_data.shape[0]
            
            for sample_size in [1, 2, 3, available_wsi_count]:  
                sampled_data = patient_data.sample(n=sample_size, replace=True)
                
                avg_sigmoid_0 = sampled_data['sigmoid_0'].mean()
                avg_sigmoid_1 = sampled_data['sigmoid_1'].mean()
                avg_sigmoid_2 = sampled_data['sigmoid_2'].mean()
                
                binary_0 = 1 if avg_sigmoid_0 > 0.5 else 0
                binary_1 = 1 if avg_sigmoid_1 > 0.5 else 0
                binary_2 = 1 if avg_sigmoid_2 > 0.5 else 0
                
                final_prediction = binary_0 + binary_1 + binary_2  
                
                patient_predictions.append({
                    'patient_id': patient_id,
                    'sample_size': "all" if sample_size > 3 else sample_size,
                    'age_group': patient_data['age_group'].iloc[0],  
                    'final_prediction': final_prediction,
                    'repetition': repetition + 1  # Track repetition
                })
    
    return pd.DataFrame(patient_predictions)




def plot_tsne(tsne_df, color='age_group', custom_palette=None, vmin=None, vmax=None, 
              figsize=(8, 6), point_size=3, alpha=0.7, save_pt=None, ax=None):

    fea_df = tsne_df.copy()
    
    if "attention" in color:
        cmap = "coolwarm"
        vmin = vmin if vmin is not None else fea_df[color].min()
        vmax = vmax if vmax is not None else fea_df[color].max()
    else:
        if custom_palette is None:
            default_palettes = {
                "age_group": {0: '#262262', 1: '#87ACC5', 2: '#00A261', 3: '#FFF200'},
                "Cluster": {0: '#8da0cb', 1: '#66c2a5', 2: '#ffd92f', 3: '#b3b3b3'}
            }
            custom_palette = default_palettes.get(color, default_palettes["age_group"])
        fea_df['scatter_color'] = fea_df[color].map(custom_palette)
        cmap = None  

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if cmap:
        scatter = ax.scatter(
            fea_df["tsne1"], fea_df["tsne2"], c=fea_df[color], cmap=cmap, 
            alpha=alpha, s=point_size, vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color.replace("_", " ").title())
    else:
        scatter = ax.scatter(
            fea_df["tsne1"], fea_df["tsne2"], c=fea_df['scatter_color'], 
            alpha=alpha, s=point_size
        )
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=custom_palette[label], markersize=10) 
                   for label in custom_palette]
        legend = ax.legend(handles, custom_palette.keys(), title=color.replace("_", " ").title(), 
                           loc="center right", bbox_to_anchor=(1.2, 0.5), frameon=False)

    ax.set_title(f't-SNE Colored by {color.replace("_", " ").title()}')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(False)

    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    if ax is None:
        plt.show()



# def plot_RidgePlot(tsne_df, label="age_group", save_pt=None):
#     custom_palette = {0: '#262262', 1: '#87ACC5', 2: '#00A261', 3: '#FFF200'}
    
#     g = sns.FacetGrid(tsne_df, row=label, hue=label, aspect=15, height=1, palette=custom_palette)
#     g.map(sns.kdeplot, "tsne1",
#           bw_adjust=.5, clip_on=False,
#           fill=True, linewidth=1.5, alpha=0.8)  
#     g.map(sns.kdeplot, "tsne1", clip_on=False, color="w", lw=2, bw_adjust=.5, alpha=0) 
#     g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

#     def label_func(x, color, label):
#         ax = plt.gca()
#         ax.text(-0.1, 0.1, f"Rank {label}", fontweight="bold", color="black",  
#                 ha="left", va="center", transform=ax.transAxes)

#     g.map(label_func, "tsne1")
#     g.figure.subplots_adjust(hspace=-.25)
#     g.set_titles("")
#     g.set(yticks=[], ylabel="")
#     g.despine(bottom=True, left=True)

#     for ax in g.axes.flat:
#         ax.set_facecolor('none')  # Subplot background is transparent
#         ax.spines['top'].set_visible(False)  # Remove top spine (border)
#         ax.spines['right'].set_visible(False)  # Remove right spine (border)
#         ax.spines['left'].set_visible(False)  # Remove left spine (border)
#         ax.spines['bottom'].set_visible(False)  # Remove bottom spine (border)
#         ax.grid(False)  # Remove gridlines

#     fig = plt.gcf()
#     fig.set_size_inches(5, 5) 
#     if save_pt is not None:
#         plt.savefig(save_pt, format="pdf", dpi=300, transparent=True)
#     plt.show()





# def plot_2D_DensityPlot(tsne_df, label="age_group", max_categories=4, save_pt=None):
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     sns.set(style="white")
#     axes = axes.flatten()
#     for i in range(max_categories):
#         category = np.unique(tsne_df[label])[i]
#         subset = tsne_df[tsne_df[label] == category]

#         sns.kdeplot(
#             x=subset["tsne1"],
#             y=subset["tsne2"],
#             fill=True,
#             alpha=0.5,
#             ax=axes[i],
#             cmap="viridis"
#         )

#         axes[i].set_title(f"Density Plot for {category}")
#         axes[i].set_xlabel("t-SNE 1")
#         axes[i].set_ylabel("t-SNE 2")

#     plt.tight_layout()
#     plt.suptitle("t-SNE Density Plots by Category", y=1.02)
#     if save_pt:
#         plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
#     plt.show()




def highlight_pattern_in_tsne(tsne_df, label='Cluster', cmap="coolwarm", 
                              max_size=50, min_size=10, 
                              max_alpha=1.0, min_alpha=0.2, 
                              figsize=None, save_pt=None):
    fea_df = tsne_df.copy()
    unique_clusters = sorted(fea_df[label].unique())
    num_clusters = len(unique_clusters)
    grid_size = int(np.ceil(np.sqrt(num_clusters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    sns.set_style("white")  
    colormap = plt.colormaps[cmap] 
    
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
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    blue_color = colormap(0)
    red_color = colormap(0.999)  
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=blue_color, markersize=6, label='Other clusters'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=red_color, markersize=10, label='Highlighted cluster')
    ]
    legend = fig.legend(
        handles=legend_handles, 
        loc="center left", 
        bbox_to_anchor=(0.9, 0.5), 
        fontsize=12, frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for legend space
    if save_pt:
        plt.savefig(save_pt, bbox_inches="tight", dpi=300)
    plt.show()




def parse_wsi_id(patch_id):
    if "K" in patch_id: # SGK
        wsi_id = patch_id.split("_")[0]
    elif " HE" in patch_id: # NKI
        wsi_id = patch_id.split(" HE")[0]
    elif "_HE" in patch_id: # NKI
        wsi_id = patch_id.split("_HE")[0]
    elif "_FPE_" in patch_id: # KHP
        wsi_id = "_".join(patch_id.split("_")[:3])
    elif "Human" in patch_id:
        wsi_id = patch_id.split("_HE.vsi")[0]
    else: # BCI
        wsi_id = "_".join(patch_id.split("_")[:-7])
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





def get_Cluster_example(tsne_df, cluster_id, im_num, WSIs="/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/WSIs"):
    patch_ids = tsne_df.loc[tsne_df["Cluster"] == cluster_id, "patch_id"].values.tolist()
    random.shuffle(patch_ids)

    img_list = []
    attempted = 0

    for patch_id in patch_ids:
        if len(img_list) >= im_num:
            break
        try:
            wsi_id = parse_wsi_id(patch_id)
            wsi_path_list = glob.glob(f"{WSIs}/*/{wsi_id}*.*")
            
            if not wsi_path_list:
                print(f"[WARNING] No WSI found for {wsi_id}. Skipping.")
                continue

            wsi = openslide.OpenSlide(wsi_path_list[0])
            x, y, patch_size = get_xy(patch_id)
            im = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            img_list.append(im)
            attempted += 1

        except Exception as e:
            print(f"[ERROR] Skipping patch {patch_id}: {e}")
            continue

    print(f"[INFO] Cluster {cluster_id}: Collected {len(img_list)} valid patches (attempted {attempted}).")

    return img_list





def compute_adjusted_pvalues_global(df, features, group_label, group_order, save_pt=None):
    p_value_table = []  
    for feature in features:
        p_values = []
        comparisons = []

        # Group the data by the group_label
        grouped_data = [df[df[group_label] == group][feature].dropna() for group in group_order]
        
        # Perform Kruskal-Wallis test across all groups
        stat, p = kruskal(*grouped_data)
        
        # Store the p-value for the feature
        p_values.append(p)
        comparisons.append((feature, "All Groups"))

        # Adjust p-values using Benjamini-Hochberg (FDR)
        _, adj_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        # Store the adjusted p-value and significance
        for (feature, comparison), adj_p in zip(comparisons, adj_p_values):
            significance = '***' if adj_p < 0.001 else '**' if adj_p < 0.01 else '*' if adj_p < 0.05 else ''
            p_value_table.append({'Feature': feature, 'Comparison': comparison, 'Adjusted p-value': adj_p, 'Significance': significance})

    p_value_df = pd.DataFrame(p_value_table)
    if save_pt is not None:
        p_value_df.to_csv(save_pt, index=False)

    return p_value_df




def compute_adjusted_pvalues(df, features, group_label, group_order, save_pt=None):
    p_value_table = []  
    for feature in features:
        p_values = []
        comparisons = []

        for (c1, c2) in combinations(group_order, 2):
            group1 = df[df[group_label] == c1][feature]
            group2 = df[df[group_label] == c2][feature]
            
            if len(group1) > 0 and len(group2) > 0:
                stat, p = ranksums(group1, group2)
                p_values.append(p)
                comparisons.append((feature, c1, c2))

        # Adjust p-values using Benjamini-Hochberg (FDR)
        _, adj_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        for (feature, cluster1, cluster2), adj_p in zip(comparisons, adj_p_values):
            # Assign significance based on adjusted p-value
            significance = '***' if adj_p < 0.001 else '**' if adj_p < 0.01 else '*' if adj_p < 0.05 else ''
            p_value_table.append({'Feature': feature, 'Group 1': cluster1, 'Group 2': cluster2, 'Adjusted p-value': adj_p, 'Significance': significance})

    p_value_df = pd.DataFrame(p_value_table)
    if save_pt is not None:
        p_value_df.to_csv(save_pt, index=False)

    return p_value_df




def compute_whisker_limits(df, feature, group_label, group_order, showfliers):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

    if showfliers:
        uppers = []
        lowers = []
        
        for group in group_order:
            group_data = df[df[group_label] == group][feature]
            uppers.append(max(group_data))
            lowers.append(min(group_data))
            
        upper = max(uppers)
        lower = min(lowers)
        y_range = upper - lower
        y_pos = upper + 0.03 * y_range  # Increase the offset space for p-value annotations

    else:
        upper_whiskers = []
        lower_whiskers = []
        
        for group in group_order:
            group_data = df[df[group_label] == group][feature]
            
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_whisker_limit = Q1 - 1.5 * IQR
            upper_whisker_limit = Q3 + 1.5 * IQR
            
            # Append whisker limits
            upper_whiskers.append(upper_whisker_limit)
            lower_whiskers.append(lower_whisker_limit)
            
            # Optionally: Clip the data for the actual whiskers (for visualizations)
            # This step can help ensure the whiskers are correctly visualized without outliers
            group_data_clipped = group_data.clip(lower=lower_whisker_limit, upper=upper_whisker_limit)
    
        # Determine the y-position and range for the whiskers
        upper_whisker_max = max(upper_whiskers)
        lower_whisker_min = min(lower_whiskers)
        y_range = upper_whisker_max - lower_whisker_min
        y_pos = upper_whisker_max + 0.03 * y_range  # Adjust the y_pos for the plot

    return y_pos, y_range




def annotate_pvalues(ax, pval_df, group_order, y_pos, y_range):
    """Annotate violin plot with significance bars at incrementally moved y_pos."""
    offset = 0.08 * y_range  
    last_y_pos = y_pos
    for i, row in pval_df.iterrows():
        g1, g2, significance = row["Group 1"], row["Group 2"], row["Significance"]
        if significance:
            x1, x2 = group_order.index(g1), group_order.index(g2)
            ax.plot([x1, x1, x2, x2], [last_y_pos, last_y_pos + 0.02 * y_range , last_y_pos + 0.02 * y_range , last_y_pos], lw=1.2, c="black")
            ax.text((x1 + x2) / 2, last_y_pos + 0.01 * y_range , significance, ha="center", va="bottom", fontsize=12, color="black")
            last_y_pos += offset  




def violin_boxplot_with_pvalue(df, feature, group_label, group_order, group_colors, pval_df=None, save_pt=None, ax=None, figsize= (8, 6)):
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df_filtered = df.groupby(group_label, group_keys=False).apply(
        lambda g: g[g[feature].between(g[feature].quantile(0.01), g[feature].quantile(0.99))]
    )

    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # Use the existing figure if ax is passed
        
    sns.violinplot(data=df_filtered, x=group_label, y=feature, hue=group_label,
               order=group_order, palette=group_colors, inner=None, density_norm="area",
               width=0.8, cut=0, ax=ax)
    
    sns.boxplot(
        data=df_filtered, x=group_label, y=feature, order=group_order, 
        showfliers=True, width=0.2, 
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "black", "markeredgecolor": "black", "alpha": 0.8},
        boxprops={"facecolor": "white", "alpha": 0.7, "edgecolor": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        ax=ax
    )

    
    if pval_df is not None:
        y_pos, y_range = compute_whisker_limits(df_filtered, feature, group_label, group_order, showfliers=True)
        annotate_pvalues(ax, pval_df.loc[pval_df['Feature'] == feature, :].copy(), group_order, y_pos, y_range)

    
    ax.set_xlabel("Group", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Violin & Box Plot of Attention Value: {feature}", fontsize=12)
    if save_pt is not None:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)

    return fig  




def boxplot_with_pvalue(df, feature, group_label, group_order, group_colors, pval_df=None, save_pt=None, ax=None, figsize=(8, 6), box_width=0.5):
    df[feature] = pd.to_numeric(df[feature], errors="coerce")  # Ensure numeric values
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  

    df_filtered = df.groupby(group_label, group_keys=False).apply(
        lambda g: g[g[feature].between(g[feature].quantile(0.01), g[feature].quantile(0.99))]
    )

    color_list = [group_colors[cl] for cl in group_order if cl in group_colors]

    sns.boxplot(
        data=df_filtered, x=group_label, y=feature, order=group_order, 
        palette=color_list,  # Use mapped colors
        showfliers=False, width=box_width, 
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "black", "markeredgecolor": "black", "alpha": 0.8},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        ax=ax
    )

    if pval_df is not None:
        y_pos, y_range = compute_whisker_limits(df_filtered, feature, group_label, group_order, showfliers=False)
        annotate_pvalues(ax, pval_df.loc[pval_df['Feature'] == feature, :].copy(), group_order, y_pos, y_range)

    ax.set_xlabel("Group", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Box Plot of {feature}", fontsize=12)

    if save_pt is not None:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)

    return fig




def lobulemask_fromAnnotation(wsi_path=None, anno_pt=None):
    slide = openslide.OpenSlide(wsi_path)
    
    with open(anno_pt, "r") as f:
        shapes = json.load(f)

    level = slide.level_count - 1
    scale_factor = 1 / slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]

    mask = np.full((height, width, 3), np.nan, dtype=np.float32)

    class_colors = {
        "3": (33, 103, 172),  # Blue
        "2": (191, 212, 179), # Light Green
        "1": (179, 31, 44)    # Red
    }

    for shape in shapes.get("features", []):
        try:
            points = np.array(shape["geometry"]["coordinates"][0], dtype=np.float32)
            points *= scale_factor
            points = points.astype(int)

            cls = shape["properties"].get("classification", {}).get("name", "")
            color = class_colors.get(cls, (0, 0, 0))  # Default to black if class is unknown
            
            if cls in class_colors:
                cv2.drawContours(mask, [points], -1, color=color, thickness=1)
                cv2.fillPoly(mask, [points], color=color)
            else:
                print(f"[WARNING] Unknown class '{cls}' in annotation file. Skipping...")

        except Exception as e:
            print(f"[ERROR] Failed to process annotation: {e}")

    mask[np.isnan(mask)] = 0
    mask = mask.astype(np.uint8)

    return mask





def plot_branch_attention_heatmap(df, branch, upscale_factor=64, ax=None):
    df = df.dropna(subset=['coord_X', 'coord_Y']).copy()
    df[f'attention_{branch}'] = pd.to_numeric(df[f'attention_{branch}'], errors='coerce')
    attention = df[f'attention_{branch}'].values

    img_width = int(df['coord_X'].max()) + 1
    img_height = int(df['coord_Y'].max()) + 1
    img = np.full((img_height, img_width), np.nan)
    for x, y, a in zip(df['coord_X'], df['coord_Y'], attention):
        img[int(y), int(x)] = a  

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = None

    abs_max = np.nanmax(np.abs(attention))
    vmin, vmax = -abs_max, abs_max
    img_upscaled = zoom(img, upscale_factor, order=1)  # Bilinear interpolation

    norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)
    im = ax.imshow(img_upscaled, cmap='seismic', norm=norm, interpolation='bilinear')

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, aspect=16)
    tick_values = np.array([-1, -0.1, 0, 0.1, 1])  
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(["-1", "-0.1", "0", "0.1", "1"])

    cbar.set_label('Attention Value', fontsize=8)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(8)
    ax.set_title(f'Attention Heatmap (Branch {branch})', fontsize=10)
    ax.axis('off')

    if created_fig:
        plt.tight_layout()
        return fig
    return None




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




def train_kmeans(reference, n_clusters=4):
    embedding_columns = [f'embedding_{i}' for i in range(512)]  # Adjust if necessary
    reference = reference.copy()
    reference[embedding_columns] = reference[embedding_columns].apply(pd.to_numeric, errors='coerce')
    reference.dropna(subset=embedding_columns, inplace=True)
    assert reference[embedding_columns].isnull().sum().sum() == 0, "Reference data contains NaN values."
    X_train = reference[embedding_columns].values
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_model.fit(X_train)
    return kmeans_model



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




def highlight_a_WSI_in_tsne(new_df, reference, cluster_colors, save_pt):
    embedding_columns = [f'embedding_{i}' for i in range(512)]
    X_train = np.array(reference.loc[:, embedding_columns])
    X_new = new_df[embedding_columns].values
    
    pca = PCA(n_components=50)
    combined_data_pca = pca.fit_transform(np.vstack([X_train, X_new]))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    combined_tsne = tsne.fit_transform(combined_data_pca)
    
    reference_tsne = combined_tsne[:len(reference), :]
    new_data_tsne = combined_tsne[len(reference):, :]
    new_df['tsne1'] = new_data_tsne[:, 0]
    new_df['tsne2'] = new_data_tsne[:, 1]
    reference['tsne1'] = reference_tsne[:, 0]
    reference['tsne2'] = reference_tsne[:, 1]
    
    marker_styles = {'type1': 'o', 'type2': 'o', 'type3': 'o'}  # Type 1 = Square, Type 2 = Triangle, Type 3 = Diamond
    lobule_colors = {'type1': '#B31F2C', 'type2': '#BFD4B3', 'type3': '#2167AC'}

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_reference = ax.scatter(reference['tsne1'], reference['tsne2'],
                                    c=[cluster_colors[label] for label in reference['Cluster']], 
                                    alpha=0.05, label='Reference', s=10)
    
    for lobule_type, marker in marker_styles.items():
        subset = new_df[new_df['lobule_type'] == lobule_type]
        ax.scatter(subset['tsne1'], subset['tsne2'], 
                   c=[lobule_colors[label] for label in subset['lobule_type']], 
                   marker=marker, 
                   label=f'Lobule Type {lobule_type}', 
                   s=20, alpha=0.7, 
                   edgecolors='black', 
                   linewidths=0.2)
    
    ax.set_title('t-SNE with Cluster and Lobule Type')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend()
    
    if save_pt:
        fig.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    return new_df, reference




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
    patch_size = int(np.unique(WSI_df["patch_size"]))
    wsi = openslide.OpenSlide(wsi_path)
    level_dimensions = wsi.level_dimensions[level]
    wsi_img = wsi.read_region((0, 0), level, level_dimensions).convert("RGBA")
    scale_factor = wsi.level_downsamples[level]  

    mpp = float(wsi.properties.get('openslide.mpp-x', 0)) * scale_factor  
    if mpp == 0:  
        raise ValueError("Microns per pixel (mpp) information is missing in the WSI metadata.")
    
    scale_bar_length_mm = 2  
    scale_bar_length_px = int((scale_bar_length_mm * 1000) / mpp)  

    draw = ImageDraw.Draw(wsi_img)
    box_size = int(patch_size / scale_factor)
    fill_opacity = 255  
    border_width = 0  

    for _, row in WSI_df.iterrows():
        x_orig, y_orig = row['x_orig'], row['y_orig']
        cluster = row['Cluster']
        cluster_color = cluster_colors.get(cluster, '#000000')
        rgb_color = mcolors.hex2color(cluster_color)
        rgba_color = tuple(int(c * 255) for c in rgb_color) + (fill_opacity,)

        x_scaled = x_orig / scale_factor
        y_scaled = y_orig / scale_factor

        draw.rectangle([x_scaled - box_size / 2, y_scaled - box_size / 2,
                        x_scaled + box_size / 2, y_scaled + box_size / 2], 
                       fill=rgba_color, outline="black")

    margin = 50  
    x_start = margin
    y_start = wsi_img.height - margin
    x_end = x_start + scale_bar_length_px
    y_end = y_start + 10  
    draw.rectangle([x_start, y_start, x_end, y_end], fill="black", outline="black")
    
    # # Add text label (2mm)
    # font_size = 50
    # draw.text((x_start, y_start - font_size - 5), "2mm", fill="black")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wsi_img)
    ax.axis('off')

    cmap = mcolors.ListedColormap([cluster_colors[i] for i in range(4)])
    norm = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=4)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical', 
                        shrink=0.5, fraction=0.02, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels([0, 1, 2, 3])

    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')

    plt.show()
    
    return WSI_df, wsi_img



def plot_cluster_proportion_for_a_WSI(WSI_df, save_pt=None, normalize=False):
    phenotype_counts = WSI_df['Cluster'].value_counts().sort_index()
    ordered_phenotypes = [0, 1, 2, 3]

    for phenotype in ordered_phenotypes:
        if phenotype not in phenotype_counts.index:
            phenotype_counts[phenotype] = 0

    phenotype_counts = phenotype_counts[ordered_phenotypes].sort_index()

    if normalize:
        phenotype_counts = phenotype_counts / phenotype_counts.sum()  # Convert to proportions

    cluster_colors = {0: '#8da0cb', 1: '#66c2a5', 2: '#ffd92f', 3: '#b3b3b3'}

    fig, ax = plt.subplots(figsize=(3, 3))
    
    bottom = np.zeros(1)  # Initialize stacking base
    for phenotype in ordered_phenotypes:
        ax.bar(0, phenotype_counts[phenotype], color=cluster_colors[phenotype], label=f'P{phenotype}', bottom=bottom)
        bottom += phenotype_counts[phenotype]  # Update stacking position

    ax.set_title("Cluster Proportions for WSI" if normalize else "Cluster Counts for WSI")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_xticks([0])
    ax.set_xticklabels(["WSI"])
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    
    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')

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




def plot_stacked_bar_and_age_group_annotation(df, figsize=(12, 10), save_pt=None):
    cluster_counts = df.groupby(['wsi_id', 'Cluster']).size().reset_index(name='Count')
    total_counts = cluster_counts.groupby('wsi_id')['Count'].transform('sum')
    cluster_counts['Proportion'] = cluster_counts['Count'] / total_counts
    cluster_pivot = cluster_counts.pivot_table(index='wsi_id', columns='Cluster', values='Proportion', aggfunc='sum')
    wsi_order = df[['wsi_id', 'age']].drop_duplicates().sort_values('age')['wsi_id']
    cluster_pivot = cluster_pivot.loc[wsi_order]
    cluster_pivot = cluster_pivot[[0, 1, 3, 2]]  # 0->P0, 1->P1, 3->P3, 2->P2

    cluster_colors = {0: '#8da0cb', 1: '#66c2a5', 2: '#ffd92f', 3: '#b3b3b3'}
    age_group_colors = {0: '#262262', 1: '#87ACC5', 2: '#00A261', 3: '#FFF200'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [0.2, 0.8]})
    
    age_groups = df[['wsi_id', 'age_group']].drop_duplicates().sort_values('wsi_id')
    age_groups = age_groups.set_index('wsi_id').loc[wsi_order].reset_index()
    age_group_numeric = age_groups['age_group'].map({0: 0, 1: 1, 2: 2, 3: 3})
    age_group_matrix = pd.DataFrame(age_group_numeric.values, columns=['Age Group'], index=age_groups['wsi_id'])

    custom_cmap = ListedColormap([age_group_colors[i] for i in age_group_colors.keys()])

    sns.heatmap(age_group_matrix.T, cmap=custom_cmap, cbar=False, linewidths=0, linecolor='white', 
                xticklabels=False, yticklabels=False, ax=ax1)
    ax1.set_xlabel('WSI', fontsize=10)
    ax1.set_title('Age Group Annotations (Heatmap)', fontsize=12)

    cluster_pivot.plot(kind='bar', stacked=True, color=[cluster_colors[i] for i in cluster_pivot.columns], width=0.95, ax=ax2)
    ax2.set_xlabel("WSI", fontsize=12)
    ax2.set_ylabel("Proportion of P0-P3 by Cluster", fontsize=12)
    ax2.set_title("Stacked Barplot of Proportions by Cluster for Each WSI", fontsize=16)
    ax2.legend(title="Cluster", labels=["P0", "P1", "P3", "P2"], bbox_to_anchor=(1, 1))

    plt.tight_layout()
    if save_pt:
        plt.savefig(save_pt, bbox_inches='tight')
        print(f"Plot saved at {save_pt}")
    else:
        plt.show()




# def Cluster_proportion_heatmap(tsne_df, save_pt=None):
#     proportions = tsne_df.groupby(['age_group', 'Cluster']).size().unstack(fill_value=0)
#     proportions = proportions.div(proportions.sum(axis=1), axis=0)  # Normalize to 100%
#     proportions = proportions.reset_index().melt(id_vars="age_group", value_vars=proportions.columns, var_name="Cluster", value_name="proportion")
    
#     heatmap_data = proportions.pivot(index="age_group", columns="Cluster", values="proportion")
#     desired_order = ['P0', 'P1', 'P2', 'P3']
#     available_columns = [col for col in desired_order if col in heatmap_data.columns]
#     if len(available_columns) == len(desired_order):
#         # Reorder the columns
#         heatmap_data = heatmap_data[available_columns]
#     else:
#         print("Mismatch in columns. Available columns:", heatmap_data.columns)

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Proportion'}, fmt='.2f',
#                 annot_kws={'size': 16})
#     plt.title("Heatmap of Cluster Proportions by Age Group", fontsize=12)
#     plt.xlabel("Cluster", fontsize=12)
#     plt.ylabel("Age Group", fontsize=12)
#     plt.tight_layout()
#     if save_pt is not None:
#         plt.savefig(fname=save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
#     plt.show()




# def Cluster_proportion_barplot(tsne_df, save_pt):
#     proportions = tsne_df.groupby(['age_group', 'Cluster']).size().unstack(fill_value=0)
#     proportions = proportions.div(proportions.sum(axis=1), axis=0)  # Normalize to 100%
#     proportions = proportions.reset_index().melt(id_vars="age_group", value_vars=proportions.columns, 
#                                                  var_name="Cluster", value_name="proportion")
    
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=proportions, x='age_group', y='proportion', hue='Cluster', palette='Set2')
#     plt.title("Proportional Barplot of Cluster Proportions by Age Group", fontsize=14)
#     plt.xlabel("Age Group", fontsize=12)
#     plt.ylabel("Proportion", fontsize=12)
#     plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     if save_pt is not None:
#         plt.savefig(save_pt, format="pdf", dpi=300)
#     plt.show()



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


    

def plot_multiple(img_list, caption_list=None, grid_x=4, grid_y=4, figure_size=(10, 10), title=None, save_pt=None):
    fig, axes = plt.subplots(grid_y, grid_x, figsize=figure_size)
    axes = axes.flatten()

    # Loop through the images and plot them on the subplots
    for i, img in enumerate(img_list):
        axes[i].imshow(img)
        axes[i].axis('off')  # Hide axes
        if caption_list:
            axes[i].set_title(str(caption_list[i]), fontsize=8)

    # Turn off axes for any unused subplots
    for j in range(len(img_list), len(axes)):
        axes[j].axis('off')

    # Set the overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    # Adjust the layout and make sure there's no clipping
    plt.tight_layout()

    # Save or display the figure
    if save_pt:
        plt.savefig(save_pt, bbox_inches='tight', dpi=300)
        print(f"[INFO] Saved plot to {save_pt}")
        plt.close(fig)
    else:
        plt.show()

    return fig




def sample_k_patches_showStainGeneralisation(wsi, patch_ids, num=25, aug=False):
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

