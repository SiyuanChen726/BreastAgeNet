# System and Path Management
import os
from pathlib import Path
import sys
from tqdm import tqdm

# Scientific Computing
import numpy as np
import pandas as pd

# Image Processing
from PIL import Image
import openslide
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning and Data Manipulation
from sklearn.manifold import TSNE
import h5py
import random

# Custom Modules
# sys.path.append('/scratch/users/k21066795/prj_normal/RandStainNA')
# from randstainna import RandStainNA

    
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader



def plot_oneline(img_list, caption_list, figure_size, save_pt=None):
    """
    Plots a single row of images with captions.

    Parameters:
    - img_list (list of np.array or images): List of images to display.
    - caption_list (list of str or float): List of captions or values to display as titles for each image.
    - figure_size (tuple): Size of the entire figure (width, height).
    - save_pt (str, optional): Path to save the figure as an image. If None, the figure won't be saved.
    """
    # Create the figure and axes objects.
    fig, axes = plt.subplots(1, len(img_list), figsize=figure_size)
    
    # Iterate through the images and display them.
    for index in range(len(img_list)):
        axes[index].imshow(img_list[index])
        axes[index].axis("off")  # Hide axis for a cleaner image display
        
        caption_i = caption_list[index]
        
        # Set title based on caption type (string or numeric).
        if isinstance(caption_i, str):
            axes[index].set_title(f"{caption_i}")
        else:
            axes[index].set_title(f"{np.around(caption_i, 2)}")

    # Save the figure if the path is provided.
    if save_pt is not None:
        plt.savefig(save_pt, pad_inches=0, bbox_inches="tight", dpi=300)
    
    # Show the plot if it's not being saved.
    plt.show()


    
    
def plot_multiple(img_list, caption_list, grid_x, grid_y, figure_size, title=None, save_pt=None):
    """
    Plots multiple images in a grid format with optional captions.

    Parameters:
    - img_list (list of np.array or images): List of images to display.
    - caption_list (list of str or float): List of captions or values to display as titles for each image.
    - grid_x (int): Number of rows in the grid.
    - grid_y (int): Number of columns in the grid.
    - figure_size (tuple): Size of the entire figure (width, height).
    - title (str, optional): Title for the entire figure. Default is None.
    - save_pt (str, optional): Path to save the figure as an image. If None, the figure won't be saved.
    """
    # Create the figure and axes objects.
    fig, axes = plt.subplots(grid_x, grid_y, figsize=figure_size)

    # Set the figure title if provided.
    if title is not None:
        plt.suptitle(title, fontsize=16)

    # Iterate through the grid and plot images with captions.
    counter = 0
    for x in range(grid_x):
        for y in range(grid_y):
            if counter < len(img_list):
                axes[x][y].imshow(img_list[counter])
                axes[x][y].axis("off")  # Hide axis for a cleaner image display
                
                if caption_list is not None and counter < len(caption_list):
                    axes[x][y].set_title(f"{caption_list[counter]}")
                counter += 1
            else:
                axes[x][y].axis("off")  # Hide unused subplots

    # Save the figure if the path is provided.
    if save_pt is not None:
        plt.savefig(save_pt, pad_inches=0, bbox_inches="tight", dpi=300)
    
    # Show the plot if it's not being saved.
    plt.show()

    
    


def get_xy(patch_id):
    """
    Extracts the x, y coordinates on the WSI (40x by default) and patch size from a patch id.

    Parameters:
    - patch_id (str): A string in the format '..._{grid_x}_{grid_y}_{patch_size}', where grid_x, grid_y, and patch_size are integers.

    Returns:
    - tuple: (x, y, patch_size), where x and y are the pixel coordinates, and patch_size is the size of the patch.
    """
    try:
        # Split the patch_id to extract coordinates and patch size.
        x, y, patch_size = patch_id.split("_")[-3:]
        x, y, patch_size = int(x), int(y), int(patch_size)
        return x * patch_size, y * patch_size, patch_size
    except ValueError as e:
        raise ValueError(f"Invalid patch_id format: {patch_id}. Expected format '..._x_y_patch_size'.") from e




def sample_k_patches(wsi, patch_ids, num=25, aug=False):
    """
    Samples 'num' patches from a WSI (Whole Slide Image) based on provided patch IDs.
    
    Optionally applies augmentation to the sampled patches.

    Parameters:
    - wsi (OpenSlide object): The WSI object to read patches from.
    - patch_ids (list of str): List of patch identifiers to sample from.
    - num (int): The number of patches to sample. Defaults to 25.
    - aug (bool): Whether to apply augmentation to the patches. Defaults to False.

    Returns:
    - img_list (list of PIL Image objects): List of sampled patches as PIL Image objects.
    - Augimg_list (list of PIL Image objects, optional): List of augmented patches (if `aug=True`).
    """
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


    # Shuffle the list of patch IDs to ensure randomness
    random.shuffle(patch_ids)

    # Determine the number of patches to sample (min between requested and available patches)
    lens = min(len(patch_ids), num)

    # Loop through the selected patches
    for i in range(lens):
        patch_id = patch_ids[i]
        
        # Get the x, y coordinates and patch size using the patch ID
        x, y, patch_size = get_xy(patch_id)
        
        # Read the region from the WSI at the given coordinates and size
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

    # Return only img_list if no augmentation is required
    if not aug:
        return img_list
    return img_list, Augimg_list



def plot_tsne(fea_df, color='age_group', point_size=3, vmin=None, vmax=None, save_pt=None):
    """
    Creates a t-SNE plot colored by a specified feature.

    Parameters:
    - fea_df (DataFrame): DataFrame containing the t-SNE components and the coloring feature.
    - color (str): Column name in `fea_df` used for coloring the points. Can be 'age_group' or 'cluster'.
    - save_pt (str, optional): If provided, the plot will be saved to this file path.
    """
    plt.figure(figsize=(10, 6))
    
    # Handle color based on 'age_group' or 'cluster'
    if color == "age_group":
        cmap = plt.get_cmap('viridis', fea_df['age_group'].nunique())  # Use 'viridis' colormap for age group
    elif color == "Cluster":
        cmap = plt.get_cmap('Set3')
        indices = [0, 4, 8, 11]  # Indices of the colors to retrieve
        colors = [cmap(i / 11) for i in indices]  # 'Set3' has 12 colors, so divide by 11
        cmap = LinearSegmentedColormap.from_list("CustomSet3", colors)
    elif "attention" in color:
        fea_df[color] = pd.to_numeric(fea_df[color], errors='coerce') 
        cmap = "viridis"

    # Ensure there are no NaN values in the color column for plotting
    if fea_df[color].isnull().any():
        print(f"Warning: NaN values found in {color}. These will be excluded from the plot.")
        fea_df = fea_df.dropna(subset=[color])
    
    if vmin is None:
        vmin = fea_df[color].min()  # or a specific value, e.g., 0
    if vmax is None:
        vmax = fea_df[color].max()  # or a specific value, e.g., 1
    # Create scatter plot
    scatter = plt.scatter(
        fea_df["tsne1"], 
        fea_df["tsne2"], 
        c=fea_df[color], 
        cmap=cmap, 
        alpha=0.7, 
        s=point_size,
        vmin=vmin,  # Set minimum color scale value
        vmax=vmax   # Set maximum color scale value
    )
    
    # Add a color bar
    plt.colorbar(scatter, label=f'{color}')
    
    # Set plot titles and labels
    plt.title(f't-SNE Visualization Colored by {color}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # If save_pt is provided, save the plot
    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    # Display the plot
    plt.show()

    


def plot_density_by_age(fea_df, max_categories=4, save_pt=None):
    """
    Plot density plots based on t-SNE results for different age categories.

    Parameters:
    - fea_df (DataFrame): DataFrame containing the t-SNE components and category codes.
    - max_categories (int): Maximum number of unique categories to plot. Default is 4 (for 2x2 grid).
    - save_pt (str, optional): If provided, the plot will be saved to this file path.
    """
    # Set up the figure with subplots (2x2 by default)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.set(style="white")

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the categories and create a density plot for each one
    for i in range(max_categories):
        category = unique_categories[i]
        subset = fea_df[fea_df['age_group'] == category]

        sns.kdeplot(
            x=subset["tsne1"],
            y=subset["tsne2"],
            fill=True,
            alpha=0.5,
            ax=axes[i],
            cmap="viridis"
        )

        # Set subplot titles and labels
        axes[i].set_title(f"Density Plot for {category}")
        axes[i].set_xlabel("t-SNE 1")
        axes[i].set_ylabel("t-SNE 2")


    # Adjust layout and add a main title
    plt.tight_layout()
    plt.suptitle("t-SNE Density Plots by Category", y=1.02)

    # Save the plot if a save path is provided
    if save_pt:
        plt.savefig(save_pt, dpi=300, bbox_inches='tight', pad_inches=0.1)

    # Show the plot
    plt.show()




def parse_wsi_id(patch_id):
    """
    Parses the WSI ID from a given patch ID based on different cohorts' naming.
    
    Parameters:
    - patch_id (str): The patch identifier from which to extract the WSI ID.
    
    Returns:
    - wsi_id (str): The extracted WSI ID.
    """
    
    if " HE" in patch_id: # NKI
        wsi_id = patch_id.split(" HE")[0]
    elif "_FPE_" in patch_id: # KHP
        wsi_id = "_".join(patch_id.split("_")[:3])
    else: # rest
        wsi_id = patch_id.split("_")[0]
    
    return wsi_id



def paste_HE_on_tsne(fea_df, WSI_folder, max_dim=100, n_samples=1000, image_size=(4000, 3000), random_state=42):
    """
    Generates a composite image of H&E-stained tissue patches arranged according to their t-SNE projections.
    
    Parameters:
    - fea_df (pd.DataFrame): DataFrame containing t-SNE projections ('tsne1', 'tsne2') and patch ids.
    - WSI_folder (str): Path to the folder containing the WSI files.
    - max_dim (int): Maximum dimension to resize patches to for the final image (default is 100).
    - n_samples (int): Number of patches to sample per cluster (default is 1000).
    - image_size (tuple): Size of the final image canvas (default is (4000, 3000)).
    - random_state (int): Random seed for reproducibility (default is 42).
    
    Returns:
    - full_image (PIL.Image): The resulting composite image with patches placed according to their t-SNE positions.
    """
    
    # Ensure reproducibility by setting the random seed
    random.seed(random_state)
    
    # Sample the dataframe by clusters
    random_df = fea_df.groupby('Cluster').sample(n=n_samples, replace=True, random_state=random_state)

    # Normalize t-SNE projections to [0, 1] for consistent positioning
    tx, ty = random_df["tsne1"], random_df["tsne2"]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    # Initialize the full image (canvas)
    width, height = image_size
    full_image = Image.new('RGBA', (width, height))

    # Process each patch, extract, resize, and paste onto the canvas
    for patch_id, tsne_x, tsne_y in zip(random_df.index, tx, ty):
        wsi_id = parse_wsi_id(patch_id)
        wsi_path = f"{WSI_folder}/*/{wsi_id}*.*"
        
        try:
            # Open the WSI using OpenSlide and extract the patch
            wsi = openslide.OpenSlide(wsi_path)
            x, y, patch_size = get_xy(patch_id)
            im = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        except Exception as e:
            print(f"Error processing WSI {wsi_id} for patch {patch_id}: {e}")
            continue  # Skip this patch if there's an issue

        # Resize the tile to fit within max_dim
        tile = Image.fromarray(np.array(im))
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)))

        # Calculate the position to paste the tile on the full image
        x_pos = int((width - max_dim) * tsne_x)
        y_pos = int((height - max_dim) * tsne_y)

        # Paste the resized tile onto the full image at the appropriate position
        full_image.paste(tile, (x_pos, y_pos), mask=tile.convert('RGBA'))

    return full_image




class WSIDataset(Dataset):
    def __init__(self, df, bag_size):
        """
        Initializes the dataset.

        Args:
            df (pd.DataFrame): A DataFrame containing the data.
            bag_size (int): The size of each bag (number of patches per sample).
        """
        self.data = df.values  # The patch data (features)
        self.patch_ids = df.index  # The patch IDs (from DataFrame index)
        self.bag_size = bag_size  # The bag size (how many patches in each bag)
        self.num_patches = self.data.shape[0]  # Total number of patches

    
    def __len__(self):
        """
        Returns the number of bags in the dataset.

        A "bag" is a collection of patches. If the total number of patches
        is not an exact multiple of the bag size, the last bag will contain
        fewer patches.
        """
        return int(np.ceil(self.num_patches / self.bag_size))

    
    def __getitem__(self, idx):
        """
        Returns a bag of patches and their corresponding patch IDs for a given index.

        Args:
            idx (int): Index of the bag (not the individual patch).
        
        Returns:
            patch_ids (np.array): Array of patch IDs for the bag.
            patches (torch.Tensor): Tensor of patches for the bag.
        """
        start_idx = idx * self.bag_size  # Start index for the bag
        end_idx = min(start_idx + self.bag_size, self.num_patches)  # End index for the bag
        patches = self.data[start_idx:end_idx]  # Get the patches for the current bag
        patch_ids = self.patch_ids[start_idx:end_idx]  # Get the patch IDs for the current bag

        # Pad if not enough patches to form a full bag
        if len(patches) < self.bag_size:
            padding = np.zeros((self.bag_size - len(patches), patches.shape[1]))  # Pad with zeros
            patches = np.vstack([patches, padding])  # Add padding to patches
            patch_ids = np.append(patch_ids, [''] * (self.bag_size - len(patch_ids)))  # Add empty strings for patch IDs

        # Convert patch_ids to a list (or numpy array)
        patch_ids = list(patch_ids)  # Convert to a list (or np.array)

        # Return patch_ids as a list and patches as a tensor
        return patch_ids, torch.tensor(patches, dtype=torch.float32)

 

def WSI_loader(df, batch_size=256, bag_size=250, shuffle=False):
    """
    Creates a DataLoader for the WSIDataset.

    Args:
        df (pd.DataFrame): A DataFrame containing the data.
        batch_size (int): The batch size for the DataLoader.
        bag_size (int): The number of patches per bag.
        shuffle (bool): Whether or not to shuffle the data.

    Returns:
        DataLoader: A PyTorch DataLoader for the WSIDataset.
    """
    dataset = WSIDataset(df, bag_size)  # Create the custom dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # Create DataLoader
    return dataloader



def get_WSIheatmap_df(model, patch_fea, patch_info):

    with h5py.File(patch_fea, "r") as file:
        bag = np.array(file["embeddings"])
        bag = np.squeeze(bag)
        img_id = np.array(file["patch_id"])
    img_id = [i.decode("utf-8") for i in img_id]
    bag_df = pd.DataFrame(bag)
    bag_df.index = img_id

    df = pd.read_csv(patch_info)
    valid_ids = list(df['patch_id'][df['cls'] == "epithelium"])

    bag_df = bag_df.loc[bag_df.index.isin(valid_ids), :]

    wsiloader = WSI_loader(bag_df, batch_size=4, bag_size=250)
    phase = 'test'
    model.eval()

    predicted_ranks = []
    df = pd.DataFrame()  # Initialize empty DataFrame
    for patch_ids, inputs in tqdm(wsiloader):
        patch_ids = np.array(patch_ids)
        patch_ids = np.transpose(patch_ids)
        patch_ids = patch_ids.flatten()
    
        with torch.set_grad_enabled(phase == 'train'):
            logits, embeddings, attentions = model(inputs)
            attentions = attentions.view(-1, attentions.shape[-1])  # Flatten attentions
            embeddings = embeddings.view(-1, embeddings.shape[-1])  # Flatten embeddings
            probs = torch.sigmoid(logits)  # Shape: [batch_size, n_classes]
            binary_predictions = (probs > 0.5).int()  # Shape: [batch_size, n_classes]
            ranks = binary_predictions.sum(dim=1)  # Shape: [batch_size]
            predicted_ranks += ranks.tolist()  # Accumulate predicted ranks
    
        combined_data = np.column_stack((patch_ids, embeddings.cpu().numpy(), attentions.cpu().numpy())) 
        dfi = pd.DataFrame(combined_data, columns=['patch_id'] + [f'embedding_{i}' for i in range(embeddings.shape[1])] + [f'attention_{i}' for i in range(attentions.shape[1])]) 
        df = pd.concat([df, dfi], axis=0)  # Append new data to the main dataframe

        coord_X, coord_Y = [], []
        
        for patch_id in df["patch_id"]:
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
        
        df["coord_X"], df["coord_Y"] = coord_X, coord_Y

    return df

    
    

def get_branch_attn(df, branch):
    # Handle NaN values by dropping rows where 'coord_X' or 'coord_Y' are NaN
    df = df.dropna(subset=['coord_X', 'coord_Y']).copy()
    
    # Ensure 'attention_{branch}' is numeric, coercing errors to NaN
    df[f'attention_{branch}'] = pd.to_numeric(df[f'attention_{branch}'], errors='coerce')
    
    # Get attention values and apply clipping if needed
    attention = df[f'attention_{branch}'].values
    
    # Determine the size of the canvas (image) based on coordinates
    img_width = int(np.max(df['coord_X'].values)) + 1
    img_height = int(np.max(df['coord_Y'].values)) + 1
    
    # Initialize the image with NaN background
    img = np.full((img_height, img_width), np.nan)
    
    # Map attention values to the corresponding coordinates
    for x, y, a in zip(df['coord_X'].values, df['coord_Y'].values, attention):
        img[int(y), int(x)] = a  # Attention values mapped to corresponding coordinates

    return img


import numpy as np
import matplotlib.pyplot as plt

def show_WSI_attnMap(img, branch, vmin=None, vmax=None, save_path=None):
    """
    Display the attention map as a heatmap and optionally save it.

    Parameters:
    - img: 2D numpy array representing the attention map.
    - branch: The branch associated with the attention map.
    - vmin: Minimum value for the color range. If None, will use the minimum value of the image.
    - vmax: Maximum value for the color range. If None, will use the maximum value of the image.
    - save_path: If provided, the heatmap will be saved at this location.
    """
    # Set default vmin and vmax if not provided
    if vmin is None:
        vmin = np.nanmin(img)  # Default to min if not provided
    if vmax is None:
        vmax = np.nanmax(img)  # Default to max if not provided

    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(6, 6))  # Keep the original size for the heatmap

    # Display the heatmap with the attention map
    im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)

    # Add a colorbar with adjusted size
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, aspect=12)
    cbar.set_label('Attention Value', fontsize=8)  # Set the font size for the colorbar label
    
    # Adjust the font size for the colorbar ticks
    cbar.ax.tick_params(labelsize=8)  # Consistent font size for tick labels
    
    # Title and axis formatting
    ax.set_title(f'Attention Heatmap for Branch {branch}', fontsize=10)
    ax.axis('off')  # Hide axes for better visualization
    
    # Apply tight layout for better spacing
    plt.tight_layout()

    # If a save path is provided, save the figure
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with 300 DPI and tight bounding box
            print(f"Figure saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    # Show the plot
    plt.show()




def lobulemask_fromAnnotation(wsi_path=None, anno_pt=None):
    slide = openslide.OpenSlide(wsi_path)
    
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
        
    return mask




def barplot_multiple_WSIs(clinic_df, save_pt):
    clinic_df_sorted = clinic_df.sort_values(by='age', ascending=True)
    pivot_df = clinic_df_sorted.groupby(['patient_id', 'final_prediction']).size().unstack(fill_value=0)
    patient_ids_sorted = clinic_df_sorted['patient_id'].drop_duplicates().values
    pivot_df = pivot_df.loc[patient_ids_sorted]
    ages = clinic_df_sorted.drop_duplicates(subset='patient_id')['age'].values
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Set the width of each bar (e.g., 0.9 instead of 0.8)
    pivot_df.plot(kind='bar', stacked=True, color=sns.color_palette("Set3", n_colors=len(pivot_df.columns)), ax=ax1, width=0.9)

    ax1.set_xlabel('Patient ID (Ordered by Age)', fontsize=12)
    ax1.set_ylabel('Count of Predicted Ranks', fontsize=12)
    
    # Optional: Add secondary y-axis with age (commented out in your original code)
    # ax2 = ax1.twinx()
    # max_bar_height = pivot_df.sum(axis=1).max()
    # scaled_ages = np.interp(ages, (ages.min(), ages.max()), (0, max_bar_height))
    # ax2.plot(range(len(patient_ids_sorted)), scaled_ages, color='black', marker='o', linestyle='-', linewidth=2, markersize=6, label='Scaled Patient Age')
    # ax2.set_ylabel('Age (Years)', fontsize=12, color='black')  # Label the secondary y-axis as "Age"
    # age_ticks = np.linspace(0, len(ages) - 1, len(ages))  # Create tick positions
    # ax2.set_yticks(np.linspace(0, max_bar_height, len(ages)))  # Adjust y-ticks
    # ax2.set_yticklabels(ages)  # Use the original, non-normalized ages for tick labels

    ax1.set_xticklabels(patient_ids_sorted, rotation=90)
    ax1.legend(title="Predicted Ranks", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_title('Stacked Predicted Ranks by Patient ID (Ordered by Age)', fontsize=14)
    plt.tight_layout()
    
    if save_pt is not None:
        plt.savefig(save_pt, format='pdf')
    
    plt.show()
