import io
import numpy as np
import random
import json
import cv2
import openslide
import umap
import umap.plot
from PIL import Image
import matplotlib.pyplot as plt
from utils_features import ensemble_feas_labels
    



def plot_multiple(img_list, caption_list, grid_x, grid_y, figure_size, cmap, save_pt=None):
    fig, axes = plt.subplots(grid_x, grid_y, figsize=figure_size)
    counter = 0
    for x in range(0, grid_x):
        for y in range(0, grid_y):
            axes[x][y].imshow(img_list[counter],cmap=cmap)
            axes[x][y].axis("off")
            axes[x][y].set_title(f"{caption_list[counter]}")
            counter += 1
    if save_pt is not None:
        plt.title(os.path.basename(save_pt).split(".png")[0])
        plt.savefig(save_pt, pad_inches=0, bbox_inches="tight")



def plot_oneline(img_list, caption_list, figure_size, save_pt=None):
    fig, axes = plt.subplots(1, len(img_list), figsize=figure_size)
 
    for index in range(0, len(img_list)):
            axes[index].imshow(img_list[index])
            axes[index].axis("off")
            caption_i = caption_list[index]
            
            if isinstance(caption_i, str):
                axes[index].set_title(f"{caption_i}")
            else:
                axes[index].set_title(f"{np.around(caption_i, 2)}")
    if save_pt is not None:
        plt.savefig(save_pt,pad_inches=0, bbox_inches="tight", dpi=300)


def generate_mask(wsi_path=None, anno_pt=None):
    slide=openslide.OpenSlide(wsi_path)
    with open(anno_pt, "r") as f:
        shapes = json.load(f)
    
    level = slide.level_count - 1
    scale_factor = 1/slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]
    
    background = np.zeros((height, width, 3), np.uint8)
    mask = np.full((height, width, 3), background, dtype=np.uint8)
    
    for shape in shapes["features"]:
        points = shape["geometry"]["coordinates"][0]
        points = np.array([(p[0], p[1]) for p in points])
        points = points * scale_factor
        points = points.astype(int)
    
        cls = shape["properties"]["classification"]["name"]
        if cls == "epithelials":
          color = (255, 0, 0)
          mask = cv2.drawContours(mask, [points], -1, color=color, thickness=1)
          mask = cv2.fillPoly(mask, [points], color=color)
        elif cls == "stroma":
          color = (0, 255, 0)
          mask = cv2.drawContours(mask, [points], -1, color=color, thickness=1)
          mask = cv2.fillPoly(mask, [points], color=color)
        elif cls == "miscellaneous":
          color = (0, 0, 255)
          mask = cv2.drawContours(mask, [points], -1, color=color, thickness=1)
          mask = cv2.fillPoly(mask, [points], color=color)
        
    return mask
    
