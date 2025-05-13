import os
import sys
import glob
import random
import h5py
import staintools
import openslide
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from skimage.measure import regionprops_table
import stardist
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

from utils.utils_model import parse_wsi_id
from utils.utils_features import Reinhard
from utils.utils_vis import get_xy


def get_xy(patch_id):
    try:
        x, y, patch_size = patch_id.split("_")[-3:]
        x, y, patch_size = int(x), int(y), int(patch_size)
        return x * patch_size, y * patch_size, patch_size
    except ValueError as e:
        raise ValueError(f"Invalid patch_id format: {patch_id}. Expected format '..._x_y_patch_size'.") from e


def crop_im(img, new_width=448,new_height=448):
    width, height = img.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img




WSIs = '/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/WSIs'                                             # the folder saving WSIs
PATCH = "/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/patches/HE"                                      # the folder to save
NUCLEI = "/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/patches/startdist"                              # the folder to save images of nuclei segmentation
nuclei_csv = "/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/RESULTs/interpretation/train_NR/train_NR_nuclei_morphology.csv"   # the file to save nuclei morphology quantification results


# load cluster predictions of the train_NR dataset
df = pd.read_csv("/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/RESULTs/interpretation/train_NR_4clusters.csv", low_memory=False)

# randomly sample patches from each cluster
patch_num = 10000
for cluster_id in [0,1,2,3]:
    patch_ids = list(df.loc[df['Cluster'] == cluster_id, "patch_id"])
    print(len(patch_ids))
    random.shuffle(patch_ids)
    
    for patch_id in patch_ids[:patch_num]:
        save_pt = f"{PATCH}/{patch_id}.tif"
        print(save_pt)
        wsi_id = parse_wsi_id(patch_id)
        try:
            wsi_path = glob.glob(f"{WSIs}/*/{wsi_id}*.*")[0]
            wsi = openslide.OpenSlide(wsi_path)
            x, y, patch_size = get_xy(patch_id)
            im = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        except Exception as e:
            print(f"Error processing WSI {wsi_id} for patch {patch_id}: {e}")
            continue  

        # save the normalised patch
        patch_im_norm = Reinhard(np.array(im))
        patch_im_norm = Image.fromarray(patch_im_norm)
        patch_im_norm = crop_im(patch_im_norm)
        patch_im_norm.save(save_pt)



# nuclei segmentation and quantification
for patch_id in os.listdir(PATCH): 
    tar_pt = f'{NUCLEI}/{patch_id.replace(".tif", "_nu.tif")}'
    if not os.path.exists(tar_pt):
        try:
            print(tar_pt)
            patch_norm = Image.open(f"{PATCH}/{patch_id}")
            patch_im_norm = np.array(patch_norm)
            
            # apply StarDist to detect nuclei and save the mask
            model = StarDist2D.from_pretrained('2D_versatile_he')
            labels, details = model.predict_instances(normalize(patch_im_norm))
            seg_im = Image.fromarray(labels.astype(bool))
            seg_im.save(tar_pt)
            print(f"{tar_pt} saved!")
            # plot_contours(patch_im, details["coord"])
            
            # compute nuclei features
            nuclei_props_list = ["label", "area", "bbox_area", "extent", "eccentricity", "perimeter", "major_axis_length", "minor_axis_length", "solidity"]
            props = regionprops_table(labels, properties=nuclei_props_list)
            props_df = pd.DataFrame.from_dict(props)
            props_df["Circularity"] = (4*np.pi*props_df["area"]) / (props_df["major_axis_length"]**2)
            props_df["Elongation"] = props_df["major_axis_length"] / props_df["minor_axis_length"]
            props_df["patch_id"] = patch_id.split('.tif')[0]
            props_df["coord_x"] = details["points"][:,0]
            props_df["coord_y"] = details["points"][:,1]
            
            if not os.path.exists(nuclei_csv):
                props_df.to_csv(nuclei_csv, index=False)
            else:
                props_df.to_csv(nuclei_csv, mode="a", header=False, index=False)
        except:
            pass

