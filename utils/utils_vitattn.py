'''
adapted the code from https://github.com/facebookresearch/dino. 
'''
import os
import torch
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys
import openslide
import seaborn as sns
from PIL import Image
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import ImageGrid
from adjustText import adjust_text

from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import timm
from huggingface_hub import login, hf_hub_download

torch.multiprocessing.set_sharing_strategy('file_system')

from utils.utils_vis import plot_multiple
from utils.utils_features import Reinhard
import torch.nn as nn
from torchvision import transforms as pth_transforms



def crop_im(patch_im, crop_size = 224):
    center_x, center_y = patch_im.width // 2, patch_im.height // 2
    left = center_x - crop_size // 2
    upper = center_y - crop_size // 2
    right = center_x + crop_size // 2
    lower = center_y + crop_size // 2
    patch_im = patch_im.crop((left, upper, right, lower))
    return patch_im



def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward



def get_attentions_lastlayer(patch, threshold=0.6):
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join("/app/BreastAgeNet/weights/UNI/pytorch_model.bin"), map_location="cpu"), strict=True)
    transform = pth_transforms.Compose(
        [pth_transforms.Resize((256,256)),
         pth_transforms.CenterCrop((224, 224)),
         pth_transforms.ToTensor(),
         pth_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # add attn outputs
    model.blocks[-1].attn.forward = my_forward_wrapper(model.blocks[-1].attn)
    model = model.eval()
    im_norm = Image.fromarray(Reinhard(np.array(patch)))
    input_im = transform(im_norm)
    pixel_values = torch.unsqueeze(input_im, axis=0)
    y = model(pixel_values)
    attentions = model.blocks[-1].attn.attn_map
    nh = attentions.shape[1] # number of head
    
    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(attentions.shape)
    w_featmap = pixel_values.shape[-2] // 16
    h_featmap = pixel_values.shape[-1] // 16
    
    # keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > threshold
    idx2 = torch.argsort(idx)
    
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu()
    attentions = attentions.detach().numpy()
    
    return th_attn, attentions



def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image



def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors



def display_instances(image, th_attn, head_id, alpha=0.5):
    """
    Given thresholded attention maps, plot an overlay of attention_head_id on the original image
    """
    N = 1
    mask = th_attn[head_id]
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    masked_image = image.astype(np.uint32).copy()
    
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        
    return masked_image.astype(np.uint8)



def get_attn_overlay(patch, th_attn, cmap, save_pt):
    image = np.array(patch.resize((224, 224)))
    overlays = []
    for head_i in range(12):
        overlay_i = display_instances(image, th_attn, head_i, alpha=0.5)
        overlays.append(overlay_i)

    plot_multiple(img_list = overlays, 
              caption_list= [f"head_{i}" for i in range(12)], 
              grid_x=3, grid_y=4, figure_size=(10, 8), cmap=cmap, save_pt=save_pt)

