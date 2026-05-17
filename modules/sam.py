import os
import torch
from SAM_repo.segment_anything import sam_model_registry
import numpy as np
import matplotlib.pyplot as plt


def show_image(image, ax):
    ax.imshow(image)

    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        if color is None:
            color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_reshape = np.concatenate([np.ones([h, w, 3], dtype=mask.dtype), mask.reshape(h, w, 1)], axis=-1)
    
    mask_image = mask_reshape * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker='o', marker_size=80, linewidth=.5):
    neg_points = coords[labels==0]
    pos_points = coords[labels==1]
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker=marker, s=marker_size, edgecolor='white', linewidth=linewidth)   
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker=marker, s=marker_size, edgecolor='white', linewidth=linewidth)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='lime', facecolor=(0,0,0,0), lw=2))   


def get_img_name(img_path):
    img_name_w_suff = os.path.split(img_path)[-1]
    img_name = os.path.splitext(img_name_w_suff)[0]

    return img_name


def load_img_seg_model(sam_dir, sam_fname, sam_model_type, device=None):
    sam_checkpoint_path = os.path.join(sam_dir, sam_fname)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=device)

    return sam
