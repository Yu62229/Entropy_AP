import numpy as np
import os
import sys
from PIL import Image
from math import ceil
from scipy.stats import entropy
import time
import seaborn as sns
from skimage.measure import shannon_entropy
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import argparse

np.random.seed(47)
adv_img_path = 'adversarial_samples/'
clean_img_path = '/root/data/VOCdevkit/VOC2012/JPEGImages'
img_path = adv_img_path if adv_mode else clean_img_path
ws_ratio = 1

def load_imgs(path):
    
    assert os.path.exists(path), f"{path}: img folder not found!!!"
    
    imgs_name = [os.path.join(path, name) for name in sorted(os.listdir(path))[:10]]
    imgs = []
    for img_name in imgs_name:
        imgs.append(np.asarray(Image.open(img_name).convert("L")))
    
    return imgs

def get_heatmap(img, ws_ratio):
    (size_x, size_y) = img.shape
    size_max = max(size_x, size_y)
    size_max = max(ceil(size_max/100), 8)
    w_size = int((size_max + (size_max & 1)) * ws_ratio)
    w_size += (w_size & 1)
    strd = w_size / 2
    assert strd.is_integer(), "strd is not integer!!!"
    strd = int(strd)

    heatmap = np.zeros((size_x + w_size, size_y + w_size))
    pad_img = np.pad(img, (int(w_size/2),), "symmetric")

    # start = time.time()
    # a = np.unique(pad_img[0:w_size, 0:w_size].flatten(), return_counts=True)[1]
    # a = np.histogram(pad_img[0:w_size, 0:w_size].flatten(), range(256))[0]

    # a = entropy(a/sum(a))

    # end = time.time()
    # print(f"time: {end - start:.5f} {a}")

    ents = []
    bound = []
    for x in range(0, pad_img.shape[0] - w_size, strd):
        for y in range(0, pad_img.shape[1] - w_size, strd):
            # counts = np.unique(pad_img[x:x+w_size, y:y+w_size].flatten(), return_counts=True)[1]
            # print(f"sum: {sum(counts)}")
            # ent = entropy(counts, base = 2)
            ent = shannon_entropy(pad_img[x:x+w_size, y:y+w_size])
            ents.append(ent)
            heatmap[x:x+w_size, y:y+w_size] = ent
            bound = [x+w_size, y+w_size]
    sorted(ents, reverse=True)
    # print(f"ents: {ents[:5]}\n      {ents[-5:]}")
    x_exc = bound[0] - size_x
    y_exc = bound[1] - size_y
    # print(f"ori_size: {size_x} X {size_y}\nent_szie: {bound[0]} X {bound[1]}")
    # print(x_exc>>1)
    # print("Before slicing:\nshape: ", heatmap.shape, "\n", heatmap)
    heatmap = heatmap[(x_exc>>1)+(x_exc&1):bound[0]-(x_exc>>1), (y_exc>>1)+(y_exc&1):bound[1]-(y_exc>>1)]
    # print("After slicing:\nshpae: ", heatmap.shape, "\n", heatmap)

    return heatmap, ents


if __name__ == '__main__':
    imgs = load_imgs(img_path)
    # print(f"ent: {entropy([0.5])}")

    # test = np.random.randint(0, 256, (217, 212))
    heatmaps = []
    ents_arr = []
    for r in range(1,2):
        mode = "adv" if adv_mode else "clean"
        print(f"mode: {mode}|ratio: {r}")
        for img in imgs:
            h, e = get_heatmap(img, ws_ratio)
            heatmaps.append(h)
            ents_arr.append(e)

    mean_ent = []
    save_hmap_path = "exp/hmap/"
    save_hstg_path = "exp/hstg/"
    mode_folder = "adv/" if adv_mode else "clean/"
    save_hmap_path += mode_folder
    save_hstg_path += mode_folder

    for i, heatmap in enumerate(heatmaps):
        mean = np.mean(ents_arr[i])
        # print(f"sum:{sum(ents_arr[i]):.4f}/num:{len(ents_arr[i])}")
        map_mean = np.mean(heatmap)
        print(f"{i:2}: mean_entropy {map_mean}/{mean}")
        mean_ent.append(mean)
        plt.figure(i)
        hstg = sns.histplot(heatmap.flatten(), kde=False, color='blue', element="poly")
        hstg.get_figure().savefig(save_hstg_path+f"{i}.png")
        hmap = sns.heatmap(heatmap)
        hmap.get_figure().savefig(save_hmap_path+f"{i}.png")

    print(f"total mean: {sum(mean_ent)/len(mean_ent)}")
