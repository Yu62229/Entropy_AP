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

    bound = []
    for x in range(0, pad_img.shape[0] - w_size, strd):
        for y in range(0, pad_img.shape[1] - w_size, strd):
            # counts = np.unique(pad_img[x:x+w_size, y:y+w_size].flatten(), return_counts=True)[1]
            # print(f"sum: {sum(counts)}")
            # ent = entropy(counts, base = 2)
            ent = shannon_entropy(pad_img[x:x+w_size, y:y+w_size])
            heatmap[x:x+w_size, y:y+w_size] = ent
            bound = [x+w_size, y+w_size]
    # print(f"ents: {ents[:5]}\n      {ents[-5:]}")
    x_exc = bound[0] - size_x
    y_exc = bound[1] - size_y
    heatmap = heatmap[(x_exc>>1)+(x_exc&1):bound[0]-(x_exc>>1), (y_exc>>1)+(y_exc&1):bound[1]-(y_exc>>1)]
    # print("After slicing:\nshpae: ", heatmap.shape, "\n", heatmap)

    return heatmap

with open('exp/clean.txt', 'r') as file:
    lines = file.readlines()
    min_lines = lines[1:6]
    max_lines = lines[-5:]
    save_hmap_path = "exp/hmap/clean/"
    save_hstg_path = "exp/hstg/clean/"
    for i,line in enumerate(min_lines):
        path = line.split(" ")[1]
        img = np.asarray(Image.open(path).convert('L'))
        h_map = get_heatmap(img, 1)
        plt.figure(i)
        hstg = sns.histplot(h_map.flatten(), kde=False, color='blue', element="poly")
        hstg.get_figure().savefig(save_hstg_path+f"min{i}.png")
        hmap = sns.heatmap(h_map)
        hmap.get_figure().savefig(save_hmap_path+f"min{i}.png")
        plt.close()
    for i,line in enumerate(max_lines):
        path = line.split(" ")[1]
        img = np.asarray(Image.open(path).convert('L'))
        h_map = get_heatmap(img, 1)
        plt.figure(i)
        hstg = sns.histplot(h_map.flatten(), kde=False, color='blue', element="poly")
        hstg.get_figure().savefig(save_hstg_path+f"max{i}.png")
        hmap = sns.heatmap(h_map)
        hmap.get_figure().savefig(save_hmap_path+f"max{i}.png")
        plt.close()
        
