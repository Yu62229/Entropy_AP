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
att_img_path = 'attacked_samples/'
# clean_img_path = '/root/data/VOCdevkit/VOC2012/JPEGImages'
clean_img_path = 'exp/clean.txt'

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--clean',action='store_true')
    parser.add_argument('-w','--window_ratio',action='store_true')
    parser.add_argument('-sm','--smooth',action='store_true')
    parser.add_argument('-b','--bin',action='store_true')
    parser.add_argument('-bz','--bin_size',type=int,default=2)
    return parser.parse_args()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_imgs(path, smooth=False, rand=False, n=1000):
    
    # if(path.split()):
    #     pass

    assert os.path.exists(path), f"{path}: img folder not found!!!"
    
    namelist = sorted(os.listdir(path))
    imgs_name = [os.path.join(path, name) for name in namelist]
    if rand:
        imgs_name = random.sample(imgs_name, k=n)

    start = time.time()
    print(f"start loading {len(imgs_name)} images")
    imgs = []
    for img_name in imgs_name:
        img = np.asarray(Image.open(img_name))
        if smooth:
            img = gaussian_filter(img, sigma = 1)
        imgs.append(img)
    end = time.time()
    print(f"finish loading images, total {end - start:.2f}s")
    return imgs, imgs_name

def get_heatmap(img, ws_ratio=1, bin_mode=False, bin_size=2):
    (size_x, size_y) = img.shape
    size_max = max(size_x, size_y)
    size_max = max(ceil(size_max/100), 8)
    size_max += (size_max & 1)
    w_size = size_max * ws_ratio
    assert w_size.is_integer() if type(w_size) is float else type(w_size) is int, "w_size is not integer!!!"
    w_size = int(w_size)
    w_size += (w_size & 1)
    strd = w_size / 2
    assert strd.is_integer(), "strd is not integer!!!"
    strd = int(strd)

    heatmap = np.zeros((size_x + w_size, size_y + w_size))
    pad_img = np.pad(img, (int(w_size/2),), "symmetric")

    ents = []
    bound = []
    for x in range(0, pad_img.shape[0] - w_size, strd):
        for y in range(0, pad_img.shape[1] - w_size, strd):
            if bin_mode:
                pi = pad_img[x:x+w_size, y:y+w_size].flatten()
                pi = pi // bin_size
                counts = np.unique(pi, return_counts=True)[1]
                ent = entropy(counts, base = 2)
            else:
                ent = shannon_entropy(pad_img[x:x+w_size, y:y+w_size])
            # assert ent==slow_ent, "not equal!!"
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
