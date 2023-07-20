import numpy as np
import os
from PIL import Image
from math import ceil
from scipy.stats import entropy
import time
import seaborn as sns

img_path = 'adversarial_samples/'
ws_ratio = 1

def load_imgs(path):
    
    assert os.path.exists(path), "img folder not found!!!"
    
    imgs_name = [os.path.join(path, name) for name in os.listdir(path)]
    imgs = []
    for img_name in imgs_name:
        imgs.append(np.asarray(Image.open(img_name).convert("L")))
    
    return imgs

def get_heatmap(img, ws_ratio):
    (size_x, size_y) = img.shape
    size_max = max(size_x, size_y)
    size_max = max(ceil(size_max/100), 8)
    w_size = int(size_max + (size_max & 1)) * ws_ratio
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
            counts = np.unique(pad_img[x:x+w_size, y:y+w_size].flatten(), return_counts=True)[1]
            ent = entropy(counts)
            ents.append((ent, (x, y)))
            heatmap[x:x+w_size, y:y+w_size] = ent
            bound = [x+w_size, y+w_size]
    x_exc = bound[0] - size_x
    y_exc = bound[1] - size_y
    # print(f"ori_size: {size_x} X {size_y}\nent_szie: {bound[0]} X {bound[1]}")
    # print(x_exc>>1)
    # print("Before slicing:\nshape: ", heatmap.shape, "\n", heatmap)
    heatmap = heatmap[(x_exc>>1)+(x_exc&1):bound[0]-(x_exc>>1), (y_exc>>1)+(y_exc&1):bound[1]-(y_exc>>1)]
    # print("After slicing:\nshpae: ", heatmap.shape, "\n", heatmap)

    return heatmap


if __name__ == '__main__':
    imgs = load_imgs(img_path)

    # test = np.random.randint(0, 256, (217, 212))
    heatmaps = []
    for img in imgs:
        heatmaps.append(get_heatmap(img, ws_ratio))

    mean_ent = []
    for i, heatmap in enumerate(heatmaps):
        # print(heatmap)
        mean = np.mean(heatmap)
        print(f"{i:2}: mean entropy {mean}")

        hstg = sns.histplot(heatmap.flatten(), kde=False, color='blue')
        hstg.get_figure().savefig(f"exp/hstg{i}.png")
    # print(imgs[0].shape)
