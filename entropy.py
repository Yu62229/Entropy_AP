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

def add_noise(imgs, mean=0, sigma=0.1):

    gaussian_outs = []
    for img in imgs:
        # int -> float (標準化)
        img = img / 255
        # 隨機生成高斯 noise (float + float)
        noise = np.random.normal(mean, sigma, img.shape)
        # noise + 原圖
        gaussian_out = img + noise
        # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
        gaussian_out = np.clip(gaussian_out, 0, 1)

        # 原圖: float -> int (0~1 -> 0~255)
        gaussian_out = np.uint8(gaussian_out*255)

        gaussian_outs.append(gaussian_out)

    return gaussian_outs

def ent_test():
    args = arg_parse()

    adv_mode = not args.clean
    w_mode = args.window_ratio
    smooth = args.smooth
    bin_mode = args.bin
    bin_size = args.bin_size
    mode = "adv" if adv_mode else "clean"
    b_txt = f"|bin_size: {bin_size}" if bin_mode else ""
    print(f"mode: {mode}|ratio_mode: {w_mode}|smooth: {str(smooth)}|bin: {str(bin_mode)}{b_txt}")

    img_path = adv_img_path if adv_mode else clean_img_path

    imgs, img_name = load_imgs(img_path, smooth)

    # test = np.random.randint(0, 256, (217, 212))
    hm_100 = []
    hm_150 = []
    hm_200 = []
    ent_100 = []
    ent_150 = []
    ent_200 = []
    heatmaps = [hm_100, hm_150, hm_200]
    ents_arr = [ent_100, ent_150, ent_200]
    ws_ratio = [1, 1.5, 2] if w_mode else [1]
    for id, r in enumerate(ws_ratio):
        print(f"Start heatmap r: {r} calculate:")
        for i, img in tqdm(enumerate(imgs), total=len(imgs)):
            h, e = get_heatmap(img, r, bin_mode, bin_size)
            heatmaps[id].append(h)
            ents_arr[id].append(e)
    m100 = []
    m150 = []
    m200 = []
    mean_ent = [m100, m150, m200]
    save_hmap_path = "exp/hmap/"
    save_hstg_path = "exp/hstg/"
    clean_txt_path = "exp/clean.txt"
    mode_folder = "att/" if adv_mode else "clean/"
    save_hmap_path += mode_folder
    save_hstg_path += mode_folder
    if bin_mode:
        save_hmap_path += "bin/"
        save_hstg_path += "bin/"
        if not os.path.exists(save_hmap_path):
            os.mkdir(save_hmap_path)
        if not os.path.exists(save_hstg_path):
            os.mkdir(save_hstg_path)

    if w_mode:
        hm_len = len(heatmaps[0])
        for i in range(hm_len):
            for wr in range(3):
                heatmap = heatmaps[wr][i]
                mean = np.mean(ents_arr[wr][i])
                # print(f"sum:{sum(ents_arr[i]):.4f}/num:{len(ents_arr[i])}")
                # map_mean = np.mean(heatmap)
                # print(f"{i:2}: mean_entropy {map_mean}/{mean}")
                mean_ent[wr].append((mean, img_name[i]))
                if adv_mode:
                    plt.figure(i)
                    hstg = sns.histplot(heatmap.flatten(), kde=False, color='blue', element="poly")
                    hstg.get_figure().savefig(save_hstg_path+f"{i}_wr{wr}.png")
                    hmap = sns.heatmap(heatmap)
                    hmap.get_figure().savefig(save_hmap_path+f"{i}_wr{wr}.png")
                    plt.close()
    else:
        for i, heatmap in enumerate(heatmaps[0]):
            mean = np.mean(ents_arr[0][i])
            # print(f"sum:{sum(ents_arr[i]):.4f}/num:{len(ents_arr[i])}")
            # map_mean = np.mean(heatmap)
            # print(f"{i:2}: mean_entropy {map_mean}/{mean}")
            mean_ent.append((mean, img_name[i]))
            if adv_mode:
                plt.figure(i)
                hstg = sns.histplot(heatmap.flatten(), kde=False, color='blue', element="poly")
                hstg.get_figure().savefig(save_hstg_path+f"{i}.png")
                hmap = sns.heatmap(heatmap)
                hmap.get_figure().savefig(save_hmap_path+f"{i}.png")
                plt.close()
    if not adv_mode:
        mean_ent = sorted(mean_ent)
        with open(clean_txt_path, 'w') as file:
            file.write('min:\n')
            for i in range(10):
                file.write('name: '+mean_ent[i][1]+' ent: '+str(mean_ent[i][0])+'\n')
            file.write('max:\n')
            for i in range(10):
                file.write('name: '+mean_ent[-10+i][1]+' ent: '+str(mean_ent[-10+i][0])+'\n')

    for i in range(3):
        ents = [a for a, _ in mean_ent[i]]
        print(f"{i}: total mean: {sum(ents)/len(ents)}")

def trans_to_img(arr):
    rgb = np.reshape(arr, (3, 32, 32))
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return gray

def cifar_test():
    cifar_100 = True
    file = "../cifar-100-python/train" if cifar_100 else "../cifar-10-batches-py/test_batch"
    meta_file = "../cifar-100-python/meta" if cifar_100 else "../cifar-10-batches-py/batches.meta"
    d = unpickle(file)
    print(f"d: {d.keys()}")
    n = 100 if cifar_100 else 10
    num = [0]*n
    ent = [0]*n
    meta = unpickle(meta_file)
    print(f"meta: {meta.keys()}")
    data_key = b'data'
    label_key = b'fine_labels' if cifar_100 else b'labels'
    meta_key = b'fine_label_names' if cifar_100 else b'label_names'
    for i in range(10000):
        img = trans_to_img(d[data_key][i])
        label = d[label_key][i]
        h, e = get_heatmap(img)
        ent[label] += np.mean(e)
        num[label] += 1
    result = [x / y for x, y in zip(ent, num)]
    result = sorted([(a, meta[meta_key][i].decode('utf-8')) for i, a in enumerate(result)])
    print(result)

def paste_patch():
    patch_file = "adversarial_samples/e70v3.png"
    clean_file = 'exp/clean.txt'
    patch = Image.open(patch_file).resize((50, 50))
    with open(clean_file, 'r') as f:
        lines = f.readlines()
    min_lines = lines[1:6]
    max_lines = lines[-5:]
    lines = max_lines + min_lines
    for i,line in enumerate(lines):
        path = line.split(" ")[1]
        img = Image.open(path).resize((224, 224))
        new_img = img.paste(patch, (50,50))
        new_img.save(f'attacked_samples/{i}.png')

def face_test():
    face_path = "../img_align_celeba"
    imgs, img_names = load_imgs(face_path, rand=True, n=20000)
    total_mean = 0
    for img in tqdm(imgs):
        h,e = get_heatmap(img)
        total_mean += np.mean(e)
    total_mean /= 20000
    print(f"mean: {total_mean}")

def noise_test():
    save_path = 'exp/hmap/noise/'
    noise_path = 'noise_samples/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(noise_path):
        os.mkdir(noise_path)
    imgs,_ = load_imgs(att_img_path)
    new_imgs = add_noise(imgs, sigma=0.1)
    r = 2
    for i,img in tqdm(enumerate(new_imgs), total=len(new_imgs)):
        Image.fromarray(img).convert('RGB').save(noise_path+f'{i}.png')
        plt.figure(i)
        h,e = get_heatmap(img, ws_ratio=r)
        hmap = sns.heatmap(h)
        hmap.get_figure().savefig(save_path+f"{i}.png")
        plt.close()

def noise_attach():
    imgs,_=load_imgs(att_img_path)
    new_imgs=add_noise(imgs,sigma=0.3)
    noise_path = 'noise_samples/'
    for i,img in tqdm(enumerate(new_imgs), total=len(new_imgs)):
        Image.fromarray(img).convert('RGB').save(noise_path+f'RGB{i}.png')


if __name__ == '__main__':

    # ent_test()
    # cifar_test()
    # paste_patch()
    # face_test()
    # noise_test()
    noise_attach()