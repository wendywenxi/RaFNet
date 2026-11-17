# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import h5py   

from basicsr.utils import scandir_SIDD
# from basicsr.utils.create_lmdb import create_lmdb_for_SIDD  

def main():
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3 

    opt['input_folder'] = '/datasets/SIDD_Medium_Raw/Data'
    opt['save_folder'] = '/SIDD_Medium_Raw/train/input_crops'
    opt['crop_size'] = 512
    opt['step'] = 384
    opt['thresh_size'] = 0

    # ### 修改开始：处理 RAW noisy .mat ###
    opt['keywords'] = '_NOISY'
    extract_subimages(opt)

    opt['save_folder'] = '/datasets/SIDD_Medium_Raw/train/gt_crops'
    opt['keywords'] = '_GT'
    extract_subimages(opt)

    print("全部 RAW patch 生成完毕。")

def extract_subimages(opt):
    """Crop raw images to subimages."""
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')

    img_list = list(scandir_SIDD(input_folder, keywords=opt['keywords'], 
                                 recursive=True, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract RAW')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All RAW images processed.')


def worker(path, opt):
    """处理单张 mat raw 并裁剪为 raw patch."""
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, ext = osp.splitext(osp.basename(path))

    img_name = img_name.replace(opt['keywords'], '')

    with h5py.File(path, 'r') as f:
        img = np.array(f['x']).T

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'RAW ndim must be 2/3, but got {img.ndim}')
    
    img = np.clip(img, 0, 1)
    img = (img * 65535.0 + 0.5).astype(np.uint16) 

    # compute sliding windows
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)

    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)

            save_path = osp.join(
                opt['save_folder'], f'{img_name}_s{index:03d}.raw'
            )
            cropped_img.tofile(save_path)

    return f'Processing {img_name} ...'


if __name__ == '__main__':
    main()
