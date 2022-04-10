# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:57:37 2021

@author: Administrator
"""

import numpy as np
from PIL import Image
import os
from numpy import concatenate


def get_img_data(img_resized_dir):
    imgs = []
    img_list = os.listdir(img_resized_dir)
    for img_name in img_list:
        img_path = os.path.join(img_resized_dir, img_name)
        img = Image.open(img_path)
        try:
            r,g,b,a = img.split()  # 拆分图片为R、G、B三个分量的图片元组
        except:
            continue
        r_array = np.array(r, dtype=np.uint8).flatten()
        g_array = np.array(g, dtype=np.uint8).flatten()
        b_array = np.array(b, dtype=np.uint8).flatten()
        img_array = concatenate((r_array, g_array, b_array))
        imgs.append(img_array)
    imgs = np.array(imgs, dtype=np.uint8)
    return imgs
    

if __name__ == "__main__":
    
    resized_dir = r"R:\workspaces\tensorflow_proj\imgs\data_batch_1_resize"
    myimgs = get_img_data(resized_dir)