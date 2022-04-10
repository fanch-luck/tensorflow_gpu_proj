# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:12:58 2021

@author: Administrator
"""


from PIL import Image
import pickle
import time
import os


def get_batchs(batch_dir):
    """
    获取batch列表
    """
    bt_list = []
    for name in os.listdir(batch_dir):
        if name.startswith('data_') or name.startswith('test_'):
            bt_list.append(os.path.join(batch_dir, name))
    return bt_list


def parse_batch(batch_path):
    """
    加载batch文件，并解析出其中图片保存到文件夹
    """
    # 创建解析结果存放目录
    save_dir = batch_path + '_parsed'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # 加载batch文件
    with open(batch_path, mode='rb') as f:
        batch = pickle.load(f, encoding='latin1')
        for dt, name, lab in zip(batch["data"], batch['filenames'], batch['labels']):
        # 取得图片
            r_array, g_array, b_array = dt[0:1024], dt[1024:2048], dt[2048:3072]
            rr_array = r_array.reshape(32,32)
            gg_array = g_array.reshape(32,32)
            bb_array = b_array.reshape(32,32)
            imgr = Image.fromarray(rr_array)
            imgg = Image.fromarray(gg_array)
            imgb = Image.fromarray(bb_array)
            # img = Image.merge('RGB', [imgr,imgg,imgb]).convert('RGBA')
            img = Image.merge('RGB', [imgr,imgg,imgb])
            img_name = os.path.join(save_dir, name+'.JPEG')
            img.save(img_name)
            

if __name__ == '__main__':
    # bd = r"R:\workspaces\tensorflow_proj\cifar_ok"
    # # bd = r"R:\workspaces\tensorflow_proj\imgs_cifar"
    # for bt in get_batchs(bd):
    #     parse_batch(bt)
    bp = r"R:\workspaces\tensorflow_proj\demo_origin\batches\test_batch"
    parse_batch(bp)
    