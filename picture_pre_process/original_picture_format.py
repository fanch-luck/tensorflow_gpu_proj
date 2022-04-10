#!usr/bin/env python
# -*- coding:utf-8 -*-
# -----------------------------------------------------------
# File Name: original_picture_format
# Author:    fan20200225
# Date:      2022/4/10 0010
# Detail: 原始图像经过多种简单处理，分别初始化为不同特征的图像（作为隐写原图）
# -----------------------------------------------------------

import cv2
import numpy as np
import random
import os
print("脚本执行路径:", os.path.abspath("."))
original_dir = 'input'
formated_dir = 'output'
original_pics = os.listdir(original_dir)
print(original_pics)

for ori_pic in original_pics:
    # 读取图片
    ori_im = cv2.imread(os.path.join(original_dir, ori_pic))  # 采用os.path.join组装文件路径
    ori_height, ori_width = ori_im.shape[:2]
    im_dic = dict()

    # 0.原图拷贝
    im_dic["im_ori_0"] = ori_im

    # 1.旋转
    center = (ori_width // 2, ori_height // 2)
    im_rot = cv2.getRotationMatrix2D(center, -90, 1)  # 旋转中心坐标，逆时针旋转：-90°，缩放因子：1
    im_90 = cv2.warpAffine(ori_im, im_rot, (ori_width, ori_height))
    im_dic['im_90_1'] = im_90

    # 2.缺失
    angle = (random.randint(-60, 60))
    im_loss1 = cv2.getRotationMatrix2D(center, angle, 1)  # 旋转中心坐标，逆时针旋转：45°，缩放因子：1
    im_loss = cv2.warpAffine(ori_im, im_loss1, (ori_width, ori_height))
    im_dic['im_loss_2'] = im_loss

    # 3.反转
    im_dic['im_top_3'] = cv2.flip(ori_im, 0)  # 倒影

    # 4.镜像
    im_dic['im_left_4'] = cv2.flip(ori_im, 1)  # 镜像

    # 5.截取
    a = (random.randint(512, int(ori_height)))  # 至少两次 至多四次
    b = (random.randint(512, int(ori_width)))  # 至多两次
    im_cut = ori_im[int(ori_height / 10):a , int(ori_width / 10):b]  # 裁剪坐标为[y0:y1, x0:x1]
    im_dic['im_cut_5'] = im_cut

    # 6.缩放
    im_zoom1 = ori_im[:, ori_im.shape[1] // 3: (ori_im.shape[1] // 3) * 2]  # 基于X轴 左1/3  右2/3
    im_zoom = im_zoom1[ori_im.shape[0] // 3: (ori_im.shape[0] // 3) * 2, :]  # Y轴 同上
    im_dic['im_zoom_6'] = im_zoom

    # 7.色相
    im_dic['im_color_7'] = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

    # 8.噪音
    sigma = 25  # 标准差越大图越差
    gauss = np.random.normal(0, sigma, (ori_height, ori_width, 3))
    im_noisy = ori_im + gauss  # 加高斯
    im_noisy = np.clip(im_noisy,a_min=0,a_max=255)
    im_dic['im_noisy_8'] = im_noisy

    # 9.模糊
    im_dic['im_blur_9'] = cv2.blur(ori_im, (5, 5))  # 55blur滤波器

    # 保存
    for k in im_dic.keys():
        name = k
        img = im_dic[k]
        npimg=np.array(img)
        cv2.imwrite(os.path.join(formated_dir, os.path.splitext(ori_pic)[0] + '_' + name + '.jpg'), npimg)


if __name__ == "__main__":
    print("ok")
