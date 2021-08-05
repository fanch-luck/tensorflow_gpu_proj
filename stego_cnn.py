#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
@file: stego_cnn.py.py
@author: Administrator
@time: 2021.08.03
@description: 隐写图片5层网络训练脚本
"""

import glob as gl
import math
import numpy as np
import tensorflow as tf
import time
import sys
import cv2
from sklearn.metrics import confusion_matrix

IMAGE_SIZE = 512  # 图片大小
NUM_CHANNELS = 1   # 通道数  灰色
PIXEL_DEPTH = 255.  # 分辨率
NUM_LABELS = 2   # 连通域的数目
NUM_EPOCHS = 20  # epochs
STEGO = 50000  # 隐写图片数量
FLAGS = tf.app.flags.FLAGS

ema = tf.train.ExponentialMovingAverage(decay=0.1)
phase_train = tf.placeholder(tf.bool, name='phase_train')


def read_pgm(filename):
    """加载pgm图片文件"""
    img1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('input pgm', img1)  # 显示输入pgm
    cv2.waitKey(0)  # 保持显示，按esc关闭图片，程序继续运行
    h, w = img1.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img1
    return vis0


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')


def mean_var_with_update(batchmean, batchvar):
    ema_apply_op = ema.apply([batchmean, batchvar])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batchmean), tf.identity(batchvar)


def conv1(in1, filter_size, size_in, size_out, pooling_size, stride_size):
    # layer 1 of 6
    W_conv = weight_variable([filter_size[0], filter_size[1], size_in, size_out])
    beta = tf.Variable(tf.constant(0.0, shape=[size_out]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[size_out]), name='gamma', trainable=True)
    z_conv = conv2d(in1, W_conv)
    z_conv = tf.abs(z_conv)  # 激活ABS
    batch_mean, batch_var = tf.nn.moments(z_conv, [0, 1, 2])

    mean, var = tf.cond(phase_train,
                        mean_var_with_update(batch_mean, batch_var),
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    BN_conv = tf.nn.batch_normalization(z_conv, mean, var, beta, gamma, epsilon)  # BN批量标准化
    f_conv = tf.nn.tanh(BN_conv)  # TanH activation
    # 平均池化， 池化尺寸 ，池化步长
    out = tf.nn.avg_pool(
        f_conv,
        ksize=[1, pooling_size, pooling_size, 1],
        strides=[1, stride_size, stride_size, 1],
        padding='SAME'
    )
    return out






if __name__ == '__main__':
    pgmfile = r"G:\BossBase-1.01-cover\1.pgm"
    # pgmfile = r"G:\BossBase-1.01-hugo-alpha=0.4\1.pgm"
    read_pgm(pgmfile)

