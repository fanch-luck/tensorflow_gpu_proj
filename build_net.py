#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
@file: build_net.py
@author: Administrator
@time: 2021.07.30
@description: ...
"""
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import warnings
import os


class NetBuilder(object):
    def __init__(self, classes_num, image_num, keep_prob, epochs, batch_size, learning_rate, epsilon):
        self.train_classes_num = classes_num
        self.train_image_num_per_batch = image_num
        self.train_keep_prob = keep_prob
        self.train_epochs = epochs
        self.train_batch_size = batch_size
        self.train_learning_rate = learning_rate
        self.train_epsilon = epsilon

        self.net = None

    def layer_conv2d(self, input_layer, kernel_win_scale:tuple, strides:tuple, _padding='same', _activation=tf.nn.tanh,
                     _kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)):
        """构建卷积层"""
        layer = tf.layers.conv2d(
            input_layer,
            kernel_win_scale,
            strides,
            padding=_padding,
            activation=_activation,
            kernel_initializer=_kernel_initializer)
        return layer

    def layer_pool2d(self, input_layer, pool_win_scale:tuple, strides:tuple, _padding='same', pooltype='average'):
        """构建池化层"""
        layer = None
        if pooltype == 'average':
            layer = tf.layers.average_pooling2d(
                input_layer,
                pool_win_scale,
                strides,
                padding=_padding
            )
        elif pooltype == 'max':
            layer = tf.layers.max_pooling2d(
                input_layer,
                pool_win_scale,
                strides,
                padding=_padding
            )
        else:
            print('error. unknown pool layer type: {}'.format(pooltype))
            exit(1)
        return layer

    def layer_fully_connect(self, input_layer, num_outputs, _activation_fn='tanh', dropout=False):
        """构件全连接层
        input_layer: 输入层
        num_outputs: 整数或长整数，层中输出单元的数量。
        _activation_fn: 激活函数名称
        dropout:
        """
        fn = None
        if _activation_fn == 'tanH':
            fn = tf.nn.tanh
        elif _activation_fn == 'relu':
            fn = tf.nn.relu
        else:
            print('error. unknown activation function name: {}'.format(_activation_fn))
            exit(1)
        layer = tf.contrib.layers.fully_connected(
            input_layer,
            num_outputs,
            activation_fn=fn
        )
        if dropout:
            layer = tf.nn.dropout(
                layer,
                keep_prob=self.train_keep_prob
            )
        return layer

    def def_logits(self, input_layer, num_outputs=10, logits_name='logits_'):
        """定义logits"""
        layer = tf.contrib.layers.fully_connected(  # 全连接
            input_layer,
            num_outputs,
            activation_fn=None
        )
        logits = tf.identity(layer, name=logits_name)  # identify：输出与输入具有相同形状和内容的张量。将操作和名称关联
        return logits

    def def_accuracy(self, logits_layer, targets_layer, accuracy_name='accuracy_'):
        """定义accuracy（预测正确率）"""
        correct_pred = tf.equal(tf.argmax(logits_layer, 1), tf.argmax(targets_layer, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=accuracy_name)
        return accuracy

    def def_cost_optimizer(self, logits_layer, target_layer):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_layer, labels=target_layer))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.train_learning_rate,
            epsilon=self.train_epsilon
        ).minimize(cost)
        return cost, optimizer

    def def_inputs_and_targets(self, inputs_shape, inputs_name, targets_shape, targets_name, data_type=tf.float32):
        inputs = tf.placeholder(data_type, inputs_shape, inputs_name)
        targets = tf.placeholder(data_type, targets_shape, targets_name)
        return inputs, targets

    def update_net(self, layer):
        self.net = layer
        return self.net


if __name__ == '__main__':
    classnums = 10
    imagenumperbatch = 10000
    keepprob = 0.6
    epochs = 20
    batchsize = 128
    learningrate = 1e-5
    epsilon = 1e-5
    net = NetBuilder(classnums, imagenumperbatch, keepprob, epochs, batchsize, learningrate, epsilon)

    # layer 1
    z_c = None  # 图片信息输入矩阵
    f_conv1  =