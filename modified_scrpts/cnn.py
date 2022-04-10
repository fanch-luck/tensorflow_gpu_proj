#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
@file: cnn.py
@author: Administrator
@time: 2021.07.06
@description: ...
"""

# CIFAR图像识别-CNN
# 该部分将使用CNN实现图像识别

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# # 加载数据
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    加载单批量的数据
    参数：
    cifar10_dataset_folder_path: 数据存储目录
    batch_id: 指定batch的编号
    """
    batch_file = os.path.join(cifar10_dataset_folder_path, 'data_batch_' + str(batch_id))
    with open(batch_file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    # features and labels
    print(batch.keys())
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def get_train_data(train_data_path):
    # 加载我的训练数据
    # 共有5个batch的训练数据
    x_tra, y_tra = load_cfar10_batch(train_data_path, 1)
    show_pic(x_tra)  # 显示一组获取的训练集图片
    for i in range(2, 6):
        features, labels = load_cfar10_batch(train_data_path, i)
        x_tra, y_tra = np.concatenate([x_tra, features]), np.concatenate([y_tra, labels])
        # show_pic(x_tra)  # 显示一组获取的训练集图片
    return x_tra, y_tra


def get_test_data(test_data_path):
    # 加载测试数据
    test_batch_file = os.path.join(test_data_path, 'test_batch')
    with open(test_batch_file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
        #####################################################################################
        # 测试图片数据处理
        # 1.从data键取测试图片一维矩阵，并重置为4纬矩阵（10000，3，32，32 >> 单张图片、单个图层、单行像素、单个像素（32*32））
        # 2.矩阵转置为（10000,32,32,3 >> 单张图片、单行像素、单个像素、单个像素）
        # 3.显示图片（意思即为从pikle读取图片数据后重新拼接还原）

        x_tst = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        y_tst = batch['labels']
        # show_pic(x_tst)  # 显示获取的训练集图片
        return x_tst, y_tst


def show_pic(imgdata):
    # 显示图片
    fig, axes = plt.subplots(nrows=4, ncols=25, sharex=True, sharey=True, figsize=(80, 12))
    imgs = imgdata[-100:]
    for image, row in zip([imgs[:25], imgs[25:50], imgs[50:75], imgs[75:100]], axes):
        for img, ax in zip(image, row):
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)


def data_process(x_train, y_train, x_test, y_test):
    # - 输入数据的处理
    # - 标签处理
    # 对输入数据进行归一化

    minmax = MinMaxScaler()

    # 重塑
    x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_test_rows = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

    # 归一化
    x_train_mm = minmax.fit_transform(x_train_rows)
    x_test_mm = minmax.fit_transform(x_test_rows)

    # 重新变为32 x 32 x 3
    x_train_processed = x_train_mm.reshape(x_train_mm.shape[0], 32, 32, 3)
    x_test_processed = x_test_mm.reshape(x_test_mm.shape[0], 32, 32, 3)
    print(x_test_processed, y_test)

    # ### 目标变量处理
    # 对目标变量进行one-hot编码
    lb = LabelBinarizer().fit(np.array(range(n_class)))

    y_train_processed = lb.transform(y_train)
    y_test_processed = lb.transform(y_test)

    return x_train_processed, y_train_processed, x_test_processed, y_test_processed


def build_net():
    # # 构建网络
    # ### 参数设置
    keep_prob = 0.6
    inputs_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='inputs_')
    targets_ = tf.placeholder(tf.float32, [None, n_class], name='targets_')

    # 第一层卷积加池化
    # 32 x 32 x 3 to 32 x 32 x 64
    conv1 = tf.layers.conv2d(inputs_, 64, (2, 2), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # 32 x 32 x 64 to 16 x 16 x 64
    conv1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')

    # 第二层卷积加池化
    # 16 x 16 x 64 to 16 x 16 x 128
    conv2 = tf.layers.conv2d(conv1, 128, (4, 4), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # 16 x 16 x 128 to 8 x 8 x 128
    conv2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')

    # 重塑输出
    shape = np.prod(conv2.get_shape().as_list()[1:])
    conv2 = tf.reshape(conv2, [-1, shape])

    # 第一层全连接层
    # 8 x 8 x 128 to 1 x 1024
    fc1 = tf.contrib.layers.fully_connected(conv2, 1024, activation_fn=tf.nn.relu)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # 第二层全连接层
    # 1 x 1024 to 1 x 512
    fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu)

    # logits层
    # 1 x 512 to 1 x 10
    logits_ = tf.contrib.layers.fully_connected(fc2, 10, activation_fn=None)
    logits_ = tf.identity(logits_, name='logits_')

    # cost & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=targets_))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    # accuracy
    correct_pred = tf.equal(tf.argmax(logits_, 1), tf.argmax(targets_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    return inputs_, targets_, cost, optimizer, accuracy


def training(inputs_, targets_, cost, optimizer, accuracy, save_model_path):
    # # 训练模型
    epochs = 10
    batch_size = 64
    img_shape = x_train.shape
    
    count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch_i in range(img_shape[0] // batch_size - 1):
                feature_batch = x_train_[batch_i * batch_size: (batch_i + 1) * batch_size]
                label_batch = y_train_[batch_i * batch_size: (batch_i + 1) * batch_size]
                train_loss, _ = sess.run([cost, optimizer],
                                         feed_dict={inputs_: feature_batch,
                                                    targets_: label_batch})

                val_acc = sess.run(accuracy,
                                   feed_dict={inputs_: x_val,
                                              targets_: y_val})

                if count % 1 == 0:
                    print(
                        'Epoch {:>2}, Train Loss {:.4f}, Validation Accuracy {:4f} '.format(epoch + 1, train_loss,
                                                                                            val_acc))
                count += 1
        # 存储参数
        saver = tf.train.Saver()
        saver.save(sess, save_model_path)


def calc_acc(save_model_path, x_test, y_test):
    # # 测试结果
    loaded_graph = tf.Graph()
    test_batch_size = 100  # 测试集大小（规模）
    with tf.Session(graph=loaded_graph) as sess:
        # 加载模型
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)
        # 加载tensor
        loaded_x = loaded_graph.get_tensor_by_name('inputs_:0')
        loaded_y = loaded_graph.get_tensor_by_name('targets_:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits_:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        # 计算test的准确率
        test_batch_acc_total = 0
        test_batch_count = 0
        print("Begin test...")
        print("{}, {}".format(x_test.shape[0], test_batch_size))
        for batch_i in range(x_test.shape[0] // test_batch_size - 1):
            test_feature_batch = x_test[batch_i * test_batch_size: (batch_i + 1) * test_batch_size]
            test_label_batch = y_test[batch_i * test_batch_size: (batch_i + 1) * test_batch_size]
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch})
            test_batch_count += 1
        print('Test Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))


if __name__ == "__main__":
    ####################################################################
    # GPU设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    # 使用设置session = tf.Session(config=tf_config)
    ####################################################################

    warnings.filterwarnings("ignore")
    # get_ipython().run_line_magic('matplotlib', 'inline')

    cifar10_path = r'E:\tensor_resorce\cifar-10-batches-py'
    print(cifar10_path)
    n_class = 10  # 原版数据集总共10类
    print('n class: {}'.format(n_class))
    print("TensorFlow Version: %s" % tf.__version__)

    x_train_data, y_train_data = get_train_data(cifar10_path)
    x_test_data, y_test_data = get_test_data(cifar10_path)
    # print(x_test, y_test)
    
    x_train, y_train, x_test, y_test = data_process(x_train_data, y_train_data, x_test_data, y_test_data)
    
    # 划分train与val
    # train_ratio = 0.8
    # x_train_, x_val, y_train_, y_val = train_test_split(x_train,  ##########x_train 还不行
    #                                                    y_train,
    #                                                    train_size=train_ratio, random_state=123)
    #
    # inputs_, targets_, cost, optimizer, accuracy = build_net()
    #
    # save_model_path = r'E:\tensor_resorce\model\test_cifar'

    # training(inputs_, targets_, cost, optimizer, accuracy, save_model_path)

    # testing(save_model_path, x_test, y_test)
