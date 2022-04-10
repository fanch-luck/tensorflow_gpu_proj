# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:28:08 2021

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# 从测试集中随机挑选一张图片看测试结果


def data_process(img_dir):
    """
    打开一张随机或确定的图片，并处理为可以提供模型识别的数据类型
    """
    # 打开图片
    image_path = None
    if os.path.isdir(img_dir):
        imgs = os.listdir(img_dir)
        img_num = len(imgs)
        idn = np.random.randint(0, img_num)
        image = imgs[idn]
        image_path = os.path.join(img_dir, image)
    elif os.path.isfile(img_dir):
        image_path = img_dir
    else:
        print('error to open a image.')
        exit(1)
    print(image_path)
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()

    # 进行图片数据处理(step 1)
    image = image.resize([32, 32])
    if os.path.splitext(image_path)[-1].lower() == '.png':
        r, g, b = image.split()[:3]  # 处理png图片，去掉透明通道，只保留RGB三个通道
        newimg = Image.merge('RGB', [r, g, b])
    else:
        newimg = image
    image_arr = np.array(newimg, dtype=np.uint8)
    image_arr_reshaped = image_arr.reshape(1, 3, 32, 32)
    image_data = image_arr_reshaped.transpose(0, 2, 3, 1)

    # 进行图片数据处理(step 2)
    minmax = MinMaxScaler()  # 创建归一化实例
    image_data_rows = image_data.reshape(image_data.shape[0], 32 * 32 * 3)  # 重塑
    image_data_rows_mm = minmax.fit_transform(image_data_rows)  # 归一化
    image_data_processed = image_data_rows_mm.reshape(image_data_rows_mm.shape[0], 32, 32, 3)

    return image_data_processed


def identify(save_model_path, img_arr):
    # 测试结果
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # 加载模型
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)
        # 加载tensor
        loaded_x = loaded_graph.get_tensor_by_name('inputs_:0')
        # loaded_y = loaded_graph.get_tensor_by_name('targets_:0')
        # 计算test的分类预测概率（log(p/(1-p))）
        loaded_logits = loaded_graph.get_tensor_by_name('logits_:0')
        # 计算test的准确率
        # loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        print("Begin identify...")
        test_feature = img_arr
        test_logit = sess.run(
             loaded_logits,
             feed_dict={loaded_x: test_feature})
        # max_index = np.argmax(test_logit)  # 获取最大值所在的索引
        #         # print("预测结果")
        #         # print(test_logit)
        #         # print("最佳目标：{}, {}\n可能性: {}".format(
        #         #     max_index,
        #         #     labels[max_index],
        #         #     test_logit[0][max_index]))

        # 利用softmax来获取概率
        probabilities = tf.nn.softmax(test_logit)
        # 获取最大概率的标签位置
        correct_prediction = tf.argmax(test_logit, 1)

        # 获取预测结果
        with tf.Session(graph=loaded_graph) as sess1:
            probabilities, label = sess1.run([probabilities, correct_prediction])
            # 获取此标签的概率
            probability = probabilities[0][label]
            # 预测结果
            print(probabilities, label)
            print(probability, labels[label[0]])


if __name__ == "__main__":
    imagefile = r"E:\tensor_resorce\images_for_test_one\cat"
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    save_model_path = r"E:\tensor_resorce\model\test_cifar"
    image_arr = data_process(imagefile)
    identify(save_model_path, image_arr)