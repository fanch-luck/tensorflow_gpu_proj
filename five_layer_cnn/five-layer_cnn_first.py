#!/usr/bin/env python
# -*- coding=utf-8 -*-

import glob as gl
import math
import numpy as np
import tensorflow as tf
import time
import sys
import cv2
from sklearn.metrics import confusion_matrix
import time

IMAGE_SIZE = 512
NUM_CHANNELS = 1
PIXEL_DEPTH = 255.
NUM_LABELS = 2
NUM_EPOCHS = 1  # 原2000
STEGO = 50000
BATCH_SIZE = 64  # 原64
FLAGS = tf.app.flags.FLAGS  #


def read_pgm(filename):
    img1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img1
    return vis0


# 该方法，用于多组算法隐写图片进行训练时读取图片。在本脚本中未被调用
# This method is used to read cover and stego images.
# We consider that stego images can be steganographied with differents keys(in practice this seems to be inefficient...)
def extract_data(indexes):
    cover_dir = FLAGS.cover_dir
    stego_dir = FLAGS.stego_dir

    nbImages = len(indexes)
    data = np.ndarray(
        shape=(nbImages, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=np.float64)
    labels = []

    for i in range(nbImages):
        if indexes[i] < STEGO:
            # Load covers
            filename = cover_dir + str(random_images[indexes[i]] + 1) + ".pgm"
            # print filename
            image = read_pgm(filename)
            data[i, :, :, 0] = (image / PIXEL_DEPTH) - 0.5
            labels = labels + [[1.0, 0.0]]
        else:
            # Load stego
            new_index = indexes[i] - STEGO
            filename = stego_dir + str(random_images[new_index] + 1) + "_" + str(k_key) + ".pgm"
            # print filename
            image = read_pgm(filename)
            data[i, :, :, 0] = (image / PIXEL_DEPTH) - 0.5
            labels = labels + [[0.0, 1.0]]

    labels = np.array(labels)

    return (data, labels)


# Same version but with one key per stego image
def extract_data_single(indexes):
    cover_dir = FLAGS.cover_dir
    stego_dir = FLAGS.stego_dir

    nbImages = len(indexes)
    data = np.ndarray(
        shape=(nbImages, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=np.float64)
    labels = []
    for i in range(nbImages):
        if indexes[i] < STEGO:  # 前5000的indexes[i]都是0~10000的值，data列表添加存载体图片
            # Load covers
            filename = cover_dir + str(random_images[indexes[i]] + 1) + ".pgm"
            # print filename
            image = read_pgm(filename)
            data[i, :, :, 0] = (image / PIXEL_DEPTH) - 0.5  # 将图片数据处理后存入data矩阵
            labels = labels + [[1.0, 0.0]]  # 将标签表示为一个可以进行分类的矩阵
        else:  # 后5000都是50000~60000的值，data列表添加隐写图片
            # Load stego
            new_index = indexes[i] - STEGO
            filename = stego_dir + str(random_images[new_index] + 1) + ".pgm"
            # print filename
            image = read_pgm(filename)
            data[i, :, :, 0] = (image / PIXEL_DEPTH) - 0.5
            labels = labels + [[0.0, 1.0]]

    labels = np.array(labels)
    return (data, labels)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name='W_init')  # 按照shape定义的纬度，截断的产生正态分布的随机数张量（随机数与均值的差值若大于两倍的标准差，则重新生成）。
    w = tf.Variable(initial, name='W')
    tf.summary.histogram('weights', w)
    return w


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name='B_init')  # 按照shape定义的纬度，创建一个常数张量
    b = tf.Variable(initial, name='B')
    tf.summary.histogram('bias', b)
    return b


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def show_layer(name,  tensor_64):
    # 用于tensorboard显示层输出结果（图像）
    for i in range(tensor_64.shape[-1]):
        begin = [0, 0, 0, i]
        size = [1, -1, -1, 1]
        tensor_1 = tf.slice(tensor_64, begin, size)
        tf.summary.image(name, tensor_1, max_outputs=BATCH_SIZE)
    return


def scalar_logger(tf_writer, name, value, step):
    # 用于tensorboard显示单一、变化的参数
    v = tf.Summary.Value(tag=name, simple_value=value)
    s = tf.Summary(value=[v])
    tf_writer.add_summary(s, step)
    return


tf.app.flags.DEFINE_string('cover_dir', 'E:\\tensor_resorce\\BossBaseDataSet1.01\\BossBase-1.01-cover\\', """Directory containing cover images.""")
tf.app.flags.DEFINE_string('stego_dir', 'E:\\tensor_resorce\\BossBaseDataSet1.01\\BossBase-1.01-hugo-alpha0.4\\', """directory containing stego images.""")
tf.app.flags.DEFINE_string('stego_test_dir', '', """directory containing stego images.""")
tf.app.flags.DEFINE_string('network', '', """Pretrained network.""")
tf.app.flags.DEFINE_string('seed', '2', """Seed.""")
tf.app.flags.DEFINE_string('batch_size', '{}'.format(BATCH_SIZE), """batch size.""")

network = FLAGS.network
seed = int(FLAGS.seed)  # 如果没有设置seed参数的取值，那么每次执行程序所产生的随机数或者随机序列均不等。

BATCH_SIZE = int(FLAGS.batch_size)

tf.set_random_seed(seed)

# 定义一个可交互的TensorFlow会话（重复使用，减少传递sess作为参数的次数），# 未指定session时，其作为默认会话被调用
sess = tf.InteractiveSession()

# 1 - Define the input x_image
x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1), name='X')
x_image = x

# tensorboard 显示输入图像
x255 = (x + 0.5) * PIXEL_DEPTH
show_image = tf.reshape(x255, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])  # 64 512 512 1
tf.summary.image('show_image', show_image, BATCH_SIZE)

# 2 - Define the expected output y_image
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2), name='Y')
y_image = y

# print(x_image.get_shape())
# print(y_image.get_shape())

# ########## A - Definition of the CNN ##########
# #### 0 - Paremeter used in the Batch-Normalization
epsilon = 1e-4

# #### 1 - High-pass filtering definition (F_0)，tf.cast，将张量转换为指定的新类型
F_0 = tf.cast(tf.constant([[[[-1 / 12.]], [[2 / 12.]], [[-2 / 12.]], [[2 / 12.]], [[-1 / 12.]]],
                           [[[2 / 12.]], [[-6 / 12.]], [[8 / 12.]], [[-6 / 12.]], [[2 / 12.]]],
                           [[[-2 / 12.]], [[8 / 12.]], [[-12 / 12.]], [[8 / 12.]], [[-2 / 12.]]],
                           [[[2 / 12.]], [[-6 / 12.]], [[8 / 12.]], [[-6 / 12.]], [[2 / 12.]]],
                           [[[-1 / 12.]], [[2 / 12.]], [[-2 / 12.]], [[2 / 12.]], [[-1 / 12.]]]]), "float")

# #### 2 - Definition of the first convolutional layer - input image => 1 feature map
# Convolution without F_0 (search for another filter 5x5) - PADDING

with tf.name_scope('high_pass'):
    z_c = tf.nn.conv2d(tf.cast(x_image, "float"), F_0, strides=[1, 1, 1, 1], padding='SAME')  # 对原始图像进行高通滤波

phase_train = tf.placeholder(tf.bool, name='phase_train')  # 占位符，训练阶段


# #### Definition of a function for the following convolution layers - size_in feature maps => size_out feature maps
def my_conv_layer(in1, filter_height, filter_width, size_in, size_out, pooling_size, stride_size, active, fabs,
                  padding_type):
    with tf.name_scope('conv'):
        # 定义卷积核
        # Convolution with filter_height x filter_width filters
        W_conv = weight_variable([filter_height, filter_width, size_in, size_out])

        # ### 卷积操作
        z_conv = conv2d(in1, W_conv)

        # ### 绝对值激活函数操作
        if fabs == 1:
            # Absolute activation
            z_conv = tf.abs(z_conv)

        # ### BN层的参数预处理
        # Batch normalization 批量归一化
        beta = tf.Variable(tf.constant(0.0, shape=[size_out]), name='beta', trainable=True)  # 给定初值并定义变量
        gamma = tf.Variable(tf.constant(1.0, shape=[size_out]), name='gamma', trainable=True)  # 给定初值并定义变量
        batch_mean, batch_var = tf.nn.moments(z_conv, [0, 1, 2])  # 获取z_conv的均值和方差
        # 滑动平均, 增加参数稳定性
        ema = tf.train.ExponentialMovingAverage(decay=0.1)  # previously 0.3

        def mean_var_with_update():
            # 控制计算的过程顺序？
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies(
                    [ema_apply_op]):  # 控制流程https://blog.csdn.net/hu_guan_jie/article/details/78495297
                return tf.identity(batch_mean), tf.identity(batch_var)  # 可以在图中新增节点

        # 如果phase_train值为1，赋值等于函数mean_var_with_update的返回结果，否则赋值等于另一个函数返回结果
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))  # lambda函数表达式，冒号后两个表达式作为函数返回结果
        BN_conv = tf.nn.batch_normalization(z_conv, mean, var, beta, gamma, epsilon, name='BN')
        # ### 根据需要设置激活函数
        if active == 1:
            # TanH activation
            f_conv = tf.nn.tanh(BN_conv)
            tf.summary.histogram('f_conv_tanh', f_conv)
        else:
            # ReLU activation
            f_conv = tf.nn.relu(BN_conv)
            tf.summary.histogram('f_conv_relu', f_conv)

        # ### 平均池化操作
        # Average pooling  - pooling_size x pooling_size - stride_size - PADDING
        out = tf.nn.avg_pool(
            f_conv,
            ksize=[1, pooling_size, pooling_size, 1],
            strides=[1, stride_size, stride_size, 1],
            padding=padding_type
        )
        # tensorboard 显示中间层图像

        mid_layer = (out + 0.5) * PIXEL_DEPTH
        show_layer('mid_layer', mid_layer)  # 显示层输出
        return out


# 神经网络分层，每一层的输出作为下一层的输入
# #### 3 - Definition of the second convolutional layer - 1 feature maps => 8 feature map
f_conv2 = my_conv_layer(z_c, 5, 5, 1, 8, 5, 2, 1, 1, 'SAME')
f_conv2_shape = f_conv2.get_shape().as_list()
print(f_conv2_shape)

# #### 4 - Definition of the third convolutional layer - 8 feature maps => 16 feature map
f_conv3 = my_conv_layer(f_conv2, 5, 5, 8, 16, 5, 2, 1, 0, 'SAME')
f_conv3_shape = f_conv3.get_shape().as_list()
print(f_conv3_shape)

# #### 5 - Definition of the fourth convolutional layer - 16 feature maps => 32 feature maps
f_conv4 = my_conv_layer(f_conv3, 1, 1, 16, 32, 5, 2, 0, 0, 'SAME')
f_conv4_shape = f_conv4.get_shape().as_list()
print(f_conv4_shape)

# #### 6 - Definition of the fifth convolutional layer - 32 feature maps => 64 feature maps
f_conv5 = my_conv_layer(f_conv4, 1, 1, 32, 64, 5, 2, 0, 0, 'SAME')
f_conv5_shape = f_conv5.get_shape().as_list()
print(f_conv5_shape)

# #### 7 - Definition of the sixth convolutional layer - 64 feature maps => 128 feature maps
f_conv6 = my_conv_layer(f_conv5, 1, 1, 64, 128, 5, 2, 0, 0, 'SAME')
f_conv6_shape = f_conv6.get_shape().as_list()
print(f_conv6_shape)

# #### 8 - Definition of the sixth convolutional layer - 128 feature maps => 256 feature maps
f_conv7 = my_conv_layer(f_conv6, 1, 1, 128, 256, 16, 1, 0, 0, 'VALID')
f_conv7_shape = f_conv7.get_shape().as_list()
print(f_conv7_shape)

# #### 9 - Reshaping the final output of the convolutional part
f_conv_shape = f_conv7.get_shape().as_list()
f_conv = tf.reshape(f_conv7, [f_conv_shape[0], f_conv_shape[1] * f_conv_shape[2] * f_conv_shape[3]])


# Definition of a function for a fully connected layer
# - input vector of size_in components => output vector of neurons outputs
def my_fullcon_layer(in1, size_in, neurons):
    # Convolution with filter_height x filter_width filters
    # 进行全连接层的定义，该函数未被调用？可能性：多隐写种类时可改为该方法
    with tf.name_scope('fc_multi_stego'):
        W_full = weight_variable([size_in, neurons])
        b_full = bias_variable([neurons])
        out = tf.nn.tanh(tf.matmul(in1, W_full) + b_full)
        tf.summary.histogram('W_full', W_full)
        tf.summary.histogram('b_full', b_full)
        tf.summary.histogram('fc_out', out)
        return out


# Without the hidden layer - input = 128 features - output = 2 softmax neurons outputs
with tf.name_scope('fc'):
    W_fc = weight_variable([256, 2])
    b_fc = bias_variable([2])
    y_pred = tf.nn.softmax(tf.matmul(f_conv, W_fc) + b_fc)  # 重塑末层输出f_conv(BN层的反向操作)，然后进行分类
    tf.summary.histogram('W_fc', W_fc)
    tf.summary.histogram('b_fc', b_fc)
    tf.summary.histogram('y_pred_fc', y_pred)

##########
# B - Definition of the variables
##########

# Definition of the error, optimization method, etc.  计算交叉熵。
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_image * tf.log(y_pred + 1e-4))

# Training  实现算法优化器
with tf.name_scope('train'):
    train_step = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9).minimize(cross_entropy)

prediction = y_pred  # 该变量未被调用到

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_image, 1))  # 获取预测结果
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 计算预测准确率
    tf.summary.scalar('accuracy', accuracy)

rounding = tf.argmax(y_pred, 1)  # 返回张量轴上具有最大值的索引，预测结果y的。
tab = tf.placeholder(tf.float32, [None], 'batch_acc')  # 存放各个batch acc的表
reduce_accuracy = tf.reduce_mean(tab)


# tensorboard, 将数据进行合并显示
summ = tf.summary.merge_all()

##########
# C - Initialization of all variables
##########
sess.run(tf.initialize_all_variables())  # 初始化所有变量
##########

# 添加tensorboard支持
tensorboard_dir = './tensorboard/test4/'
writer = tf.summary.FileWriter(tensorboard_dir + 'hparam')
writer.add_graph(sess.graph)


##########
# E - Loading data
##########

# images are permuted according to the random number generation of the seed
# 创建随机值（0~9999）列表
random_images = np.arange(0, 10000)
np.random.seed(seed)
np.random.shuffle(random_images)
# 将随机值列表分为训练组合测试组
im_train = random_images[0:5000]
im_test = random_images[5000:10000]  # im_test：长度5000 数值分布0~10000

# #### 1 - Define training data when no given network， 进行训练数据的定义处理
# if network=='':
steg = np.add(im_train, np.ones(im_train.shape, dtype=np.int32) * STEGO)  # steg: 长度5000，数值分布0~10000，50000~60000
arr_train = np.concatenate((im_train, steg), axis=0)  # arr_train：长度10000 数值分布：前5000：0~10000，后5000：50000~60000

np.random.shuffle(arr_train)
# 按照batch_size大小分割arr_train 如[arr_train[0:64],arr_train[64:128],...,[...]]
indexes_train = [arr_train[i:i + BATCH_SIZE] for i in range(0, len(arr_train), BATCH_SIZE)]
train_size = len(indexes_train)
# print arr_train
# print indexes_train

# #### 2 - Define testing data， 进行测试数据的定义处理
steg = np.add(im_test, np.ones(im_test.shape, dtype=np.int32) * STEGO)  # steg: 长度5000，数值分布0~10000，50000~60000
arr_test = np.concatenate((im_test, steg), axis=0)  # arr_test：长度10000 数值分布：前5000：0~10000，后5000：50000~60000

# test data are shuffled
np.random.seed(seed)  # test data的随机列表保持不变
np.random.shuffle(arr_test)
# 按照batch_size大小分割test_test 如[arr_test[0:64],arr_test[64:128],...,[...]]
indexes_test = [arr_test[i:i + BATCH_SIZE] for i in range(0, len(arr_test), BATCH_SIZE)]
test_size = len(indexes_test)

##########
# F - Training or loading a network
##########
num_epochs = NUM_EPOCHS
saver = tf.train.Saver(max_to_keep=1000)
##### 1 - Train a network
key = np.arange(1, 3)


if network == '':
    print("training a network")
    start_time = time.time()
    for ep in range(num_epochs):  # 按照原设定，循环2000次
        np.random.shuffle(key)
        k_key = key[0]  # 此处未生效。这是代码设计时考虑到不同的隐写载体可以用不同的后缀进行命名
        for step in range(train_size - 1):  # train_size=157, 157*64=10048张，溢出了。所以batch只取156
            # 读取训练图片数据和标签，由于数据拼接实时进行，会拖累训练速度
            batch_index = step
            batch_data, batch_labels = extract_data_single(indexes_train[batch_index])  # 取出一个batch的训练图片数据、标签
            # 执行训练计算
            train_step.run(session=sess, feed_dict={x: batch_data, y: batch_labels, phase_train: True})

            if step % 5 == 0:
                # 用于tensorboard显示accuracy
                train_accuracy = accuracy.eval(
                    session=sess,
                    feed_dict={x: batch_data, y: batch_labels, phase_train: True}
                )
                scalar_logger(writer, 'accuracy', train_accuracy, step)

                train_summ = summ.eval(
                    session=sess,
                    feed_dict={x: batch_data, y: batch_labels, phase_train: True}
                )
                writer.add_summary(train_summ, step)

                if step % 40 == 0:
                    # 156个batch中，每过40个batch,计算一次train_accuracy和test_accuracy
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    pred_test_index = step % test_size
                    pred_test_data, pred_test_labels = extract_data_single(
                        indexes_test[pred_test_index])  # 取出一个batch的测试图片数据、标签

                    print("step %d (epoch %d), %.1f ms, showing prediction" % (step, ep, 1000 * elapsed_time))
                    train_accuracy = accuracy.eval(session=sess,
                                                   feed_dict={x: batch_data, y: batch_labels, phase_train: True})
                    print("Train accuracy - batch " + str(batch_index))
                    print(train_accuracy)

                    test_accuracy = accuracy.eval(session=sess,
                                                  feed_dict={x: pred_test_data, y: pred_test_labels,
                                                             phase_train: False})
                    print("Test accuracy - batch " + str(pred_test_index))
                    print(test_accuracy)

            if step == train_size - 1 - 1:  # 155  最后batch进行额外的处理
                global_test_predlabels = []  # 全epoch预测标签
                global_test_truelabels = []  # 全epoch真实标签
                gtest_accuracy = np.zeros(shape=(test_size), dtype=np.float32)  # 全epoch 预测准确率

                # #train accuracy only to compute update of batch normalization
                train_accuracy = accuracy.eval(session=sess,
                                               feed_dict={x: batch_data, y: batch_labels, phase_train: True})
                # 遍历所有batch，取出各个batch的测试图片数据、标签，计算batch_accuracy并打印输出
                for global_test_index in range(test_size - 1):
                    gtest_data, gtest_labels = extract_data_single(indexes_test[global_test_index])
                    batch_accuracy = accuracy.eval(session=sess,
                                                   feed_dict={x: gtest_data, y: gtest_labels, phase_train: False})
                    gtest_accuracy[global_test_index] = batch_accuracy
                    print("Global accuracy batch %d = %.3f" % (global_test_index, gtest_accuracy[global_test_index]))
                    # 拼接各个batch的预测分类结果、真实分类结果（作为全epoch的结果）
                    gtest_predlabels = rounding.eval(session=sess, feed_dict={x: gtest_data, phase_train: False})
                    global_test_predlabels = np.concatenate((global_test_predlabels, gtest_predlabels), axis=0)
                    gtest_truelabels = np.argmax(gtest_labels, 1)
                    global_test_truelabels = np.concatenate((global_test_truelabels, gtest_truelabels), axis=0)
                # 全epoch的全局正确率
                global_accuracy = reduce_accuracy.eval(session=sess, feed_dict={tab: gtest_accuracy})
                print("Global Test accuracy")
                print(global_accuracy)
                # 计算混淆矩阵以评估分类的准确性。
                print("Confusion_matrix")
                print(confusion_matrix(global_test_predlabels, global_test_truelabels))
                # 打乱训练数据，该轮训练结束后，以新的训练数据序列arr_train开始训练
                np.random.shuffle(arr_train)
                indexes_train = [arr_train[i:i + BATCH_SIZE] for i in range(0, len(arr_train), BATCH_SIZE)]
                train_size = len(indexes_train)
                print("SHUFFLE")
                # 保存训练模型
                saver.save(sess, "my-model20", global_step=ep)

            ##### 2 - Load a network
else:
    print("loading a network")
    saver.restore(sess, network)

    global_test_predlabels = []
    global_test_truelabels = []
    gtest_accuracy = np.ndarray(shape=(test_size), dtype=np.float32)
    for global_test_index in range(test_size - 1):  # 156
        gtest_data, gtest_labels = extract_data_single(indexes_test[global_test_index])
        # print gtest_labels
        batch_accuracy = accuracy.eval(session=sess,
                                       feed_dict={x: gtest_data, y: gtest_labels, phase_train.name: False})
        gtest_accuracy[global_test_index] = batch_accuracy
        print("Global accuracy batch %d = %.2f" % (global_test_index, gtest_accuracy[global_test_index]))
        gtest_predlabels = rounding.eval(session=sess, feed_dict={x: gtest_data, phase_train.name: False})
        # print gtest_predlabels
        global_test_predlabels = np.concatenate((global_test_predlabels, gtest_predlabels), axis=0)
        gtest_truelabels = np.argmax(gtest_labels, 1)
        global_test_truelabels = np.concatenate((global_test_truelabels, gtest_truelabels), axis=0)

    global_accuracy = reduce_accuracy.eval(session=sess, feed_dict={tab: gtest_accuracy})
    print("Global Test accuracy")  # 全局测试准确率
    print(global_accuracy)
    print("Confusion_matrix")
    # 混淆矩阵，预测分类结果与实际分类结果交叉比较，用于评价图片分类精度
    # 参见https://baike.baidu.com/item/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5/10087822?fr=aladdin
    print(confusion_matrix(global_test_predlabels, global_test_truelabels))
