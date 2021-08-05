#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
@file: tst_tf_gpu.py
@author: Administrator
@time: 2021.07.29
@description: ...
"""
import tensorflow as tf

hello=tf.constant('hello, tensorflow')
sess=tf.Session()
#测试tensorflow是否可以正常调用
print(sess.run(hello))

#测试gpu是否可以使用
print(tf.test.is_gpu_available())

import keras
#测试keras是否可以正常调用
print(keras.__version__)
# ————————————————
# 版权声明：本文为CSDN博主「zoujiahui_2018」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_18055167/article/details/113789386
