# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:40:12 2021
制作训练数据集、测试集之前的图片分拣重命名等操作
@author: Administrator
"""
import os
import numpy as np


class ImageSet(object):
    def __init__(self, imgroot):
        self.img_root_dir = imgroot
        self.img_class_dic = None
    
    def image_rename(self, prefix="", suffix=""):
        """
        取文件夹名称作为类别名称，请提前命名文件夹。
        """
        imgroot = self.img_root_dir
        assert os.path.isdir(imgroot)
        class_names = os.listdir(imgroot)
        class_dic = dict()
        for cname, clabel in zip(class_names, range(len(class_names))):
            class_dic[cname] = clabel
        for cname in class_names:
            imgdir = os.path.join(imgroot, cname)
            assert os.path.isdir(imgdir)
            i = 1
            for imgname in os.listdir(imgdir):
                if not suffix:
                    suffix = os.path.splitext(imgname)[-1]
                newname = prefix + "{}_{}_{}{}".format(cname, class_dic[cname], i, suffix)
                #os.rename(os.path.join(imgdir, imgname), os.path.join(imgdir, newname))  # 文件重命名
                i += 1
            print(cname)
            print("renamed: ", cname, i)
        self.img_class_dic = class_dic
        return 0

    def image_rebatch(self, databatchname='data_batch', batchnum=5, tstbatchname='test_batch'):
        """
        创建batch训练数据集文件夹，将已分类图片重新、平均、随机抽取并放入batch文件夹：
        1.data_batch: 10个分类文件夹文件的每一个文件夹图片：随机、平均的分为6份(图片不重复)，其中5份随机copy或move到batch1~bathc5
        2.test_batch: 10个分类文件夹文件的每一个文件夹图片，随机抽取6分之1数量的，copy或move到test batch
        """
        assert os.path.isdir(self.img_root_dir)
        assert self.img_class_dic
        # 创建batch文件夹
        batch_names = [databatchname + '_' + str(i) for i in range(1, batchnum+1)] + [tstbatchname]

        # 创建用于图片分发的随机numpy array
        one = np.ones(10, dtype=np.int8)
        rand_flags = one
        for i in range(2, batchnum+1):
            rand_flags = np.concatenate((rand_flags, one * i))
            i += 1
        np.random.shuffle(rand_flags)
        print(rand_flags)
        print('shape:', rand_flags.shape)
        # for cname in self.img_root_dir.keys():
        #     imgdir = os.path.join(self.img_root_dir, cname)


if __name__ == "__main__":

    imgroot = r"D:\testimges"
    iset = ImageSet(imgroot)
    iset.image_rename('h1_22', '.JPEG')
    iset.image_rebatch()


