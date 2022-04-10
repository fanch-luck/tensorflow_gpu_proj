
from PIL import Image
from numpy import *
 
import numpy as np
import os
import pickle
 
 
def img_resize(img_dir, img_dim):
    img_resized_dir = img_dir + '_resize'
    os.makedirs(img_resized_dir, exist_ok=True)
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGBA')
            x_new = img_dim
            y_new = img_dim
            out = img.resize((x_new, y_new), Image.ANTIALIAS)
            out.save('{}/{}.png'.format(img_resized_dir, img_name))
            print('Images in {} are resized as {}×{}.\n'.format(img_dir, img_dim, img_dim))
        except Exception:
            continue
    return img_resized_dir


def get_filenames_and_labels(img_resized_dir):
    filenames = []
    labels = []
    img_list = os.listdir(img_resized_dir)
    for img_name in img_list:
        filenames.append(img_name)
        img_name_str = img_name.split('.')[0]
        try:
            label = int(img_name_str.split('_')[1])
            # label = int(img_name_str[-1])
            labels.append(label)
        except ValueError:
            pass
    return filenames, labels
 
 
def get_img_data(img_resized_dir):
    imgs = []
    # count = 0
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
        # count += 1
        # print('Get {} images of {}'.format(count, img_resized_dir))
    imgs = np.array(imgs, dtype=np.uint8)
 
    return imgs
 
 
if __name__ == '__main__':
    img_dir='.\\images'
    img_dim = 32
    img_dir_names = ['test_batch']  # 1个测试批次
    num_data_batch = 5 # 2个训练批次
    for i in range(1, num_data_batch + 1):   
        img_dir_names.append('data_batch_' + str(i))  # 1个测试批次 + 多个训练批次
    count = 0
    for img_dir_name in img_dir_names:
        img_dir = '.\\images\\' + img_dir_name
        filepath = '.\\images_cifar\\' + img_dir_name
        img_resized_dir = img_resize(img_dir, img_dim=32)
 
        data_batch = {}
 
        if 'test' in filepath:
            data_batch['batch_label'] = 'testing batch 1 of 1'
        else:
            count += 1
            batch_label = 'training batch ' + str(count) + ' of ' + str(num_data_batch)
            data_batch['batch_label'] = batch_label
 
        filenames, labels = get_filenames_and_labels(img_resized_dir)
        data = get_img_data(img_resized_dir)
 
        data_batch['filenames'] = filenames
        data_batch['labels'] = labels
        data_batch['data'] = data
 
        with open(filepath, 'wb') as f:
            pickle.dump(data_batch, f)
    
    # meta: 图片类型信息及标签
    img_classes = '.\\images_cifar\\' + 'batches.meta'
    meta = {
        "label_names": ['mglsbstego', 'LSBsteg', 'outguess', 'steghide', 'stego'],
        "num_cases_per_batch": 1300,
        "num_vis": 3072
        }
    with open(img_classes, 'wb') as f:
        pickle.dump(meta, f)
print('----------------------finish------------------------')