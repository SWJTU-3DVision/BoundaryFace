#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2021/5/28 下午 08:51
# @file : generate_label_flip.py
# @desc:  generate xxx ratio closed-set noise
'''
import os
import random
import numpy as np
import math

def make_noise(dataset_root, rate, data_file_flip_root):
    class_nums = len(os.listdir(dataset_root))
    # print(class_nums)
    label = 0
    f = open(os.path.join(data_file_flip_root, 'data_file_flip_' + str(int(rate * 100)) + '.txt'), 'w')
    f1 = open(os.path.join(data_file_flip_root, 'noise_flip_' + str(int(rate * 100)) + '.txt'), 'w')
    for root, dirs, files in os.walk(dataset_root):
        if root == dataset_root:
            continue
        random.shuffle(files)
        count = 0
        all_count = math.floor(len(files) * rate)
        for file in files:
            if count < all_count:
                random_label = random.randint(0, class_nums-1)
                while random_label == label:
                    random_label = random.randint(0, class_nums - 1)
                f.write(os.path.join(root, file) + '  ' + str(random_label) + '\n')
                f1.write(os.path.join(root, file) + '  ' + str(label) + '\n')
                count = count + 1
            else:
                f.write(os.path.join(root, file) + '  ' + str(label) + '\n')
        label = label + 1
    f.close()
    f1.close()

if __name__ == '__main__':
    dataset_root = r'D:\train_webface\align_webface_112x112'
    rate = 0.2
    data_file_flip_root = r'D:\train_webface\closed_list'
    make_noise(dataset_root, rate, data_file_flip_root)
