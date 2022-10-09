#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2021/6/9 下午 02:46
# @file : generate_outlier_update.py
# @desc: 产生开集噪声
'''
import os
import random
import math

# train_root: 训练集所在路径
# origin_list_path: 训练list文件(直接在该文件上添加外集噪声)
# outliers_num: 要产生的外集噪声数目
# data_outliers_root: 外集噪声来源目录(megaface)
def make_noise(train_root, origin_list_path, rate, outlier_source_root, data_file_outlier_root):
    class_to_num_dict = {}
    class_nums = 10572  # 训练数据集类别数目
    # 统计每个类有多少个样本，形成字典dict
    for root, dirs, files in os.walk(train_root):
        if root == train_root:
            continue
        class_to_num_dict[root.split('\\')[-1]] = len(files)
    # print(class_to_num_dict)
    # print(len(class_to_num_dict))
    # return
    outlier_list = [] # 外集噪声路径列表
    number = 0
    stop = False
    for root, dirs, files in os.walk(outlier_source_root):
        # if root == outlier_source_root:
        #     continue
        for file in files:
            if file.split('.')[-1] == 'json':
                continue
            outlier_list.append(os.path.join(root, file))
            number = number + 1
            # 存储50w外集噪声待使用
            if number > 500000:
                stop = True
                break
        if stop:
            break
    print('一共存储的外集噪声数量：', len(outlier_list))
    random.shuffle(outlier_list)
    # 在origin_list_path上添加20%开集噪声
    f = open(os.path.join(data_file_outlier_root, 'data_file_outlier_' + str(int(rate * 100)) + '.txt'), 'w')
    # 专用来保存添加的外集样本路径
    f1 = open(os.path.join(data_file_outlier_root, 'noise_outlier_' + str(int(rate * 100)) + '.txt'), 'w')

    count = 0

    remain_Iter = -1 # 类剩余要替换的噪声数
    flag = 'xxx'
    with open(origin_list_path) as f2:
        img_label_list = f2.read().splitlines()
        for line in img_label_list:
            img, label = line.split()
            if(img.split('\\')[-2] != flag):
                flag = img.split('\\')[-2]
                class_total_num = class_to_num_dict[flag]
                class_total_noise = math.floor(class_total_num * rate) # 每个类的噪声数量
                remain_Iter = class_total_num - class_total_noise
            if remain_Iter > 0:
                remain_Iter = remain_Iter - 1
                f.write(img + '  ' + label + '\n')
            else:
                random_label = random.randint(0, class_nums - 1)
                if count > number-1:
                    print('now count:', count)
                    count = 0
                    random.shuffle(outlier_list)
                f.write(outlier_list[count] + '  ' + str(random_label) + '\n')
                f1.write(outlier_list[count] + '  ' + str(random_label) + '\n')
                count = count + 1




if __name__ == '__main__':
    train_root = r'H:\train_webface\align_webface_112x112'
    origin_list_path = r'C:\Users\swjtu\Desktop\data_file_flip_10.txt'
    rate = 0.3
    outlier_source_root = r'H:\FlickrFinal2'
    data_file_outlier_root = r'C:\Users\swjtu\Desktop' # 保存产生的外集噪声
    make_noise(train_root, origin_list_path, rate, outlier_source_root, data_file_outlier_root)