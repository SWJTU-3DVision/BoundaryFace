#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2022/7/19 下午 16:12
# @file : train_DDP.py
# @desc: 对train_DDP.py产生的model做测试 LFW,AgeDB-30,CFP-FP,CALFW,CPLFW,SLLFW,RFW
'''
import numpy as np
import scipy.io
import os
import torch.utils.data
from backbone import cbam
from data_processor.lfw import LFW
from data_processor.agedb import AgeDB30
from data_processor.cfp import CFP_FP
from data_processor.calfw import CALFW
from data_processor.cplfw import CPLFW
from data_processor.rfw import RFW
import torchvision.transforms as transforms
import argparse


rfw_root = []
rfw_file_list = []
rfw_feature_save_path = []


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(feature_path='../../result/cur_epoch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(feature_path)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)

    return ACCs

def loadModel(data_root, file_list, backbone_net, gpus='0', resume=None, flag='None'):

    if backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        print(backbone_net, ' is not available!')

    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # conventional_training use
    # net.load_state_dict(torch.load(resume)['net_state_dict'])
    # distributed_training use
    net.load_state_dict(torch.load(resume)['state_dict'])
    net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    if flag == 'lfw' or flag == 'sllfw':
        dataset = LFW(data_root, file_list, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                 shuffle=False, num_workers=0, drop_last=False)
    elif flag == 'agedb':
        dataset = AgeDB30(data_root, file_list, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                 shuffle=False, num_workers=0, drop_last=False)
    elif flag == 'cfp_fp':
        dataset = CFP_FP(data_root, file_list, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                 shuffle=False, num_workers=0, drop_last=False)
    elif flag == 'calfw':
        dataset = CALFW(data_root, file_list, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=False, num_workers=0, drop_last=False)
    elif flag == 'cplfw':
        dataset = CPLFW(data_root, file_list, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=False, num_workers=0, drop_last=False)
    elif flag == 'rfw':
        dataset = RFW(data_root, file_list, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=512,
                                             shuffle=False, num_workers=0, drop_last=False)
    else:
        print('dataset is None !')
        print('dataloader is None !')
        pass

    return net.eval(), device, dataset, loader

def getFeatureFromTorch(feature_save_dir, net, device, data_set, data_loader):
    featureLs = None
    featureRs = None
    count = 0
    for data in data_loader:
        for i in range(len(data)):
            data[i] = data[i].to(device)
        count += data[0].size(0)
        #print('extracing deep features from the face pair {}...'.format(count))
        with torch.no_grad():
            res = [net(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        # print(featureL.shape, featureR.shape)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureLs.shape, featureRs.shape)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': data_set.folds, 'flag': data_set.flags}
    scipy.io.savemat(feature_save_dir, result)


def testAccOnLfw(data_root, file_list, backbone_net, gpus, resume, feature_save_path, current_iter):
    net, device, lfw_dataset, lfw_loader = loadModel(data_root, file_list, backbone_net, gpus,
                                                     resume, 'lfw')
    getFeatureFromTorch(feature_save_path, net, device, lfw_dataset, lfw_loader)
    ACCs = evaluation_10_fold(feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('LFW AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('LFW Iter', current_iter)

def testAccOnAgeDB(data_root, file_list, backbone_net, gpus, resume, feature_save_path, current_iter):
    net, device, agedb_dataset, agedb_loader = loadModel(data_root, file_list, backbone_net, gpus,
                                                     resume, 'agedb')
    getFeatureFromTorch(feature_save_path, net, device, agedb_dataset, agedb_loader)
    ACCs = evaluation_10_fold(feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('AgeDB-30 AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('AgeDB-30 Current_Iter', current_iter)


def testAccOnCFP_FP(data_root, file_list, backbone_net, gpus, resume, feature_save_path, current_iter):
    net, device, cfp_fp_dataset, cfp_fp_loader = loadModel(data_root, file_list, backbone_net, gpus,
                                                     resume, 'cfp_fp')
    getFeatureFromTorch(feature_save_path, net, device, cfp_fp_dataset, cfp_fp_loader)
    ACCs = evaluation_10_fold(feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('CFP_FP AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('CFP_FP Current_Iter', current_iter)

def testAccOnCALFW(data_root, file_list, backbone_net, gpus, resume, feature_save_path, current_iter):
    net, device, calfw_dataset, calfw_loader = loadModel(data_root, file_list, backbone_net, gpus,
                                                     resume, 'calfw')
    getFeatureFromTorch(feature_save_path, net, device, calfw_dataset, calfw_loader)
    ACCs = evaluation_10_fold(feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('CALFW AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('CALFW Current_Iter', current_iter)

def testAccOnCPLFW(data_root, file_list, backbone_net, gpus, resume, feature_save_path, current_iter):
    net, device, cplfw_dataset, cplfw_loader = loadModel(data_root, file_list, backbone_net, gpus,
                                                     resume, 'cplfw')
    getFeatureFromTorch(feature_save_path, net, device, cplfw_dataset, cplfw_loader)
    ACCs = evaluation_10_fold(feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('CPLFW AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('CPLFW Current_Iter', current_iter)

def testAccOnSLLFW(data_root, file_list, backbone_net, gpus, resume, feature_save_path, current_iter):
    net, device, sllfw_dataset, sllfw_loader = loadModel(data_root, file_list, backbone_net, gpus,
                                                     resume, 'sllfw')
    getFeatureFromTorch(feature_save_path, net, device, sllfw_dataset, sllfw_loader)
    ACCs = evaluation_10_fold(feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('SLLFW AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('SLLFW Current_Iter', current_iter)

def testAccOnRFW(data_root, file_list, backbone_net, gpus, resume, feature_save_path, current_iter):
    print('------------' * 2)
    print('------------' * 2)
    # Asian
    net, device, Asian_dataset, Asian_loader = loadModel(data_root[0], file_list[0], backbone_net, gpus,
                                                     resume, 'rfw')
    getFeatureFromTorch(feature_save_path[0], net, device, Asian_dataset, Asian_loader)
    ACCs = evaluation_10_fold(feature_save_path[0])
    # for i in range(len(ACCs)):
    #     print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    # print('--------'*2)
    print('Asian AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('Asian Current_Iter', current_iter)

    # Caucasian
    net, device, Caucasian_dataset, Caucasian_loader = loadModel(data_root[1], file_list[1], backbone_net, gpus,
                                                     resume, 'rfw')
    getFeatureFromTorch(feature_save_path[1], net, device, Caucasian_dataset, Caucasian_loader)
    ACCs = evaluation_10_fold(feature_save_path[1])
    # for i in range(len(ACCs)):
    #     print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------'*2)
    print('Caucasian AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('Caucasian Current_Iter', current_iter)

    # Indian
    net, device, Indian_dataset, Indian_loader = loadModel(data_root[2], file_list[2], backbone_net, gpus,
                                                     resume, 'rfw')
    getFeatureFromTorch(feature_save_path[2], net, device, Indian_dataset, Indian_loader)
    ACCs = evaluation_10_fold(feature_save_path[2])
    # for i in range(len(ACCs)):
    #     print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------'*2)
    print('Indian AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('Indian Current_Iter', current_iter)

    # African
    net, device, African_dataset, African_loader = loadModel(data_root[3], file_list[3], backbone_net, gpus,
                                                     resume, 'rfw')
    getFeatureFromTorch(feature_save_path[3], net, device, African_dataset, African_loader)
    ACCs = evaluation_10_fold(feature_save_path[3])
    # for i in range(len(ACCs)):
    #     print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------'*2)
    print('African AVE    {:.4f}'.format(np.mean(ACCs) * 100))
    print('African Current_Iter', current_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--backbone_net', type=str, default='CBAM_50', help='CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--gpus', type=str, default='0', help='gpu number')


    # 以下四个可变
    # common  (起始的权重文件路径)
    parser.add_argument('--resume', type=str, default='F:/MS1M_ArcFace/Epoch_0.ckpt',
                        help='The path pf save model')
    # LFW相关路径
    parser.add_argument('--lfw_root', type=str, default='D:/LFW/lfw_align_112', help='The path of lfw data')
    parser.add_argument('--lfw_file_list', type=str, default='D:/LFW/pairs.txt', help='The path of lfw data')
    parser.add_argument('--lfw_feature_save_path', type=str, default='../../result/cur_epoch_lfw_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # AgeDB-30相关路径
    parser.add_argument('--agedb_root', type=str, default=r'D:/AgeDB-30/agedb30_align_112', help='The path of agedb data')
    parser.add_argument('--agedb_file_list', type=str, default='D:/AgeDB-30/agedb_30_pair.txt', help='The path of agedb data')
    parser.add_argument('--agedb_feature_save_path', type=str, default='../../result/cur_epoch_agedb_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # CFP-FP相关路径
    parser.add_argument('--cfp_fp_root', type=str, default='D:/CFP-FP/CFP_FP_aligned_112', help='The path of cfp_fp data')
    parser.add_argument('--cfp_fp_file_list', type=str, default='D:/CFP-FP/cfp_fp_pair.txt', help='The path of cfp_fp data')
    parser.add_argument('--cfp_fp_feature_save_path', type=str, default='../../result/cur_epoch_cfp_fp_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # CALFW相关路径
    parser.add_argument('--calfw_root', type=str, default='D:/calfw/calfw',
                        help='The path of calfw data')
    parser.add_argument('--calfw_file_list', type=str, default='D:/calfw/calfw_pair.txt',
                        help='The path of calfw data')
    parser.add_argument('--calfw_feature_save_path', type=str, default='../../result/cur_epoch_calfw_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # CPLFW相关路径
    parser.add_argument('--cplfw_root', type=str, default='D:/cplfw/cplfw',
                        help='The path of cplfw data')
    parser.add_argument('--cplfw_file_list', type=str, default='D:/cplfw/cplfw_pair.txt',
                        help='The path of cplfw data')
    parser.add_argument('--cplfw_feature_save_path', type=str, default='../../result/cur_epoch_cplfw_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # SLLFW相关路径
    parser.add_argument('--sllfw_root', type=str, default='D:/LFW/lfw_align_112',
                        help='The path of sllfw data')
    parser.add_argument('--sllfw_file_list', type=str, default='D:/LFW/SLLFW_changed.txt',
                        help='The path of sllfw data')
    parser.add_argument('--sllfw_feature_save_path', type=str, default='../../result/cur_epoch_sllfw_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # RFW相关路径
    # Asian
    parser.add_argument('--Asian_root', type=str, default='D:/RFW_crop/Asian',
                        help='The path of Asian data')
    parser.add_argument('--Asian_file_list', type=str, default='D:/txts/Asian/Asian_pairs.txt',
                        help='The path of Asian data')
    parser.add_argument('--Asian_feature_save_path', type=str, default='../../result/cur_epoch_Asian_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # Caucasian
    parser.add_argument('--Caucasian_root', type=str, default='D:/RFW_crop/Caucasian',
                        help='The path of Caucasian data')
    parser.add_argument('--Caucasian_file_list', type=str, default='D:/txts/Caucasian/Caucasian_pairs.txt',
                        help='The path of Caucasian data')
    parser.add_argument('--Caucasian_feature_save_path', type=str, default='../../result/cur_epoch_Caucasian_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # Indian
    parser.add_argument('--Indian_root', type=str, default='D:/RFW_crop/Indian',
                        help='The path of Indian data')
    parser.add_argument('--Indian_file_list', type=str, default='D:/txts/Indian/Indian_pairs.txt',
                        help='The path of Indian data')
    parser.add_argument('--Indian_feature_save_path', type=str, default='../../result/cur_epoch_Indian_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # African
    parser.add_argument('--African_root', type=str, default='D:/RFW_crop/African',
                        help='The path of African data')
    parser.add_argument('--African_file_list', type=str, default='D:/txts/African/African_pairs.txt',
                        help='The path of African data')
    parser.add_argument('--African_feature_save_path', type=str, default='../../result/cur_epoch_African_result.mat',
                        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()

    rfw_root.append(args.Asian_root)
    rfw_root.append(args.Caucasian_root)
    rfw_root.append(args.Indian_root)
    rfw_root.append(args.African_root)

    rfw_file_list.append(args.Asian_file_list)
    rfw_file_list.append(args.Caucasian_file_list)
    rfw_file_list.append(args.Indian_file_list)
    rfw_file_list.append(args.African_file_list)

    rfw_feature_save_path.append(args.Asian_feature_save_path)
    rfw_feature_save_path.append(args.Caucasian_feature_save_path)
    rfw_feature_save_path.append(args.Indian_feature_save_path)
    rfw_feature_save_path.append(args.African_feature_save_path)
    print('load model at:', args.resume)

    # 分别在各个数据集上进行测试
    LFW_flag = True
    AgeDB_flag = True
    CFP_FP_flag = True
    CALFW_flag = True
    CPLFW_flag = True
    SLLFW_flag = True
    RFW_flag = True

    # LFW
    if LFW_flag:
        testAccOnLfw(args.lfw_root, args.lfw_file_list, args.backbone_net, args.gpus, args.resume,
                     args.lfw_feature_save_path, args.resume.split('/')[-1].split('_')[1])

    # AgeDB-30
    if AgeDB_flag:
        testAccOnAgeDB(args.agedb_root, args.agedb_file_list, args.backbone_net, args.gpus, args.resume,
                     args.agedb_feature_save_path, args.resume.split('/')[-1].split('_')[1])

    # CFP_FP
    if CFP_FP_flag:
        testAccOnCFP_FP(args.cfp_fp_root, args.cfp_fp_file_list, args.backbone_net, args.gpus, args.resume,
                     args.cfp_fp_feature_save_path, args.resume.split('/')[-1].split('_')[1])

    # CALFW
    if CALFW_flag:
        testAccOnCALFW(args.calfw_root, args.calfw_file_list, args.backbone_net, args.gpus, args.resume,
                        args.calfw_feature_save_path, args.resume.split('/')[-1].split('_')[1])

    # CPLFW
    if CPLFW_flag:
        testAccOnCPLFW(args.cplfw_root, args.cplfw_file_list, args.backbone_net, args.gpus, args.resume,
                        args.cplfw_feature_save_path, args.resume.split('/')[-1].split('_')[1])
      

    # SLLFW
    if SLLFW_flag:
        testAccOnSLLFW(args.sllfw_root, args.sllfw_file_list, args.backbone_net, args.gpus, args.resume,
                       args.sllfw_feature_save_path, args.resume.split('/')[-1].split('_')[1])

    # RFW
    if RFW_flag:
        testAccOnRFW(rfw_root, rfw_file_list, args.backbone_net, args.gpus, args.resume,
                       rfw_feature_save_path, args.resume.split('/')[-1].split('_')[1])
        




