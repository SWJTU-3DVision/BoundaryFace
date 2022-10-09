#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2022/7/10 下午 03:50
# @file : train.py
# @desc: torch:1.1.0 torchvisioin:0.3.0
'''

import os
import sys
import torch.utils.data
sys.path.append('../../')
from datetime import datetime
from backbone.cbam import CBAMResNet
from utils.visualize import Visualizer
from utils.log import init_log
from data_processor.train_dataset import CASIAWebFace
from data_processor.lfw import LFW
from data_processor.calfw import CALFW
from data_processor.cplfw import CPLFW
from data_processor.agedb import AgeDB30
from data_processor.cfp import CFP_FP
from data_processor.rfw import RFW
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from eval.eval_lfw import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse

from head.ArcFace import ArcMarginProduct
from head.boundaryMargin1 import BoundaryMargin1
from head.boundaryMargin import BoundaryMargin
from head.ms_origin_margin import Ms_origin_margin
from head.curricularMargin import CurricularFace


def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = os.path.join(args.save_dir, args.model_pre + args.backbone.upper() + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    _print("file list: {}".format(args.train_file_list))
    _print("backbone: {}".format(args.backbone))
    _print("margin_type: {}".format(args.margin_type))
    _print("information: ")
    _print("s: {}".format(args.scale_size))
    _print("batchsize: {}".format(args.batch_size))
    _print("resume: {}".format(args.resume))
    _print("feature_dim: {}".format(args.feature_dim))


    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    # validation dataset
    trainset = CASIAWebFace(args.train_root, args.train_file_list, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
    # test dataset
    lfwdataset = LFW(args.lfw_test_root, args.lfw_file_list, transform=transform)
    lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=128,
                                             shuffle=False, num_workers=0, drop_last=False)
    calfwdataset = CALFW(args.calfw_test_root, args.calfw_file_list, transform=transform)
    calfwloader = torch.utils.data.DataLoader(calfwdataset, batch_size=128,
                                             shuffle=False, num_workers=0, drop_last=False)
    cplfwdataset = CPLFW(args.cplfw_test_root, args.cplfw_file_list, transform=transform)
    cplfwloader = torch.utils.data.DataLoader(cplfwdataset, batch_size=128,
                                             shuffle=False, num_workers=0, drop_last=False)
    sllfwdataset = LFW(args.sllfw_test_root, args.sllfw_file_list, transform=transform)
    sllfwloader = torch.utils.data.DataLoader(sllfwdataset, batch_size=128,
                                             shuffle=False, num_workers=0, drop_last=False)
    agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, transform=transform)
    agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=128,
                                            shuffle=False, num_workers=0, drop_last=False)
    cfpfpdataset = CFP_FP(args.cfpfp_test_root, args.cfpfp_file_list, transform=transform)
    cfpfploader = torch.utils.data.DataLoader(cfpfpdataset, batch_size=128,
                                              shuffle=False, num_workers=0, drop_last=False)
    Asian_dataset = RFW(args.Asian_root, args.Asian_file_list, transform=transform)
    Asian_loader = torch.utils.data.DataLoader(Asian_dataset, batch_size=128,
                                         shuffle=False, num_workers=0, drop_last=False)
    Caucasian_dataset = RFW(args.Caucasian_root, args.Caucasian_file_list, transform=transform)
    Caucasian_loader = torch.utils.data.DataLoader(Caucasian_dataset, batch_size=128,
                                         shuffle=False, num_workers=0, drop_last=False)
    Indian_dataset = RFW(args.Indian_root, args.Indian_file_list, transform=transform)
    Indian_loader = torch.utils.data.DataLoader(Indian_dataset, batch_size=128,
                                         shuffle=False, num_workers=0, drop_last=False)
    African_dataset = RFW(args.African_root, args.African_file_list, transform=transform)
    African_loader = torch.utils.data.DataLoader(African_dataset, batch_size=128,
                                         shuffle=False, num_workers=0, drop_last=False)

    # define backbone and margin layer
    if args.backbone == 'Res50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Res100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        print(args.backbone, ' is not available!')

    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'boundaryMargin1':
        margin = BoundaryMargin1(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'boundaryMargin':
        margin = BoundaryMargin(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'MV-Arc-Softmax':
        margin = Ms_origin_margin(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'curricularMargin':
        margin = CurricularFace(args.feature_dim, trainset.class_nums)
    else:
        print(args.margin_type, 'is not available!')

    currentEpoch = 1
    last_epoch_test = -1
    if args.resume:
        print('resume the model parameters from: ', args.net_path, args.margin_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
        iters = torch.load(args.net_path)['iters']
        # print(iters)
        currentEpoch = torch.load(args.net_path)['currentEpoch']
        last_epoch_test = currentEpoch - 1
        print('当前模型的epoch:', currentEpoch)
        margin.load_state_dict(torch.load(args.margin_path)['net_state_dict'])

    lr_test = 0.1
    if currentEpoch>=6 and currentEpoch<12:
        lr_test = 0.01
    elif currentEpoch>=12 and currentEpoch<19:
        lr_test = 0.001
    elif currentEpoch>=19:
        lr_test = 0.0001
    print('对应学习率:', lr_test)
    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4, 'initial_lr': lr_test},
        {'params': margin.parameters(), 'weight_decay': 5e-4, 'initial_lr': lr_test}
    ], lr=lr_test, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 12, 19], gamma=0.1, last_epoch=last_epoch_test) # for webface 30
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10, 18, 22], gamma=0.1, last_epoch=last_epoch_test) # for ms1mv2 24

    net = net.to(device)
    margin = margin.to(device)

    total_iters = 0
    vis = Visualizer(env=args.model_pre + args.backbone)
    for epoch in range(currentEpoch, args.total_epoch + 1):
        exp_lr_scheduler.step()
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        net.train()

        since = time.time()
        epoch_start_time = time.time()
        for data in trainloader:
            img, label = data[0].to(device), data[1].to(device)
            img_path = data[2]
            optimizer_ft.zero_grad()

            raw_logits = net(img)
            output, rectified_label = margin(raw_logits, label, epoch, img_path)
            total_loss = criterion(output, rectified_label)

            # boundaryMargin
            # output, right, rectified_label = margin(raw_logits, label, epoch, img_path)
            # total_loss = criterion(output, rectified_label) + right

            total_loss.backward()


            optimizer_ft.step()

            total_iters += 1
            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = rectified_label.size(0)
                correct = (np.array(predict.cpu()) == np.array(rectified_label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                vis.plot_curves({'softmax loss': total_loss.item()}, iters=total_iters, title='train loss',
                                xlabel='iters', ylabel='train loss')
                vis.plot_curves({'train accuracy': correct / total}, iters=total_iters, title='train accuracy', xlabel='iters',
                                ylabel='train accuracy')

                _print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, epoch, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'currentEpoch': epoch,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'currentEpoch': epoch,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.test_freq == 0:
                net.eval()
                # test model on lfw
                getFeatureFromTorch('../../result/cur_lfw_result.mat', net, device, lfwdataset, lfwloader)
                lfw_accs = evaluation_10_fold('../../result/cur_lfw_result.mat')
                _print('LFW Ave Accuracy: {:.4f}'.format(np.mean(lfw_accs) * 100))

                # test model on AgeDB30
                getFeatureFromTorch('../../result/cur_agedb30_result.mat', net, device, agedbdataset, agedbloader)
                age_accs = evaluation_10_fold('../../result/cur_agedb30_result.mat')
                _print('AgeDB-30 Ave Accuracy: {:.4f}'.format(np.mean(age_accs) * 100))

                # test model on CFP-FP
                getFeatureFromTorch('../../result/cur_cfpfp_result.mat', net, device, cfpfpdataset, cfpfploader)
                cfp_accs = evaluation_10_fold('../../result/cur_cfpfp_result.mat')
                _print('CFP-FP Ave Accuracy: {:.4f}'.format(np.mean(cfp_accs) * 100))

                # test model on calfw
                getFeatureFromTorch('../../result/cur_epoch_calfw_result.mat', net, device, calfwdataset, calfwloader)
                calfw_accs = evaluation_10_fold('../../result/cur_epoch_calfw_result.mat')
                _print('CALFW Ave Accuracy: {:.4f}'.format(np.mean(calfw_accs) * 100))

                # test model on cplfw
                getFeatureFromTorch('../../result/cur_epoch_cplfw_result.mat', net, device, cplfwdataset, cplfwloader)
                cplfw_accs = evaluation_10_fold('../../result/cur_epoch_cplfw_result.mat')
                _print('CPLFW Ave Accuracy: {:.4f}'.format(np.mean(cplfw_accs) * 100))

                # test model on sllfw
                getFeatureFromTorch('../../result/cur_epoch_sllfw_result.mat', net, device, sllfwdataset, sllfwloader)
                sllfw_accs = evaluation_10_fold('../../result/cur_epoch_sllfw_result.mat')
                _print('SLLFW Ave Accuracy: {:.4f}'.format(np.mean(sllfw_accs) * 100))

                # test model on RFW
                if args.RFW_Test:
                    getFeatureFromTorch(args.Asian_feature_save_path, net, device, Asian_dataset, Asian_loader)
                    Asian_accs = evaluation_10_fold(args.Asian_feature_save_path)
                    print('--' * 13)
                    _print('Asian Ave Accuracy: {:.4f}'.format(np.mean(Asian_accs) * 100))

                    getFeatureFromTorch(args.Caucasian_feature_save_path, net, device, Caucasian_dataset, Caucasian_loader)
                    Caucasian_accs = evaluation_10_fold(args.Caucasian_feature_save_path)
                    _print('Caucasian Ave Accuracy: {:.4f}'.format(np.mean(Caucasian_accs) * 100))

                    getFeatureFromTorch(args.Indian_feature_save_path, net, device, Indian_dataset, Indian_loader)
                    Indian_accs = evaluation_10_fold(args.Indian_feature_save_path)
                    _print('Indian Ave Accuracy: {:.4f}'.format(np.mean(Indian_accs) * 100))

                    getFeatureFromTorch(args.African_feature_save_path, net, device, African_dataset, African_loader)
                    African_accs = evaluation_10_fold(args.African_feature_save_path)
                    _print('African Ave Accuracy: {:.4f}'.format(np.mean(African_accs) * 100))

                if args.RFW_Test:
                    vis.plot_curves(
                        {'lfw': np.mean(lfw_accs), 'agedb-30': np.mean(age_accs), 'cfp-fp': np.mean(cfp_accs),
                         'calfw': np.mean(calfw_accs), 'cplfw': np.mean(cplfw_accs), 'sllfw': np.mean(sllfw_accs),
                         'Asian': np.mean(Asian_accs), 'Caucasian': np.mean(Caucasian_accs), 'Indian': np.mean(Indian_accs), 'African': np.mean(African_accs)},
                        iters=total_iters,
                        title='test accuracy', xlabel='iters', ylabel='test accuracy')
                else:
                    vis.plot_curves({'lfw': np.mean(lfw_accs), 'agedb-30': np.mean(age_accs), 'cfp-fp': np.mean(cfp_accs), 'calfw': np.mean(calfw_accs), 'cplfw': np.mean(cplfw_accs), 'sllfw': np.mean(sllfw_accs)}, iters=total_iters,
                                title='test accuracy', xlabel='iters', ylabel='test accuracy')
                net.train()
        epoch_end_time = time.time()
        print(f"一次epoch训练时间为：{epoch_end_time-epoch_start_time}")
        # exp_lr_scheduler.step()
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--train_root', type=str, default='H:/train_webface/align_webface_112x112',
                        help='train image root')
    parser.add_argument('--train_file_list', type=str, default='H:/train_webface/data_file_flip_20.txt',
                        help='train list')
    parser.add_argument('--lfw_test_root', type=str, default='H:/LFW/lfw_align_112', help='lfw image root')
    parser.add_argument('--lfw_file_list', type=str, default='H:/LFW/pairs.txt', help='lfw pair file list')
    parser.add_argument('--calfw_test_root', type=str, default='H:/calfw/calfw', help='calfw image root')
    parser.add_argument('--calfw_file_list', type=str, default='H:/calfw/calfw_pair.txt', help='calfw pair file list')
    parser.add_argument('--cplfw_test_root', type=str, default='H:/cplfw/cplfw', help='cplfw image root')
    parser.add_argument('--cplfw_file_list', type=str, default='H:/cplfw/cplfw_pair.txt', help='cplfw pair file list')
    parser.add_argument('--sllfw_test_root', type=str, default='H:/LFW/lfw_align_112', help='sllfw image root')
    parser.add_argument('--sllfw_file_list', type=str, default='H:/LFW/SLLFW_changed.txt', help='sllfw pair file list')
    parser.add_argument('--agedb_test_root', type=str, default='H:/AgeDB-30/agedb30_align_112', help='agedb image root')
    parser.add_argument('--agedb_file_list', type=str, default='H:/AgeDB-30/agedb_30_pair.txt',
                        help='agedb pair file list')
    parser.add_argument('--cfpfp_test_root', type=str, default='H:/CFP-FP/CFP_FP_aligned_112', help='agedb image root')
    parser.add_argument('--cfpfp_file_list', type=str, default='H:/CFP-FP/cfp_fp_pair.txt', help='agedb pair file list')
    # RFW相关路径
    # Asian
    parser.add_argument('--Asian_root', type=str, default='H:/RFW_crop/Asian',
                        help='The path of Asian data')
    parser.add_argument('--Asian_file_list', type=str, default='H:/txts/Asian/Asian_pairs.txt',
                        help='The path of Asian data')
    parser.add_argument('--Asian_feature_save_path', type=str, default='../../result/cur_epoch_Asian_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # Caucasian
    parser.add_argument('--Caucasian_root', type=str, default='H:/RFW_crop/Caucasian',
                        help='The path of Caucasian data')
    parser.add_argument('--Caucasian_file_list', type=str, default='H:/txts/Caucasian/Caucasian_pairs.txt',
                        help='The path of Caucasian data')
    parser.add_argument('--Caucasian_feature_save_path', type=str, default='../../result/cur_epoch_Caucasian_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # Indian
    parser.add_argument('--Indian_root', type=str, default='H:/RFW_crop/Indian',
                        help='The path of Indian data')
    parser.add_argument('--Indian_file_list', type=str, default='H:/txts/Indian/Indian_pairs.txt',
                        help='The path of Indian data')
    parser.add_argument('--Indian_feature_save_path', type=str, default='../../result/cur_epoch_Indian_result.mat',
                        help='The path of the extract features save, must be .mat file')

    # African
    parser.add_argument('--African_root', type=str, default='H:/RFW_crop/African',
                        help='The path of African data')
    parser.add_argument('--African_file_list', type=str, default='H:/txts/African/African_pairs.txt',
                        help='The path of African data')
    parser.add_argument('--African_feature_save_path', type=str, default='../../result/cur_epoch_African_result.mat',
                        help='The path of the extract features save, must be .mat file')

    parser.add_argument('--backbone', type=str, default='Res50_IR',
                        help='Res50_IR, SERes50_IR, Res100_IR, SERes100_IR')
    parser.add_argument('--margin_type', type=str, default='boundaryMargin',
                        help='curricularMargin, MV-Arc-Softmax, boundaryMargin, boundaryMargin1, ArcFace')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=30, help='total epochs')

    parser.add_argument('--save_freq', type=int, default=3000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=3000, help='test frequency')
    parser.add_argument('--resume', type=int, default=False, help='resume model')
    parser.add_argument('--net_path', type=str,
                        default=r'H:\model\ArcFace_filp30_open10_RES50_IR_20220825_005739/Iter_030000_net.ckpt',
                        help='resume model')
    parser.add_argument('--margin_path', type=str,
                        default=r'H:\model\ArcFace_filp30_open10_RES50_IR_20220825_005739/Iter_033000_margin.ckpt',
                        help='resume model')
    parser.add_argument('--save_dir', type=str, default='H:/model', help='model save dir')
    parser.add_argument('--model_pre', type=str, default='BoundaryFace_filp20_', help='model prefix')
    parser.add_argument('--RFW_Test', type=int, default=True, help='whether or not test model on RFW')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')

    args = parser.parse_args()

    train(args)