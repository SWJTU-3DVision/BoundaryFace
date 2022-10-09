#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2021/5/6 下午 09:42
# @file : my_extract_feature.py
# @desc:
'''
import sys
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append('..')
#from .utils.extractor.feature_extractor import CommonExtractor
from test_protocol.utils.extractor.feature_extractor import CommonExtractor

from data_processor.test_dataset import CommonTestDataset
from backbone import cbam

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='extract features for megaface.')
    conf.add_argument("--data_conf_file", type = str, default='./data_conf.yaml',
                      help = "The path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type = str, default='CBAM_50',
                      help = "CBAM_50")
    conf.add_argument('--batch_size', type = int, default = 512)
    conf.add_argument('--feature_dim', type = int, default = 512)
    conf.add_argument('--model_path', type = str, default = '/home/xun/wsj/FaceX-Zoo-main/training_mode/distributed_training/MS1M_MV-Arc-Softmax/Epoch_23.ckpt',
                      help = 'MS1M-BoundaryF1.ckpt, MS1M_BoundaryFace.ckpt, MS1M_CurricularFace.ckpt, MS1MV2_ArcFace.ckpt, MS1MV2_BoundaryFace.ckpt, MS1MV2_BoundaryFace2.ckpt')
    conf.add_argument('--feats_root', type = str, default = r'/ssd2t/megaface_result/mv-softmax_ms1m_megaface_feats',
                      help = 'arc_ms1m_facescrub_feats, arc_ms1m_megaface_feats')
    args = conf.parse_args()
    with open(args.data_conf_file) as f:
        data_conf = yaml.safe_load(f)['MegaFace']
        cropped_face_folder = data_conf['cropped_face_folder']
        image_list_file = data_conf['image_list_file']
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file, False),
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    # define model.
    if args.backbone_type == 'CBAM_50':
        model = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone_type == 'CBAM_50_SE':
        model = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone_type == 'CBAM_100':
        model = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone_type == 'CBAM_100_SE':
        model = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        pass

    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    model = model.to(device)

    # extract feature.
    feature_extractor = CommonExtractor('cuda:0')
    feature_extractor.extract_offline(args.feats_root, model, data_loader)
