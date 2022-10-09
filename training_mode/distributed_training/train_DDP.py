#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2022/7/6 下午 12:00
# @file : train_DDP.py
# @desc:
'''
import os
import sys
import shutil
import argparse
import logging as logger

import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
import torchvision.transforms as transforms
from data_processor.train_dataset import CASIAWebFace


# 日志保存到当前文件所在目录的log目录下
logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   filename=os.path.join('log', 'boundaryFace0.3_ms1m_log.log'),
                   datefmt='%Y-%m-%d %H:%M:%S')
torch.backends.cudnn.benchmark = True


class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, backbone_factory, head_factory):
        """Init face model by backbone factorcy and head factory.

        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head()

    def forward(self, data, label, epoch, img_path, flag=None):
        feat = self.backbone(data)
        if flag == None:
            pred, rectified_label = self.head(feat, label, epoch, img_path)
            return pred, rectified_label
        else:
            pred, right, rectified_label = self.head(feat, label, epoch, img_path)
            return pred, right, rectified_label


def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, args):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images, labels, img_path) in enumerate(data_loader):
        images = images.to(args.local_rank)
        labels = labels.to(args.local_rank)
        labels = labels.squeeze()
        if args.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        elif args.head_type == 'boundaryMargin':
            outputs, right, rectified_label = model(images, labels, cur_epoch, img_path, flag='mining')
            loss = criterion(outputs, rectified_label) + right
        else:
            outputs, rectified_label = model(images, labels, cur_epoch, img_path)
            loss = criterion(outputs, rectified_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        if args.local_rank == 0 and batch_idx % args.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            # print(loss.item())
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' %
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            args.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            args.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        # if args.local_rank == 0 and (batch_idx + 1) % args.save_freq == 0:
        #     saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
        #     state = {
        #         'state_dict': model.module.state_dict(),
        #         'epoch': cur_epoch,
        #         'batch_id': batch_idx
        #     }
        #     torch.save(state, os.path.join(args.out_dir, saved_name))
        #     logger.info('Save checkpoint %s to disk.' % saved_name)
    if args.local_rank == 0:
        saved_name = 'Epoch_%d.ckpt' % cur_epoch
        state = {'state_dict': model.module.backbone.state_dict(),
                 'epoch': cur_epoch, 'batch_id': batch_idx}
        torch.save(state, os.path.join(args.out_dir, saved_name))
        logger.info('Save checkpoint %s to disk...' % saved_name)

def train(args):
    """Total training procedure.
    """
    print("Use GPU: {} for training".format(args.local_rank))
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir=args.tensorboardx_logdir)
        args.writer = writer
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        console = logger.StreamHandler()
        console.setLevel(logger.INFO)
        logger.getLogger('').addHandler(console)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    args.rank = dist.get_rank()
    print('args.rank: ', dist.get_rank())
    print('args.get_world_size: ', dist.get_world_size())
    print('is_nccl_available: ', dist.is_nccl_available())
    args.world_size = dist.get_world_size()
    # trainset = ImageDataset(args.data_root, args.train_file)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    trainset = CASIAWebFace(args.data_root, args.train_file, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)

    # 原来
    # backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    # head_factory = HeadFactory(args.head_type, args.head_conf_file)
    # model = FaceModel(backbone_factory, head_factory)
    # model = model.to(args.local_rank)
    # model.train()
    # for ps in model.parameters():
    #     dist.broadcast(ps, 0)
    # # DDP
    # model = torch.nn.parallel.DistributedDataParallel(
    #     module=model,
    #     broadcast_buffers=False,
    #     device_ids=[args.local_rank]
    # )
    # criterion = torch.nn.CrossEntropyLoss().to(args.local_rank)
    # ori_epoch = 0
    # parameters = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(parameters, lr=args.lr,
    #                       momentum=args.momentum, weight_decay=1e-4)
    # lr_schedule = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.milestones, gamma=0.1)
    # loss_meter = AverageMeter()

    # 现在
    criterion = torch.nn.CrossEntropyLoss().to(args.local_rank)
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    head_factory = HeadFactory(args.head_type, args.head_conf_file)
    model = FaceModel(backbone_factory, head_factory)
    model = model.to(args.local_rank)
    model.train()
    for ps in model.parameters():
        #dist.broadcast(ps, 0, async_op=True)
        dist.broadcast(ps, 0)
    # DDP
    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        broadcast_buffers=False,
        device_ids=[args.local_rank]
    )
    optimizer = optim.SGD([
        {'params': model.module.backbone.parameters(), 'weight_decay': 5e-4, 'initial_lr': args.lr},
        {'params': model.module.head.parameters(), 'weight_decay': 5e-4, 'initial_lr': args.lr}
    ], lr=args.lr, momentum=args.momentum, nesterov=True)
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)
    # model = model.to(args.local_rank)
    ori_epoch = 0
    loss_meter = AverageMeter()

    for epoch in range(ori_epoch, args.epoches):
        # model.train()
        train_one_epoch(train_loader, model, optimizer,
                        criterion, epoch, loss_meter, args)
        dist.barrier()
        lr_schedule.step()
    dist.destroy_process_group()


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument('--local_rank', type=int, default=0, help='local_rank')
    conf.add_argument("--data_root", type=str, default='/home/xun/datasets/train_ms1m/MS1M-Origin-img',
                      help="The root folder of training set.")
    conf.add_argument("--train_file", type=str, default='/home/xun/datasets/train_ms1m/ms1m_list.txt',
                      help="The training file path. data_file_flip_30.txt webface_list.txt")
    conf.add_argument("--backbone_type", type=str, default='Res50_IR',
                      help="Res50_IR.")
    conf.add_argument("--backbone_conf_file", type=str, default='../backbone_conf.yaml',
                      help="the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type=str, default='boundaryMargin',
                      help="MvArcMargin, curricularMargin, boundaryMargin, boundaryMargin1, ArcFace.")
    conf.add_argument("--head_conf_file", type=str, default='../head_conf.yaml',
                      help="the path of head_conf.yaml.")
    conf.add_argument('--lr', type=float, default=0.1,
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str, default='MS1M_BoundaryFace0.3',
                      help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=24,
                      help='The training epoches.')
    conf.add_argument('--step', type=str, default='10,18,22',
                      help='Step for lr.')
    conf.add_argument('--print_freq', type=int, default=100,
                      help='The print frequency for training state.')
    conf.add_argument('--save_freq', type=int, default=3000,
                      help='The save frequency for training state.')
    conf.add_argument('--batch_size', type=int, default=32,
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type=float, default=0.9,
                      help='The momentum for sgd.')
    # log日志存放位置
    conf.add_argument('--log_dir', type=str, default='log',
                      help='The directory to save log.log')
    # 打开 tensorboard: tensorboard --logdir ./training_mode/distributed_training/MS1M_BoundaryFace0.3_tensorboard
    conf.add_argument('--tensorboardx_logdir', type=str, default='MS1M_BoundaryFace0.3_tensorboard',
                      help='The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type=str, default='mv_epoch_8.pt',
                      help='The path of pretrained model')
    conf.add_argument('--resume', '-r', action='store_true', default=False,
                      help='Whether to resume from a checkpoint.')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    train(args)
