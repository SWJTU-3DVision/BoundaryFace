#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2021/5/30 下午 08:46
# @file : boundaryMargin1.py
# @desc:
'''
import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class BoundaryMargin1(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False, epoch_start=7):
        super(BoundaryMargin1, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.epoch_start = epoch_start

    def forward(self, x, label, epoch, img_path):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        if epoch > self.epoch_start:
            right2 = one_hot * cosine.detach()
            left2 = (1.0 - one_hot) * phi.detach()
            left_max2, argmax2 = torch.max(left2.detach(), dim=1)
            max_index2 = argmax2.detach()
            right_max2, _ = torch.max(right2.detach(), dim=1)
            sub2 = left_max2 - right_max2
            zero2 = torch.zeros_like(sub2)
            temp2 = torch.where(sub2 > 0, sub2, zero2)
            # non_zero_index2 = torch.nonzero(temp2.detach(), as_tuple=True)
            non_zero_index2 = torch.nonzero(temp2.detach())
            numpy_index2 = torch.squeeze(non_zero_index2, 1)
            # numpy_index2 = non_zero_index2[0]
            with open(r'/home/xun/wsj/FaceX-Zoo-main/MS1M_saved/savedPath_' + str(epoch) + '.txt', 'a') as f:
                for index in numpy_index2:
                    one_hot_line = torch.zeros((1, self.out_feature)).cuda()
                    one_hot_line.scatter_(1, max_index2[index].unsqueeze(-1).view(-1, 1), 1)
                    one_hot[index] = one_hot_line
                    f.write(img_path[index] + '\t' + str(max_index2.cpu().numpy()[index]) + '\n')
                # f.write('-------batch-------' + '\n')
            rectified_label = torch.topk(one_hot, 1)[1].squeeze(1).cuda()
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s

        if epoch <= self.epoch_start:
            rectified_label = label
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s

        return output, rectified_label


if __name__ == '__main__':
    pass

