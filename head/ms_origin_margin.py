#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
# @Author : wushijie
# @Time : 2021/5/4 下午 04:38
# @file : ms_origin_margin.py
# @desc: Mis-classified vector guided Loss for FR
'''
import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class Ms_origin_margin(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, t=0.2, easy_margin=False):
        super(Ms_origin_margin, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.t = t
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label,  epoch, img_path):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        phi = phi.clamp(-1, 1)
        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output = output * self.s
        # print(output)

        batch_size = label.detach().size(0)
        # ground truth
        gt = phi.detach()[torch.arange(batch_size), label.reshape(batch_size)]
        gt = gt.unsqueeze(1)

        zero = torch.zeros_like(cosine)
        one = torch.ones_like(cosine)
        temp = 1 - one_hot
        final = temp * torch.where(output.detach() >= gt, one, zero) * self.t
        output = output + final
        output = output.clamp(-1, 1)
        # print(output)
        output = output * self.s
        return output, label


if __name__ == '__main__':
    x = torch.rand(3, 2).cuda()

    labels = torch.randint(high=4, size=(3, 1)).cuda()
    # print('label', labels)

    margin = Ms_origin_margin(in_feature=2, out_feature=4).cuda()
    out, label = margin(x, labels)
    print(out)