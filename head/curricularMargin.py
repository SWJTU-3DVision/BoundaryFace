import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, device_id=0, m=0.5, s=32.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label, epoch, img_path):

        # print(self.kernel)
        # nu = self.kernel.detach().numpy()
        # print(nu)
        # numpy.savetxt('test.txt', nu)
        # print(type(self.kernel))
        # 1/0
        # embbedings = l2_norm(embbedings, axis = 1)
        embbedings = F.normalize(embbedings)

        # kernel_norm = l2_norm(self.kernel, axis = 0)
        kernel_norm = F.normalize(self.kernel, dim=0)
        #temp_kernel_norm = kernel_norm.cuda(self.device_id[0])
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t

        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, label
        # return output, origin_cos * self.s

if __name__ == '__main__':
    pass