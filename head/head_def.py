import sys
import yaml
sys.path.append('../../')
from head.boundaryMargin1 import BoundaryMargin1
from head.boundaryMargin import BoundaryMargin
from head.ArcFace import ArcMarginProduct
from head.curricularMargin import CurricularFace
from head.ms_origin_margin import Ms_origin_margin

class HeadFactory:
    """Factory to produce head according to the head_conf.yaml
    
    Attributes:
        head_type(str): which head will be produce.
        head_param(dict): parsed params and it's value.
    """
    def __init__(self, head_type, head_conf_file):
        self.head_type = head_type
        with open(head_conf_file) as f:
            head_conf = yaml.safe_load(f)
            # head_conf = yaml.load(f) 原始
            self.head_param = head_conf[head_type]
        print('head param:')
        print(self.head_param)
    def get_head(self):
        if self.head_type == 'ArcFace':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.head_param['num_class'] # number of classes in the training set.
            scale = self.head_param['scale'] # the scaling factor for cosine values.
            head = ArcMarginProduct(feat_dim, num_class, scale)
        elif self.head_type == 'boundaryMargin1':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512
            num_class = self.head_param['num_class'] # number of classes in the training set.
            start = self.head_param['epoch_start'] # cos(theta + margin_arc).
            head = BoundaryMargin1(feat_dim, num_class, epoch_start=start)
        elif self.head_type == 'boundaryMargin':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512
            num_class = self.head_param['num_class'] # number of classes in the training set.
            start = self.head_param['epoch_start'] # cos(theta + margin_arc).
            head = BoundaryMargin(feat_dim, num_class, epoch_start=start)
        elif self.head_type == 'curricularMargin':
            feat_dim = self.head_param['feat_dim']  # dimension of the output features, e.g. 512
            num_class = self.head_param['num_class']  # number of classes in the training set.
            head = CurricularFace(feat_dim, num_class)
        elif self.head_type == 'MvArcMargin':
            feat_dim = self.head_param['feat_dim']  # dimension of the output features, e.g. 512
            num_class = self.head_param['num_class']  # number of classes in the training set.
            head = Ms_origin_margin(feat_dim, num_class)
        else:
            pass
        return head
