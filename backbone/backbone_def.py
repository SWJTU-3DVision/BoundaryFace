"""
@author: Jun Wang 
@date: 20201019 
@contact: jun21wangustc@gmail.com    
"""

import sys
import yaml
sys.path.append('../../')
from backbone.cbam import CBAMResNet

class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  parsed params and it's value. 
    """
    def __init__(self, backbone_type, backbone_conf_file):
        self.backbone_type = backbone_type
        with open(backbone_conf_file) as f:
            backbone_conf = yaml.safe_load(f)
            # backbone_conf = yaml.load(f)  #origin
            self.backbone_param = backbone_conf[backbone_type]
        print('backbone param:')
        print(self.backbone_param)

    def get_backbone(self):
        if self.backbone_type == 'Res50_IR':
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            backbone = CBAMResNet(50, feature_dim=feat_dim, mode='ir')
        else:
            pass
        return backbone
