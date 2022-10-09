"""
@author: Jun Wang
@date: 20201012 
@contact: jun21wangustc@gmail.com
"""  

import sys
import argparse
import yaml
sys.path.append('..')
from test_protocol.megaface.megaface_evaluator import CommonMegaFaceEvaluator

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='megaface test protocal in python.')
    conf.add_argument("--data_conf_file", type = str, default='./data_conf.yaml',
                      help = "The path of data_conf.yaml.")
    conf.add_argument("--max_rank", type = int, default=5,
                      help = "Rank N accuray..")
    conf.add_argument("--facescrub_feature_dir", type = str, default='/ssd2t/megaface_result/mv-softmax_ms1m_facescrub_feats_clean',
                      help = "The dir of facescrub features.")
    conf.add_argument("--megaface_feature_dir", type = str, default='/ssd2t/megaface_result/mv-softmax_ms1m_megaface_feats_clean',
                      help = "The dir of megaface features.")
    conf.add_argument("--is_concat", type = int, default=0,
                      help = "If the feature is concated by two nomalized features.")
    args = conf.parse_args()
    with open(args.data_conf_file) as f:
        data_conf  = yaml.safe_load(f)['MegaFace']
        facescrub_json_list = data_conf['facescrub_list']
        megaface_json_list = data_conf['megaceface_list']
    is_concat = True if args.is_concat == 1 else False
    megaFaceEvaluator = CommonMegaFaceEvaluator(
        facescrub_json_list, megaface_json_list,
        args.facescrub_feature_dir, args.megaface_feature_dir,
        is_concat)
