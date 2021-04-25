

"""
Source:
    https://github.com/DeepTrial/Optic-Disc-Unet
    https://arxiv.org/pdf/1804.03999v3.pdf
"""

from Collected_Libraries.AttnUnet.perception.infers.segmention_infer import SegmentionInfer
from Collected_Libraries.AttnUnet.perception.metric.segmention_metric import *
from Collected_Libraries.AttnUnet.configs.utils.config_utils import process_config
import os
import argparse

parser = argparse.ArgumentParser(description='Train and Test UPANets with PyTorch')
parser.add_argument('--img_dir', default='./tmp_images', type=str, )
parser.add_argument('--result_dir', default='./results_AttnUnet', type=str, )
args = parser.parse_args()

ImgesDIR=args.img_dir 
savepath=args.result_dir 

def main_test():
    print('[INFO] Reading Configs...')
    config = None

    try:
        config = process_config('Collected_Libraries/AttnUnet/configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)


    print('[INFO] Predicting...')

    try:
        os.stat(savepath)
    except:
        os.makedirs(savepath) 

    infer = SegmentionInfer( config, ImgesDIR, savepath) #ImgesDIR
    infer.predict()


if __name__ == '__main__':
    main_test()
