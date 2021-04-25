
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""

from perception.infers.segmention_infer import SegmentionInfer
from perception.metric.segmention_metric import *
from configs.utils.config_utils import process_config
import os

def main_test():
    print('[INFO] Reading Configs...')
    config = None

    try:
        config = process_config('configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    ImgesDIR='./test/origin/'
    savepath='./test/results/'
    print('[INFO] Predicting...')

    try:
        os.stat(savepath)
    except:
        os.makedirs(savepath) 

    infer = SegmentionInfer( config, ImgesDIR, savepath) #ImgesDIR
    infer.predict()


if __name__ == '__main__':
    main_test()
