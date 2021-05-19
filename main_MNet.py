# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:55:24 2021

@author: meslami
"""
"""
Source: 
    mnet_deep_cdr    
    https://github.com/HzFu/MNet_DeepCDR
    https://doi.org/10.1109/TMI.2018.2791488
""" 

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from time import time
from Fn_Functions import *
import glob
import os
import argparse
from skimage.transform import rescale

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', default='./tmp_images', type=str, )
parser.add_argument('--result_dir', default='./results_MNet', type=str, )
args = parser.parse_args()

imgespath=args.img_dir 
savepath=args.result_dir 

try:
    os.stat(savepath)
except:
    os.makedirs(savepath) 

DiscSeg_model, CDRSeg_model= Fn_Initialize_MNet()  

for root, dirs, files in os.walk(imgespath):
    for filename in files:
        print('===> filename', filename)
        scale=1
        img=io.imread(os.path.join(imgespath,filename)) 

        if img.shape[0]<1500:
            
            scale=1500/img.shape[0]
            print(scale)
            img = rescale(img, scale, anti_aliasing=False)

        Method='MNet'
        run_start=time()
        Mask_2=Fn_MNet_apply(img, 
                             DiscSeg_model, CDRSeg_model)
        run_end=time()
        print('=> Processed by: {}, running time: {:.2f}'.format(
         Method, run_end - run_start   ))
        if scale>1:
            Mask_2 = rescale(Mask_2, 1/scale, anti_aliasing=False)  
        io.imsave(os.path.join(savepath,filename[:-3]+'png'),Mask_2, check_contrast=False)
        
