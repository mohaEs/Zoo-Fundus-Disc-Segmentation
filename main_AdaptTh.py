# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:55:24 2021

@author: meslami
"""

'''
Source: 
    Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation-
    https://github.com/NupurBhaisare/Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation-
    https://doi.org/10.1109/SPIN.2015.7095384 
''' 


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from time import time
from Fn_Functions import *
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', default='./tmp_images', type=str, )
parser.add_argument('--result_dir', default='./results_AdaptTh', type=str, )
args = parser.parse_args()

imgespath=args.img_dir 
savepath=args.result_dir 

try:
    os.stat(savepath)
except:
    os.makedirs(savepath) 

for root, dirs, files in os.walk(imgespath):
    for filename in files:
        print('===> filename', filename)
        
        img=io.imread(os.path.join(imgespath,filename))           
        
        Method='AdaptiveThreshold'
        run_start=time()
        DiskMask_1, CupMask_1=Fn_segment_AdaptiveThreshold(img)
        DiskMask_1=DiskMask_1.astype(dtype=int)        
        run_end=time()
        print('=> Processed by: {}, running time: {:.2f}'.format(
         Method, run_end - run_start    ))
        io.imsave(os.path.join(savepath,filename),DiskMask_1, check_contrast=False)
                
        
        
        
        

        
    
    


