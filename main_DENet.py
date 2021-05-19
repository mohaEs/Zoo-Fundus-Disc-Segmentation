#

"""
source:
    https://github.com/HzFu/DENet_GlaucomaScreen
    https://doi.org/10.1109/TMI.2018.2837012
    https://doi.org/10.1109/TMI.2018.2791488
"""

import numpy as np
import scipy
import scipy.io as sio
from scipy.misc import imsave
import skimage
from skimage.transform import rescale, resize

from tensorflow.keras.preprocessing import image

from skimage.measure import label, regionprops
from skimage.transform import rotate 
from time import time
import cv2

import os
import glob
import argparse

import Collected_Libraries.DENet.Model_Disc_Seg as DiscSegModel
from Collected_Libraries.DENet.utils import BW_img

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', default='./tmp_images', type=str, )
parser.add_argument('--result_dir', default='./results_DENet', type=str, )
args = parser.parse_args()

data_img_path=args.img_dir 
data_save_path=args.result_dir 


try:
    os.stat(data_save_path)
except:
    os.makedirs(data_save_path) 


Img_Seg_size = 640
Img_Scr_size = 400
ROI_Scr_size = 224

pre_model_DiscSeg = './Collected_Libraries/DENet/pre_model/pre_model_DiscSeg.h5'

seg_model = DiscSegModel.DeepModel(Img_Seg_size)
seg_model.load_weights(pre_model_DiscSeg, by_name=True)

Method='DENet'

for root, dirs, files in os.walk(data_img_path):
    for filename in files:
        print('===> filename', filename)
        org_img = np.asarray(image.load_img(os.path.join(data_img_path, filename)))
        run_start=time()
        img_scale = 2048.0 / org_img.shape[0] 
        org_img = scipy.misc.imresize(org_img, (2048, int(org_img.shape[1]*img_scale), 3))

        # disc segmentation
        temp_img = scipy.misc.imresize(org_img, (Img_Seg_size, Img_Seg_size, 3))
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
        [prob_6, prob_7, prob_8, prob_9, prob_10] = seg_model.predict([temp_img])    
        disc_map = np.reshape(prob_10, (Img_Seg_size, Img_Seg_size))

        # disc_map=scipy.misc.imresize(disc_map, (org_img.shape[0] , org_img.shape[1] , 3))
        # run_end=time()
        # print('=> Processed by: {}, running time: {:.2f}'.format(
        #  Method, run_end - run_start    ))
        # skimage.io.imsave(os.path.join(data_save_path,filename),disc_map)

        # disc_map[0:round(disc_map.shape[0] / 5), :] = 0
        # disc_map[-round(disc_map.shape[0] / 5):, :] = 0
        disc_map = BW_img(disc_map, 0.25)
        disc_map=scipy.misc.imresize(1*disc_map, (org_img.shape[0] , org_img.shape[1] , 3))
        run_end=time()
        print('=> Processed by: {}, running time: {:.2f}'.format(
         Method, run_end - run_start    ))
        skimage.io.imsave(os.path.join(data_save_path,filename[:-3]+'png'),disc_map, check_contrast=False)

