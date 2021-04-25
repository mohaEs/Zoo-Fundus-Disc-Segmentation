#
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
from utils import BW_img, Deep_Screening, Disc_Crop

import cv2

import os
import glob
import argparse

import Model_Disc_Seg as DiscSegModel

parser = argparse.ArgumentParser(description='Train and Test UPANets with PyTorch')
parser.add_argument('--img_dir', default='./tmp_images', type=str, )
parser.add_argument('--result_dir', default='./results_DENet', type=str, )
args = parser.parse_args()

data_img_path=args.img_dir 
data_save_path=args.result_dir 
data_img_path = './test_image/'


try:
    os.stat(data_save_path)
except:
    os.makedirs(data_save_path) 


Img_Seg_size = 640
Img_Scr_size = 400
ROI_Scr_size = 224

pre_model_DiscSeg = './pre_model/pre_model_DiscSeg.h5'



seg_model = DiscSegModel.DeepModel(Img_Seg_size)
seg_model.load_weights(pre_model_DiscSeg, by_name=True)


for root, dirs, files in os.walk(data_img_path):
    for filename in files:
        print('===> filename', filename)
        org_img = np.asarray(image.load_img(data_img_path + filename))

        img_scale = 2048.0 / org_img.shape[0] 
        org_img = scipy.misc.imresize(org_img, (2048, int(org_img.shape[1]*img_scale), 3))
        #org_img = rescale(org_img, img_scale, anti_aliasing=False)
        # skimage.io.imsave('org_img.png',org_img)

        # disc segmentation
        temp_img = scipy.misc.imresize(org_img, (Img_Seg_size, Img_Seg_size, 3))
        #temp_img = resize(org_img, (Img_Seg_size, Img_Seg_size, 3))
        # skimage.io.imsave('temp_img.png',temp_img)
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
        [prob_6, prob_7, prob_8, prob_9, prob_10] = seg_model.predict([temp_img])    
        disc_map = np.reshape(prob_10, (Img_Seg_size, Img_Seg_size))
        # print(np.max(prob_6))
        # print(np.max(prob_7))
        # print(np.max(prob_8))
        # print(np.max(prob_9))
        # print(np.max(prob_10))
        disc_map=scipy.misc.imresize(disc_map, (org_img.shape[0] , org_img.shape[1] , 3))
        skimage.io.imsave(os.path.join(data_save_path,filename),disc_map)
                
        disc_map[0:round(disc_map.shape[0] / 5), :] = 0
        disc_map[-round(disc_map.shape[0] / 5):, :] = 0
        disc_map = BW_img(disc_map, 0.25)

