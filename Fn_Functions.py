# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:58:51 2021

@author: meslami
"""

import os
from sys import modules
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from scipy import signal
import math
import cv2
import numpy as np
from PIL import Image
from pkg_resources import resource_filename
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image

''' M_NET '''
import Collected_Libraries.MNet.Model_DiscSeg as DiscModel 
import Collected_Libraries.MNet.Model_MNet as MNetModel
from Collected_Libraries.MNet.mnet_utils import pro_process, BW_img, disc_crop, mk_dir, files_with_ext


#%%

'''
Source: 
    Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation-
    https://github.com/NupurBhaisare/Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation-
    https://doi.org/10.1109/SPIN.2015.7095384 
''' 

def Fn_segment_AdaptiveThreshold(image):
    """
    Code from

    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    

    # FUNCTION TO SEGMENT CUP AND DISK
    #image = fundus image
    #plot_seg = plots the segmented image
    #plt_hist = plots the histogram of red and green channel before and after smoothing

    # image = image[400:1400,500:1600,:] #cropping the fundus image to ger region of interest

    Abo,Ago,Aro = cv2.split(image)  #splitting into 3 channels
    #Aro = clahe.apply(Aro)
    Ago = clahe.apply(Ago)
    M = 60    #filter size
    filter = signal.gaussian(M, std=6) #Gaussian Window
    filter=filter/sum(filter)
    STDf = filter.std()  #It'standard deviation
    

    Ar = Aro - Aro.mean() - Aro.std() #Preprocessing Red
    
    Mr = Ar.mean()                           #Mean of preprocessed red
    SDr = Ar.std()                           #SD of preprocessed red
    Thr = 0.5*M - STDf - Ar.std()            #Optic disc Threshold
    #print(Thr)

    Ag = Ago - Ago.mean() - Ago.std()		 #Preprocessing Green
    Mg = Ag.mean()                           #Mean of preprocessed green
    SDg = Ag.std()                           #SD of preprocessed green
    Thg = 0.5*Mg +2*STDf + 2*SDg + Mg        #Optic Cup Threshold
    #print(Thg)
    
#    
#    hist,bins = np.histogram(Ag.ravel(),256,[0,256])   #Histogram of preprocessed green channel
#    histr,binsr = np.histogram(Ar.ravel(),256,[0,256]) #Histogram of preprocessed red channel
#    
    r,c = Ag.shape
    Dd = np.zeros(shape=(r,c)) #Segmented disc image initialization
    Dc = np.zeros(shape=(r,c)) #Segmented cup image initialization

    #Using obtained threshold for thresholding of the fundus image
    for i in range(1,r):
        for j in range(1,c):
            if Ar[i,j]>Thr:
                Dd[i,j]=255
            else:
                Dd[i,j]=0

    for i in range(1,r):
        for j in range(1,c):
        
            if Ag[i,j]>Thg:
                Dc[i,j]=1
            else:
                Dc[i,j]=0
    
    Disk=Dd
    Cup=Dc
    #Saving the segmented image in the same place as the code folder      
#    cv2.imwrite('disk.png',Disk)
#    plt.imsave('cup.png',Cup)
    
    return Disk, Cup


#%% 
'''
Source: 
    mnet_deep_cdr    
    https://github.com/HzFu/MNet_DeepCDR
    https://doi.org/10.1109/TMI.2018.2791488
''' 

def Fn_Initialize_MNet():
    
    DiscROI_size = 600
    DiscSeg_size = 640
    CDRSeg_size = 400
    parent_dir = './Collected_Libraries/MNet'
    
    DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
    DiscSeg_model.load_weights(os.path.join(parent_dir, 'deep_model', 'Model_DiscSeg_ORIGA.h5'))
    
    CDRSeg_model = MNetModel.DeepModel(size_set=CDRSeg_size)
    CDRSeg_model.load_weights(os.path.join(parent_dir, 'deep_model', 'Model_MNet_REFUGE.h5'))
    
    return (DiscSeg_model, CDRSeg_model )
    
def Fn_MNet_apply(img, DiscSeg_model, CDRSeg_model):
    DiscROI_size = 600
    DiscSeg_size = 640
    CDRSeg_size = 400
    
    # load image
    org_img = img 
    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscSeg_model.predict([temp_img])
    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
    disc_region, err_xy, crop_xy = disc_crop(org_img, DiscROI_size, C_x, C_y)

    # Disc and Cup segmentation by M-Net

    Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size / 2, DiscROI_size / 2),
                                       DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS), -90)

    temp_img = pro_process(Disc_flat, CDRSeg_size)
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    [_, _, _, _, prob_10] = CDRSeg_model.predict(temp_img)

    # Extract mask
    prob_map = np.reshape(prob_10, (prob_10.shape[1], prob_10.shape[2], prob_10.shape[3]))
    disc_map = np.array(Image.fromarray(prob_map[:, :, 0]).resize((DiscROI_size, DiscROI_size)))
    cup_map = np.array(Image.fromarray(prob_map[:, :, 1]).resize((DiscROI_size, DiscROI_size)))
    disc_map[-round(DiscROI_size / 3):, :] = 0
    cup_map[-round(DiscROI_size / 2):, :] = 0
    De_disc_map = cv2.linearPolar(rotate(disc_map, 90), (DiscROI_size / 2, DiscROI_size / 2),
                                  DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    De_cup_map = cv2.linearPolar(rotate(cup_map, 90), (DiscROI_size / 2, DiscROI_size / 2),
                                 DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    De_disc_map = np.array(BW_img(De_disc_map, 0.5), dtype=int)
    De_cup_map = np.array(BW_img(De_cup_map, 0.5), dtype=int)


    # Save raw mask
    ROI_result = np.array(BW_img(De_disc_map, 0.5), dtype=int) + np.array(BW_img(De_cup_map, 0.5), dtype=int)
    Img_result = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.uint8)
    Img_result[crop_xy[0]:crop_xy[1], crop_xy[2]:crop_xy[3], ] = ROI_result[err_xy[0]:err_xy[1], err_xy[2]:err_xy[3], ]
    Img_result[(Img_result > 1) & (Img_result < 3)]=255
    Img_result[(Img_result > 0) & (Img_result < 2)]=128    
    #.astype(np.uint8)
    return(Img_result)
