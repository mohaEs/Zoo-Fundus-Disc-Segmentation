#
import numpy as np
import scipy
import scipy.io as sio
from scipy.misc import imsave
import skimage
from skimage.transform import rescale, resize

from tensorflow.keras.preprocessing import image
import pandas as pd
from skimage.measure import label, regionprops
from skimage.transform import rotate 
from skimage import measure, io
from time import time
import cv2
from tqdm import tqdm
import os
import glob
import argparse

import Collected_Libraries.DENet.Model_Disc_Seg as DiscSegModel
from Collected_Libraries.DENet.utils import BW_img

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', default='./images_short', type=str, )
parser.add_argument('--result_dir', default='./results_DENet', type=str, )
parser.add_argument('--csv_file', default='./metadata.csv', type=str, )
args = parser.parse_args()

data_img_path=args.img_dir 
data_save_path=args.result_dir 


try:
    os.stat(data_save_path)
except:
    os.makedirs(data_save_path) 


data_df=pd.read_csv(args.csv_file)
timeoftest=data_df['timeoftest']
devicename=data_df['deviceserialnumber']

data_df.insert(11, "DiscSeg", '-')
data_df.insert(12, "DiscSegLoc", '-')


Img_Seg_size = 640
Img_Scr_size = 400
ROI_Scr_size = 224

pre_model_DiscSeg = './Collected_Libraries/DENet/pre_model/pre_model_DiscSeg.h5'

seg_model = DiscSegModel.DeepModel(Img_Seg_size)
seg_model.load_weights(pre_model_DiscSeg, by_name=True)


Method='DENet'

for i in tqdm(range(len(data_df))):
    
    filename=str(int(devicename[i]))+'_'+str(int(timeoftest[i]))+'.jpg'
#    print('===> filename', filename)
    
    try:
        org_img = np.asarray(image.load_img(os.path.join(data_img_path, filename)))
    except:
        continue
    
    run_start=time()
    img_scale = 2048.0 / org_img.shape[0] 
    temp_img = scipy.misc.imresize(org_img, (2048, int(org_img.shape[1]*img_scale), 3))

    # disc segmentation
    temp_img = scipy.misc.imresize(temp_img, (Img_Seg_size, Img_Seg_size, 3))
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
#    print('=> Processed by: {}, running time: {:.2f}'.format(
#     Method, run_end - run_start    ))
    
    
#    io.imsave('mask.png', disc_map, check_contrast=False)
    skimage.io.imsave(os.path.join(data_save_path,filename[:-3]+'png'),disc_map, check_contrast=False)

    disc_map = BW_img(disc_map, 0.25)
    label=measure.label(disc_map)
    props=measure.regionprops(label)
    
    if len(props)==0:
        data_df.loc[i,'DiscSeg']='NotFound'
        continue
        
    props_area_sorted=sorted(
            props,
            key=lambda r: r.area,
            reverse=True,
    )
        
    try: 
        center=props_area_sorted[0].centroid
        center_row=int(center[0])
        center_col=int(center[1])
        area=props_area_sorted[0].area
        
        area_rate=area/(org_img.shape[1]*org_img.shape[0])
        
        r_min=props_area_sorted[0].minor_axis_length
        r_max=props_area_sorted[0].major_axis_length
    except:
        p=0
        

    if area_rate<1e-3:
        data_df.loc[i,'DiscSeg']='VerySmall'
    elif r_max/r_min>3:
        data_df.loc[i,'DiscSeg']='NotEclipse'
    else:
        data_df.loc[i,'DiscSeg']='Found'    
    
    img_size_row=org_img.shape[0]
    img_size_col=org_img.shape[1]
    centerbox=np.zeros(shape=(img_size_row,img_size_col), dtype=np.int)
    centerbox[int(img_size_row/3):int(2*img_size_row/3), 
              int(img_size_row/3):int(2*img_size_row/3)]=1 
    
              
    masked_center=np.multiply(centerbox, label)
    nonzeros=np.nonzero(masked_center)    
    if len(nonzeros[0]) !=0:
        data_df.loc[i,'DiscSegLoc']='Center' 
    else:
        data_df.loc[i,'DiscSegLoc']='NotCenter' 
#        set_value('C', 'x', 10)
        
data_df.to_csv('data_disc_added.csv')

        

