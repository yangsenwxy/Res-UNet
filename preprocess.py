import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import sys
import glob
import shutil
from keras.preprocessing.image import random_rotation,random_shift

def raw2nii(raw_img_path,slice_index,processed_dir,sample_name,is_gt=False):
    save_dir = ''
    if(is_gt):
        save_dir += processed_dir + 'GT/' + sample_name
    else:
        save_dir += processed_dir + 'BBOX/' + sample_name
    if(os.path.exists(save_dir)):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    raw_img = sitk.ReadImage(raw_img_path)
    
    arr = sitk.GetArrayFromImage(raw_img)
    arr = arr.transpose((2,1,0))
    affine = np.diag([1,1,1,1])

    for i in range(slice_index[0],slice_index[1]+1):
        nii_img = arr[:,:,i]
        nii_img = nib.Nifti1Image(nii_img,affine)
        nib.save(nii_img,save_dir + '/z_' + str(i) + '.nii')

def random_enhance(img_arr,lab_arr):
    sample_num = img_arr.shape[0]
    for i in range(sample_num):
            img = img_arr[i,:,:,0]
            label = lab_arr[i,:,:,0]
            merge = np.array((img,label))
            merge = random_rotation(merge,20)
            merge = random_shift(merge,0.2,0.2)
            img_arr[i,:,:,:] = np.expand_dims(merge[0,:,:],2)
            lab_arr[i,:,:,:] = np.expand_dims(merge[1,:,:],2)
    return img_arr,lab_arr  
        

if __name__ == '__main__':
    raw_dir = '/home/albelt/NoseGT/'
    processed_dir = '/home/albelt/NoseData/'
    bbox_paths = glob.glob(raw_dir + '/**/Nose?_boundingboxcut.mhd',recursive=True)
    bbox_paths += glob.glob(raw_dir + '/**/Nose??_boundingboxcut.mhd',recursive=True)
    bbox_paths.sort()
    gt_paths = []
    for bbox_path in bbox_paths:
        sample_name = bbox_path.split('/')[-2]
        gt_path = glob.glob(raw_dir + sample_name + '/GT*.mhd',recursive=True)
        assert gt_path != '', 'No matching GT file'
        gt_paths += gt_path
    gt_paths.sort()

    sample_num = len(bbox_paths)
    sampling_index = {}
    with open(raw_dir + 'sampling_index2.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            name, low, high = line.split(',')
            sampling_index[str(name)] = [int(low)-1,int(high)-1]
    total_img = 0
    for i in range(sample_num):
        sample_name = bbox_paths[i].split('/')[-2]
        sample_index = sampling_index[str(sample_name)]
        assert sample_index[1] > sample_index[0],'sample_index[1] should be large than sample_index[0]'
        raw2nii(bbox_paths[i],sample_index,processed_dir,sample_name,False)
        raw2nii(gt_paths[i],sample_index,processed_dir,sample_name,True)
        total_img += (sample_index[1] - sample_index[0] + 1)
    print('Finished')
    print('\t\t--total images:{0}'.format(total_img))

    
