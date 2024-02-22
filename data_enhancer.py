import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import random
import os
from utils import reading_3d

def rotater_3d(ndarr, angle):
    ndarr = rotate(ndarr, angle, axes=(0, 1), reshape=False)
    return ndarr

def scaler_3d(ndarr, scale):
    x = ndarr.shape[0]
    h = ndarr.shape[2]
    ndarr = zoom(ndarr, scale)
    y = ndarr.shape[0]
    z = ndarr.shape[2]
    print(z)
    if scale > 1:
        center_idx = y//2
        center_idh = z//2
        start = center_idx-x//2
        starth = center_idh-h//2
        end = center_idx+x//2
        endh = starth+h
        ndarr = ndarr[start:end, start:end, starth:endh]
    elif scale < 1:
        delta = x-y
        deltah = h-z
        start = delta//2
        starth = deltah//2
        end = delta-start
        endh = deltah-starth
        ndarr = np.pad(ndarr, ((start,end),(start,end),(starth,endh)))
    return ndarr


def data_enhance(in_num, out_num, angle=5, coff=1.05):
    in_index = '%03d'%in_num
    out_index = '%03d'%out_num
    fla = nib.load(f'dataset_segmentation/train/{in_index}/{in_index}_fla.nii.gz')
    fla_data_origin = fla.get_fdata()
    fla_aff = fla.affine
    seg = nib.load(f'dataset_segmentation/train/{in_index}/{in_index}_seg.nii.gz')
    seg_data_origin = seg.get_fdata()
    seg_aff = seg.affine

    angle = random.randint(-angle, angle)
    fla_data = rotater_3d(fla_data_origin, angle)
    seg_data = rotater_3d(seg_data_origin, angle)
    scale = random.uniform(1/coff, coff)
    #scale = coff
    fla_data = scaler_3d(fla_data, scale)
    seg_data = scaler_3d(seg_data, scale)
    fla_data[fla_data<1e-3] = 0
    seg_data[seg_data<0.3] = 0
    seg_data[seg_data>=0.3] = 1

    if fla_data.shape != (240,240,155) or seg_data.shape != (240,240,155):
        print(f'{out_index} shape error!, output is {fla_data.shape}')
        return False
    else:
        #reading_3d(fla_data_origin, seg_data_origin, 77)
        reading_3d(fla_data, seg_data, 77)
        # os.makedirs(f'dataset_segmentation/train/{out_index}', exist_ok=True)
        # nib.Nifti1Image(fla_data,fla_aff).to_filename(f'dataset_segmentation/train/{out_index}/{out_index}_fla.nii.gz')
        # nib.Nifti1Image(seg_data,seg_aff).to_filename(f'dataset_segmentation/train/{out_index}/{out_index}_seg.nii.gz')
        # print(f'{out_index} is saved')
        # return True


if __name__ == '__main__':
    # for i in range(1, 211):
    #     ok = data_enhance(i, i+210)
    #     if ok == False:
    #         break
    data_enhance(20, 299)