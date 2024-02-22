import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import random
from copy import deepcopy
from memory_profiler import profile

def rotater_3d(ndarr, angle):
    rotated_arr = rotate(ndarr, angle, axes=(0, 1), reshape=False)
    return rotated_arr

def scaler_3d(ndarr, scale):
    x = ndarr.shape[0]
    ndarr = zoom(ndarr, scale)
    y = ndarr.shape[0]
    if scale > 1:
        center_idx = y//2
        start = center_idx-x//2
        end = center_idx+x//2
        ndarr = ndarr[start:end, start:end, start:end]
    elif scale < 1:
        delta = x-y
        start = delta//2
        end = delta-start
        ndarr = np.pad(ndarr, ((start,end),(start,end),(start,end)))
    return ndarr



class niidataset(torch.utils.data.Dataset):
    def __init__(self, use_data_list, rotate=False, scale=False, data_path='dataset_segmentation/train'):
        self.use_data_list = use_data_list
        self.data_path = data_path
        self.rotate = rotate
        self.scale = scale
        self.fla_list = []
        self.seg_list = []

        for file_index in self.use_data_list:
            fla, seg = self.load_nii(file_index)
            self.fla_list.append(fla.tolist())
            self.seg_list.append(seg.tolist())


    def load_nii(self, file_index, angle=10, coff=1.1):
        file = '%03d'%file_index
        fla = nib.load(self.data_path + '/' + file + '/' + file + '_fla.nii.gz').get_fdata(dtype=np.float32)
        seg = nib.load(self.data_path + '/' + file + '/' + file + '_seg.nii.gz').get_fdata(dtype=np.float32)
        #fla = np.pad(fla, ((8,8),(8,8),(51,50))) # (240,240,155)->(256,256,256)
        #seg = np.pad(seg, ((8,8),(8,8),(51,50))) # (240,240,155)->(256,256,256)
        fla = np.pad(fla, ((0,0),(0,0),(43,42))) # (240,240,155)->(240,240,240)
        seg = np.pad(seg, ((0,0),(0,0),(43,42))) # (240,240,155)->(240,240,240)

        # Rotate the data
        if self.rotate:
            angle = random.randint(-angle, angle)
            fla = rotater_3d(fla, angle)
            seg = rotater_3d(seg, angle)
        # Scale the data
        if self.scale:
            scale = random.uniform(1/coff, coff)
            fla = scaler_3d(fla, scale)
            seg = scaler_3d(seg, scale)

        fla = deepcopy(fla[np.newaxis,:,:,:]) # (240,240,240)->(1,240,240,240)
        seg = deepcopy(seg[np.newaxis,:,:,:]) # (240,240,240)->(1,240,240,240)
        return fla, seg

    ### Not complete: NOT READY FOR DATA ROTATE AND SCALE!!! need data enhancement like noise-reducing
    def __getitem__(self, index):
        return self.fla_list[index], self.seg_list[index]

    ### Not complete: NOT READY FOR DATA ROTATE AND SCALE!!!
    def __len__(self):
        return len(self.seg_list)
    

### Not complete
class niidataset2d(torch.utils.data.Dataset):
    def __init__(self, use_data_list, rotate_num=0, scale_num=0, data_path='dataset_segmentation/train'):
        pass

if __name__ == '__main__':
    g = niidataset([1,3,5])
    pred ,target= g[1]
    #print(len(target))
