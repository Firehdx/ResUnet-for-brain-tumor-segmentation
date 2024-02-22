import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import random

def rotater_3d(ndarr, angle):
    ndarr = rotate(ndarr, angle, axes=(0, 1), reshape=False)
    return ndarr

def scaler_3d(ndarr, scale):
    x = ndarr.shape[0]
    h = ndarr.shape[2]
    ndarr = zoom(ndarr, scale)
    y = ndarr.shape[0]
    z = ndarr.shape[2]
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


class niidataset(torch.utils.data.Dataset):
    def __init__(self, use_data_list, data_path='dataset_segmentation/train', rotate = False, scale=False):
        self.use_data_list = use_data_list
        self.data_path = data_path
        self.rotate = rotate
        self.scale = scale


    def load_nii(self, file_index, angle=5, coff=1.05):
            file = '%03d'%file_index
            fla = nib.load(self.data_path + '/' + file + '/' + file + '_fla.nii.gz').get_fdata(dtype=np.float32)
            seg = nib.load(self.data_path + '/' + file + '/' + file + '_seg.nii.gz').get_fdata(dtype=np.float32)
            #fla = np.pad(fla, ((8,8),(8,8),(51,50))) # (240,240,155)->(256,256,256)
            #seg = np.pad(seg, ((8,8),(8,8),(51,50))) # (240,240,155)->(256,256,256)
            fla = np.pad(fla, ((0,0),(0,0),(43,42))) # (240,240,155)->(240,240,240)
            seg = np.pad(seg, ((0,0),(0,0),(43,42))) # (240,240,155)->(240,240,240)
            
            angle = random.randint(-angle, angle)
            if self.rotate:
                fla = rotate(fla, angle, axes=(0, 1), reshape=False)
                seg = rotate(seg, angle, axes=(0, 1), reshape=False)
            fla /= np.max(fla)

            # # Rotate the data
            # if self.rotate:
            #     angle = random.randint(-angle, angle)
            #     fla = rotater_3d(fla, angle)
            #     seg = rotater_3d(seg, angle)
            # # Scale the data
            # if self.scale:
            #     print('s')
            #     scale = random.uniform(1/coff, coff)
            #     fla = scaler_3d(fla, scale)
            #     seg = scaler_3d(seg, scale)
            # if self.rotate or self.scale:
            #     print('p')
            #     fla[fla<1e-3] = 0
            #     seg[seg<0.5] = 0
            #     seg[seg>=0.5] = 1
            # print('f')
            fla = fla[np.newaxis,:,:,:] # (240,240,240)->(1,240,240,240)
            seg = seg[np.newaxis,:,:,:] # (240,240,240)->(1,240,240,240)
            return fla, seg

    ### Not complete: need data enhancement like noise-reducing
    def __getitem__(self, index):
        index = self.use_data_list[index]
        return self.load_nii(index)

    def __len__(self):
        return len(self.use_data_list)
    


if __name__ == '__main__':
    g = niidataset([1,3,5])
    pred = g[1][0]
    target = g[1][1]
    print(target.shape)
