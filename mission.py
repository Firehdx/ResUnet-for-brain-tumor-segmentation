import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from Unet3d import Modified3DUNet
from dataloader import niidataset
import nibabel as nib
import utils
import matplotlib.pyplot as plt
import os


# model loading
model_path = 'lr=0.01_ep=210_wd=1e-06.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Modified3DUNet(in_channels=1, n_classes=1, base_n_filter=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print('finish model loading')


# data loading
data_path = 'dataset_segmentation/test/test_fla'
save_path = 'dataset_segmentation/test/test_seg1'

for index in range(211, 252):
    file = '%03d'%index
    print(f'loading {file}')
    nii = nib.load(data_path + '/' + file + '/' + file + '_fla.nii.gz')
    fla = nii.get_fdata(dtype=np.float32)
    aff = nii.affine
    fla = np.pad(fla, ((0,0),(0,0),(43,42)))
    fla /= np.max(fla)
    fla = fla[np.newaxis,np.newaxis,:,:,:]
    fla = torch.from_numpy(fla)
    with torch.no_grad():
        seg = torch.sigmoid(model(fla)).squeeze().numpy()
    print(f'cutting {file}')
    seg = seg[:,:,43:198]
    seg[seg<0.5]=0
    seg[seg>=0.5]=1


    if seg.shape != (240,240,155):
        print(f'{file} shape error!, output is {seg.shape}')
        break
    else:
        os.makedirs(save_path + '/' + file, exist_ok=True)
        nib.Nifti1Image(seg ,aff).to_filename(save_path + '/' + file + '/' + file + '_seg.nii.gz')
        print(f'{file} is saved')
        