import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import random
import re
#from monai.losses.dice import DiceLoss


# fla: [0, inf), seg: 0 or 1

def reading_nii(num, cut=77):
    index = '%03d'%num
    fla = nib.load(f'dataset_segmentation/train/{index}/{index}_fla.nii.gz').get_fdata()
    seg = nib.load(f'dataset_segmentation/train/{index}/{index}_seg.nii.gz').get_fdata()
    print(fla.shape)
    reading_3d_mean(fla, seg)


def reading_3d(fla, seg, cut):
    fla = fla[:,:,cut]
    seg = seg[:,:,cut]
    #plt.imshow(pic, cmap='bone')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(fla, cmap='bone')
    axes[1].imshow(seg, cmap='bone')
    plt.show()

def reading_3d_mean(fla, seg, axis=2):
    fla = np.mean(fla, axis=axis)
    seg = np.mean(seg, axis=axis)
    #plt.imshow(pic, cmap='bone')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(fla, cmap='bone')
    axes[1].imshow(seg, cmap='bone')
    plt.show()

def reading_3d_3p(fla, seg, predict, cut=77, mode='mean'):
    if mode == 'mean':
        fla = np.mean(fla, axis=2)
        seg = np.mean(seg, axis=2)
        predict = np.mean(predict, axis=2)
    else:
        fla = fla[:,:,cut]
        seg = seg[:,:,cut]
        predict = predict[:,:,cut]
    #plt.imshow(pic, cmap='bone')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(fla, cmap='bone')
    axes[1].imshow(seg, cmap='bone')
    axes[2].imshow(predict, cmap='bone')
    plt.show()

def reading_3d_4p(fla, seg, predict1, predict2, cut=77, mode='mean'):
    if mode == 'mean':
        fla = np.mean(fla, axis=2)
        seg = np.mean(seg, axis=2)
        predict1 = np.mean(predict1, axis=2)
        predict2 = np.mean(predict2, axis=2)
    else:
        fla = fla[:,:,cut]
        seg = seg[:,:,cut]
        predict1 = predict1[:,:,cut]
        predict2 = predict2[:,:,cut]
    #plt.imshow(pic, cmap='bone')
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0][0].imshow(fla, cmap='bone')
    axes[0][1].imshow(seg, cmap='bone')
    axes[1][0].imshow(predict1, cmap='bone')
    axes[1][1].imshow(predict2, cmap='bone')
    plt.show()


def DiceLoss(targets, logits, ep=1e-8):
    batch_size = targets.shape[0]
    logits = torch.from_numpy(logits).view(batch_size, -1).type(torch.float32)
    targets = torch.from_numpy(targets).view(batch_size, -1).type(torch.float32)
    intersection = (logits * targets).sum(-1)
    dice_score = (2. * intersection + ep) / ((logits + targets).sum(-1) + ep) # Laplace smooth
    return torch.mean(1. - dice_score)

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
        endh = center_idh+h//2
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


def loss_test():
    a = np.array([[[[0,1,0,1,0,0],
                   [0,0,1,1,0,0]],

                  [[0,1,1,0,1,0],
                   [0,1,1,0,0,0]],

                  [[1,1,0,0,0,0],
                   [1,1,0,0,0,0]]]])
    
    
    b = np.array([[[[0,0,1,1,0,0],
                   [0,0,1,1,0,0]],

                  [[0,1,1,0,0,0],
                   [0,1,1,0,0,0]],

                  [[1,1,0,0,0,0],
                   [1,1,0,0,0,0]]]])
    

    # a = np.array([[0,0,1,1,0,0], [0,0,1,1,0,0]])
    # b = np.array([[0,0,1,1,0,0], [0,0,1,1,0,0]])
    
    loss = DiceLoss(a,b)
    print(loss)


def test_rotate_scale():
    a = torch.Tensor(240,240,240)
    print(a.shape)
    a = rotater_3d(a,10)
    print(a.shape)
    a = scaler_3d(a, 0.7777)
    print(a.shape)

def test_shape():
    fla_shape = 0;
    seg_shape = 0;
    for num in range(1,211):
        index = '%03d'%num
        fla = nib.load(f'dataset_segmentation/train/{index}/{index}_fla.nii.gz').get_fdata()
        seg = nib.load(f'dataset_segmentation/train/{index}/{index}_seg.nii.gz').get_fdata()
        if fla.shape == (240,240,155):
            fla_shape += 1
        if seg.shape == (240,240,155):
            seg_shape += 1
    print(f'correct shape num: {fla_shape}, {seg_shape}')



def loss_lr_fig(path='log.txt'):
    epoch = []
    train_loss = []
    val_loss = []
    lr = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = re.split('/|,|:|\n| ', lines[i])
            epoch.append(i)
            train_loss.append(float(lines[i][7]))
            val_loss.append(float(lines[i][12]))
            if i < 5:
                lr.append(0.01)
            elif i >=50:
                lr.append(0.0001)
            else:
                lr.append(0.001)
    
    fig = plt.figure(num=1, figsize=(12,6))
    ax1=fig.add_subplot(121)
    ax1.plot(epoch, train_loss, label='train loss')
    ax1.plot(epoch, val_loss, label='validate loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax1.legend()
    ax2=fig.add_subplot(122)
    ax2.plot(epoch, lr)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.show()

def relu_fig():
    x = torch.linspace(-5,5,200)
    y_relu = torch.nn.functional.relu(x)
    y_leaky = torch.nn.functional.leaky_relu(x, 0.1)

    plt.plot(x, y_relu, label='ReLU')
    plt.plot(x, y_leaky, label='Leaky ReLU')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #reading_nii(248, 77)
    #loss_test()
    #test_rotate_scale()
    # fla = nib.load(f'dataset_segmentation/train/001/001_fla.nii.gz').get_fdata()
    # seg = nib.load(f'dataset_segmentation/train/001/001_seg.nii.gz').get_fdata()
    # fla2 = rotater_3d(fla, 30)
    # reading_3d(fla, fla2, 77)
    # test_shape()
    #loss_lr_fig()
    #reading_nii(4)
    relu_fig()
