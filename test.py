import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from Unet3d import Modified3DUNet#, DiceLoss
from dataloader import niidataset
import nibabel as nib
import utils


# model loading
model_path = 'lr=0.01_ep=210_wd=1e-06.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Modified3DUNet(in_channels=1, n_classes=1, base_n_filter=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
print('finish model loading')


# testset loading
data_path = 'dataset_segmentation/train'
# test_set = [i for i in range(100, 121)]
# testset = niidataset(use_data_list=test_set, data_path=data_path)
# test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)
# print('finish data loading')

# loss function
def DiceLoss(logits, targets, eps=1e-8):
    log_prob = torch.sigmoid(logits)
    batch_size = targets.size(0)
    log_prob = log_prob.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    intersection = (log_prob * targets).sum(-1)
    dice_score = (2. * intersection + eps) / (log_prob.sum(-1) + targets.sum(-1) + eps)
    return torch.mean(1. - dice_score)

# testing
model.eval()
# test_loss = 0
# sample = 0
# with torch.no_grad():
#     for data_t, target_t in test_loader:
#         print(f'sample{sample} testing')
#         #data_t = torch.nn.functional.normalize(data_t, p=2, dim=(2,3,4))
#         data_t, target_t = data_t.to(device), target_t.to(device)
#         output_pre = model(data_t)
#         loss = DiceLoss(output_pre, target_t)
#         test_loss += loss.item()
#         sample +=1
# test_loss /= len(test_loader)

# print(test_loss)

# visualizing
file_index = 210
file = '%03d'%file_index
fla_orign = nib.load(data_path + '/' + file + '/' + file + '_fla.nii.gz').get_fdata(dtype=np.float32)
seg = nib.load(data_path + '/' + file + '/' + file + '_seg.nii.gz').get_fdata(dtype=np.float32)
fla_orign = np.pad(fla_orign, ((0,0),(0,0),(43,42)))
seg = np.pad(seg, ((0,0),(0,0),(43,42)))
fla = fla_orign[np.newaxis,np.newaxis,:,:,:]
fla /= np.max(fla)
fla = torch.from_numpy(fla)

with torch.no_grad():
    seg_predict = torch.sigmoid(model(fla)).squeeze()
seg_predict = seg_predict.numpy()

utils.reading_3d_3p(fla_orign, seg, seg_predict, cut = 120, mode='cut')