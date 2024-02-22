import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from torch.autograd import Function
from itertools import repeat
from Unet3d import Modified3DUNet
from dataloader import niidataset
from scipy.ndimage import rotate
import random


def DiceLoss(logits, targets, eps=1e-8):
    log_prob = torch.sigmoid(logits)
    batch_size = targets.size(0)
    log_prob = log_prob.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    intersection = (log_prob * targets).sum(-1)
    dice_score = (2. * intersection + eps) / (log_prob.sum(-1) + targets.sum(-1) + eps)
    return torch.mean(1. - dice_score)





### Not complete: need to use schedular
def train(net, train_loader, validate_loader, num_epochs, learning_rate, weight_decay=0, save_path=''):
    net.train() # model is updating the parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, [5, 50], gamma=0.1)
    loss_f = DiceLoss
    #loss_f = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("cuda" if torch.cuda.is_available() else "cpu")
    f = open(save_path + f'lr={learning_rate}_ep={num_epochs}_wd={weight_decay}.txt', 'w')
    f.write(("cuda" if torch.cuda.is_available() else "cpu") + '\n')

    #training
    for epoch in range(num_epochs):
        train_loss = 0
        net.train() # model is updating the parameters
        for data, target in train_loader:
            #data = torch.nn.functional.normalize(data, p=2, dim=(2,3,4))
            #data /= torch.max(data)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = loss_f(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        #validating
        net.eval()  # model do not update the params
        val_loss = 0
        with torch.no_grad():
            for data_t, target_t in validate_loader:
                #data_t = torch.nn.functional.normalize(data_t, p=2, dim=(2,3,4))
                #data_t /= torch.max(data_t)
                data_t, target_t = data_t.to(device), target_t.to(device)
                output_pre = net(data_t)
                val_loss += loss_f(output_pre, target_t).item()

        val_loss /= len(validate_loader)

        message = f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}, Validate Loss: {val_loss}'
        print(message)
        f.write(message + '\n')
        if epoch%10 == 0:
            torch.save(net.state_dict(), save_path + f'lr={learning_rate}_ep={epoch}_wd={weight_decay}.pth')

    f.close()
    torch.save(net.state_dict(), save_path + f'lr={learning_rate}_ep={num_epochs}_wd={weight_decay}_final.pth')
    print("Model saved.")



if __name__ == '__main__':
    lr = 1e-2
    num_epoch = int(5e2)
    batch_size = 2
    weight_decay = 1e-6
    workers = 8
    data_path = 'dataset_segmentation/train'
    model_path = ''
    net = Modified3DUNet(in_channels=1, n_classes=1, base_n_filter=8)
    train_set = [i for i in range(1, 200)]
    val_set = [i for i in range(200, 211)]

    trainset = niidataset(use_data_list=train_set, data_path=data_path, rotate=True)
    valset = niidataset(use_data_list=val_set, data_path=data_path)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=workers)
    print('finish data loading')

    train(net=net, train_loader=train_loader, validate_loader=val_loader,
           num_epochs=num_epoch, learning_rate=lr, save_path=model_path,weight_decay=weight_decay)

