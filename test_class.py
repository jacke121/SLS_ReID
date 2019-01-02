# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import numpy as np
# from torch.autograd import Variable
import torch.nn.functional as m_func
from torch.utils.data import dataloader
from torchvision import transforms

import time

from data.market1501 import Market1501
from nets.model import ft_net_dense
from nets.mobile_net import MobileNet2
from utils.random_erasing import RandomErasing

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default=r'\\192.168.55.73\Team-CV\dataset\origin_all_datas_0814', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=10, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet')
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((288, 144), interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

# if opt.PCB:
#     transform_train_list = [
#         transforms.Resize((384, 192), interpolation=3),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]
#     transform_val_list = [
#         transforms.Resize(size=(384, 192), interpolation=3),  # Image.BICUBIC
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}
m_dataset=Market1501(data_dir, data_transforms['val'], "train")
train_loader = dataloader.DataLoader(m_dataset,shuffle=True,batch_size=opt.batchsize,num_workers=0)


use_gpu = torch.cuda.is_available()

# inputs, classes = next(iter(train_loader))


def train_model(model, criterion):
    since = time.time()

    # Each epoch has a training and validation phase
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    batch_len=len(train_loader)
    for step, data in enumerate(train_loader):
        # get the inputs
        inputs, labels,path = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs, labels = inputs, labels
        time1=time.time()
        outputs = model(inputs)
        print("{time:.3f}s".format(time=time.time()-time1))
        outputs=m_func.log_softmax(outputs, dim=1)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        # backward + optimize only if in training phase
        # statistics
        running_loss += loss.item()
        current_correct=float(torch.sum(preds == labels.data))/len(labels)

        print(np.asarray(path)[np.where(preds != labels.data)])
        print(np.asarray(labels.data)[np.where(preds != labels.data)])
        running_corrects += current_correct
        # if step%2==0:
        #     print("batch {}/{} Loss:{:.4f} acc:{:.4f} avacc:{:.4f}".format \
        #       (step,batch_len, loss.item(), current_correct,running_corrects/(step+1)))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model

if opt.use_dense:
    model = ft_net_dense(opt.num_classes)
else:
    # model = ft_net(opt.num_classes)
    model = MobileNet2(num_classes=173)

#print(model)
state_dict = torch.load(r"E:\github\SLS_ReID\models\m_v2\ft_ResNet50\1.0000_26.pth")
model.load_state_dict(state_dict)
if use_gpu:
    model = model.cuda()

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    model = train_model(model, criterion)

