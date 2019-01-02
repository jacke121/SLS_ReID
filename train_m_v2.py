# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision import transforms

import time
import os

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
parser.add_argument('--batchsize', default=76, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet')
parser.add_argument('--num_classes', type=int, default=173, help='')
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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.5

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        need_val=False
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                loader = train_loader
            else:
                if not need_val:
                    continue
                loader=train_loader
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_len=len(loader)
            for step, data in enumerate(loader):
                # get the inputs
                inputs, labels,path = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                lr = scheduler.get_lr()[0]
                running_loss += loss.item()
                current_correct=float(torch.sum(preds == labels.data))/len(labels)
                running_corrects += current_correct
                if step%2==0:
                    print("{} Epoch {}/{} batch {}/{} Loss:{:.4f} acc:{:.4f} avacc:{:.4f} lr:{:.4f}".format \
                      (phase,epoch, num_epochs - 1,step,batch_len, loss.item(), current_correct,running_corrects/(step+1),lr))
            if running_corrects/(step+1)>0.7:
                need_val=True
            else:
                need_val = False
            # deep copy the model
            if phase == 'val':
                if running_corrects/(step+1)<best_acc:
                    continue
                best_acc=running_corrects/(step+1)
                last_model_wts = model.state_dict()
                #if epoch % 10 == 9:
                save_network(model, epoch,(running_corrects/(step+1)))
                #draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model

x_epoch = []

def save_network(network, epoch_label,acc):
    save_filename = '%0.4f_%s.pth' % (acc,epoch_label)
    save_path = os.path.join('./models/m_v2', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

criterion = nn.CrossEntropyLoss()

if opt.use_dense:
    model = ft_net_dense(opt.num_classes)
else:
    model = MobileNet2(num_classes=173)
if use_gpu:
    model = model.cuda()

ignored_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.01},
    {'params': model.fc.parameters(), 'lr': 0.1},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

m_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=30)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

dir_name = os.path.join('./models/m_v2', name)
os.makedirs(dir_name, exist_ok=True)

if __name__ == "__main__":
    model = train_model(model, criterion, optimizer_ft, m_lr_scheduler,
                    num_epochs=10000)

