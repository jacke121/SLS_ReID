# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import configparser
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision import transforms

import time
import os

from data.market1501 import Market1501
from nets.model import ft_net, ft_net_dense
from utils.random_erasing import RandomErasing

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default=r'\\192.168.55.73\Team-CV\dataset\origin_all_datas_0807', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=84, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet')
parser.add_argument('--num_classes', type=int, default=123, help='')
parser.add_argument('--pretrain_snapshot', type=str, default='')
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

# inputs, classes = next(iter(train_loader))

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
def write_ini(ini_name,accuracy,error_list):
    _conf_file = configparser.ConfigParser()
    f2 = open(ini_name, 'w')
    f2.close()
    _conf_file.read(ini_name)
    _conf_file.add_section("Accuracy")  # 增加section
    _conf_file.set("Accuracy", "acc", str(accuracy))  # 给新增的section 写入option

    _conf_file.add_section("error_list")
    for k, file in enumerate(error_list):
        _conf_file.set("error_list", str(k), file)
    print('write to ini %s' % ini_name)
    _conf_file.write(open(ini_name, 'w'))
def evaluate(model):
    # checkpoint_paths = {'58': r'\\192.168.25.58\Team-CV\checkpoints\torch_yolov3'}

    checkpoint_paths = {'39': r'F:\Team-CV\SLS_ReID\models\model\ft_ResNet50/'}
    # checkpoint_paths = {'68': r'E:\github\YOLOv3_PyTorch\evaluate\weights'}
    post_weights = {k: 0 for k in checkpoint_paths.keys()}
    weight_index = {k: 0 for k in checkpoint_paths.keys()}
    time_inter = 10

    while 1:
        for key,checkpoint_path in checkpoint_paths.items():
            os.makedirs(checkpoint_path + '/result', exist_ok=True)
            checkpoint_weights = os.listdir(checkpoint_path)
            checkpoint_result = os.listdir(checkpoint_path + '/result')
            checkpoint_result = [cweight.split("_")[2][:-4] for cweight in checkpoint_result if cweight.endswith('ini')]
            checkpoint_weights =[cweight for cweight in checkpoint_weights if cweight.endswith('pth')]

            if weight_index[key]>=len(checkpoint_weights):
                print('weight_index[key]',weight_index[key],len(checkpoint_weights))
                time.sleep(time_inter)
                continue
            if post_weights[key] == checkpoint_weights[weight_index[key]]:
                print('post_weights[key]', post_weights[key])
                time.sleep(time_inter)
                continue
            post_weights[key] = checkpoint_weights[weight_index[key]]

            if post_weights[key].endswith("_.pth"):#检查权重是否保存完
                print("post_weights[key].split('_')",post_weights[key].split('_'))
                time.sleep(time_inter)
                continue
            if checkpoint_weights[weight_index[key]].split("_")[1][:-8] in checkpoint_result:
                print('weight_index[key] +',weight_index[key])
                weight_index[key] += 1
                time.sleep(time_inter//20)
                continue
            weight_index[key] += 1
            try:
                if opt.pretrain_snapshot:  # Restore pretrain model
                    state_dict = torch.load(opt.pretrain_snapshot)
                    print("loading model from %s"%opt.pretrain_snapshot)
                    model.load_state_dict(state_dict)
                else:
                    state_dict = torch.load(os.path.join(checkpoint_path, post_weights[key]))
                    print("loading model from %s" % os.path.join(checkpoint_path, post_weights[key]))
                    model.load_state_dict(state_dict)
            except Exception as E:
                print(E)
                time.sleep(time_inter)
                continue
            print("Start eval.")# Start the eval loop
            n_gt = 0
            correct = 0
            imagepath_list = []

            with torch.no_grad():
                time1=datetime.datetime.now()
                model.train(False)  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                batch_len = len(train_loader)
                for step, data in enumerate(train_loader):
                    # get the inputs
                    inputs, labels = data
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    # statistics
                    running_loss += loss.item()
                    current_correct = float(torch.sum(preds == labels.data)) / len(labels)
                    running_corrects += current_correct
                    if step % 2 == 0:
                        print("batch {}/{} Loss:{:.4f} acc:{:.4f} avacc:{:.4f}".format \
                                  ( step, batch_len, loss.item(), current_correct, running_corrects / (step + 1)))

            Mean_Average = float(running_corrects / (step + 1))
            print('Mean Average Precision: %.4f' % Mean_Average)
            name=post_weights[key].replace(".pth","")
            ini_name = os.path.join(checkpoint_path+'/result/', '%.4f_%s_%s.ini'%((float(name.split("_")[2])+Mean_Average)/2,name.split("_")[2],name.split("_")[1]))
            write_ini(ini_name, Mean_Average, imagepath_list)

if opt.use_dense:
    model = ft_net_dense(opt.num_classes)
else:
    model = ft_net(opt.num_classes)

#print(model)

if use_gpu:
    model = model.cuda()
# state_dict = torch.load(r"F:\Team-CV\SLS_ReID\models\model\ft_ResNet50\net_2114_0.9996.pth")
# model.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()


ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())


dir_name = os.path.join('./models/model', name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    evaluate(model)