# coding='utf-8'
from __future__ import print_function, division
import datetime
from shutil import copyfile

import nmslib

def query():
    # create a random matrix to index
    data = np.random.randn(10000, 128).astype(np.float32)
    time1=datetime.datetime.now()
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=False)
    print("time1", (datetime.datetime.now() - time1).microseconds)
    time1 = datetime.datetime.now()
    # query for the nearest neighbours of the first datapoint
    ids, distances = index.knnQuery(data[0], k=10)
    print("time2",(datetime.datetime.now()-time1).microseconds)
    print(ids,distances)
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    # neighbours = index.knnQueryBatch(data, k=10, num_threads=4)
    # print(neighbours)


import argparse
import torch
import numpy as np


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
parser.add_argument('--data_dir', default=r'\\192.168.55.73\Team-CV\dataset\origin_all_datas_0807', type=str, help='training dir path')
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
m_dataset=Market1501(data_dir, data_transforms['val'], "test")
train_loader = dataloader.DataLoader(m_dataset,shuffle=True,batch_size=opt.batchsize,num_workers=0)

use_gpu = torch.cuda.is_available()


def eval_model(model):
    since = time.time()
    # Each epoch has a training and validation phase
    model.eval()
    batch_len=len(train_loader)
    features=[]
    paths=[]
    for step, data in enumerate(train_loader):
        if step>400:
            break
        # get the inputs
        inputs, labels,path = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs, labels = inputs, labels
        outputs = model(inputs)
        for index, feature in enumerate(outputs):
            features.append(np.asarray(feature.data))
            paths.append(str(path[index]))

    features=np.asarray(features).astype(np.float32)
    # data = np.random.randn(m_dataset.__len__(), 173).astype(np.float32)
    time1 = datetime.datetime.now()
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(features)
    index.createIndex({'post': 2}, print_progress=False)
    print("time1", (datetime.datetime.now() - time1).microseconds,"data_len",features.__len__())
    time1 = datetime.datetime.now()

    q_index=0
    id_s=list(range(0,features.__len__()))
    querys=[]
    while(True):
        if q_index>features.__len__():
            break
        # query for the nearest neighbours of the first datapoint
        ids, distances = index.knnQuery(features[q_index], k=100)
        print("knnQuery time", (datetime.datetime.now() - time1).microseconds)
        ids=ids[distances<0.2]
        print(ids, distances)

        chaji=list(set(id_s).difference(set(ids)))  # b中有而a中没有的
        if len(chaji)==0:
            break
        q_index=min(chaji)
        result=data_dir+"/result/"+str(q_index)
        os.makedirs(result,exist_ok=True)
        for id in ids:
            if id in id_s:
                id_s.remove(id)
            print("   ",paths[id])
            copyfile(paths[id], result + '/' + os.path.basename(paths[id]))

if opt.use_dense:
    model = ft_net_dense(opt.num_classes)
else:
    # model = ft_net(opt.num_classes)
    model = MobileNet2(num_classes=173)

#print(model)
state_dict = torch.load(r"E:\github\SLS_ReID\models\m_v2\ft_ResNet50\1.0000_286.pth")
model.load_state_dict(state_dict)
if use_gpu:
    model = model.cuda()

if __name__ == "__main__":
    model = eval_model(model)

