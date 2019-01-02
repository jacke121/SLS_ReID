from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os

from torch.utils.data import dataloader

from data.market1501 import Market1501
from nets.MobileNetV2 import *
import argparse
from torchvision import datasets, transforms

def train_model(args, model, criterion, optimizer, scheduler, num_epochs):
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

    # if args.erasing_p > 0:
    #     transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    #
    # if args.color_jitter:
    #     transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
    #                                                    hue=0)] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    m_dataset = Market1501(args.data_dir, data_transforms['val'], "train")
    train_loader = dataloader.DataLoader(m_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0)

    since = time.time()
    resumed = False

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1,num_epochs):

        need_val=False
        for phase in ['train','val']:
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch+1)
                    resumed = True
                else:
                    if not need_val:
                        continue
                    scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            batch_len = len( train_loader)
            for step, data in enumerate(train_loader):
                # get the inputs
                inputs, labels, path = data
            # Iterate over data.
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                lr = scheduler.get_lr()[0]
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                current_correct = float(torch.sum(preds == labels.data)) / len(labels)
                running_corrects += current_correct
                if step % 2 == 0:
                    print("{} Epoch {}/{} batch {}/{} Loss:{:.4f} acc:{:.4f} avacc:{:.4f} lr:{:.4f}".format \
                              (phase, epoch, num_epochs - 1, step, batch_len, loss.item(), current_correct,
                               running_corrects / (step + 1), lr))
            if running_corrects / (step + 1) > 0.7:
                need_val = True
            else:
                need_val = False

        if phase == 'val':
            save_filename = '%0.4f_%s.pth' % ( (running_corrects / (step + 1)), epoch)
            save_path = os.path.join('./models/m_v2',  save_filename)
            torch.save(model.cpu().state_dict(), save_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data_dir', default=r'\\192.168.55.73\Team-CV\dataset\origin_all_datas_0814', type=str)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_class', type=int, default=98)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--erasing_p', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default="output")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    model = mobilenetv2_19(num_classes = args.num_class)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=args.num_epochs)