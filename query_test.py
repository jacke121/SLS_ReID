# coding='utf-8'
import numpy as np
from scipy.spatial.distance import cdist

from utils import utility
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking
import datetime
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torchvision import transforms

from data.market1501 import Market1501
from nets.model import ft_net

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

class Trainer():
    def __init__(self, args, model, ckpt):
        self.args = args

        data_transforms = {
            'val': transforms.Compose(transform_val_list),
        }

        dataset_test = Market1501(args.data_dir, data_transforms['val'], "test")
        self.test_loader = dataloader.DataLoader(dataset_test, shuffle=True, batch_size=args.batchsize, num_workers=0)
        dataset_query = Market1501(args.data_dir, data_transforms['val'], "query")
        self.query_loader = dataloader.DataLoader(dataset_query, shuffle=True, batch_size=args.batchsize, num_workers=0)


        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.testset =dataset_test# getattr(module, args.data_test)(args, test_transform, 'test')
        self.queryset =dataset_query# getattr(module, args.data_test)(args, test_transform, 'query')

        self.ckpt = ckpt
        self.model = model
        self.lr = 0.
        self.device = torch.device('cpu' if args.cpu else 'cuda')

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)
    def extract_feature(self, loader):
        features = torch.FloatTensor()
        for (inputs, labels) in loader:
            # ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            ff = torch.FloatTensor(inputs.size(0), 123).zero_()
            for i in range(2):
                if i == 1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
        return features
    def test(self):
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.train(False)
        # self.model.eval()
        time_old = datetime.datetime.now()
        self.ckpt.add_log(torch.zeros(1, 5))
        qf = self.extract_feature(self.query_loader).numpy()
        gf = self.extract_feature(self.test_loader).numpy()

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf)
        print('time_cost' ,(datetime.datetime.now( ) -time_old).microseconds)
        r = cmc(dist, self.queryset.ids, self.testset.ids, separate_camera_set=False,
                single_gallery_shot=True,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids)

        self.ckpt.log[-1, 0] = m_ap
        self.ckpt.log[-1, 1] = r[0]
        self.ckpt.log[-1, 2] = r[2]
        self.ckpt.log[-1, 3] = r[4]
        self.ckpt.log[-1, 4] = r[9]
        best = self.ckpt.log.max(0)
        print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
                m_ap,
                r[0], r[2], r[4], r[9],
                best[0][0],
                (best[1][0] + 1 ) *self.args.test_every
            )
        )
        if not self.args.test_only:
            self.ckpt.save(self, 1, is_best=((best[1][0] + 1 ) *self.args.test_every == 1))
if __name__ == '__main__':
    args = EasyDict()
    args.re_rank=True
    args.cpu=False
    args.test_only=False
    args.num_classes=173
    args.height=256
    args.width=128
    args.batchsize=48
    args.data_dir=r'\\192.168.55.73\Team-CV\dataset\origin_all_datas_0807'
    args.load=''
    args.test_every=20
    args.save='test'
    args.reset=True
    args.data_test="Market1501"
    model = ft_net(args.num_classes)

    # print(model)

    model = model.cuda()
    state_dict = torch.load(r"F:\Team-CV\SLS_ReID\models\model\ft_ResNet50\net_675_0.8042.pth")
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()

    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    ckpt = utility.checkpoint(args)
    m_test= Trainer(args,model,ckpt)
    m_test.test()