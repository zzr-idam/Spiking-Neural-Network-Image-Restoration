import os
import torch

import time,math

import dataset
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim, fft
from torch.nn import functional as F
from torchvision.utils import save_image
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding
from unet_model import UNet


def train(args):
    model = UNet(T=2, n_channels=3)

    model = model.cuda()
    
    
    encoder = encoding.PoissonEncoder(step_mode='m')

    t = 4


    functional.set_step_mode(model, step_mode='m')
    
    # model.load_state_dict(torch.load('/home/wangdi/Retinex/POLED/model/modelLoss120.pth')) #加载pth文件

    mse = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    optimizer.zero_grad()

    gt_folder = '/home/sylvanas/Wanghan/data_train/gts4/'
    input_folder = '/home/sylvanas/Wanghan/data_train/li4/'

    train_loader = dataset.style_loader(input_folder, gt_folder, 512, 1)

    num_batch = len(train_loader)

    for epoch in range(args.epoch):

        for idx, batch in tqdm(enumerate(train_loader), total=2):

            total_iter = epoch * num_batch + idx
            input = batch[0].unsqueeze(0).repeat(2, 1, 1, 1, 1).float().cuda()
            gtimg = batch[1].float().cuda()
            
            print(input.shape)
            print(gtimg.shape)

            optimizer.zero_grad()

            output = model(input)

            loss = mse(output, gtimg)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            functional.reset_net(model) 

            if np.mod(total_iter + 1, 1) == 0:
                print('{}, Epoch:{} Iter:{} total loss: {}'.format(args.save_dir, epoch, total_iter, loss.item()))

            if not os.path.exists(args.save_dir + '/image'):
                os.mkdir(args.save_dir + '/image')

        if epoch % 1 == 0:
            # content = torch.log(content)
            # output = torch.log(output)

            # out_image = torch.cat([output[0:3], gtimg[0:3]], dim=0)
            # save_image(out_image, args.save_dir + '/image/train{}.jpg'.format(epoch))
            torch.save(model.state_dict(),
                       '/home/sylvanas/Wanghan/model' + '/modelLoss{}.pth'.format(epoch))  # 指定pth文件存放路径


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--epoch', default=2000, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--save_dir', default='/home/sylvanas/Wanghan/result', type=str)  # 指定中间结果图片存放路径
    parser.add_argument('--guided_map_kernel_size', default=3, type=int)
    parser.add_argument('--pixelshuffle_ratio', default=2, type=int)
    parser.add_argument('--guided_map_channels', default=16, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with torch.autograd.set_detect_anomaly(True):

        train(args)
