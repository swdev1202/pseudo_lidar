from __future__ import print_function
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import skimage
import skimage.io
import skimage.transform

import numpy as np
import time
import math
from utils import preprocess
from torchsummary import summary
from models import *

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/scratch/datasets/kitti2015/testing/',
                    help='select data')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save_path', type=str, default='finetune_1000', metavar='S',
                    help='path to save the predict')
parser.add_argument('--save_figure', default=False, help='if true, save the png file, not the png file')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
torch.backends.cudnn.benchmark = True

# https://discuss.pytorch.org/t/how-can-i-crop-half/41767/6
device = 'cuda' if args.cuda else 'cpu'

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  


test_left_img, test_right_img = DA.dataloader(args.datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.to(device)

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()
        print(model)

        # if args.cuda:
        #    # imgL = torch.FloatTensor(imgL).cuda()
        #    # imgR = torch.FloatTensor(imgR).cuda()
        #     imgL = torch.tensor(imgL, dtype=torch.half, requires_grad=False).to(device)
        #     imgR = torch.tensor(imgR, dtype=torch.half, requires_grad=False).to(device)
        # else:
        #     imgL = torch.tensor(imgL, dtype=torch.half, requires_grad=False)
        #     imgR = torch.tensor(imgR, dtype=torch.half, requires_grad=False)

        # imgL, imgR= Variable(imgL), Variable(imgR)
        imgL = torch.tensor(imgL, dtype=torch.half, requires_grad=False).to(device)
        print(imgL)
        print(type(imgL))
        imgR = torch.tensor(imgR, dtype=torch.half, requires_grad=False).to(device)

        with torch.no_grad():
            print('here')
            output = model(imgL, imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp


def main():
   processed = preprocess.get_transform(augment=False)
   if not os.path.isdir(args.save_path):
       os.makedirs(args.save_path)


   for inx in range(len(test_left_img)):

       imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]]) # (batch, color_channel, H, W) = (1,3,2056,2464)
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       # pad to (2064, 2464) ... Argoverse Original Image Size = (2056,2464)
       top_pad = 2064-imgL.shape[2] # 8
       left_pad = 2464-imgL.shape[3] # 0
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       top_pad   = 2064-imgL_o.shape[0]
       left_pad  = 2464-imgL_o.shape[1]
       img = pred_disp[top_pad:,:-left_pad]
       print(test_left_img[inx].split('/')[-1])
       if args.save_figure:
           skimage.io.imsave(args.save_path+'/'+test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))
       else:
           np.save(args.save_path+'/'+test_left_img[inx].split('/')[-1][:-4], img)

if __name__ == '__main__':
   main()






