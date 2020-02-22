from __future__ import print_function
import argparse
import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import skimage
import skimage.io
import skimage.transform

import numpy as np
import time

from utils import preprocess 
from models import *

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/scratch/datasets/kitti2015/testing/',
                    help='select model')
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
parser.add_argument('--save_figure', action='store_true', help='if true, save the png file, not the npy file')
parser.add_argument('--fullsize', action='store_true', help='if true, use a fullsize image')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print('CUDA available? ', args.cuda)

if(args.fullsize):
    top_pad_const = 2064
    left_pad_const = 2464
else:
    top_pad_const = 544
    left_pad_const = 1248

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
    print(args.model, ' model selected')
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()
cudnn.benchmark = True

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)

    # getting rid of module. in state dict
    substring = 'module.'
    state_dict_tmp = OrderedDict()
    for k in state_dict:
        new_k = k[len(substring):] if k.startswith(substring) else k
        state_dict_tmp[new_k] = state_dict[k]
    state_dict = state_dict_tmp

    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
            imgL = torch.FloatTensor(imgL).cuda()
            imgR = torch.FloatTensor(imgR).cuda()
            imgL, imgR= Variable(imgL), Variable(imgR)
            
            with torch.no_grad():
                output = model(imgL,imgR)
            output = torch.squeeze(output)
            pred_disp = output.data.cpu().numpy()
            
            return pred_disp
        else:
            print('I will not work without cuda')
            return None

def main():
    processed = preprocess.get_transform(augment=False)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    for inx in range(len(test_left_img)):
        # read left / right images
        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))

        if(args.fullsize == False):
            # downsample to (H/4 , W/2)
            imgL_o = skimage.transform.downscale_local_mean(imgL_o, (4,2,1))
            imgR_o = skimage.transform.downscale_local_mean(imgR_o, (4,2,1))

        # process the image
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        # reshape to match the batch size (inference batch size = 1)
        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]]) #(B, C, H, W)
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        top_pad = top_pad_const-imgL.shape[2]
        left_pad = left_pad_const-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))

        top_pad   = top_pad_const-imgL_o.shape[0]
        left_pad  = left_pad_const-imgL_o.shape[1]
        img = pred_disp[top_pad:,:-left_pad]
        print(test_left_img[inx].split('/')[-1])
        if args.save_figure:
            skimage.io.imsave(args.save_path+'/'+test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))
        else:
            np.save(args.save_path+'/'+test_left_img[inx].split('/')[-1][:-4], img)

if __name__ == '__main__':
   main()






