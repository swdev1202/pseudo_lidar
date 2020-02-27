import argparse
import os

import numpy as np
import scipy.misc as ssc

import kitti_util


def project_disp_to_points(calib, disp, max_high, base):
    disp[disp < 0] = 0
    baseline = base
    mask = disp > 0
    if(args.debug): print(f'currently, mask shape = {mask.shape}')
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    if(args.debug): print(f'rows = {rows}, cols = {cols}')
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    if(args.debug): print(f'c = {c}, r = {r}')
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    if(args.debug): print(f'points shape = {points.shape}')
    cloud = calib.project_image_to_velo(points)
    if(args.debug): print(f'cloud shape = {cloud.shape}')
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    if(args.debug): print(f'valid shape = {valid.shape}')
    return cloud[valid]

def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--calib_dir', type=str,
                        default='~/Kitti/object/training/calib')
    parser.add_argument('--disparity_dir', type=str,
                        default='~/Kitti/object/training/predicted_disparity')
    parser.add_argument('--save_dir', type=str,
                        default='~/Kitti/object/training/predicted_velodyne')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument('--datatype', type=str, default='KITTI')
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()

    assert os.path.isdir(args.disparity_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    disps = [x for x in os.listdir(args.disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
    disps = sorted(disps)

    if(args.debug):
        print('is it depth? ', args.is_depth)

    for fn in disps:
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)

        if fn[-3:] == 'png':
            disp_map = ssc.imread(args.disparity_dir + '/' + fn)
        elif fn[-3:] == 'npy':
            disp_map = np.load(args.disparity_dir + '/' + fn)
            print(disp_map.shape)
        else:
            assert False

        if(args.datatype == 'KITTI'):
            base = 0.54
        else:
            base = 0.2986

        if not args.is_depth:
            disp_map = (disp_map*256).astype(np.uint16)/256.
            lidar = project_disp_to_points(calib, disp_map, args.max_high, base)
        else:
            disp_map = (disp_map).astype(np.float32)/256.
            lidar = project_depth_to_points(calib, disp_map, args.max_high)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        if(args.debug): print(f'final lidar points = {lidar}')
        lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))
