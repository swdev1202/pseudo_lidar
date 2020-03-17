import argparse
import os

import numpy as np
import scipy.misc as ssc

import kitti_util


def project_disp_to_points(calib, disp, max_high, base):
    disp[disp < 0] = 0
    baseline = base
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
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
    parser = argparse.ArgumentParser(description='Generate Lidar from disparity or depth')
    parser.add_argument('--calib_dir', type=str,
                        default='~/Kitti/object/training/calib')
    parser.add_argument('--masked_disparity_dir', type=str,
                        default='~/Kitti/object/training/predicted_disparity')
    parser.add_argument('--velo_dir', type=str, default='~/Kitti')
    parser.add_argument('--save_dir', type=str,
                        default='~/Kitti/object/training/predicted_velodyne')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument('--datatype', type=str, default='KITTI')

    args = parser.parse_args()

    assert os.path.isdir(args.masked_disparity_dir)
    assert os.path.isdir(args.calib_dir)
    assert os.path.isdir(args.velo_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    disps = [x for x in os.listdir(args.masked_disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
    disps = sorted(disps)

    for fn in disps:
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)

        # original point cloud
        velo_file = '{}/{}.bin'.format(args.velo_dir, predix)
        velo = np.fromfile(velo_file, dtype=np.float32).reshape((-1,4))[:, :3]

        if fn[-3:] == 'png':
            disp_map = ssc.imread(args.masked_disparity_dir + '/' + fn)
        elif fn[-3:] == 'npy':
            disp_map = np.load(args.masked_disparity_dir + '/' + fn)
            print(disp_map.shape)
        else:
            assert False

        if(args.datatype == 'KITTI'):
            base = 0.54
        else:
            base = 0.2986

        if not args.is_depth:
            disp_map = (disp_map*256).astype(np.uint16)/256.
            pseudo_lidar = project_disp_to_points(calib, disp_map, args.max_high, base)
        else:
            disp_map = (disp_map).astype(np.float32)/256.
            pseudo_lidar = project_depth_to_points(calib, disp_map, args.max_high)
        
        # pad 1 in the indensity dimension
        # pseudo_lidar = np.concatenate([pseudo_lidar, np.ones((pseudo_lidar.shape[0], 1))], 1)
        # pseudo_lidar = pseudo_lidar.astype(np.float32)

        # concatenate pseudo lidar and original lidar
        combined_lidar = np.vstack((velo,pseudo_lidar))
        combined_lidar = np.concatenate([combined_lidar, np.ones((combined_lidar.shape[0], 1))], 1)
        combined_lidar = combined_lidar.astype(np.float32)

        combined_lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))
