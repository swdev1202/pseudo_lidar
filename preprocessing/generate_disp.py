import argparse
import os
import numpy as np
import imageio
import kitti_util

def generate_dispariy_from_velo(pc_velo, height, width, calib, cam_baseline):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    baseline = cam_baseline

    disp_map = (calib.f_u * baseline) / depth_map
    print(disp_map.shape)
    return disp_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str, default='~/Kitti/object/training/')
    parser.add_argument('--split_file', type=str, default='~/Kitti/object/train.txt')
    parser.add_argument('--dataset', type=str, default='KITTI')
    parser.add_argument('--scale', nargs='+', type=float, default=[1.0,1.0]) # scale (H,W)
    args = parser.parse_args()

    assert os.path.isdir(args.data_path)
    lidar_dir = args.data_path + '/velodyne/'
    calib_dir = args.data_path + '/calib/'
    image_dir = args.data_path + '/image_2/'
    disparity_dir = args.data_path + '/disparity' + str(int(args.scale[0]) + str(int(args.scale[1])) + '/'

    assert os.path.isdir(lidar_dir)
    assert os.path.isdir(calib_dir)
    assert os.path.isdir(image_dir)

    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)

    lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'bin']
    lidar_files = sorted(lidar_files)

    assert os.path.isfile(args.split_file)
    with open(args.split_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    for fn in lidar_files:
        predix = fn[:-4]
        if predix not in file_names:
            continue
        # load calibration info
        calib_file = '{}/{}.txt'.format(calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)

        # scale the calibration
        calib.scale_P(args.scale[0], args.scale[1])

        # load point cloud
        lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 4))[:, :3]
        image_file = '{}/{}.png'.format(image_dir, predix)
        image = imageio.imread(image_file)
        height, width = image.shape[:2]

        if(args.dataset == 'KITTI'):
            baseline = 0.54
        else: # ARGO stereo cam baseline distance
            baseline = 0.2986
            
        disp = generate_dispariy_from_velo(lidar, height, width, calib, baseline)
        np.save(disparity_dir + '/' + predix, disp)
        print('Finish Disparity {}'.format(predix))
