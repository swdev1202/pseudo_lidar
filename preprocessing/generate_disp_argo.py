import argparse
import argoverse
import numpy as np
import os
import shutil

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.camera_stats import get_image_dims_for_camera as get_dim

NUM_TRAIN = 4

# def generate_dispariy_from_velo(pc_velo, height, width, calib):
#     pts_2d = calib.project_velo_to_image(pc_velo)
#     fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
#                (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
#     fov_inds = fov_inds & (pc_velo[:, 0] > 2)
#     imgfov_pc_velo = pc_velo[fov_inds, :]
#     imgfov_pts_2d = pts_2d[fov_inds, :]
#     imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
#     depth_map = np.zeros((height, width)) - 1
#     imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
#     for i in range(imgfov_pts_2d.shape[0]):
#         depth = imgfov_pc_rect[i, 2]
#         depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
#     baseline = 0.54

#     disp_map = (calib.f_u * baseline) / depth_map
#     return disp_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity for Argoverse Dataset')
    parser.add_argument('--data_path', type=str, default='~/argoverse-tracking/')
    args = parser.parse_args()

    for i in range(NUM_TRAIN+1):
        train_dir = args.data_path + 'train' + str(i) + '/'

        argoverse_loader = ArgoverseTrackingLoader(train_dir)
        print('Total Number of Logs in train', str(i), ': ', len(argoverse_loader))

        cam_left = argoverse_loader.CAMERA_LIST[7] # left pair of the stereo camera
        cam_right = argoverse_loader.CAMERA_LIST[8] # right pair of the stereo camera
        (cam_width, cam_height) = get_dim(cam_left)
        baseline = 0.2986 # page 4 of the paper (https://arxiv.org/pdf/1911.02620.pdf)
        
        for log_id in argoverse_loader.log_list:
            # 


    assert os.path.isdir(args.data_path)
    lidar_dir = args.data_path + '/lidar/'
    calib_file = args.data_path + 'vehicle_calibration_info.json'
    image_dir = args.data_path + '/stereo_front_left/'
    disparity_dir = args.data_path + '/disparity/'

    assert os.path.isdir(lidar_dir)
    assert os.path.isfile(calib_file)
    assert os.path.isdir(image_dir)

    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)

    # Argoverse Lidar point clouds are stored as .ply format
    lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'ply']
    lidar_files = sorted(lidar_files)

    # assert os.path.isfile(args.split_file)
    # with open(args.split_file, 'r') as f:
    #     file_names = [x.strip() for x in f.readlines()]

    for fn in lidar_files:
        predix = fn[:-4]
        #if predix not in file_names:
        #    continue

        calib_file = '{}/{}.txt'.format(calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)
        # load point cloud
        lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 4))[:, :3]
        image_file = '{}/{}.png'.format(image_dir, predix)
        image = ssc.imread(image_file)
        height, width = image.shape[:2]
        disp = generate_dispariy_from_velo(lidar, height, width, calib)
        np.save(disparity_dir + '/' + predix, disp)
        print('Finish Disparity {}'.format(predix))

