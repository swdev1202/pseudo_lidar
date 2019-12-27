import argparse
import argoverse
import numpy as np
import os
import shutil

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.camera_stats import get_image_dims_for_camera as get_dim

# Argoverse Dataset Specific
NUM_TRAIN = 4
BASELINE = 0.2986 #page 4 of the paper (https://arxiv.org/pdf/1911.02620.pdf)
STEREO_IMG_WIDTH = 2464
STEREO_IMG_HEIGHT = 2056
LEFT_STEREO_CAM = 'stereo_front_left'

def generate_disparity_from_velo_argo(pc_path, scene_log):
    pc = load_ply(pc_path) # load point cloud
    calib = scene_log.get_calibration(LEFT_STEREO_CAM) # load calibration

    # uv => nx3 points in image coord + depth
    uv = calib.project_ego_to_image(pc) # 3D point cloud -> 2d pixel coordinates
    print(uv.shape)

    fov_inds = (uv[:, 0] < STEREO_IMG_WIDTH - 1) & (uv[:, 0] >= 0) & \
               (uv[:, 1] < STEREO_IMG_HEIGHT - 1) & (uv[:, 1] >= 0)
    fov_inds = fov_inds & (pc[:, 0] > 2)
    
    imgfov_pc_velo = pc[fov_inds, :]
    imgfov_pc_rect = calib.project_ego_to_cam(imgfov_pc_velo)
    imgfov_pts_2d = uv[fov_inds, :]
    depth_map = np.zeros((STEREO_IMG_HEIGHT, STEREO_IMG_WIDTH)).astype(int)
    imgfov_pts_2d = np.round(imgfov_pts_2d).astpye(int)
    print(imgfov_pts_2d.shape)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    
    disp_map = (calib.fu * BASELINE) / depth_map
    return disp_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity for Argoverse Dataset')
    parser.add_argument('--data_path', type=str, default='~/argoverse-tracking/')
    parser.add_argument('--train_size', type=int, default=4)
    args = parser.parse_args()

    for i in range(NUM_TRAIN+1):
        train_dir = args.data_path + 'train' + str(i) + '/'
        print('Currently Processing Train ', str(i))

        argoverse_loader = ArgoverseTrackingLoader(train_dir)
        print('Total Number of Logs in train', str(i), ': ', len(argoverse_loader))

        for log_index, log_id in enumerate(argoverse_loader.log_list):
            curr_log = argoverse_loader.get(log_id)
            curr_dir = train_dir + str(log_id) + '/'

            # create a ground-truth disparity directory
            disp_dir = curr_dir + 'disparity/'

            if not os.path.isdir(disp_dir):
                os.makedirs(disp_dir)
            
            # point to valid file directories to generate ground-truth disparity
            lidar_dir = curr_dir + 'sync_lidar/'

            assert os.path.isdir(lidar_dir)

            # Argoverse Lidar point clouds are stored as .ply format
            lidar_files_list = [x for x in os.listdir(lidar_dir) if x[-3:] == 'ply']
            lidar_files_list = sorted(lidar_files_list)
            print('Total LiDAR point clouds for this log = ', len(lidar_files_list))

            for fn in lidar_files_list:
                lidar_file_path = lidar_dir + fn
                lidar_file_name = fn.split('.')[0]
                grnd_truth_disp = generate_disparity_from_velo_argo(lidar_file_path, curr_log)
                np.save(disp_dir + lidar_file_name, grnd_truth_disp)
                np.savetxt(disp_dir + lidar_file_name + '.out', grnd_truth_disp)
    print("complete!")