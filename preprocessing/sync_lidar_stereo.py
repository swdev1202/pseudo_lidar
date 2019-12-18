import argparse
import shutil
import os

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.synchronization_database import SynchronizationDB

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synchronize LiDAR and stereo images')
    parser.add_argument('--data_path', type=str, default='~/argoverse-tracking/')
    parser.add_argument('--train_size', type=int, default=4)
    args = parser.parse_args()

    for i in range(1, args.train_size+1):
        train_dir = args.data_path + 'train' + str(i) + '/'
        argoverse_loader = ArgoverseTrackingLoader(train_dir)
        print('Total number of logs:',len(argoverse_loader))
        argoverse_loader.print_all()

        for log_index, log_id in enumerate(argoverse_loader.log_list):
            # current directory full path
            curr_dir = train_dir + log_id + '/'

            # create directories to sort out synchronized point cloud and images
            sync_lidar_dir = curr_dir + 'sync_lidar/'

            if not os.path.isdir(sync_lidar_dir):
                os.makedirs(sync_lidar_dir)

            # load the current log
            curr_log = argoverse_loader[log_index]

            # create a synchronization db to extract synced lidar, left, and right
            db = SynchronizationDB(train_dir, log_id)
            
            # get all left, right image paths + lidar point cloud paths (absolute)
            unique_stereo_left = curr_log.image_timestamp_list['stereo_front_left']
            unique_stereo_right = curr_log.image_timestamp_list['stereo_front_right']
            unique_lidar = curr_log.lidar_list

            lidar_sync_left = []
            lidar_sync_right = []
            for left_img in unique_stereo_left:
                lidar_sync_left.append(db.get_closest_lidar_timestamp(left_img, log_id))

            for right_img in unique_stereo_right:
                lidar_sync_right.append(db.get_closest_lidar_timestamp(right_img, log_id))

            # only when their lidar timestamps are synchronized
            if(lidar_sync_left == lidar_sync_right):
                for stamp in lidar_sync_left:
                    for lidar_idx in range(len(unique_lidar)):
                        if(str(stamp) in unique_lidar[lidar_idx]):
                            lidar_filename = unique_lidar[lidar_idx].split('/')[-1]
                            shutil.copy2(unique_lidar[lidar_idx], sync_lidar_dir + lidar_filename)
            else:
                print("Left and Right Images Cannot be Synchronized.", log_id)
            
            print(log_id, " Complete")
        
        print("Train ", i, " Complete")