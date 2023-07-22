import pandas as pd
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

def get_transform_mat(rotation, translation, scale):

    r_mat = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


# get camera position
def get_camera_position(cameraMatrix, R, t, img_size, color=[1,0,0]):
    
    R = R.reshape((3,3))

    # image plane 
    camera_plane = np.array([[0,           0, img_size[1], img_size[1]],
                             [0, img_size[0], img_size[0],           0],
                             [1,           1,           1,           1]])

    # because origin one is too big
    camera_plane = camera_plane / 6 

    # map camera plane to 3d space    
    v = np.linalg.pinv(cameraMatrix) @ camera_plane
    camera_to_space = (np.linalg.pinv(R) @ v + t)   #3x4

    # add center
    camera_to_space = np.concatenate((camera_to_space, t), axis=1) #3x5
    camera_to_space = camera_to_space.T # 5x3
 
    # set up line
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(camera_to_space),
        lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )
    colors = np.tile(color, (8, 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


# load point cloud function
def load_point_cloud(points3D_df: pd.DataFrame):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def displayPointcloud(points3D_df, R_list, t_list, cameraMatrix, img_size):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    point_cloud = load_point_cloud(points3D_df)
    vis.add_geometry(point_cloud)

    for i in range(R_list.shape[0]):
        
        R = Rotation.from_quat(R_list[i]).as_matrix()
        t = t_list[i].reshape(3,1)
        line_set = get_camera_position(cameraMatrix, R, t, img_size)
        vis.add_geometry(line_set)

    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)
    vis.run()
    vis.destroy_window()   