#!/usr/bin/env python3
"""
Extract point clouds from multiple RGB-D cameras (Wrist + External).
Uses manual color-depth alignment by projecting depth points onto the color image.
Generates:
1. Camera coordinate frame point clouds
2. Global coordinate frame point clouds
3. Combined global point clouds
"""
import os
import numpy as np
import cv2
import open3d as o3d
import pybullet as p
import pybullet_data
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# --- CONFIG ---
BAG_PATH = "/home/skills/varun/latest_jan/jan_24_1"
EXTRACTED_DATA_ROOT = "/home/skills/varun/latest_jan/extracted_data_all/jan_24_1"
OUTPUT_ROOT = "/home/skills/varun/latest_jan/point_clouds_aligned_test/jan_24_1/point_clouds_aligned"

URDF_PATH = "/home/skills/varun/Point-Cloud-Processing/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6

# Extrinsics: Depth to Color (Manual Alignment)
# Assuming both cameras are the same model (e.g., RealSense D435)
T_COLOR_OPTICAL_DEPTH_OPTICAL = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)
T_DEPTH_TO_COLOR = np.linalg.inv(T_COLOR_OPTICAL_DEPTH_OPTICAL)

# Camera 1 (Wrist): EEF -> Depth Camera Optical Frame
EEF_TO_WRIST_CAM = np.array([
    [-0.0140339,  -0.99989573,  0.00340287,  0.05830656 ],
    [ 0.99975296, -0.0140904,  -0.01718981,  0.02025594],
    [ 0.01723596,  0.00316079,  0.99984645,  0.03893953],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float32)

# Camera 2 (External): Base -> Depth Camera Optical Frame
BASE_TO_EYE_CAM = np.array([
    [ 0.9882, -0.0163,  0.1522,  0.3611],
    [-0.1524, -0.0092,  0.9883, -0.8294],
    [-0.0147, -0.9998, -0.0116,  0.4017],
    [ 0.0,     0.0,     0.0,     1.0   ]
], dtype=np.float32)

CAMERA_INFO_TOPICS = {
    'cam1_color': '/camera1/camera1/color/camera_info',
    'cam1_depth': '/camera1/camera1/depth/camera_info',
    'cam2_color': '/camera2/camera2/color/camera_info',
    'cam2_depth': '/camera2/camera2/depth/camera_info',
}

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def extract_intrinsics(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    topics_info = reader.get_all_topics_and_types()
    def get_type(topic_name):
        tlist = [t.type for t in topics_info if t.name == topic_name]
        return get_message(tlist[0]) if tlist else None
    
    intrinsics = {}
    while reader.has_next():
        topic, msg_data, ts = reader.read_next()
        for key, topic_name in CAMERA_INFO_TOPICS.items():
            if topic == topic_name and key not in intrinsics:
                msg_type = get_type(topic_name)
                if msg_type:
                    msg = deserialize_message(msg_data, msg_type)
                    K = np.array(msg.k).reshape(3, 3)
                    intrinsics[key] = {'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
        if len(intrinsics) == len(CAMERA_INFO_TOPICS): break
    return intrinsics

def compute_eef_pose(robot_id, joint_angles):
    for j in range(len(joint_angles)): p.resetJointState(robot_id, j, joint_angles[j])
    state = p.getLinkState(robot_id, EEF_INDEX)
    pos, quat = np.array(state[0]), np.array(state[1])
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T

def deproject(depth, params):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (u - params['cx']) * z / params['fx']
    y = (v - params['cy']) * z / params['fy']
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def align_colors(pts_cam, rgb, color_params, T_depth_to_color):
    points_homo = np.column_stack([pts_cam, np.ones(len(pts_cam))])
    pts_color_frame = (T_depth_to_color @ points_homo.T).T[:, :3]
    z = pts_color_frame[:, 2]
    u = (pts_color_frame[:, 0] * color_params['fx'] / np.maximum(z, 0.001)) + color_params['cx']
    v = (pts_color_frame[:, 1] * color_params['fy'] / np.maximum(z, 0.001)) + color_params['cy']
    h, w, _ = rgb.shape
    u_idx, v_idx = np.round(u).astype(int), np.round(v).astype(int)
    in_image = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
    mask = (z > 0.01) & in_image
    colors = np.zeros((len(pts_cam), 3), dtype=np.float32)
    colors[mask] = rgb[v_idx[mask], u_idx[mask]] / 255.0
    return colors, mask

def process():
    intrinsics = extract_intrinsics(BAG_PATH)
    p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF_PATH, useFixedBase=True)
    
    # Setup directories
    for cam in ['camera1', 'camera2', 'combined']:
        os.makedirs(os.path.join(OUTPUT_ROOT, "global_frame", cam), exist_ok=True)

    states = np.loadtxt(os.path.join(EXTRACTED_DATA_ROOT, "states.txt"))
    for i in range(len(states)):
        T_base_eef = compute_eef_pose(robot, states[i, :6])
        T_base_cam1 = T_base_eef @ EEF_TO_WRIST_CAM
        T_base_cam2 = BASE_TO_EYE_CAM
        
        cams_data = []
        for cam_idx, T_base_cam in enumerate([T_base_cam1, T_base_cam2], 1):
            cam_name = f"camera{cam_idx}"
            rgb = cv2.cvtColor(cv2.imread(os.path.join(EXTRACTED_DATA_ROOT, cam_name, "rgb", f"rgb_{i:05d}.png")), cv2.COLOR_BGR2RGB)
            depth = np.load(os.path.join(EXTRACTED_DATA_ROOT, cam_name, "depth", f"depth_{i:05d}.npy"))
            
            pts_cam = deproject(depth, intrinsics[f'cam{cam_idx}_depth'])
            colors, proj_mask = align_colors(pts_cam, rgb, intrinsics[f'cam{cam_idx}_color'], T_DEPTH_TO_COLOR)
            
            mask = proj_mask & (depth.reshape(-1) > 0.01) & (depth.reshape(-1) < 5.0)
            pts_v, clrs_v = pts_cam[mask], colors[mask]
            
            pts_g = (T_base_cam @ np.column_stack([pts_v, np.ones(len(pts_v))]).T).T[:, :3]
            cams_data.append((pts_g, clrs_v))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_g)
            pcd.colors = o3d.utility.Vector3dVector(clrs_v)
            o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "global_frame", cam_name, f"pc_{i:05d}.ply"), pcd)

        # Combined
        pts_c = np.vstack([d[0] for d in cams_data])
        clrs_c = np.vstack([d[1] for d in cams_data])
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(pts_c)
        pcd_c.colors = o3d.utility.Vector3dVector(clrs_c)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "global_frame", "combined", f"pc_{i:05d}.ply"), pcd_c)

        if (i+1) % 50 == 0: print(f"  Processed {i+1} frames...")
    p.disconnect()

if __name__ == "__main__":
    process()
