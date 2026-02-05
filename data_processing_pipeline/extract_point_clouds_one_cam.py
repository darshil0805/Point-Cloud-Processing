#!/usr/bin/env python3
"""
Extract point clouds from RGB-D data from a single external camera.
Uses manual color-depth alignment by projecting depth points onto the color image.
Generates:
1. Camera coordinate frame point clouds
2. Global coordinate frame point clouds (using static extrinsics)
"""
import os
import numpy as np
import cv2
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# --- CONFIG ---
BAG_PATH = "/media/skills/RRC HDD A/cross-emb/real-lab-data/feb-01-3"
EXTRACTED_DATA_ROOT = "/media/skills/RRC HDD A/cross-emb/real-lab-data/feb-01-3/extracted_data_one_camera/feb-01-3"
OUTPUT_ROOT = "/media/skills/RRC HDD A/cross-emb/real-lab-data/feb-01-3/point_clouds_aligned_test/feb-01-3/point_clouds_aligned"

# Extrinsics: Depth to Color (Manual Alignment)
T_DEPTH_TO_COLOR = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)
#T_DEPTH_TO_COLOR = np.linalg.inv(T_COLOR_OPTICAL_DEPTH_OPTICAL)

# Camera extrinsics (Base -> Depth Camera Optical Frame)
# Note: If calibration was done for color frame, this should be adjusted.
BASE_TO_EXT_CAM = np.array([
    [-0.5777,  0.4385, -0.6884,  0.7814],
    [ 0.7992,  0.1327, -0.5862,  0.4271],
    [-0.1657, -0.8889, -0.4271,  0.5267],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float32)

# Topics for camera info
CAMERA_INFO_TOPICS = {
    'color': '/camera/camera/color/camera_info',
    'depth': '/camera/camera/depth/camera_info',
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
    
    # Setup directories
    for subdir in ['camera_frame', 'global_frame']:
        os.makedirs(os.path.join(OUTPUT_ROOT, subdir), exist_ok=True)

    rgb_path = os.path.join(EXTRACTED_DATA_ROOT, "camera", "rgb")
    depth_path = os.path.join(EXTRACTED_DATA_ROOT, "camera", "depth")
    rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
    
    for i in range(len(rgb_files)):
        rgb = cv2.cvtColor(cv2.imread(os.path.join(rgb_path, f"rgb_{i:05d}.png")), cv2.COLOR_BGR2RGB)
        depth = np.load(os.path.join(depth_path, f"depth_{i:05d}.npy"))
        
        pts_cam = deproject(depth, intrinsics['depth'])
        colors, proj_mask = align_colors(pts_cam, rgb, intrinsics['color'], T_DEPTH_TO_COLOR)
        
        mask = proj_mask & (depth.reshape(-1) > 0.01) & (depth.reshape(-1) < 5.0)
        pts_v, clrs_v = pts_cam[mask], colors[mask]
        
        # Save camera frame
        pcd_cam = o3d.geometry.PointCloud()
        pcd_cam.points = o3d.utility.Vector3dVector(pts_v)
        pcd_cam.colors = o3d.utility.Vector3dVector(clrs_v)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "camera_frame", f"pc_{i:05d}.ply"), pcd_cam)
        
        # Save global frame
        pts_g = (BASE_TO_EXT_CAM @ np.column_stack([pts_v, np.ones(len(pts_v))]).T).T[:, :3]
        pcd_global = o3d.geometry.PointCloud()
        pcd_global.points = o3d.utility.Vector3dVector(pts_g)
        pcd_global.colors = o3d.utility.Vector3dVector(clrs_v)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_ROOT, "global_frame", f"pc_{i:05d}.ply"), pcd_global)

        if (i+1) % 50 == 0: print(f"  Processed {i+1} frames...")

if __name__ == "__main__":
    process()
