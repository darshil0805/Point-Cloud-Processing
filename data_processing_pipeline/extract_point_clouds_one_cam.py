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
BAG_PATH = "/home/skills/varun/latest_jan/jan_21_2"
EXTRACTED_DATA_ROOT = "/home/skills/varun/latest_jan/extracted_data_all/jan_21_2"
OUTPUT_ROOT = "/home/skills/varun/latest_jan/point_clouds_aligned_test/jan_21_2/point_clouds_aligned"

# Extrinsics: Depth to Color (Manual Alignment)
T_COLOR_OPTICAL_DEPTH_OPTICAL = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)
T_DEPTH_TO_COLOR = np.linalg.inv(T_COLOR_OPTICAL_DEPTH_OPTICAL)

# Camera extrinsics (Base -> Depth Camera Optical Frame)
# Note: If calibration was done for color frame, this should be adjusted.
BASE_TO_EXT_CAM = np.array([
    [ 0.9997, -0.0232,  0.0054,  0.425 ],
    [-0.0048,  0.0249,  0.9997, -0.7518],
    [-0.0234, -0.9994,  0.0248,  0.3408],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float32)

# Topics for camera info
CAMERA_INFO_TOPICS = {
    'color_info': '/camera/camera/color/camera_info',
    'depth_info': '/camera/camera/depth/camera_info',
}

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def extract_camera_intrinsics(bag_path):
    """Extract camera intrinsics from camera_info topics."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    
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
                    intrinsics[key] = {
                        'K': K,
                        'width': msg.width,
                        'height': msg.height,
                        'fx': K[0, 0],
                        'fy': K[1, 1],
                        'cx': K[0, 2],
                        'cy': K[1, 2]
                    }
        if len(intrinsics) == len(CAMERA_INFO_TOPICS):
            break
    return intrinsics

def deproject(depth, fx, fy, cx, cy):
    """Convert depth image to 3D point cloud in camera coordinates."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

def project_and_sample(points_cam, rgb, color_params, T_depth_to_color):
    """Project 3D points to color image and sample colors."""
    # 1. Transform to color frame
    points_homo = np.column_stack([points_cam, np.ones(len(points_cam))])
    pts_color_frame = (T_depth_to_color @ points_homo.T).T[:, :3]
    
    # 2. Project to 2D
    z = pts_color_frame[:, 2]
    u = (pts_color_frame[:, 0] * color_params['fx'] / np.maximum(z, 0.001)) + color_params['cx']
    v = (pts_color_frame[:, 1] * color_params['fy'] / np.maximum(z, 0.001)) + color_params['cy']
    
    # 3. Sample colors
    h, w, _ = rgb.shape
    u_idx = np.round(u).astype(int)
    v_idx = np.round(v).astype(int)
    
    in_image = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
    mask = (z > 0.01) & in_image
    
    colors = np.zeros((len(points_cam), 3), dtype=np.float32)
    colors[mask] = rgb[v_idx[mask], u_idx[mask]] / 255.0
    
    return colors, mask

def transform_pointcloud(points, transform_matrix):
    """Transform point cloud using 4x4 homogeneous transformation matrix."""
    points_homo = np.column_stack([points, np.ones(len(points))])
    points_transformed = (transform_matrix @ points_homo.T).T
    return points_transformed[:, :3]

def create_point_cloud(points, colors):
    """Create Open3D point cloud from points and colors."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return pcd

# -------------------------
# MAIN
# -------------------------
def process_trajectory(traj_name, bag_path, extracted_data_root, output_root):
    print(f"\nProcessing: {traj_name}")
    intrinsics = extract_camera_intrinsics(bag_path)
    
    if 'color_info' not in intrinsics or 'depth_info' not in intrinsics:
        print("❌ Error: Missing intrinsics")
        return

    # Paths
    traj_path = os.path.join(extracted_data_root, traj_name)
    rgb_path = os.path.join(traj_path, "camera", "rgb")
    depth_path = os.path.join(traj_path, "camera", "depth")
    
    output_dirs = {
        'camera_frame': os.path.join(output_root, traj_name, "camera_camera_frame"),
        'global_frame': os.path.join(output_root, traj_name, "camera_global_frame"),
    }
    for d in output_dirs.values(): os.makedirs(d, exist_ok=True)
    
    rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
    
    for i in range(len(rgb_files)):
        rgb = cv2.imread(os.path.join(rgb_path, f"rgb_{i:05d}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = np.load(os.path.join(depth_path, f"depth_{i:05d}.npy"))
        
        # 1. Deproject with Depth Intrinsics
        pts_cam = deproject(depth, **{k: intrinsics['depth_info'][k] for k in ['fx', 'fy', 'cx', 'cy']})
        
        # 2. Project and Sample with Color Intrinsics
        colors, projection_mask = project_and_sample(
            pts_cam, rgb, intrinsics['color_info'], T_DEPTH_TO_COLOR
        )
        
        # 3. Filter
        depth_mask = (depth.reshape(-1) > 0.01) & (depth.reshape(-1) < 5.0)
        final_mask = projection_mask & depth_mask & np.isfinite(pts_cam).all(axis=1)
        
        pts_valid = pts_cam[final_mask]
        colors_valid = colors[final_mask]
        
        # Save camera frame
        pcd_cam = create_point_cloud(pts_valid, colors_valid)
        o3d.io.write_point_cloud(os.path.join(output_dirs['camera_frame'], f"pc_{i:05d}.ply"), pcd_cam)
        
        # Save global frame
        pts_global = transform_pointcloud(pts_valid, BASE_TO_EXT_CAM)
        pcd_global = create_point_cloud(pts_global, colors_valid)
        o3d.io.write_point_cloud(os.path.join(output_dirs['global_frame'], f"pc_{i:05d}.ply"), pcd_global)

        if (i+1) % 50 == 0: print(f"  Processed {i+1} frames...")

if __name__ == "__main__":
    traj_name = os.path.basename(BAG_PATH.rstrip('/'))
    process_trajectory(traj_name, BAG_PATH, EXTRACTED_DATA_ROOT, OUTPUT_ROOT)
