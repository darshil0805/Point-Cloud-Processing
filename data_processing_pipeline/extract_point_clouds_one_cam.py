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

T_DEPTH_TO_COLOR = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)

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

def process(bag_path, extracted_data_root, output_root, base_to_ext_cam):
    intrinsics = extract_intrinsics(bag_path)
    
    # Setup directories
    for subdir in ['camera_frame', 'global_frame']:
        os.makedirs(os.path.join(output_root, subdir), exist_ok=True)

    rgb_path = os.path.join(extracted_data_root, "camera", "rgb")
    depth_path = os.path.join(extracted_data_root, "camera", "depth")
    
    if not os.path.exists(rgb_path):
        print(f"ERROR: RGB path does not exist: {rgb_path}")
        return
    
    rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
    
    for i in range(len(rgb_files)):
        rgb_file = os.path.join(rgb_path, f"rgb_{i:05d}.png")
        depth_file = os.path.join(depth_path, f"depth_{i:05d}.npy")
        
        if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_file)
        
        pts_cam = deproject(depth, intrinsics['depth'])
        colors, proj_mask = align_colors(pts_cam, rgb, intrinsics['color'], T_DEPTH_TO_COLOR)
        
        mask = proj_mask & (depth.reshape(-1) > 0.01) & (depth.reshape(-1) < 5.0)
        pts_v, clrs_v = pts_cam[mask], colors[mask]
        
        # Save camera frame
        pcd_cam = o3d.geometry.PointCloud()
        pcd_cam.points = o3d.utility.Vector3dVector(pts_v)
        pcd_cam.colors = o3d.utility.Vector3dVector(clrs_v)
        o3d.io.write_point_cloud(os.path.join(output_root, "camera_frame", f"pc_{i:05d}.ply"), pcd_cam)
        
        # Save global frame
        pts_g = (base_to_ext_cam @ np.column_stack([pts_v, np.ones(len(pts_v))]).T).T[:, :3]
        pcd_global = o3d.geometry.PointCloud()
        pcd_global.points = o3d.utility.Vector3dVector(pts_g)
        pcd_global.colors = o3d.utility.Vector3dVector(clrs_v)
        o3d.io.write_point_cloud(os.path.join(output_root, "global_frame", f"pc_{i:05d}.ply"), pcd_global)

        if (i+1) % 50 == 0: print(f"  Processed {i+1} frames...")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract point clouds from RGB-D data.")
    parser.add_argument("--bag_path", type=str, required=True)
    parser.add_argument("--extracted_data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--extrinsics", type=str, help="Comma-separated 16 floats for BASE_TO_EXT_CAM matrix")
    args = parser.parse_args()

    if args.extrinsics:
        ext = np.fromstring(args.extrinsics, sep=',').reshape(4, 4)
    else:
        # Default fallback if not provided
        ext = np.array(BASE_TO_EXT_CAM, dtype=np.float32)

    process(args.bag_path, args.extracted_data_root, args.output_root, ext)

if __name__ == "__main__":
    main()
