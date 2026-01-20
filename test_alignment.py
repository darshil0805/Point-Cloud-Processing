#!/usr/bin/env python3
import os
import numpy as np
import cv2
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Configurations
BAG_PATH = "/home/skills/varun/puru_test/sample_bag"
RGB_FILE = "/home/skills/varun/puru_test/sample_bag/extracted_data_one_camera/sample_bag/camera/rgb/rgb_00000.png"
DEPTH_FILE = "/home/skills/varun/puru_test/sample_bag/extracted_data_one_camera/sample_bag/camera/depth/depth_00000.npy"
OUTPUT_FILE = "/home/skills/varun/puru_test/sample_bag/extracted_data_one_camera/sample_bag/alignment_test/alignment_test_result.ply"

# Extrinsics (Depth to Color)
T_COLOR_OPTICAL_DEPTH_OPTICAL = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)
T_DEPTH_TO_COLOR = np.linalg.inv(T_COLOR_OPTICAL_DEPTH_OPTICAL)

CAMERA_INFO_TOPICS = {
    'ext_color_info': '/camera/camera/color/camera_info',
    'ext_depth_info': '/camera/camera/depth/camera_info',
}

def extract_camera_intrinsics(bag_path):
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
                    intrinsics[key] = {'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2], 'K': K}
        if len(intrinsics) == len(CAMERA_INFO_TOPICS): break
    return intrinsics

def deproject(depth, params):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (u - params['cx']) * z / params['fx']
    y = (v - params['cy']) * z / params['fy']
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def test():
    print("Extracting intrinsics from bag...")
    intrinsics = extract_camera_intrinsics(BAG_PATH)
    if 'ext_color_info' not in intrinsics or 'ext_depth_info' not in intrinsics:
        print("Error: Could not extract intrinsics from bag.")
        return

    print("Loading images...")
    rgb = cv2.imread(RGB_FILE)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = np.load(DEPTH_FILE)

    print("Generating aligned point cloud...")
    # 1. Deproject using DEPTH intrinsics
    pts_depth_frame = deproject(depth, intrinsics['ext_depth_info'])
    
    # 2. Project to COLOR image
    points_homo = np.column_stack([pts_depth_frame, np.ones(len(pts_depth_frame))])
    pts_color_frame = (T_DEPTH_TO_COLOR @ points_homo.T).T[:, :3]
    
    z = pts_color_frame[:, 2]
    u = (pts_color_frame[:, 0] * intrinsics['ext_color_info']['fx'] / np.maximum(z, 0.001)) + intrinsics['ext_color_info']['cx']
    v = (pts_color_frame[:, 1] * intrinsics['ext_color_info']['fy'] / np.maximum(z, 0.001)) + intrinsics['ext_color_info']['cy']
    
    h, w, _ = rgb.shape
    u_idx = np.round(u).astype(int)
    v_idx = np.round(v).astype(int)
    
    in_image = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
    mask = (z > 0.01) & in_image & (depth.reshape(-1) > 0.01) & (depth.reshape(-1) < 5.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_depth_frame[mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb[v_idx[mask], u_idx[mask]] / 255.0)
    
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd)
    print(f"Alignment test complete. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    test()
