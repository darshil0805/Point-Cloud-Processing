#!/usr/bin/env python3
"""
Process all extracted trajectories to generate point clouds for CAMERA 2 ONLY.
Loops over all directories in /home/skills/varun/latest_jan/extracted_data_all
and provides options for extrinsics, depth transforms, and intrinsics.
"""
import os
import glob
import numpy as np
import cv2
import open3d as o3d
import pybullet as p
import pybullet_data
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# -------------------------
# CONFIGURATION
# -------------------------
URDF_PATH = "/home/skills/varun/Point-Cloud-Processing/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6  # flange link (link_eef)

# Grouped Extrinsics (Eye camera is camera2)
EXTRINSICS = {
    "BASE_TO_EYE_CAM": np.array([
        [ 0.9882, -0.0163,  0.1522,  0.3611],
        [-0.1524, -0.0092,  0.9883, -0.8294],
        [-0.0147, -0.9998, -0.0116,  0.4017],
        [ 0.0,     0.0,     0.0,     1.0   ]
    ], dtype=np.float32)
}

T_COLOR_OPTICAL_DEPTH_OPTICAL = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)

CAMERA_INFO_TOPICS = {
    'camera2_color_info': '/camera2/camera2/color/camera_info',
    'camera2_depth_info': '/camera2/camera2/depth/camera_info',
}

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def extract_camera_intrinsics(bag_path):
    """Extract camera intrinsics from camera_info topics."""
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", "")
        )
    except Exception as e:
        print(f"❌ Error opening bag {bag_path}: {e}")
        return None
    
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
                        'K': K, 'width': msg.width, 'height': msg.height,
                        'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]
                    }
        if len(intrinsics) == len(CAMERA_INFO_TOPICS):
            break
    return intrinsics

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

def transform_pointcloud(points, transform_matrix):
    points_homo = np.column_stack([points, np.ones(len(points))])
    points_transformed = (transform_matrix @ points_homo.T).T
    return points_transformed[:, :3]

def create_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return pcd

def filter_valid_points(points, colors, min_depth=0.01, max_depth=5.0):
    mask = (np.isfinite(points).all(axis=1) & (points[:, 2] > min_depth) & (points[:, 2] < max_depth))
    return points[mask], colors[mask]

# -------------------------
# MAIN PROCESSING
# -------------------------
def process_trajectory(traj_name, bag_path, extracted_data_root, output_root, options):
    """Process a single trajectory and generate point clouds for camera 2."""
    print(f"\nProcessing trajectory (Camera 2): {traj_name}")
    
    intrinsics = extract_camera_intrinsics(bag_path)
    if not intrinsics: return

    if options['use_color_intrinsics']:
        if 'camera2_color_info' not in intrinsics:
            print("❌ Skipping: Missing camera 2 color intrinsics")
            return
        cam2_params = intrinsics['camera2_color_info']
    else:
        if 'camera2_depth_info' not in intrinsics:
            print("❌ Skipping: Missing camera 2 depth intrinsics")
            return
        cam2_params = intrinsics['camera2_depth_info']
    
    traj_path = os.path.join(extracted_data_root, traj_name)
    camera2_rgb_path = os.path.join(traj_path, "camera2", "rgb")
    camera2_depth_path = os.path.join(traj_path, "camera2", "depth")
    states_file = os.path.join(traj_path, "states.txt")
    
    for p_ in [camera2_rgb_path, camera2_depth_path, states_file]:
        if not os.path.exists(p_):
            print(f"❌ Missing data: {p_}")
            return
    
    states = np.loadtxt(states_file, dtype=np.float32)
    N = len(states)
    
    output_dirs = {
        'cam2_camera': os.path.join(output_root, traj_name, "camera2_camera_frame"),
        'cam2_global': os.path.join(output_root, traj_name, "camera2_global_frame"),
        'camera_poses': os.path.join(output_root, traj_name, "camera_poses"),
    }
    for d in output_dirs.values(): os.makedirs(d, exist_ok=True)
    
    cam2_poses = []
    
    for i in range(N):
        T_base_cam2 = options['extrinsics']['BASE_TO_EYE_CAM']
        
        if options['use_depth_offset']:
            T_base_cam2 = T_base_cam2 @ T_COLOR_OPTICAL_DEPTH_OPTICAL
        
        cam2_poses.append(T_base_cam2.flatten())
        
        # Camera 2
        rgb2_path = os.path.join(camera2_rgb_path, f"rgb_{i:05d}.png")
        depth2_path = os.path.join(camera2_depth_path, f"depth_{i:05d}.npy")

        rgb2, depth2 = None, None
        if os.path.exists(rgb2_path) and os.path.exists(depth2_path):
            rgb2 = cv2.imread(rgb2_path)
            try:
                depth2 = np.load(depth2_path)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load depth2 at frame {i}: {e}")

        if rgb2 is not None and depth2 is not None:
            rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)
            pts2_cam = depth_to_pointcloud(depth2, cam2_params['fx'], cam2_params['fy'], cam2_params['cx'], cam2_params['cy'])
            cols2 = (rgb2.reshape(-1, 3) / 255.0).astype(np.float32)
            p2_c, c2 = filter_valid_points(pts2_cam, cols2)
            o3d.io.write_point_cloud(os.path.join(output_dirs['cam2_camera'], f"pc_{i:05d}.ply"), create_point_cloud(p2_c, c2))
            p2_g = transform_pointcloud(p2_c, T_base_cam2)
            o3d.io.write_point_cloud(os.path.join(output_dirs['cam2_global'], f"pc_{i:05d}.ply"), create_point_cloud(p2_g, c2))
        
        if (i+1) % 100 == 0: print(f"  Processed {i+1}/{N} frames...")

    np.savetxt(os.path.join(output_dirs['camera_poses'], "camera2_poses.txt"), np.vstack(cam2_poses), fmt="%.6f")

if __name__ == "__main__":
    EXTRACTED_ROOT = "/home/skills/varun/latest_jan/extracted_data_all"
    BAGS_ROOT = "/home/skills/varun/latest_jan"

    use_color_intrinsics = True
    use_depth_offset = False

    output_suffix = "_cam2_only"
    if use_color_intrinsics: output_suffix += "_color_intr"
    else: output_suffix += "_depth_intr"
    if use_depth_offset: output_suffix += "_with_offset"
    else: output_suffix += "_no_offset"

    # Save to a slightly different path to avoid mixing with dual-cam data
    OUTPUT_ROOT = f"/media/skills/RRC HDD A/cross-emb/Processed_Data_Rea_Training/latest_jan/point_clouds{output_suffix}"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    options = {
        'use_color_intrinsics': use_color_intrinsics,
        'use_depth_offset': use_depth_offset,
        'extrinsics': EXTRINSICS
    }

    # Get all directories in extracted_root
    trajs = sorted([d for d in os.listdir(EXTRACTED_ROOT) if os.path.isdir(os.path.join(EXTRACTED_ROOT, d))])
    print(f"\nFound {len(trajs)} trajectories to process for Camera 2.")

    for traj in trajs:
        bag_path = os.path.join(BAGS_ROOT, traj)
        if not os.path.exists(bag_path):
            print(f"⚠️ Warning: Bag path {bag_path} not found. Skipping {traj}.")
            continue
            
        process_trajectory(traj, bag_path, EXTRACTED_ROOT, OUTPUT_ROOT, options)

    print(f"\n✅ All processing complete. Outputs in: {OUTPUT_ROOT}")
