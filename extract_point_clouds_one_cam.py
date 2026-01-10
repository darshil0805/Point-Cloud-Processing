#!/usr/bin/env python3
"""
Extract point clouds from RGB-D data from a single external camera.
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

BAG_PATH = "/home/skills/varun/nitin_vis_jan8_no_gripper/jan9_recording1"
EXTRACTED_DATA_ROOT = "/home/skills/varun/nitin_vis_jan8_no_gripper/jan9_recording1/extracted_data_one_camera"
OUTPUT_ROOT = "/home/skills/varun/nitin_vis_jan8_no_gripper/jan9_recording1/extracted_data_one_camera/point_clouds_depth"

# --- COLOR TO DEPTH OFFSET (RealSense Baseline) ---
# Use this if your extrinsics are for 'color_optical_frame' but points are in 'depth_optical_frame'
# and they ARE NOT already aligned in the driver.
USE_COLOR_TO_DEPTH_OFFSET = True 

T_COLOR_OPTICAL_DEPTH_OPTICAL = np.array([
    [ 1.0,      0.0,      0.0,     -0.059 ],
    [ 0.0,      1.0,      0.0,    0.0    ],
    [ 0.0,      0.0,      1.0,    0.0    ],
    [ 0.0,      0.0,      0.0,      1.0    ]
], dtype=np.float32)

# --- IMPORTANT: Intrinsics Selection ---
USE_COLOR_INTRINSICS = False  # True = RGB intrinsics (recommended for RealSense)

# Camera extrinsics (Base -> Camera)
BASE_TO_EXT_CAM = np.array([
    [ 0.9985, -0.0394,  0.0382,  0.2724],
    [-0.0373,  0.0231,  0.9990, -0.7900],
    [-0.0402, -0.9990,  0.0216,  0.3268],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float32)

# Topics for camera info
CAMERA_INFO_TOPICS = {
    'ext_color_info': '/camera/camera/color/camera_info',
    'ext_depth_info': '/camera/camera/depth/camera_info',
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
    
    # Read first message from each camera_info topic
    while reader.has_next():
        topic, msg_data, ts = reader.read_next()
        
        for key, topic_name in CAMERA_INFO_TOPICS.items():
            if topic == topic_name and key not in intrinsics:
                msg_type = get_type(topic_name)
                if msg_type:
                    msg = deserialize_message(msg_data, msg_type)
                    # Extract intrinsics matrix K
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
        
        # Break if we have all intrinsics
        if len(intrinsics) == len(CAMERA_INFO_TOPICS):
            break
    
    print("\n=== Camera Intrinsics ===")
    for key, params in intrinsics.items():
        print(f"\n{key}:")
        print(f"  Resolution: {params['width']}x{params['height']}")
        print(f"  fx={params['fx']:.2f}, fy={params['fy']:.2f}")
        print(f"  cx={params['cx']:.2f}, cy={params['cy']:.2f}")
    
    return intrinsics

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """Convert depth image to 3D point cloud in camera coordinates."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Standard deprojection
    Z = depth.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

def transform_pointcloud(points, transform_matrix):
    """Transform point cloud using 4x4 homogeneous transformation matrix."""
    # Add homogeneous coordinate
    points_homo = np.column_stack([points, np.ones(len(points))])
    # Apply transformation
    points_transformed = (transform_matrix @ points_homo.T).T
    # Return 3D points
    return points_transformed[:, :3]

def create_point_cloud(points, colors):
    """Create Open3D point cloud from points and colors."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return pcd

def filter_valid_points(points, colors, min_depth=0.01, max_depth=5.0):
    """Filter out invalid points (NaN, inf, out of range)."""
    # Mask for valid points
    mask = (
        np.isfinite(points).all(axis=1) & 
        (points[:, 2] > min_depth) & 
        (points[:, 2] < max_depth)
    )
    return points[mask], colors[mask]

# -------------------------
# MAIN PROCESSING
# -------------------------
def process_trajectory(traj_name, bag_path, extracted_data_root, output_root):
    """Process a single trajectory and generate point clouds."""
    print(f"\n{'='*60}")
    print(f"Processing trajectory: {traj_name} (External Camera Only)")
    print(f"{'='*60}")
    
    # Extract camera intrinsics from bag
    intrinsics = extract_camera_intrinsics(bag_path)
    
    # Select intrinsics based on configuration
    if USE_COLOR_INTRINSICS:
        print("\n📌 Using COLOR camera intrinsics (for aligned depth)")
        if 'ext_color_info' not in intrinsics:
            print("❌ ERROR: Missing external color camera intrinsics")
            return
        cam_intrinsics = intrinsics['ext_color_info']
    else:
        print("\n📌 Using DEPTH camera intrinsics (for raw/non-aligned depth)")
        if 'ext_depth_info' not in intrinsics:
            print("❌ ERROR: Missing external depth camera intrinsics")
            return
        cam_intrinsics = intrinsics['ext_depth_info']
    
    # Paths to extracted data
    traj_path = os.path.join(extracted_data_root, traj_name)
    camera_rgb_path = os.path.join(traj_path, "camera", "rgb")
    camera_depth_path = os.path.join(traj_path, "camera", "depth")
    
    # Check if paths exist
    for path in [camera_rgb_path, camera_depth_path]:
        if not os.path.exists(path):
            print(f"❌ ERROR: Path not found: {path}")
            return
    
    # Count frames from RGB directory
    rgb_files = sorted([f for f in os.listdir(camera_rgb_path) if f.endswith('.png')])
    N = len(rgb_files)
    print(f"\nFound {N} RGB frames to process")
    
    # Create output directories
    output_dirs = {
        'camera_frame': os.path.join(output_root, traj_name, "camera_camera_frame"),
        'global_frame': os.path.join(output_root, traj_name, "camera_global_frame"),
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print("\n=== Processing frames ===")
    for i in range(N):
        # Camera extrinsics (static)
        T_base_cam = BASE_TO_EXT_CAM
        
        if USE_COLOR_TO_DEPTH_OFFSET:
            T_base_cam = T_base_cam @ T_COLOR_OPTICAL_DEPTH_OPTICAL
        
        # Load RGB and depth
        rgb_file = os.path.join(camera_rgb_path, f"rgb_{i:05d}.png")
        depth_file = os.path.join(camera_depth_path, f"depth_{i:05d}.npy")
        
        if os.path.exists(rgb_file) and os.path.exists(depth_file):
            rgb = cv2.imread(rgb_file)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = np.load(depth_file)
            
            # Generate point cloud in camera coordinates
            pts_cam = depth_to_pointcloud(
                depth, 
                cam_intrinsics['fx'], 
                cam_intrinsics['fy'],
                cam_intrinsics['cx'], 
                cam_intrinsics['cy']
            )
            colors = (rgb.reshape(-1, 3) / 255.0).astype(np.float32)
            
            # Filter valid points
            pts_cam_valid, colors_valid = filter_valid_points(pts_cam, colors)
            
            # Save camera frame point cloud
            pcd_cam = create_point_cloud(pts_cam_valid, colors_valid)
            o3d.io.write_point_cloud(
                os.path.join(output_dirs['camera_frame'], f"pc_{i:05d}.ply"),
                pcd_cam,
                write_ascii=False
            )
            
            # Transform to global frame
            pts_global = transform_pointcloud(pts_cam_valid, T_base_cam)
            pcd_global = create_point_cloud(pts_global, colors_valid)
            o3d.io.write_point_cloud(
                os.path.join(output_dirs['global_frame'], f"pc_{i:05d}.ply"),
                pcd_global,
                write_ascii=False
            )
        else:
            print(f"⚠️ Warning: Missing camera data for frame index {i}")
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{N} frames...")
    
    print(f"\n✓ Processing complete!")
    print(f"\n=== Output Summary ===")
    print(f"Camera (camera frame): {output_dirs['camera_frame']}")
    print(f"Camera (global frame): {output_dirs['global_frame']}")
    print(f"\nTotal frames processed: {N}")

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    # Get trajectory name from bag path
    traj_name = os.path.basename(BAG_PATH.rstrip('/'))
    
    process_trajectory(
        traj_name=traj_name,
        bag_path=BAG_PATH,
        extracted_data_root=EXTRACTED_DATA_ROOT,
        output_root=OUTPUT_ROOT
    )
