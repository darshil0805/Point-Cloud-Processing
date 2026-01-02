#!/usr/bin/env python3
"""
Extract point clouds from RGB-D data in multiple coordinate frames.
Generates:
1. Camera coordinate frame point clouds (individual cameras)
2. Global coordinate frame point clouds (individual cameras)
3. Combined point clouds in global coordinate frame
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

# -------------------------
# CONFIG
# -------------------------
BAG_PATH = "/home/skills/varun/dual_data/joint_trajectory_1"
EXTRACTED_DATA_ROOT = "/home/skills/varun/dual_data/extracted_data"
OUTPUT_ROOT = "/home/skills/varun/dual_data/point_clouds_opt_frame"

# URDF path for FK computation
URDF_PATH = "/home/skills/varun/Point-Cloud-Processing/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6  # flange link (link_eef)

# --- CALIBRATION DEBUG TOGGLES ---
# Since Cam2 is accurate, we keep base rotation at 0.
BASE_ROTATION_Z = 0.0 

# Frame convention correction:
# Most hand-eye calibration (like easy_handeye) outputs the transform to 'camera_link'.
# However, depth points are in 'camera_color_optical_frame'.
# This toggle applies the standard 90-degree ROS rotation.
USE_ROS_OPTICAL_CONVENTION = True 

# D455 Specific: If calibration was done on Color Lens, 
# but deprojection uses Depth (raw IR) lens, we need the baseline offset.
# D455 depth-to-color baseline is approx 59mm along X-optical.
# If deprojecting from depth, we need to shift points back to the color center 
# so the hand-eye calibration (which was likely done wrt color) matches.
D455_BASELINE_OFFSET = 0.059 # meters. Set to 0.0 if you use aligned depth or calibrated to depth.

# If the wrist cloud is "inverted" or on the wrong side, try this:
INVERT_WRIST_EXTRINSICS = False 

# If left/right is swapped in the wrist view, toggle this:
FLIP_WRIST_CLOUD_Y = False

# --- IMPORTANT: Intrinsics Selection ---
USE_COLOR_INTRINSICS = False # User preferred False for better results

# Camera1 extrinsics (EEF -> Camera_Link/Color Lens)
EEF_TO_WRIST_CAM = np.array([
    [-0.00179952,  0.02153973,  0.99976637,  0.072252  ],
    [ 0.01309415, -0.99968177,  0.02156148, -0.05377971],
    [ 0.99991265,  0.01312989,  0.0015169,   0.03106089],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float32)


if USE_ROS_OPTICAL_CONVENTION:
    # Rotate points from Optical frame (Z-forward) to Link frame (X-forward)
    T_opt_link = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # If using depth intrinsics but calibrated to color, add the baseline shift
    if not USE_COLOR_INTRINSICS and D455_BASELINE_OFFSET != 0:
        # Shift Optical X by baseline to align Depth origin with Color origin
        T_depth_to_color = np.eye(4, dtype=np.float32)
        T_depth_to_color[0, 3] = D455_BASELINE_OFFSET
        EEF_TO_WRIST_CAM = EEF_TO_WRIST_CAM @ T_opt_link @ T_depth_to_color
    else:
        EEF_TO_WRIST_CAM = EEF_TO_WRIST_CAM @ T_opt_link

if INVERT_WRIST_EXTRINSICS:
    EEF_TO_WRIST_CAM = np.linalg.inv(EEF_TO_WRIST_CAM)

# Camera2 extrinsics (Base -> Camera)
BASE_TO_EYE_CAM = np.array([
    [ 0.983,   0.0274, -0.1813,  0.7653],
    [ 0.1803,  0.0373,  0.9829, -0.6609],
    [ 0.0337, -0.9989,  0.0317,  0.369 ],
    [ 0.0,     0.0,     0.0,     1.0    ]
], dtype=np.float32)

# Topics for camera info
CAMERA_INFO_TOPICS = {
    'camera1_color_info': '/camera1/camera1/color/camera_info',
    'camera1_depth_info': '/camera1/camera1/depth/camera_info',
    'camera2_color_info': '/camera2/camera2/color/camera_info',
    'camera2_depth_info': '/camera2/camera2/depth/camera_info',
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

def pos_quat_to_matrix(pos, quat):
    """Convert position [x,y,z] and quaternion [x,y,z,w] to 4x4 homogeneous matrix."""
    R = p.getMatrixFromQuaternion(quat)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array(R, dtype=np.float32).reshape(3, 3)
    T[:3, 3] = pos
    return T

def get_base_transform():
    """Returns the transformation matrix for the robot base."""
    T = np.eye(4, dtype=np.float32)
    c, s = np.cos(BASE_ROTATION_Z), np.sin(BASE_ROTATION_Z)
    T[0:2, 0:2] = [[c, -s], [s, c]]
    return T

def compute_eef_pose(robot_id, joint_angles):
    """Compute end effector pose from joint angles using FK."""
    for j in range(len(joint_angles)):
        p.resetJointState(robot_id, j, joint_angles[j])
    
    # Trigger FK calculation
    p.getLinkState(robot_id, EEF_INDEX) 
    
    state = p.getLinkState(robot_id, EEF_INDEX)
    pos = np.array(state[0], dtype=np.float32)
    quat = np.array(state[1], dtype=np.float32)
    
    T_rel = pos_quat_to_matrix(pos, quat)
    # Apply base rotation
    return get_base_transform() @ T_rel

def depth_to_pointcloud(depth, fx, fy, cx, cy, flip_y=False):
    """Convert depth image to 3D point cloud in camera coordinates."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Standard deprojection
    Z = depth.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    if flip_y:
        X = -X
        
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
    print(f"Processing trajectory: {traj_name}")
    print(f"{'='*60}")
    
    # Extract camera intrinsics from bag
    intrinsics = extract_camera_intrinsics(bag_path)
    
    # Select intrinsics based on configuration
    if USE_COLOR_INTRINSICS:
        print("\n📌 Using COLOR camera intrinsics (for aligned depth)")
        print("   ⚠️  Make sure your depth is 'aligned to color'!")
        
        if 'camera1_color_info' not in intrinsics or 'camera2_color_info' not in intrinsics:
            print("❌ ERROR: Missing color camera intrinsics")
            return
        
        cam1_intrinsics = intrinsics['camera1_color_info']
        cam2_intrinsics = intrinsics['camera2_color_info']
    else:
        print("\n📌 Using DEPTH camera intrinsics (for raw/non-aligned depth)")
        print("   ⚠️  Note: Extrinsic calibration should be wrt depth sensor!")
        
        if 'camera1_depth_info' not in intrinsics or 'camera2_depth_info' not in intrinsics:
            print("❌ ERROR: Missing depth camera intrinsics")
            return
        
        cam1_intrinsics = intrinsics['camera1_depth_info']
        cam2_intrinsics = intrinsics['camera2_depth_info']
    
    # Paths to extracted data
    traj_path = os.path.join(extracted_data_root, traj_name)
    
    camera1_rgb_path = os.path.join(traj_path, "camera1", "rgb")
    camera1_depth_path = os.path.join(traj_path, "camera1", "depth")
    camera2_rgb_path = os.path.join(traj_path, "camera2", "rgb")
    camera2_depth_path = os.path.join(traj_path, "camera2", "depth")
    states_file = os.path.join(traj_path, "states.txt")
    
    # Check if paths exist
    for path in [camera1_rgb_path, camera1_depth_path, camera2_rgb_path, camera2_depth_path, states_file]:
        if not os.path.exists(path):
            print(f"❌ ERROR: Path not found: {path}")
            return
    
    # Load states (joint angles + gripper state)
    states = np.loadtxt(states_file, dtype=np.float32)
    N = len(states)
    print(f"\nFound {N} frames to process")
    
    # Create output directories
    output_dirs = {
        # Camera coordinate frame
        'cam1_camera_frame': os.path.join(output_root, traj_name, "camera1_camera_frame"),
        'cam2_camera_frame': os.path.join(output_root, traj_name, "camera2_camera_frame"),
        # Global coordinate frame (individual)
        'cam1_global_frame': os.path.join(output_root, traj_name, "camera1_global_frame"),
        'cam2_global_frame': os.path.join(output_root, traj_name, "camera2_global_frame"),
        # Combined global frame
        'combined_global': os.path.join(output_root, traj_name, "combined_global_frame"),
        # EEF poses
        'eef_poses': os.path.join(output_root, traj_name, "eef_poses"),
        # Camera poses
        'camera_poses': os.path.join(output_root, traj_name, "camera_poses"),
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize PyBullet for FK
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF_PATH, useFixedBase=True)
    
    # Storage for poses
    eef_poses_list = []
    cam1_poses_list = []
    cam2_poses_list = []
    
    print("\n=== Processing frames ===")
    for i in range(N):
        # Get joint angles (first 6 values from state)
        joint_angles = states[i, :6]
        # Compute EEF pose using FK
        T_base_eef = compute_eef_pose(robot, joint_angles)
        eef_poses_list.append(T_base_eef.flatten())

        # Compute camera poses in global frame
        T_base_cam1 = T_base_eef @ EEF_TO_WRIST_CAM  # Camera1 on wrist
        #T_base_cam1 = EEF_TO_WRIST_CAM
        T_base_cam2 = BASE_TO_EYE_CAM  # Camera2 fixed in world
        
        cam1_poses_list.append(T_base_cam1.flatten())
        cam2_poses_list.append(T_base_cam2.flatten())
        
        # Load RGB and depth for camera 1
        rgb1_file = os.path.join(camera1_rgb_path, f"rgb_{i:05d}.png")
        depth1_file = os.path.join(camera1_depth_path, f"depth_{i:05d}.npy")
        
        if os.path.exists(rgb1_file) and os.path.exists(depth1_file):
            rgb1 = cv2.imread(rgb1_file)
            rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
            depth1 = np.load(depth1_file)
            
            # Generate point cloud in camera coordinates
            pts1_cam = depth_to_pointcloud(
                depth1, 
                cam1_intrinsics['fx'], 
                cam1_intrinsics['fy'],
                cam1_intrinsics['cx'], 
                cam1_intrinsics['cy'],
                flip_y=FLIP_WRIST_CLOUD_Y
            )
            colors1 = (rgb1.reshape(-1, 3) / 255.0).astype(np.float32)
            
            # Filter valid points
            pts1_cam_valid, colors1_valid = filter_valid_points(pts1_cam, colors1)
            
            # Save camera frame point cloud
            pcd1_cam = create_point_cloud(pts1_cam_valid, colors1_valid)
            o3d.io.write_point_cloud(
                os.path.join(output_dirs['cam1_camera_frame'], f"pc_{i:05d}.ply"),
                pcd1_cam,
                write_ascii=False
            )
            
            # Transform to global frame
            pts1_global = transform_pointcloud(pts1_cam_valid, T_base_cam1)
            pcd1_global = create_point_cloud(pts1_global, colors1_valid)
            o3d.io.write_point_cloud(
                os.path.join(output_dirs['cam1_global_frame'], f"pc_{i:05d}.ply"),
                pcd1_global,
                write_ascii=False
            )
        else:
            print(f"⚠️ Warning: Missing camera1 data for frame {i}")
            pts1_global = None
            colors1_valid = None
        
        # Load RGB and depth for camera 2
        rgb2_file = os.path.join(camera2_rgb_path, f"rgb_{i:05d}.png")
        depth2_file = os.path.join(camera2_depth_path, f"depth_{i:05d}.npy")
        
        if os.path.exists(rgb2_file) and os.path.exists(depth2_file):
            rgb2 = cv2.imread(rgb2_file)
            rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)
            depth2 = np.load(depth2_file)
            
            # Generate point cloud in camera coordinates
            pts2_cam = depth_to_pointcloud(
                depth2,
                cam2_intrinsics['fx'],
                cam2_intrinsics['fy'],
                cam2_intrinsics['cx'],
                cam2_intrinsics['cy']
            )
            colors2 = (rgb2.reshape(-1, 3) / 255.0).astype(np.float32)
            
            # Filter valid points
            pts2_cam_valid, colors2_valid = filter_valid_points(pts2_cam, colors2)
            
            # Save camera frame point cloud
            pcd2_cam = create_point_cloud(pts2_cam_valid, colors2_valid)
            o3d.io.write_point_cloud(
                os.path.join(output_dirs['cam2_camera_frame'], f"pc_{i:05d}.ply"),
                pcd2_cam,
                write_ascii=False
            )
            
            # Transform to global frame
            pts2_global = transform_pointcloud(pts2_cam_valid, T_base_cam2)
            pcd2_global = create_point_cloud(pts2_global, colors2_valid)
            o3d.io.write_point_cloud(
                os.path.join(output_dirs['cam2_global_frame'], f"pc_{i:05d}.ply"),
                pcd2_global,
                write_ascii=False
            )
        else:
            print(f"⚠️ Warning: Missing camera2 data for frame {i}")
            pts2_global = None
            colors2_valid = None
        
        # Create combined point cloud in global frame
        if pts1_global is not None and pts2_global is not None:
            pts_combined = np.vstack([pts1_global, pts2_global])
            colors_combined = np.vstack([colors1_valid, colors2_valid])
            pcd_combined = create_point_cloud(pts_combined, colors_combined)
            o3d.io.write_point_cloud(
                os.path.join(output_dirs['combined_global'], f"pc_{i:05d}.ply"),
                pcd_combined,
                write_ascii=False
            )
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{N} frames...")
    
    # Disconnect PyBullet
    p.disconnect()
    
    # Save EEF and camera poses
    eef_poses_array = np.vstack(eef_poses_list)
    cam1_poses_array = np.vstack(cam1_poses_list)
    cam2_poses_array = np.vstack(cam2_poses_list)
    
    np.savetxt(os.path.join(output_dirs['eef_poses'], "eef_poses.txt"), eef_poses_array, fmt="%.6f")
    np.savetxt(os.path.join(output_dirs['camera_poses'], "camera1_poses.txt"), cam1_poses_array, fmt="%.6f")
    np.savetxt(os.path.join(output_dirs['camera_poses'], "camera2_poses.txt"), cam2_poses_array, fmt="%.6f")
    
    print(f"\n✓ Processing complete!")
    print(f"\n=== Output Summary ===")
    print(f"Camera 1 (camera frame): {output_dirs['cam1_camera_frame']}")
    print(f"Camera 1 (global frame): {output_dirs['cam1_global_frame']}")
    print(f"Camera 2 (camera frame): {output_dirs['cam2_camera_frame']}")
    print(f"Camera 2 (global frame): {output_dirs['cam2_global_frame']}")
    print(f"Combined (global frame): {output_dirs['combined_global']}")
    print(f"EEF poses: {output_dirs['eef_poses']}/eef_poses.txt")
    print(f"Camera poses: {output_dirs['camera_poses']}/")
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
