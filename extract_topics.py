#!/usr/bin/env python3
"""
Extract and synchronize ROS2 bag topics using gripper state as master timestamp.
Extracts data from dual cameras (camera1 and camera2) with different extrinsics.
"""
import os
from bisect import bisect_left
import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import argparse

# -------------------------
# CONFIG
# -------------------------
# Defaults (can be overridden by CLI)
BAG_PATH = "/home/skills/varun/latest_jan/jan_25_1"
OUTPUT_ROOT = "/home/skills/varun/latest_jan/jan_25_1/extracted_data"

# Topics to extract (based on actual bag topics)
TOPICS = {
    'gripper_state': '/gripper/state',
    'gripper_command': '/gripper/command',
    'joint_states': '/ufactory/joint_states',
    'camera1_rgb': '/camera1/camera1/color/image_raw',
    'camera1_depth': '/camera1/camera1/depth/image_rect_raw',
    'camera1_color_info': '/camera1/camera1/color/camera_info',
    'camera1_depth_info': '/camera1/camera1/depth/camera_info',
    'camera2_rgb': '/camera2/camera2/color/image_raw',
    'camera2_depth': '/camera2/camera2/depth/image_rect_raw',
    'camera2_color_info': '/camera2/camera2/color/camera_info',
    'camera2_depth_info': '/camera2/camera2/depth/camera_info',
    'joint_command': '/lite6_traj_controller/joint_trajectory',
}

# -------------------------
# HELPERS
# -------------------------
def inspect_bag(bag_path):
    """Inspect bag file and return available topics."""
    print(f"\n=== Inspecting bag: {bag_path} ===")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    
    topics_info = reader.get_all_topics_and_types()
    print(f"\nFound {len(topics_info)} topics:")
    for topic in topics_info:
        print(f"  - {topic.name} ({topic.type})")
    
    return topics_info

def align_to(reference_ts, target_ts):
    """Align target timestamps to reference timestamps."""
    idx = []
    if len(target_ts) == 0:
        return [0] * len(reference_ts)
    for ts in reference_ts:
        pos = bisect_left(target_ts, ts)
        if pos == 0:
            idx.append(0)
        elif pos == len(target_ts):
            idx.append(len(target_ts) - 1)
        else:
            if abs(target_ts[pos] - ts) < abs(target_ts[pos-1] - ts):
                idx.append(pos)
            else:
                idx.append(pos - 1)
    return idx

def convert_rgb_msg(msg):
    """Convert ROS Image message to numpy array."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape(h, w, 3)

def convert_depth_msg(msg):
    """Convert ROS depth Image message to numpy array (in meters)."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    return arr.astype(np.float32) / 1000.0

def extract_gripper_state(msg):
    """Extract gripper state value from message."""
    if hasattr(msg, 'data'):
        return float(msg.data)
    elif hasattr(msg, 'position'):
        if isinstance(msg.position, (list, tuple)):
            return float(msg.position[0]) if len(msg.position) > 0 else 0.0
        return float(msg.position)
    else:
        print("⚠️ Warning: Unknown gripper state format")
        return 0.0

def extract_joint_positions(msg):
    """Extract joint positions from JointState message."""
    if hasattr(msg, 'position'):
        return np.array(msg.position, dtype=np.float32)
    else:
        print("⚠️ Warning: No position field in joint state")
        return np.zeros(6, dtype=np.float32)

def extract_controller_positions(msg):
    """Extract positions from controller state message."""
    # Try feedback.positions first
    if hasattr(msg, "feedback") and hasattr(msg.feedback, "positions"):
        pos = list(msg.feedback.positions)
        if len(pos) >= 6:
            return np.array(pos[:6], dtype=np.float32)

    # Try other common fields
    candidate_fields = [
        ("reference", "positions"),
        ("output", "positions"),
        ("error", "positions"),
        ("actual", "positions"),
        ("desired", "positions"),
    ]
    for obj, field in candidate_fields:
        if hasattr(msg, obj):
            obj_val = getattr(msg, obj)
            if hasattr(obj_val, field):
                arr = getattr(obj_val, field)
                if isinstance(arr, (list, tuple)) and len(arr) >= 6:
                    return np.array(arr[:6], dtype=np.float32)

    print("⚠️ Warning: controller_state has no usable joint position fields")
    return None

def extract_cmd_positions(msg):
    """Extract joint positions from JointTrajectory message."""
    if hasattr(msg, 'points') and len(msg.points) > 0:
        if hasattr(msg.points[0], 'positions'):
            return np.array(msg.points[0].positions, dtype=np.float32)[:6]
    return np.zeros(6, dtype=np.float32)

# -------------------------
# MAIN PROCESSING
# -------------------------
def process_bag(bag_path, output_root):
    """Process bag file and extract synchronized data from dual cameras."""
    print(f"\n=== Processing bag: {bag_path} ===")
    
    # First, inspect the bag
    topics_info = inspect_bag(bag_path)
    
    # Initialize reader
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    
    # Get message types
    def get_type(topic_name):
        if not topic_name:
            return None
        tlist = [t.type for t in topics_info if t.name == topic_name]
        return get_message(tlist[0]) if tlist else None
    
    msg_types = {key: get_type(topic_name) for key, topic_name in TOPICS.items()}
    
    # Data buffers
    data = {}
    for key in TOPICS.keys():
        data[key] = {'msgs': [], 'ts': []}
    
    # Read all messages
    print("\n=== Reading messages ===")
    while reader.has_next():
        topic, msg_data, ts = reader.read_next()
        
        for key, topic_name in TOPICS.items():
            if topic == topic_name and msg_types[key]:
                msg = deserialize_message(msg_data, msg_types[key])
                data[key]['msgs'].append(msg)
                data[key]['ts'].append(ts)
    
    # Convert timestamps to numpy arrays
    for key in data.keys():
        data[key]['ts'] = np.array(data[key]['ts'])
    
    # Print counts
    print("\nMessage counts:")
    for key in data.keys():
        print(f"  {key}: {len(data[key]['msgs'])}")
    
    # Check if we have gripper state data
    if len(data['gripper_state']['msgs']) == 0:
        print("\n❌ ERROR: No gripper state messages found.")
        return
    
    # Synchronize to gripper state timestamps
    print("\n=== Synchronizing to gripper state timestamps ===")
    master_ts = data['gripper_state']['ts']
    N = len(master_ts)
    print(f"Master frames: {N}")
    
    # Compute alignment indices for all topics
    indices = {}
    for key in data.keys():
        if key != 'gripper_state' and len(data[key]['msgs']) > 0:
            indices[key] = align_to(master_ts, data[key]['ts'])
        else:
            indices[key] = None
    
    # Create output directories - organized by trajectory first
    traj_name = os.path.basename(bag_path.rstrip('/'))
    traj_out = os.path.join(output_root, traj_name)
    
    # Camera 1 outputs
    camera1_rgb_out = os.path.join(traj_out, "camera1", "rgb")
    camera1_depth_out = os.path.join(traj_out, "camera1", "depth")
    
    # Camera 2 outputs
    camera2_rgb_out = os.path.join(traj_out, "camera2", "rgb")
    camera2_depth_out = os.path.join(traj_out, "camera2", "depth")
    
    for d in [camera1_rgb_out, camera1_depth_out, camera2_rgb_out, camera2_depth_out]:
        os.makedirs(d, exist_ok=True)
    
    # Process synchronized frames
    print("\n=== Extracting and saving data ===")
    states = []
    gripper_commands = []
    gripper_states_list = []
    cmd_positions_list = []
    saved_frames = 0
    
    for i in range(N):
        frame_saved = True
        
        # Get gripper state (master timestamp)
        gripper_state_msg = data['gripper_state']['msgs'][i]
        gripper_state_val = extract_gripper_state(gripper_state_msg)
        gripper_states_list.append(gripper_state_val)
        
        # Get gripper command
        gripper_cmd_val = 0.0
        if indices['gripper_command'] is not None:
            gripper_cmd_msg = data['gripper_command']['msgs'][indices['gripper_command'][i]]
            if hasattr(gripper_cmd_msg, 'data'):
                gripper_cmd_val = float(gripper_cmd_msg.data)
        gripper_commands.append(gripper_cmd_val)
        
        # Get joint states
        joint_pos = None
        if indices['joint_states'] is not None:
            joint_msg = data['joint_states']['msgs'][indices['joint_states'][i]]
            joint_pos = extract_joint_positions(joint_msg)
        
        # Get joint commands
        cmd_pos = np.zeros(6, dtype=np.float32)
        if indices['joint_command'] is not None:
            cmd_msg = data['joint_command']['msgs'][indices['joint_command'][i]]
            cmd_pos = extract_cmd_positions(cmd_msg)
        cmd_positions_list.append(cmd_pos)
        
        # Save Camera 1 RGB
        if indices['camera1_rgb'] is not None:
            try:
                rgb_msg = data['camera1_rgb']['msgs'][indices['camera1_rgb'][i]]
                rgb_arr = convert_rgb_msg(rgb_msg)
                rgb_path = os.path.join(camera1_rgb_out, f"rgb_{i:05d}.png")
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"⚠️ Warning: Failed to save Camera1 RGB frame {i}: {e}")
                frame_saved = False
        
        # Save Camera 1 Depth
        if indices['camera1_depth'] is not None:
            try:
                depth_msg = data['camera1_depth']['msgs'][indices['camera1_depth'][i]]
                depth_arr = convert_depth_msg(depth_msg)
                depth_path = os.path.join(camera1_depth_out, f"depth_{i:05d}.npy")
                np.save(depth_path, depth_arr)
            except Exception as e:
                print(f"⚠️ Warning: Failed to save Camera1 depth frame {i}: {e}")
        
        # Save Camera 2 RGB
        if indices['camera2_rgb'] is not None:
            try:
                rgb_msg = data['camera2_rgb']['msgs'][indices['camera2_rgb'][i]]
                rgb_arr = convert_rgb_msg(rgb_msg)
                rgb_path = os.path.join(camera2_rgb_out, f"rgb_{i:05d}.png")
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"⚠️ Warning: Failed to save Camera2 RGB frame {i}: {e}")
                frame_saved = False
        
        # Save Camera 2 Depth
        if indices['camera2_depth'] is not None:
            try:
                depth_msg = data['camera2_depth']['msgs'][indices['camera2_depth'][i]]
                depth_arr = convert_depth_msg(depth_msg)
                depth_path = os.path.join(camera2_depth_out, f"depth_{i:05d}.npy")
                np.save(depth_path, depth_arr)
            except Exception as e:
                print(f"⚠️ Warning: Failed to save Camera2 depth frame {i}: {e}")
        
        # Build state vector (6 joints + gripper state)
        if joint_pos is not None and frame_saved:
            state_vec = np.hstack([joint_pos[:6], np.array([gripper_state_val], dtype=np.float32)])
            states.append(state_vec)
            saved_frames += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{N} frames...")
    
    print(f"\n✓ Saved {saved_frames} frames")
    
    # Save states and actions
    if len(states) > 0:
        states = np.vstack(states).astype(np.float32)
        gripper_commands_arr = np.array(gripper_commands[:len(states)], dtype=np.float32)
        gripper_states_arr = np.array(gripper_states_list[:len(states)], dtype=np.float32)
        
        actual_joint_states = states[:, :6]
        cmd_joint_states = np.vstack(cmd_positions_list[:len(states)]).astype(np.float32)
        
        # 2. Commanded actions (stored directly - per user request)
        commanded_actions = np.zeros_like(states, dtype=np.float32)
        commanded_actions[:, :6] = cmd_joint_states
        commanded_actions[:, 6] = gripper_commands_arr
        
        # 1. Actual deltas (still used for pruning frames where robot was stuck)
        actual_actions = np.zeros_like(states, dtype=np.float32)
        actual_actions[1:, :6] = actual_joint_states[1:] - actual_joint_states[:-1]
        actual_actions[1:, 6] = gripper_states_arr[1:] - gripper_states_arr[:-1]
        actual_actions[0, :] = 0.0
        
        # Remove frames where all ACTUAL actions are zero (robot was stationary/trajectory had issues)
        nonzero_idx = np.any(actual_actions != 0, axis=1)
        
        # Count how many frames will be removed
        num_zero_frames = np.sum(~nonzero_idx)
        if num_zero_frames > 0:
            print(f"\n  Filtering out {num_zero_frames} frames with all-zero actions...")
            
            # Get indices of frames to remove
            removed_frames = np.where(~nonzero_idx)[0]
            
            # Remove corresponding image files
            for idx in removed_frames:
                for path_template in [
                    (camera1_rgb_out, f"rgb_{idx:05d}.png"),
                    (camera1_depth_out, f"depth_{idx:05d}.npy"),
                    (camera2_rgb_out, f"rgb_{idx:05d}.png"),
                    (camera2_depth_out, f"depth_{idx:05d}.npy"),
                ]:
                    try:
                        os.remove(os.path.join(path_template[0], path_template[1]))
                    except FileNotFoundError:
                        pass
            
            # Filter states and actions
            states_filtered = states[nonzero_idx]
            actions_filtered = commanded_actions[nonzero_idx]
            
            # Renumber remaining files sequentially
            print(f"  Renumbering remaining {len(states_filtered)} frames...")
            remaining_indices = np.where(nonzero_idx)[0]
            for new_idx, old_idx in enumerate(remaining_indices):
                for old_name, new_name in [
                    (os.path.join(camera1_rgb_out, f"rgb_{old_idx:05d}.png"),
                     os.path.join(camera1_rgb_out, f"rgb_{new_idx:05d}.png")),
                    (os.path.join(camera1_depth_out, f"depth_{old_idx:05d}.npy"),
                     os.path.join(camera1_depth_out, f"depth_{new_idx:05d}.npy")),
                    (os.path.join(camera2_rgb_out, f"rgb_{old_idx:05d}.png"),
                     os.path.join(camera2_rgb_out, f"rgb_{new_idx:05d}.png")),
                    (os.path.join(camera2_depth_out, f"depth_{old_idx:05d}.npy"),
                     os.path.join(camera2_depth_out, f"depth_{new_idx:05d}.npy")),
                ]:
                    if os.path.exists(old_name) and old_idx != new_idx:
                        os.rename(old_name, new_name + ".tmp")
            
            # Second pass: rename .tmp files to final names
            for new_idx in range(len(states_filtered)):
                for base_path, prefix, ext in [
                    (camera1_rgb_out, "rgb_", ".png"),
                    (camera1_depth_out, "depth_", ".npy"),
                    (camera2_rgb_out, "rgb_", ".png"),
                    (camera2_depth_out, "depth_", ".npy"),
                ]:
                    tmp_name = os.path.join(base_path, f"{prefix}{new_idx:05d}{ext}.tmp")
                    final_name = os.path.join(base_path, f"{prefix}{new_idx:05d}{ext}")
                    if os.path.exists(tmp_name):
                        os.rename(tmp_name, final_name)
        else:
            states_filtered = states
            actions_filtered = commanded_actions
        
        # Save filtered states and actions
        states_file = os.path.join(traj_out, "states.txt")
        actions_file = os.path.join(traj_out, "actions.txt")
        np.savetxt(states_file, states_filtered, fmt="%.6f")
        np.savetxt(actions_file, actions_filtered, fmt="%.6f")
        
        print(f"\n✓ States saved: {states_file} (shape: {states_filtered.shape})")
        print(f"✓ Actions saved: {actions_file} (shape: {actions_filtered.shape})")
    
    print(f"\n✓ Processing complete!")
    print(f"  Output directory: {output_root}")
    print(f"  Camera 1 RGB: {camera1_rgb_out}")
    print(f"  Camera 1 Depth: {camera1_depth_out}")
    print(f"  Camera 2 RGB: {camera2_rgb_out}")
    print(f"  Camera 2 Depth: {camera2_depth_out}")

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract topics from a ROS2 bag.")
    parser.add_argument("--bag_path", type=str, default=BAG_PATH, help="Path to the ROS2 bag directory.")
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT, help="Output root directory.")
    args = parser.parse_args()
    
    process_bag(args.bag_path, args.output_root)

