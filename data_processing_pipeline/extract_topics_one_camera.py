#!/usr/bin/env python3
"""
Extract and synchronize ROS2 bag topics using gripper state as master timestamp.
Extracts data from a single camera with topics under the 'camera' namespace.
"""
import os
from bisect import bisect_left
import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

TOPICS = {
    'gripper_state': '/gripper/state',
    'gripper_command': '/gripper/command',
    'joint_states': '/ufactory/joint_states',
    'camera_rgb': '/camera/camera/color/image_raw',
    'camera_depth': '/camera/camera/depth/image_rect_raw',
    'camera_color_info': '/camera/camera/color/camera_info',
    'camera_depth_info': '/camera/camera/depth/camera_info',
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
        print("Warning: Unknown gripper state format")
        return 0.0

def extract_joint_positions(msg):
    """Extract joint positions from JointState message."""
    if hasattr(msg, 'position'):
        return np.array(msg.position, dtype=np.float32)
    else:
        print("Warning: No position field in joint state")
        return np.zeros(6, dtype=np.float32)

def extract_gripper_command(msg):
    """Extract gripper command value from message."""
    if hasattr(msg, 'data'):
        return float(msg.data)
    else:
        print("Warning: Unknown gripper command format")
        return 0.0

def extract_cmd_positions(msg):
    """Extract joint positions from JointTrajectory message."""
    if hasattr(msg, 'points') and len(msg.points) > 0:
        if hasattr(msg.points[0], 'positions'):
            return np.array(msg.points[0].positions, dtype=np.float32)[:6]
    return np.zeros(6, dtype=np.float32)

def process_bag(bag_path, output_root):
    """Process bag file and extract synchronized data from a single camera."""
    print(f"\n=== Processing bag (Single Camera): {bag_path} ===")
    
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
    
    # Master Timestep: Gripper Command (usually lowest frequency)
    if len(data['gripper_command']['msgs']) == 0:
        print("\n ERROR: No gripper command messages found.")
        return
    
    # Synchronize to gripper command timestamps
    print("\n=== Synchronizing to gripper command (master) timestamps ===")
    master_ts = data['gripper_command']['ts']
    N = len(master_ts)
    print(f"Master frames (N): {N}")
    
    # Compute alignment indices for all topics relative to master_ts
    indices = {}
    for key in data.keys():
        if key != 'gripper_command' and len(data[key]['msgs']) > 0:
            indices[key] = align_to(master_ts, data[key]['ts'])
        else:
            indices[key] = None
    
    # Create output directories
    traj_name = os.path.basename(bag_path.rstrip('/'))
    traj_out = os.path.join(output_root, traj_name)
    
    camera_rgb_out = os.path.join(traj_out, "camera", "rgb")
    camera_depth_out = os.path.join(traj_out, "camera", "depth")
    
    for d in [camera_rgb_out, camera_depth_out]:
        os.makedirs(d, exist_ok=True)
    
    # Process synchronized frames
    print("\n=== Extracting and saving data (Filtering stationary frames) ===")
    states = []
    actions = []
    saved_frames = 0
    prev_state = None
    
    for i in range(N):
        # 1. Get Gripper Command (direct action value)
        gripper_cmd_msg = data['gripper_command']['msgs'][i]
        gripper_cmd_val = extract_gripper_command(gripper_cmd_msg)
        
        # 2. Get synchronized Joint States (for current state)
        joint_pos = None
        if indices['joint_states'] is not None:
            joint_msg = data['joint_states']['msgs'][indices['joint_states'][i]]
            joint_pos = extract_joint_positions(joint_msg)
        else:
            continue # Should not happen if joint_states topic exists
            
        # 3. Get synchronized Joint Commands (for actions)
        cmd_pos = np.zeros(6, dtype=np.float32)
        if indices['joint_command'] is not None:
            cmd_msg = data['joint_command']['msgs'][indices['joint_command'][i]]
            cmd_pos = extract_cmd_positions(cmd_msg)

        # 4. Get synchronized Gripper State
        gripper_state_val = 0.0
        if indices['gripper_state'] is not None:
            gripper_state_msg = data['gripper_state']['msgs'][indices['gripper_state'][i]]
            gripper_state_val = extract_gripper_state(gripper_state_msg)
        
        # Build current state vector
        current_state = np.hstack([joint_pos[:6], np.array([gripper_state_val], dtype=np.float32)])
        
        # Check for change from previous state
        if prev_state is not None:
            if np.allclose(current_state, prev_state, atol=1e-5):
                continue
        
        # 4. Save Camera Data (Now that we know the state changed)
        frame_saved = True
        
        # Save Camera RGB
        if indices['camera_rgb'] is not None:
            try:
                rgb_msg = data['camera_rgb']['msgs'][indices['camera_rgb'][i]]
                rgb_arr = convert_rgb_msg(rgb_msg)
                rgb_path = os.path.join(camera_rgb_out, f"rgb_{saved_frames:05d}.png")
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Warning: Failed to save Camera RGB frame {i}: {e}")
                frame_saved = False
        
        # Save Camera Depth
        if indices['camera_depth'] is not None:
            try:
                depth_msg = data['camera_depth']['msgs'][indices['camera_depth'][i]]
                depth_arr = convert_depth_msg(depth_msg)
                depth_path = os.path.join(camera_depth_out, f"depth_{saved_frames:05d}.npy")
                np.save(depth_path, depth_arr)
            except Exception as e:
                print(f"Warning: Failed to save Camera depth frame {i}: {e}")

        if frame_saved:
            # Action: [commanded_joints(6), gripper_command(1)]
            action_vec = np.hstack([cmd_pos, np.array([gripper_cmd_val], dtype=np.float32)])
            
            states.append(current_state)
            actions.append(action_vec)
            prev_state = current_state
            saved_frames += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{N} frames...")
    
    print(f"\nSaved {saved_frames} frames")
    
    # Save states and actions
    if len(states) > 0:
        states = np.vstack(states).astype(np.float32)
        actions = np.vstack(actions).astype(np.float32)

        states_file = os.path.join(traj_out, "states.txt")
        actions_file = os.path.join(traj_out, "actions.txt")
        np.savetxt(states_file, states, fmt="%.6f")
        np.savetxt(actions_file, actions, fmt="%.6f")
        
        print(f"\nStates saved: {states_file} ({states.shape})")
        print(f"\nActions saved: {actions_file} ({actions.shape})")
    
    print(f"\nProcessing complete!")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract and synchronize ROS2 bag topics.")
    parser.add_argument("--bag_path", type=str, required=True, help="Path to the ROS2 bag file.")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for extracted data.")
    args = parser.parse_args()

    process_bag(args.bag_path, args.output_root)

if __name__ == "__main__":
    main()
