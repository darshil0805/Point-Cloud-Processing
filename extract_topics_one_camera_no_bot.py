#!/usr/bin/env python3
"""
Extract and synchronize ROS2 bag topics for a single camera.
No gripper or joint states are extracted.
"""
import os
from bisect import bisect_left
import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# -------------------------
# CONFIG
# -------------------------
BAG_PATH = "/home/skills/varun/nitin_vis_jan8_no_gripper/jan9_recording1"
OUTPUT_ROOT = "/home/skills/varun/nitin_vis_jan8_no_gripper/jan9_recording1/extracted_data_one_camera"

# Topics to extract
TOPICS = {
    'camera_rgb': '/camera/camera/color/image_raw',
    'camera_depth': '/camera/camera/depth/image_rect_raw',
    'camera_color_info': '/camera/camera/color/camera_info',
    'camera_depth_info': '/camera/camera/depth/camera_info',
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
    if len(arr) == h * w * 3:
        return arr.reshape(h, w, 3)
    else:
        # Some ROS images might be encoded differently, but assuming standard for now
        return None

def convert_depth_msg(msg):
    """Convert ROS depth Image message to numpy array (in meters)."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    return arr.astype(np.float32) / 1000.0

# -------------------------
# MAIN PROCESSING
# -------------------------
def process_bag(bag_path, output_root):
    """Process bag file and extract data from a single camera synchronized to RGB."""
    print(f"\n=== Processing bag (Single Camera, No Bot): {bag_path} ===")
    
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
    
    # Check if we have RGB data to use as master
    if len(data['camera_rgb']['msgs']) == 0:
        print("\n❌ ERROR: No camera RGB messages found.")
        return
    
    # Synchronize to RGB timestamps
    print("\n=== Synchronizing to camera RGB timestamps ===")
    master_ts = data['camera_rgb']['ts']
    N = len(master_ts)
    print(f"Master frames: {N}")
    
    # Compute alignment indices for all other topics
    indices = {}
    for key in data.keys():
        if key != 'camera_rgb' and len(data[key]['msgs']) > 0:
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
    print("\n=== Extracting and saving data ===")
    saved_frames = 0
    
    for i in range(N):
        # Save Camera RGB
        try:
            rgb_msg = data['camera_rgb']['msgs'][i]
            rgb_arr = convert_rgb_msg(rgb_msg)
            if rgb_arr is not None:
                rgb_path = os.path.join(camera_rgb_out, f"rgb_{i:05d}.png")
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
            else:
                print(f"⚠️ Warning: RGB frame {i} conversion failed (None)")
                continue
        except Exception as e:
            print(f"⚠️ Warning: Failed to save Camera RGB frame {i}: {e}")
            continue
            
        # Save Camera Depth (aligned to RGB)
        if indices['camera_depth'] is not None:
            try:
                depth_msg = data['camera_depth']['msgs'][indices['camera_depth'][i]]
                depth_arr = convert_depth_msg(depth_msg)
                depth_path = os.path.join(camera_depth_out, f"depth_{i:05d}.npy")
                np.save(depth_path, depth_arr)
            except Exception as e:
                print(f"⚠️ Warning: Failed to save Camera depth frame {i}: {e}")
        
        saved_frames += 1
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{N} frames...")
    
    print(f"\n✓ Saved {saved_frames} frames")
    print(f"\n✓ Processing complete!")

if __name__ == "__main__":
    process_bag(BAG_PATH, OUTPUT_ROOT)
