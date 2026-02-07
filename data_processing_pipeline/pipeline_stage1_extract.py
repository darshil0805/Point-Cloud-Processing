#!/usr/bin/env python3
import os
import argparse
import subprocess
import numpy as np
import sys

# --- CONFIG ---
BAG_DIR = "/media/skills/RRC HDD A/cross-emb/real-lab-data"
OUTPUT_ROOT = "/media/skills/RRC HDD A/cross-emb/real-lab-data/processed_data"

# Camera extrinsics (Base -> Depth Camera Optical Frame)
BASE_TO_EXT_CAM = [
    [-0.5777,  0.4385, -0.6884,  0.7814],
    [ 0.7992,  0.1327, -0.5862,  0.4271],
    [-0.1657, -0.8889, -0.4271,  0.5267],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
]

def run_command(cmd):
    print(f"\nExecuting: {sys.executable} {' '.join(cmd)}")
    # Clear PYTHONPATH to avoid interference from other environments (like ROS)
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]
    result = subprocess.run([sys.executable] + cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Topic and Point Cloud Extraction")
    parser.add_argument("--bag_dir", type=str, default=BAG_DIR)
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT)
    
    ext_flat = [str(x) for row in BASE_TO_EXT_CAM for x in row]
    ext_str = ",".join(ext_flat)
    parser.add_argument("--extrinsics", type=str, default=ext_str)
    
    args = parser.parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    bag_paths = [os.path.join(args.bag_dir, d) for d in os.listdir(args.bag_dir) if os.path.isdir(os.path.join(args.bag_dir, d))]
    valid_bags = [p for p in bag_paths if any(f.endswith('.db3') or f.endswith('.mcap') for f in os.listdir(p))]
    
    if not valid_bags:
        print(f"No valid bags found in {args.bag_dir}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_topics_script = os.path.join(script_dir, "extract_topics_one_camera.py")
    extract_pcd_script = os.path.join(script_dir, "extract_point_clouds_one_cam.py")

    for bag_path in valid_bags:
        bag_name = os.path.basename(bag_path.rstrip('/'))
        print(f"\n=== PROCESSING BAG: {bag_name} ===")

        # 1. Extract Topics
        extracted_data_root = os.path.join(args.output_root, bag_name, "extracted_data")
        os.makedirs(extracted_data_root, exist_ok=True)
        
        cmd1 = [extract_topics_script, "--bag_path", bag_path, "--output_root", extracted_data_root]
        if not run_command(cmd1): continue

        # 2. Extract Point Clouds
        actual_extracted_path = os.path.join(extracted_data_root, bag_name)
        pcd_output_root = os.path.join(args.output_root, bag_name, "point_clouds")
        os.makedirs(pcd_output_root, exist_ok=True)
        
        cmd2 = [extract_pcd_script, "--bag_path", bag_path, "--extracted_data_root", actual_extracted_path, "--output_root", pcd_output_root, f"--extrinsics={args.extrinsics}"]
        if not run_command(cmd2): continue

        print(f"Stage 1 complete for: {bag_name}")

if __name__ == "__main__":
    main()
