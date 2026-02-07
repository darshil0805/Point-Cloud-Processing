#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys

OUTPUT_ROOT = "/media/skills/RRC HDD A/cross-emb/real-lab-data/processed_data"

def run_command(cmd):
    print(f"\nExecuting: {sys.executable} {' '.join(cmd)}")
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]
    result = subprocess.run([sys.executable] + cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Point Cloud Filtering and FPS")
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT, help="The same output_root used in Stage 1")
    args = parser.parse_args()

    if not os.path.exists(args.output_root):
        print(f"Directory not found: {args.output_root}")
        return

    bag_folders = [d for d in os.listdir(args.output_root) if os.path.isdir(os.path.join(args.output_root, d))]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_pcd_script = os.path.join(script_dir, "dp3_point_cloud_processing.py")

    for bag_name in bag_folders:
        print(f"\n=== FILTERING PCD: {bag_name} ===")
        
        pcd_input_dir = os.path.join(args.output_root, bag_name, "point_clouds", "global_frame")
        filtered_pcd_output = os.path.join(args.output_root, bag_name, "filtered_point_clouds")
        
        if not os.path.exists(pcd_input_dir):
            print(f"global_frame not found for {bag_name}, skipping...")
            continue
            
        cmd3 = [process_pcd_script, "--input_root", pcd_input_dir, "--output_root", filtered_pcd_output]
        if not run_command(cmd3): continue

        print(f"Stage 2 complete for: {bag_name}")

if __name__ == "__main__":
    main()
