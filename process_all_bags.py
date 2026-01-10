#!/usr/bin/env python3
import os
import subprocess
import glob

def process_all_bags(root_dir, output_root):
    # Find all directories in the root directory
    dirs = [d for d in glob.glob(os.path.join(root_dir, "*/")) if os.path.isdir(d)]
    dirs.sort()

    print(f"Found {len(dirs)} directories in {root_dir}")

    for bag_path in dirs:
        bag_name = os.path.basename(bag_path.rstrip('/'))
        
        # Check if it's a ROS bag (looking for metadata.yaml)
        if not os.path.exists(os.path.join(bag_path, "metadata.yaml")):
            print(f"Skipping {bag_name}: No metadata.yaml found.")
            continue

        print(f"\n>>> Processing: {bag_name}")
        
        cmd = [
            "python3", "extract_topics.py",
            "--bag_path", bag_path,
            "--output_root", output_root
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {bag_name}: {e}")
        except Exception as e:
            print(f"Unexpected error with {bag_name}: {e}")

if __name__ == "__main__":
    ROOT_DIR = "/home/skills/varun/latest_jan"
    OUTPUT_ROOT = "/home/skills/varun/latest_jan/extracted_data_all"
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_all_bags(ROOT_DIR, OUTPUT_ROOT)
