#!/usr/bin/env python3
import os
import zarr
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import shutil

# ------------------------
# Paths and parameters
# ------------------------
# Root directories for different data types
EXTRACTED_DATA_ROOT = "/home/skills/varun/latest_jan/extracted_data_all"
PC_ROOT = "/media/skills/RRC HDD A/cross-emb/Processed_Data_Rea_Training/latest_jan/point_clouds_cam2_only_color_intr_no_offset"

OUT_ZARR = "/media/skills/RRC HDD A/cross-emb/Processed_Data_Rea_Training/latest_jan/latest_jan_data.zarr"
CHUNK_SIZE = 100  # number of samples per chunk

# Subdirectory structure within each trajectory folder
PC_SUBDIR = "camera2_global_frame_filtered"
RGB_SUBDIR = os.path.join("camera2", "rgb")

# ------------------------
# Prepare storage with episode tracking
# ------------------------
all_imgs = []
all_pcs = []
all_actions = []
all_states = []
episode_ends = []  # Track cumulative end indices for each trajectory
current_idx = 0

# ------------------------
# Load data
# ------------------------
# We'll use the trajectory names from the PC_ROOT as our master list
trajectories = sorted([d for d in os.listdir(PC_ROOT) if os.path.isdir(os.path.join(PC_ROOT, d))])
print(f"Found {len(trajectories)} trajectories.")

for traj in tqdm(trajectories, desc="Trajectories"):
    # Paths for this trajectory - all from extracted_data_all
    traj_extracted = os.path.join(EXTRACTED_DATA_ROOT, traj)
    traj_pc_path = os.path.join(PC_ROOT, traj, PC_SUBDIR)
    traj_rgb_path = os.path.join(traj_extracted, RGB_SUBDIR)
    
    state_file = os.path.join(traj_extracted, "states.txt")
    action_file = os.path.join(traj_extracted, "actions.txt")

    # Check if all components exist
    if not all([os.path.exists(traj_pc_path), os.path.exists(traj_rgb_path), 
                os.path.exists(state_file), os.path.exists(action_file)]):
        print(f"Skipping {traj}: Missing one or more data components.")
        continue

    # --- Actions ---
    action_arr = np.loadtxt(action_file)
    # --- States ---
    state_arr = np.loadtxt(state_file)
    
    # Check if shape is (N, 7) or (7,) for single step trajs? Usually (N, 7)
    if action_arr.ndim == 1:
        action_arr = action_arr.reshape(1, -1)
    if state_arr.ndim == 1:
        state_arr = state_arr.reshape(1, -1)

    num_frames = state_arr.shape[0]

    # --- Images ---
    # Files are named rgb_00000.png
    img_files = sorted([f for f in os.listdir(traj_rgb_path) if f.endswith(".png")])
    if len(img_files) != num_frames:
        print(f"Warning: {traj} has {num_frames} states but {len(img_files)} images. Skipping.")
        continue

    for f in img_files:
        img_arr = np.array(Image.open(os.path.join(traj_rgb_path, f)))
        all_imgs.append(img_arr)

    # --- Point clouds ---
    # Files are named pc_00000.ply
    pc_files = sorted([f for f in os.listdir(traj_pc_path) if f.endswith(".ply")])
    if len(pc_files) != num_frames:
        print(f"Warning: {traj} has {num_frames} states but {len(pc_files)} point clouds. Skipping.")
        # Remove images added above? Or better skip earlier.
        # Simple fix: pop the images we just added
        for _ in range(len(img_files)): all_imgs.pop()
        continue

    for f in pc_files:
        pcd = o3d.io.read_point_cloud(os.path.join(traj_pc_path, f))
        points = np.asarray(pcd.points)

        if pcd.has_colors():
            colors = np.asarray(pcd.colors) * 255
            pc_arr = np.concatenate([points, colors], axis=1)
        else:
            pc_arr = np.concatenate([points, np.zeros_like(points)], axis=1)

        all_pcs.append(pc_arr)

    # Append states and actions
    all_actions.append(action_arr)
    all_states.append(state_arr)
    
    # Track episode end (cumulative index)
    current_idx += num_frames
    episode_ends.append(current_idx)

# ------------------------
# Convert lists to arrays
# ------------------------
print("\nStacking arrays...")
all_imgs = np.stack(all_imgs, axis=0)
all_pcs = np.stack(all_pcs, axis=0)
all_actions = np.vstack(all_actions)
all_states = np.vstack(all_states)
episode_ends = np.array(episode_ends, dtype=np.int64)

print("\nAll data collected:")
print(f"Total images: {all_imgs.shape}")
print(f"Total point clouds: {all_pcs.shape}")
print(f"Total actions: {all_actions.shape}")
print(f"Total states: {all_states.shape}")
print(f"Total episodes: {len(episode_ends)}")
print(f"Episode ends: {episode_ends}")

# ------------------------
# Create Zarr with proper structure
# ------------------------
if os.path.exists(OUT_ZARR):
    print(f"Overwriting existing Zarr file: {OUT_ZARR}")
    shutil.rmtree(OUT_ZARR)

zroot = zarr.open(OUT_ZARR, mode="w")
compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

# Create groups
data_group = zroot.create_group('data')
meta_group = zroot.create_group('meta')

# Chunk sizes
img_chunk_size = (CHUNK_SIZE, all_imgs.shape[1], all_imgs.shape[2], all_imgs.shape[3])
pc_chunk_size = (CHUNK_SIZE, all_pcs.shape[1], all_pcs.shape[2])
action_chunk_size = (CHUNK_SIZE, all_actions.shape[1])
state_chunk_size = (CHUNK_SIZE, all_states.shape[1])

# Save datasets to data/ group
print("Saving to Zarr...")
data_group.create_dataset('img', data=all_imgs, chunks=img_chunk_size, dtype='uint8', compressor=compressor)
data_group.create_dataset('point_cloud', data=all_pcs, chunks=pc_chunk_size, dtype='float32', compressor=compressor)
data_group.create_dataset('action', data=all_actions, chunks=action_chunk_size, dtype='float32', compressor=compressor)
data_group.create_dataset('state', data=all_states, chunks=state_chunk_size, dtype='float32', compressor=compressor)

# Save episode_ends to meta/ group
meta_group.create_dataset('episode_ends', data=episode_ends, chunks=(len(episode_ends),), dtype='int64', compressor=compressor)

print(f"\nSaved Zarr file: {OUT_ZARR}")
print("Dataset structure:")
print(f"  data/img: {all_imgs.shape}")
print(f"  data/point_cloud: {all_pcs.shape}")
print(f"  data/action: {all_actions.shape}")
print(f"  data/state: {all_states.shape}")
print(f"  meta/episode_ends: {episode_ends.shape}")
