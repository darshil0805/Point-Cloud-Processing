import os
import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops

# ---------------------------
# FARTHEST POINT SAMPLING
# ---------------------------
def farthest_point_sampling(points, num_points=2500, use_cuda=False):
    K = [num_points]

    pc = torch.from_numpy(points).float()
    if use_cuda:
        pc = pc.cuda()

    sampled, idx = torch3d_ops.sample_farthest_points(
        points=pc.unsqueeze(0), K=K
    )

    sampled = sampled.squeeze(0).cpu().numpy()
    idx = idx.squeeze(0).cpu().numpy()

    return sampled, idx


# ---------------------------
# WORKSPACE CROP + FPS
# ---------------------------
def process_point_cloud(pc_xyz, pc_rgb):
    """
    pc_xyz: (N, 3)
    pc_rgb: (N, 3)
    """

    WORK_SPACE = [
        [-0.05, 0.4],  # X (radius)
        [-0.12, 0.35],  # Y (radius)
        [0, 0.6]   # Z (height)
    ]

    mask = (
        (pc_xyz[:, 0] > WORK_SPACE[0][0]) & (pc_xyz[:, 0] < WORK_SPACE[0][1]) &
        (pc_xyz[:, 1] > WORK_SPACE[1][0]) & (pc_xyz[:, 1] < WORK_SPACE[1][1]) &
        (pc_xyz[:, 2] > WORK_SPACE[2][0]) & (pc_xyz[:, 2] < WORK_SPACE[2][1])
    )

    pc_xyz = pc_xyz[mask]
    pc_rgb = pc_rgb[mask]

    print(f" → After crop: {pc_xyz.shape[0]} points")

    if pc_xyz.shape[0] == 0:
        return None, None

    # FPS
    num_points = min(2500, pc_xyz.shape[0])
    pc_xyz_fps, idx = farthest_point_sampling(pc_xyz, num_points=num_points, use_cuda=torch.cuda.is_available())
    pc_rgb_fps = pc_rgb[idx]

    print(f" → After FPS: {pc_xyz_fps.shape[0]} points")

    return pc_xyz_fps, pc_rgb_fps


# ---------------------------
# SAVE PLY
# ---------------------------
def save_ply(path, xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


# ---------------------------
# MAIN BATCH PROCESSOR
# ---------------------------
def main():
    BASE_ROOT = "/media/skills/RRC HDD A/cross-emb/Processed_Data_Rea_Training/latest_jan/point_clouds_cam2_only_color_intr_no_offset"
    
    if not os.path.exists(BASE_ROOT):
        # Fallback to the all-cameras directory if needed
        BASE_ROOT = "/media/skills/RRC HDD A/cross-emb/Processed_Data_Rea_Training/latest_jan/point_clouds_all_color_intr_no_offset"
    
    if not os.path.exists(BASE_ROOT):
        print(f"❌ Error: Could not find base directory {BASE_ROOT}")
        return

    trajs = sorted([d for d in os.listdir(BASE_ROOT) if os.path.isdir(os.path.join(BASE_ROOT, d))])
    
    print(f"\nFound {len(trajs)} trajectories to process.")

    for traj in trajs:
        in_folder = os.path.join(BASE_ROOT, traj, "camera2_global_frame")
        out_folder = os.path.join(BASE_ROOT, traj, "camera2_global_frame_filtered")
        
        if not os.path.exists(in_folder):
            print(f"⚠️ Warning: Missing input folder {in_folder}. Skipping.")
            continue
            
        os.makedirs(out_folder, exist_ok=True)
        
        ply_files = sorted([f for f in os.listdir(in_folder) if f.endswith(".ply")])
        if not ply_files:
            continue
            
        print(f"\n=== Processing {traj}: {len(ply_files)} clouds ===")
        
        for fn in ply_files:
            in_path = os.path.join(in_folder, fn)
            out_path = os.path.join(out_folder, fn) # Keeping same name for easier downstream use
            
            print(f"\nProcessing {fn} ...")
            
            # Load
            pcd = o3d.io.read_point_cloud(in_path)
            pc_xyz = np.asarray(pcd.points)
            pc_rgb = np.asarray(pcd.colors)
            
            print(f"Loaded XYZ: {pc_xyz.shape} | RGB: {pc_rgb.shape}")
            
            if pc_xyz.size == 0:
                continue

            # Filter & Sample
            xyz_f, rgb_f = process_point_cloud(pc_xyz, pc_rgb)
            
            if xyz_f is not None:
                # Save
                save_ply(out_path, xyz_f, rgb_f)
                print(f"Saved → {out_path}")

    print("\n✅ All processing complete.")

if __name__ == "__main__":
    main()
