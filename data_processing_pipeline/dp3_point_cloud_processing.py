import os
import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops

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

def process_point_cloud(pc_xyz, pc_rgb):
    """
    pc_xyz: (N, 3)
    pc_rgb: (N, 3)
    """

    WORK_SPACE = [
        [-0.05, 0.4],  # X (radius)
        [-0.12, 0.35],  # Y (radius)
        [-0.008, 0.6]   # Z (height)
    ]

    mask = (
        (pc_xyz[:, 0] > WORK_SPACE[0][0]) & (pc_xyz[:, 0] < WORK_SPACE[0][1]) &
        (pc_xyz[:, 1] > WORK_SPACE[1][0]) & (pc_xyz[:, 1] < WORK_SPACE[1][1]) &
        (pc_xyz[:, 2] > WORK_SPACE[2][0]) & (pc_xyz[:, 2] < WORK_SPACE[2][1])
    )

    pc_xyz = pc_xyz[mask]
    pc_rgb = pc_rgb[mask]

    print(f" → After crop: {pc_xyz.shape[0]} points")

    # FPS
    pc_xyz_fps, idx = farthest_point_sampling(pc_xyz, num_points=2500, use_cuda=True)
    pc_rgb_fps = pc_rgb[idx]

    print(f" → After FPS: {pc_xyz_fps.shape[0]} points")

    return pc_xyz_fps, pc_rgb_fps

def save_ply(path, xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


def process(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    ply_files = sorted([f for f in os.listdir(input_root) if f.endswith(".ply")])

    print(f"\n=== Processing folder: {input_root} ({len(ply_files)} clouds) ===")

    for fn in ply_files:
        in_path = os.path.join(input_root, fn)
        out_path = os.path.join(output_root, fn.replace(".ply", "_filtered.ply"))

        print(f"\nProcessing {fn} ...")

        # Load
        pcd = o3d.io.read_point_cloud(in_path)
        if pcd.is_empty():
            print(f"Warning: Empty point cloud at {in_path}")
            continue
            
        pc_xyz = np.asarray(pcd.points)
        pc_rgb = np.asarray(pcd.colors)

        print(f"Loaded XYZ: {pc_xyz.shape} | RGB: {pc_rgb.shape}")

        # Filter
        xyz_f, rgb_f = process_point_cloud(pc_xyz, pc_rgb)
        # Save
        save_ply(out_path, xyz_f, rgb_f)

        print(f"Saved → {out_path}")

    print(f"\nDone. All filtered clouds saved in {output_root}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Process point clouds with FPS and workspace cropping.")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    process(args.input_root, args.output_root)

if __name__ == "__main__":
    main()