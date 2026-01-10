import rerun as rr
import open3d as o3d
import pathlib
import numpy as np
import argparse
import time
import re
import os

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def load_poses(pose_file):
    if not os.path.exists(pose_file):
        return None
    poses_flat = np.loadtxt(pose_file)
    return poses_flat.reshape(-1, 4, 4)

def main():
    parser = argparse.ArgumentParser(description="Visualize Cam vs Global point clouds in Rerun.")
    parser.add_argument("--dir", type=str, required=True, help="Trajectory directory (e.g., .../jan_25_1/point_clouds_no_offset_color/jan_25_1)")
    parser.add_argument("--cam", type=int, default=2, help="Camera index to focus on (1 or 2)")
    parser.add_argument("--save", type=str, default=None, help="Save the recording to a .rrd file")
    parser.add_argument("--serve", action="store_true", help="Serve the Rerun viewer over the web")
    parser.add_argument("--port", type=int, default=8812, help="Port to serve the web viewer on")
    parser.add_argument("--ws-port", type=int, default=9876, help="Port for the WebSocket data stream")
    args = parser.parse_args()

    base_path = pathlib.Path(args.dir)
    cam_local_dir = base_path / f"camera{args.cam}_camera_frame"
    cam_global_dir = base_path / f"camera{args.cam}_global_frame"
    combined_dir = base_path / "combined_global_frame"
    camera_poses_file = base_path / "camera_poses" / f"camera{args.cam}_poses.txt"

    if not cam_local_dir.exists():
        print(f"Error: Local directory {cam_local_dir} not found.")
        return

    # Collect files
    local_files = sorted(list(cam_local_dir.glob("*.ply")), key=natural_sort_key)
    global_files = sorted(list(cam_global_dir.glob("*.ply")), key=natural_sort_key)
    combined_files = sorted(list(combined_dir.glob("*.ply")), key=natural_sort_key)

    # Load poses
    poses = load_poses(camera_poses_file)
    if poses is not None:
        print(f"Loaded {len(poses)} camera poses.")
    else:
        print(f"Warning: Camera poses file {camera_poses_file} not found.")

    # Initialize Rerun
    rr.init(f"cam_vs_glob_cam{args.cam}", spawn=False)
    
    if args.save:
        print(f"Saving recording to {args.save}")
        rr.save(args.save)
    elif args.serve:
        print(f"Starting Rerun server:")
        print(f"  - Web Viewer: http://localhost:{args.port}")
        print(f"  - WebSocket Data: ws://localhost:{args.ws_port}")
        rr.serve(open_browser=False, web_port=args.port, ws_port=args.ws_port)
    else:
        print("Spawning Rerun viewer...")
        rr.spawn()
    
    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("local", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    num_frames = len(local_files)
    print(f"Visualizing {num_frames} frames...")

    for i in range(num_frames):
        rr.set_time_sequence("frame", i)
        
        # 1. Log Local Point Cloud (always at origin in 'local' space)
        pcd_local = o3d.io.read_point_cloud(str(local_files[i]))
        pts_local = np.asarray(pcd_local.points)
        clr_local = np.asarray(pcd_local.colors)
        rr.log("local/point_cloud", rr.Points3D(positions=pts_local, colors=clr_local))
        
        # 2. Log Global Point Cloud (pre-transformed in 'world' space)
        if i < len(global_files):
            pcd_glob = o3d.io.read_point_cloud(str(global_files[i]))
            pts_glob = np.asarray(pcd_glob.points)
            clr_glob = np.asarray(pcd_glob.colors)
            rr.log("world/global_pretransformed", rr.Points3D(positions=pts_glob, colors=clr_glob))

        # 3. Log Combined Point Cloud (reference)
        if i < len(combined_files):
            pcd_comb = o3d.io.read_point_cloud(str(combined_files[i]))
            pts_comb = np.asarray(pcd_comb.points)
            clr_comb = np.asarray(pcd_comb.colors)
            rr.log("world/combined", rr.Points3D(positions=pts_comb, colors=clr_comb))

        # 4. Log Camera Pose and Local Points transformed by Rerun
        if poses is not None and i < len(poses):
            T = poses[i]
            # Decompose T into translation and rotation
            translation = T[:3, 3]
            rotation_matrix = T[:3, :3]
            
            # Log the transform to Rerun
            rr.log(
                f"world/camera_{args.cam}",
                rr.Transform3D(
                    translation=translation,
                    mat3x3=rotation_matrix,
                    from_parent=False
                )
            )
            
            # Log the same local points under the moving camera frame
            # This is the "TRUE" cam vs glob check.
            rr.log(
                f"world/camera_{args.cam}/points",
                rr.Points3D(positions=pts_local, colors=clr_local)
            )

        if i % 50 == 0:
            print(f"Processed {i}/{num_frames}")

    print("Done.")

    if args.serve:
        print(f"\nServer is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")

if __name__ == "__main__":
    main()
