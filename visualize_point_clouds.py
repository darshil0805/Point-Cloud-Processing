import rerun as rr
import open3d as o3d
import pathlib
import numpy as np
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Visualize a sequence of PLY point clouds in Rerun.")
    parser.add_argument("--dir", type=str, default="/scratch2/cross-emb/processed/final-data/pc/sync_data_bag_5", help="Directory containing .ply files")
    parser.add_argument("--serve", action="store_true", help="Serve the Rerun viewer over the web")
    parser.add_argument("--save", type=str, default=None, help="Save the recording to a .rrd file")
    parser.add_argument("--port", type=int, default=8812, help="Port to serve the web viewer on")
    parser.add_argument("--ws-port", type=int, default=9876, help="Port for the WebSocket data stream")
    args = parser.parse_args()

    # Initialize Rerun
    rr.init("point_cloud_sequence", spawn=False)
    
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
    
    # Log world coordinate system and a marker to verify connectivity
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/origin_marker", rr.Boxes3D(half_sizes=[0.1, 0.1, 0.1], colors=[255, 0, 0]), static=True)
    
    # Path to directory
    data_dir = pathlib.Path(args.dir)
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist.")
        return

    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

    # Get all .ply files and sort them naturally
    ply_files = sorted(list(data_dir.glob("*.ply")), key=natural_sort_key)
    
    if not ply_files:
        print(f"No .ply files found in {data_dir}")
        return

    print(f"Found {len(ply_files)} PLY files. Starting visualization...")

    for i, ply_path in enumerate(ply_files):
        # Set the timeline
        rr.set_time_sequence("frame", i)
        
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(str(ply_path))
        
        # Convert to numpy
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Log to Rerun
        if colors.size > 0:
            # Open3D colors are 0-1, Rerun expects 0-255 if using uint8, or 0-1 if float
            rr.log("world/point_cloud", rr.Points3D(positions=points, colors=colors))
        else:
            rr.log("world/point_cloud", rr.Points3D(positions=points))
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(ply_files)}: {ply_path.name}")

    print("Done logging.")
    
    if args.serve:
        print(f"\nServer is running at http://localhost:{args.port} (or the server's IP).")
        print("Keep this script running to view the data.")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")

if __name__ == "__main__":
    main()
