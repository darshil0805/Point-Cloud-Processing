import rerun as rr
import zarr
import numpy as np
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Visualize Point Clouds from a DP3 Zarr file.")
    parser.add_argument("--zarr", type=str, required=True, help="Path to the .zarr file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--fps", type=float, default=10.0, help="Visualization speed (frames per second)")
    parser.add_argument("--save", type=str, default=None, help="Save the recording to a .rrd file")
    parser.add_argument("--serve", action="store_true", help="Serve the Rerun viewer over the web")
    parser.add_argument("--port", type=int, default=8812, help="Port to serve the web viewer on")
    parser.add_argument("--ws-port", type=int, default=9876, help="Port for the WebSocket data stream")
    args = parser.parse_args()

    # Open Zarr
    print(f"Opening Zarr: {args.zarr}")
    root = zarr.open(args.zarr, mode='r')
    
    # Get episode data
    episode_ends = root['meta/episode_ends'][:]
    num_episodes = len(episode_ends)
    
    if args.episode >= num_episodes:
        print(f"Error: Episode index {args.episode} out of range (max {num_episodes-1})")
        return

    start_idx = 0 if args.episode == 0 else episode_ends[args.episode - 1]
    end_idx = episode_ends[args.episode]
    
    print(f"Visualizing Episode {args.episode}: indices {start_idx} to {end_idx} ({end_idx - start_idx} frames)")

    # Data slices
    pcs = root['data/point_cloud']
    imgs = root['data/img']
    # actions = root['data/action'] # Optional

    # Initialize Rerun
    rr.init(f"zarr_vis_ep{args.episode}", spawn=False)
    
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
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for i in range(start_idx, end_idx):
        # Set timeline index
        rr.set_time_sequence("frame", i - start_idx)
        
        # 1. Point Cloud
        pc_data = pcs[i] # (1024, 6)
        xyz = pc_data[:, :3]
        rgb = pc_data[:, 3:6]
        
        # Scale RGB if needed (assuming 0-255 based on previous check)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        rr.log("world/pc", rr.Points3D(positions=xyz, colors=rgb))

        # 2. Image (if available and not empty)
        if 'img' in root['data']:
            img = imgs[i]
            rr.log("camera/rgb", rr.Image(img))

        if (i - start_idx) % 20 == 0:
            print(f"Frame {i - start_idx}/{end_idx - start_idx}")
            
    print("Done visualizing episode.")

    if args.serve:
        print(f"\nServer is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")

if __name__ == "__main__":
    main()
