import rerun as rr
import pathlib
import numpy as np
import argparse
import time
import re
import cv2 # Added for depth and RGB image processing

def main():
    parser = argparse.ArgumentParser(description="Visualize a sequence of RGB and Depth images in Rerun.")
    parser.add_argument("--depth_dir", type=str, default=None, help="Directory containing .npy depth files")
    parser.add_argument("--rgb_dir", type=str, default=None, help="Directory containing .png RGB files")
    parser.add_argument("--serve", action="store_true", help="Serve the Rerun viewer over the web")
    parser.add_argument("--save", type=str, default=None, help="Save the recording to a .rrd file")
    parser.add_argument("--port", type=int, default=8812, help="Port to serve the web viewer on")
    parser.add_argument("--ws-port", type=int, default=9876, help="Port for the WebSocket data stream")
    args = parser.parse_args()

    if not args.depth_dir and not args.rgb_dir:
        print("Error: At least one of --depth_dir or --rgb_dir must be specified.")
        return

    # Initialize Rerun
    rr.init("multimodal_robot_data", spawn=False)
    
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
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

    depth_files = []
    if args.depth_dir:
        depth_path = pathlib.Path(args.depth_dir)
        if not depth_path.exists():
            print(f"Directory {depth_path} does not exist.")
            return
        depth_files = sorted(list(depth_path.glob("*.npy")), key=natural_sort_key)
        print(f"Found {len(depth_files)} depth files.")

    rgb_files = []
    if args.rgb_dir:
        rgb_path = pathlib.Path(args.rgb_dir)
        if not rgb_path.exists():
            print(f"Directory {rgb_path} does not exist.")
            return
        rgb_files = sorted(list(rgb_path.glob("*.png")), key=natural_sort_key)
        print(f"Found {len(rgb_files)} RGB files.")

    # We'll use the largest collection to drive the loop
    max_len = max(len(depth_files), len(rgb_files))
    
    if max_len == 0:
        print("No files found in any specified directory.")
        return

    print(f"Starting visualization for {max_len} frames...")

    for i in range(max_len):
        # Set the timeline
        rr.set_time_sequence("frame", i)
        
        # Log Depth
        if i < len(depth_files):
            depth_path = depth_files[i]
            depth_data = np.load(depth_path)
            # Rerun expects float32 or uint16 for DepthImage
            rr.log("camera/depth", rr.DepthImage(depth_data, meter=1.0))

        # Log RGB
        if i < len(rgb_files):
            rgb_path = rgb_files[i]
            rgb_data = cv2.imread(str(rgb_path))
            if rgb_data is not None:
                rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
                rr.log("camera/rgb", rr.Image(rgb_data))
        
        if i % 20 == 0:
            print(f"Processed frame {i}/{max_len}")

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

