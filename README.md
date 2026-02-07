# Data Processing Pipeline for DP3

### All the relevant scripts for data processing pipeline are in the data_processing_pipeline folder

### To perform data processing for one trajectory:
```bash
cd data_processing_pipeline
source /opt/ros/humble/setup.bash
```
1. Extract topics from bag file:
```bash
   python3 extract_topics_one_camera.py --bag_path <bag_path> --output_root <output_root>
```
2. Extract point clouds from extracted data:
```bash   
   python3 extract_point_clouds_one_cam.py --bag_path <bag_path> --extracted_data_root <extracted_data_root> --output_root <output_root>
```
3. Filter point clouds:
```bash
   python3 filter_point_clouds.py --input_root <input_root> --output_root <output_root>
```

### To run them on a directory of bag files for DP3 data preparation:

```bash
cd data_processing_pipeline
```
1. Extracting topics from bag files and extracting point clouds from the extracted data
```bash
    python3 pipeline_stage1_extract.py --bag_dir <bag_dir> --output_root <output_root>
```
2. Filter point clouds
```bash
   python3 pipeline_stage2_filter.py --input_root <input_root> --output_root <output_root>
```
3. Create zarr files (format expected by DP3)
```bash
   python3 chunking_data.py
```
### Point Cloud Visualisation in Rerun:

```bash
cd data_processing_pipeline

python visualize_point_clouds.py --dir <pc_dir> --depth_dir <depth_dir> --rgb_dir <rgb_dir> --save <save_path.rrd>
```
Currently the serve functionality is not working as expected, need to debug that. I just save the rerun file on the server and open it on my local machine using rerun command

