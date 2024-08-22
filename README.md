# Lidar cone detection
Simple cone detection using lidar
- [Overview](#Overview)
- [Requirements](#Requirements)
- [Features](#Features)
- [Topics](#Topics)
- [Build and Run](#Build-and-Run)
- [FSDS lidar config](#FSDS-lidar-config)
- [Important Notes](#Important-Notes)
- [License](#License)

## Overview
This project is focused on detecting traffic cones using LIDAR data. The primary goal is to process the point cloud data, remove ground points, and cluster the remaining points to identify the positions of cones.

## Requirements
| Dependency            | Version        |
| --------------------- | -------------- |
| C++ Standard          | >=17           |
| Cmake                 | >=3.5          |
| Eigen                 |                |
| PCL                   |                |

## Features
- **Ground Plane Segmentation:** Uses RANSAC to remove ground points from the LIDAR data.
- **Clustering:** Clusters non-ground points using DBSCAN to detect potential cones.
- **Visualization:** Publishes the detected cones as markers in RViz for visualization.

## Topics
- **Subscribed Topics:**
  - `/lidar/Lidar1`: The input point cloud from the LIDAR.

- **Published Topics:**
  - `filtered_points`: The point cloud after ground removal.
  - `visualization_marker_array`: Markers for visualizing detected cones in RViz.
  - `clustered_points`: Pose array of clustered points representing detected cones.


## Build and Run
Build:
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Source the environment:
```bash
source install/setup.bash
```

Launch:
```bash
ros2 launch cone_detection cone_detection.launch.py 
```

## FSDS lidar config
Below is an example of a JSON configuration file that you might use to set the parameters for the FSDS lidar:
```json
"Lidar1": {
  "SensorType": 6,
  "Enabled": true,
  "X": 1.2, "Y": 0, "Z": 0.4,
  "Roll": 0, "Pitch": 0, "Yaw" : 0,
  "NumberOfLasers": 16,
  "PointsPerScan": 10096,
  "RotationsPerSecond": 20,
  "VerticalFOVUpper": 2,
  "VerticalFOVLower": -6,
  "HorizontalFOVStart": -45,
  "HorizontalFOVEnd": 45,
  "DrawDebugPoints": false
}

"cam1": {
  "CaptureSettings": [
  {
    "ImageType": 0,
    "Width": 1280,
    "Height": 1280,
    "FOV_Degrees": 90
  }
  ],
  "X": -0.3, "Y": 0, "Z": 0.8,
  "Pitch": 0.0,
  "Roll": 0.0,
  "Yaw": 0
}
```
Parameters are roughly matched as for Velodyne VLP-16 lidar

## Important Notes
- **Sensitivity to Lidar and Environment:** The performance and accuracy of the algorithm are highly dependent on the characteristics of the lidar sensor used and the surrounding environment. Different lidars with varying resolutions, ranges, and field of view might require unique parameter tuning for optimal performance.
- **Parameter Tuning:** To achieve the best results, you may need to adjust the filtering and clustering parameters depending on the specific lidar model and environmental conditions. Experiment with parameters such as `filter_min`, `filter_max`, and `ClusterTolerance` to find the optimal settings.

## License
This project is under an ISC license.