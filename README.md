# Сone detection
- [Overview](#Overview)
- [Requirements](#Requirements)
- [Build and Run](#Build-and-Run)
- [Models](#Models)
- [License](#License)

## Overview
The package, using the pre-trained model, detects cones in the images coming from the camera, merges the camera data with the lidar data, and determines the location of each cone relative to the lidar.

> **Note**  
> Make sure that your camera and lidar parameters match the parameters in the `params.yaml` file.

## Requirements
| Dependency            | Version        |
| --------------------- | -------------- |
| C++ Standard          | >=17           |
| Cmake                 | >=3.8          |
| OpenCV                | >=4.9          |
| ONNX Runtime          | >=1.18.1       |
| YOLO model            | >=8.1 \<9      |
| Cuda (Optional)       | >=12.4 \<=12.5 |
| cuDNN (Cuda required) | =9             |
| PCL                   | x              |
| Eigen3                | x              |

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

### Launch parameters
| Parameter | Value      | Description                    |
| --------- | ---------- | ------------------------------ |
| rviz      | bool       | Runs with RViz2                |

## Models
| Model            | Description                                                     | mAP50 | Speed |
|------------------|-----------------------------------------------------------------|-------|-------|
| Yolov8n1280.onnx | Base model, the fastest of the current lineup (n = nano).       | 0.763 | 2.8ms |
| Yolov8s1280.onnx | Bigger model (focusing on accurracy over performance).          | 0.799 | 5.1ms |
| Yolov8n1920.onnx | The most accurate model, however it compromises on performance. | 0.829 | 6.1ms |

## License
This project is under an MIT license.
