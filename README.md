# Lidar cone detection
Simple cone detection using lidar
- [Overview](#Overview)
- [Requirements](#Requirements)
- [Features](#Features)
- [Build and Run](#Build-and-Run)
- [FSDS lidar config](#FSDS-lidar-config)
- [Important Notes](#Important-Notes)
- [License](#License)

## Overview
This model, defines the cones using the model and the camera, and then by combining the camera and lidar data, obtains the location of the cone relative to the lidar. Camera-lidar fusion is realized by traformation and rotation matrixes and camera matrix which are located in `params.yaml` file. 

## Requirements
| Dependency            | Version        |
| --------------------- | -------------- |
| C++ Standard          | >=17           |
| Cmake                 | >=3.5          |
| Eigen                 |                |
| PCL                   |                |
| Armadillo             | >=11.0.1       |

Download: http://arma.sourceforge.net/download.html
```bash
tar -xvf armadillo-14.0.2.tar.xz
cd armadillo-14.0.2
mkdir build
cd build
cmake ..
make
sudo make install
```

## Features
- **Object detection**
- **Camera lidar fusion**

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

## FSDS lidar and camera config
In the config folder there is a settings.json file, which should be placed in the root of the FSDS folder. In this file the camera settings are changed for our model, as well as the lidar settings, selected in a similar way as in Lidar Ouster OS1-64 SR.
```bash
cp ~/ws/src/cone_detection/config/settings.json /home/ros/Formula-Student-Driverless-Simulator/
```

## Important Notes
- **Parameters:** For this module to work properly, it is important that the parameters match. Therefore, before starting, make sure that the parameters of the camera and lidar coincide with the parameters in the param.yaml file.

## License
This project is under an ISC license.
