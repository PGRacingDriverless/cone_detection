cmake_minimum_required(VERSION 3.8)
project(cone_detection)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
    find_package(CUDA 12.4 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else()
    set(USE_CUDA OFF)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
# find_package(pcl_ros REQUIRED)
# find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(pathplanner_msgs REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.9 REQUIRED)
find_package(Armadillo 14.0.2 REQUIRED)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

set(ONNXRUNTIME_VERSION 1.18.1)

if(USE_CUDA)
    set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime-libs/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
else()
    set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime-libs/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
endif()

set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)

include_directories(include
    ${ARMADILLO_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
    src/${PROJECT_NAME}.cpp
    src/model.cpp
)

rclcpp_components_register_node(
    ${PROJECT_NAME}
    PLUGIN "ConeDetection"
    EXECUTABLE ${PROJECT_NAME}_node
)

target_link_libraries(${PROJECT_NAME}
    ${ARMADILLO_LIBRARIES}
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

ament_auto_package(
    INSTALL_TO_SHARE
    config
    launch
)

install(FILES DESTINATION share/${PROJECT_NAME})
