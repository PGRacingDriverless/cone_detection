cmake_minimum_required(VERSION 3.5)
project(cone_detection)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --------------- OpenCV --------------- #
find_package(OpenCV 4.9 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --------------- CUDA ----------------- #
option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
    find_package(CUDA 12.4 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else()
    set(USE_CUDA OFF)
endif()

# --------------- ONNX Runtime  -------- #
set(ONNXRUNTIME_VERSION 1.18.1)

if(USE_CUDA)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
else()
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)

set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)

find_package(PCL REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

include_directories(include ${PCL_INCLUDE_DIRS})

ament_auto_add_library(${PROJECT_NAME} SHARED
    src/${PROJECT_NAME}.cpp
    src/model.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}
    PLUGIN "ConeDetection"
    EXECUTABLE ${PROJECT_NAME}_node
)

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
    INSTALL_TO_SHARE
    launch
    config
    rviz
)

install(FILES DESTINATION share/${PROJECT_NAME})
