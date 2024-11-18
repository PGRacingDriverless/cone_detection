#ifndef CONE_DETECTION_HPP
#define CONE_DETECTION_HPP

#include <rclcpp/rclcpp.hpp>
// Matrix manipulations
#include <Eigen/Dense>
// Image manipulations, msg<->Image conversion
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
// Point cloud manipulations
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
// Synchronization of messages from camera and lidar
#include <rclcpp/qos.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
// Messages
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "pathplanner_msgs/msg/cone_array.hpp"
#include "pathplanner_msgs/msg/cone.hpp"
// Image detection using a model
#include "model.hpp"
// Standard
#include <limits>
#include <string>
#include <utility>
#include <vector>
// Debug
#ifdef NDEBUG
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <chrono>
#endif

/**
 * Lidar-Camera fusion parameters. All parameters can be changed
 * in `params.yaml`.
 * @param max_len Maximum distance for the point cloud. Points farther than
 * this distance are filtered out.
 * @param min_len Minimum distance for the point cloud. Points closer than
 * this distance are filtered out.
 * @param max_fov Maximum field of view. Not used now. For interpolation.
 * @param min_fov Minimum field of view Not used now. For interpolation.
 * @param ang_res_x Angular resolution x. Not used now. For interpolation.
 * @param ang_res_y Angular resolution y. Not used now. For interpolation.
 */
typedef struct {
    float max_len;
    float min_len;
    float max_fov;
    float min_fov;
    float ang_res_x;
    float ang_res_y;
} FusionParams;

/**
 * The ROS2 node responsible for detecting the cones using the lidar and
 * camera and sending the position of the cones relative to the lidar.
 */
class ConeDetection : public rclcpp::Node {
public:
    /**
     * Constructor. Reads parameters from config, synchronizes lidar
     * and camera topics, creates subscribers and publishers.
     */
    ConeDetection(const rclcpp::NodeOptions &node_options);
private:
    /**
     * Main callback of cone detection. Receives messages from camera and
     * lidar, detects cones and merges data from these two sensors,
     * sending at the end array of detected cones `ConeArray`.
     * @param point_cloud_msg `PointCloud2` from lidar.
     * @param image_msg `Image` from camera.
     */
    void cone_detection_callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &image_msg
    );

    /**
     * Detects cones using a model.
     * @param cv_image_ptr image pointer from `cv_bridge`, received by
     * converting an `Image` message to `OpenCV` readable format.
     * @return vector of pairs of cone class name and cone box (`cv::Rect`).
     */
    std::vector<std::pair<std::string, cv::Rect>> camera_cones_detect(
        cv_bridge::CvImagePtr cv_image_ptr
    );

    /**
     * Finds the closest point of the cone relative to the lidar.
     * @param point_cloud_msg `PointCloud2` from lidar.
     * @param image_msg `Image` from camera.
     * @param detected_cones vector of pairs of cone class name and cone
     * box (`cv::Rect`) received from `camera_cones_detect()`.
     * @return vector of pairs of cone class name and cone closest point
     * (`pcl::PointXYZRGB`).
     * @todo Add interpolation to increase the number of points and accuracy.
     */
    std::vector<std::pair<std::string, pcl::PointXYZRGB>> lidar_camera_fusion(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
        const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
    );

    /** Lidar point cloud topic name. Can be changed in `params.yaml`. */
    std::string lidar_points_topic_;
    /** Сamera image topic тame. Can be changed in `params.yaml`. */
    std::string camera_image_topic_;
    //** Subscriber on lidar_points_topic_. */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> lidar_points_subscriber_;
    //** Subscriber on camera_image_topic_. */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> camera_image_subscriber_;
    /**
     * Synchronizer that synchronizes messages from the lidar and camera
     * topics using an approximate time-based policy.
     */
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>>> lidar_camera_synchronizer_;

    /** Pointer to model instance. */
    std::shared_ptr<Model> model_;

    /** Lidar-camera fusion parameters. */
    FusionParams params_;
    /** Сamera's intrinsic parameters matrix; camera calibration matrix. */
    Eigen::MatrixXf camera_matrix_;
    /** 
     * Lidar-camera transformation (rotation and translation) matrix;
     * extrinsic parameters between the sensors.
     */
    Eigen::MatrixXf transformation_matrix_;
    /** Describes the rotation of the camera regarding the lidar sensor. */
    Eigen::MatrixXf rotation_matrix_;
    /**
     * Represents the translation from lidar coordinates to camera
     * coordinates.
     */
    Eigen::MatrixXf translation_matrix_;

    /** Detected cones topic name. */
    std::string detected_cones_topic_ = "cone_array";
    /**
     * Publisher for detected cones. Publishes on the detected_cones_topic_
     * ConeArray messages from pathplanner_msgs.
     */
    rclcpp::Publisher<pathplanner_msgs::msg::ConeArray>::SharedPtr detected_cones_publisher_;

// CMake macro for debug build
#ifdef NDEBUG
    std::string detection_frames_topic_ = "cone_detection/image_detected";
    /**
     * Publishes on the detection_frames_topic_ image with labeled and
     * boxed detected cones.
     */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_frames_publisher_;
    std::string cone_markers_topic_ = "cone_detection/cone_markers";
    /** Publishes on the cone_markers_topic_ markers for visualization. */
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cone_markers_publisher_;
    /** Publishes fusion point cloud. */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fusion_point_cloud_publisher_;
    /** Publishes range image. */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr range_img_publisher_;
    /** Publishes a point cloud from overlaid on the camera image. */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr point_cloud_on_img_publisher_;
#endif
};

#endif
