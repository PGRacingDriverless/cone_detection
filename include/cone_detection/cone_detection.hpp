#ifndef CONE_DETECTION_HPP
#define CONE_DETECTION_HPP

// Show debug data in release and debug modes
#undef NDEBUG

#include <rclcpp/rclcpp.hpp>
// Matrix manipulations
#include <Eigen/Dense>
// Image manipulations, msg<->Image conversion
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
// Point cloud manipulations
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
// Synchronization of messages from camera and lidar
#include <rclcpp/qos.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
// Messages
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "common_msgs/msg/cone_array.hpp"
#include "common_msgs/msg/cone.hpp"
// Image detection using a model
#include "model.hpp"
// Standard
#include <limits>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
// Debug
#ifndef NDEBUG
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <chrono>
#endif

/**  */
struct ConeInfo {
    std::string id;
    cv::Rect bbox;
    std::vector<pcl::PointXYZ> associated_points;
    pcl::PointXYZ average_point;
    double confidence;
};

/**
 * Lidar-Camera fusion parameters. All parameters can be changed
 * in `params.yaml`.
 * @param max_len Maximum distance for the point cloud. Points farther than
 * this distance are filtered out.
 * @param min_len Minimum distance for the point cloud. Points closer than
 * this distance are filtered out.
 * @param interp_factor Interpolation factor.
 */
typedef struct {
    float max_len;
    float min_len;
    float interp_factor;
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
    /** Lidar point cloud topic name. Can be changed in `params.yaml`. */
    std::string lidar_points_topic_;
    /** Сamera image topic тame. Can be changed in `params.yaml`. */
    std::string camera_image_topic_;
    /** Subscriber on lidar_points_topic_. */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> lidar_points_subscriber_;
    /** Subscriber on camera_image_topic_. */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> camera_image_subscriber_;
    /**
     * Synchronizer that synchronizes messages from the lidar and camera
     * topics using an approximate time-based policy.
     */
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>>> lidar_camera_synchronizer_;

    /** Publisher for detected cones. */
    rclcpp::Publisher<common_msgs::msg::ConeArray>::SharedPtr detected_cones_publisher_;
// CMake macro for debug build
#ifndef NDEBUG
    /** Publishes image with labeled and boxed detected cones. */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_frames_publisher_;
    /** Publishes markers for visualization. */
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cone_markers_publisher_;
    /** Publishes filtered point cloud. */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_point_cloud_publisher_;
    /** Publishes interpolated point cloud. */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr interp_point_cloud_publisher_;
    /** Publishes a point cloud from overlaid on the camera image. */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr point_cloud_on_img_publisher_;
#endif

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

    /** Pointer to model instance. */
    std::shared_ptr<Model> model_;

    /**
     * Detects cones using a model.
     * @param cv_image_ptr image pointer from `cv_bridge`, received by
     * converting an `Image` message to `OpenCV` readable format.
     * @return vector of pairs of cone class name and cone box (`cv::Rect`).
     */
    std::vector<std::pair<std::string, cv::Rect>> detect_cones_on_img(
        cv_bridge::CvImagePtr cv_image_ptr
    );

    /** Height threshold in pixels for distance filtering. */
    int height_in_pixels;

    /**
     * Filter cones by `cv::Rect` box height in pixels.
     * @param detected_cones vector of pairs of cone class name and cone
     * box (`cv::Rect`) for filtering.
     */
    void filter_by_px_height(
        std::vector<std::pair<std::string, cv::Rect>> &detected_cones
    );

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

    /**
     * Finds the closest point of the cone relative to the lidar.
     * @param point_cloud_msg `PointCloud2` from lidar.
     * @param image_msg `Image` from camera.
     * @param detected_cones vector of pairs of cone class name and cone
     * box (`cv::Rect`) received from `detect_cones_on_img()`.
     * @return vector of pairs of cone class name and cone closest point
     * (`pcl::PointXYZ`).
     */
    std::vector<std::pair<std::string, pcl::PointXYZ>> lidar_camera_fusion(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
        const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
    );

    /**
     * Point cloud distance-based filtering. 
     * @param point_cloud point cloud for filtering.
     * @return filtered point cloud.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr distance_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
    );

    /**
     * Point cloud ground removal filtering.
     * @param point_cloud point cloud for filtering.
     * @return filtered point cloud.
     * @todo move distance threshold to config.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_removal_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
    );

    /**
     * Interpolates point cloud using KdTree.
     * @param point_cloud point cloud for filtering.
     * @return interpolated point cloud.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr interp_point_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
    );

    /** The current dynamic event (mission). */
    std::string mission_;
};

#endif
