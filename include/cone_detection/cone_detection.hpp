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
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
// Range image
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/range_image/range_image.h>
// Interpolation
#include <armadillo>
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
#include <math.h>
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

/**
 * Lidar-Camera fusion parameters. All parameters can be changed
 * in `params.yaml`.
 * @param max_len Maximum distance for the point cloud. Points farther than
 * this distance are filtered out.
 * @param min_len Minimum distance for the point cloud. Points closer than
 * this distance are filtered out.
 * @param max_fov Maximum field of view. For interpolation.
 * @param min_fov Minimum field of view For interpolation.
 * @param ang_res_x Angular resolution x. For interpolation.
 * @param ang_res_y Angular resolution y. For interpolation.
 * @param max_ang_w Maximum angle width. For interpolation.
 * @param max_ang_h Maximum angle height. For interpolation.
 * @todo Describe all params.
 */
typedef struct {
    float max_len;
    float min_len;
    float max_fov;
    float min_fov;
    float ang_res_x;
    float ang_res_y;
    float max_ang_w;
    float max_ang_h;
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
     * Filter point cloud before interpolation
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr filterPointCloudBeforeInterpolation(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud);

    /**
     * Filter range matrix after interpolation
     */
    arma::mat filterInterpolatedData(
        const arma::mat &range_matrix_interp);

    /**
     * Interpolate point cloud using range image
     * @param range_img `RangeImageSpherical` after convert on point cloud
     * @param range_matrix_interp `arma::mat` OUTPUT matrix of interpolated range values
     * @param height_matrix_interp `arma::mat` OUTPUT matrix of interpolated Z axis values
     */
    void interpolateRangeImage(
        const pcl::RangeImageSpherical::Ptr &range_img,
        arma::mat &range_matrix, arma::mat &height_matrix,
        arma::mat &range_matrix_interp, arma::mat &height_matrix_interp);

    /**
     * Finds the closest point of the cone relative to the lidar.
     * @param point_cloud_msg `PointCloud2` from lidar.
     * @param image_msg `Image` from camera.
     * @param detected_cones vector of pairs of cone class name and cone
     * box (`cv::Rect`) received from `camera_cones_detect()`.
     * @return vector of pairs of cone class name and cone closest point
     * (`pcl::PointXYZ`).
     * @todo Add interpolation to increase the number of points and accuracy.
     */
    std::vector<std::pair<std::string, pcl::PointXYZ>> lidar_camera_fusion(
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

    // I love this
    float interp_value = 10.0;

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
    /** Publishes range image. */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr range_img_publisher_;
    /** Publishes point cloud from range image. */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr range_img_cloud_publisher_;
    /** Publishes a point cloud from overlaid on the camera image. */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr point_cloud_on_img_publisher_;
#endif
};

#endif
