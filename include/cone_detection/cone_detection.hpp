#ifndef CONE_DETECTION_NODE_HPP_
#define CONE_DETECTION_NODE_HPP_

#include "rclcpp/rclcpp.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

#include "cones/cone_array.hpp"
#include "cones/cone.hpp"
#include "pathplanner_msgs/msg/cone.hpp"
#include "pathplanner_msgs/msg/cone_array.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "message_filters/time_synchronizer.h"

#include "model.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/qos.hpp>

#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>

using namespace message_filters;

class ConeDetection : public rclcpp::Node {
public:
    ConeDetection(const rclcpp::NodeOptions &node_options);

private:
    
    void cone_detection_callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &image_msg);

    std::vector<std::pair<std::string, cv::Rect>> camera_cones_detect(cv_bridge::CvImagePtr cv_image_ptr);


    // SYNC
    std::shared_ptr<Synchronizer<sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>>> sync_;
    std::shared_ptr<Subscriber<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<Subscriber<sensor_msgs::msg::PointCloud2>> pc2_sub_;        

    rclcpp::Publisher<pathplanner_msgs::msg::ConeArray>::SharedPtr detected_cones_pub_;


    std::string detected_cones_topic_ = "detected_cones";
    std::string lidar_points_topic_;
    std::string camera_image_topic_;
    std::string frame_id_;

    std::shared_ptr<Model> model_;
    Eigen::Matrix4f camera_to_lidar_;

    bool debug_mode_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_frames_publisher_;
    std::string detection_frames_topic_ = "camera/image_detected";
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_cones_publisher_;
    std::string markers_cones_topic_ = "detected_cones_markers";
    int marker_id_;
};

#endif // CONE_DETECTION_NODE_HPP_