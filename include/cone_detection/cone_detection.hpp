#ifndef CONE_DETECTION_NODE_HPP_
#define CONE_DETECTION_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>

#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <pcl/kdtree/kdtree.h>

#include <pcl/point_types.h>

using std::placeholders::_1;

class ConeDetector : public rclcpp::Node {
public:
    ConeDetector(const rclcpp::NodeOptions &node_options);

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    //void publish_markers_to_RVIZ(const std::vector<Cluster> &clusters);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_pub_;

    std::string lidar_points_topic_;
    std::string frame_id_;
    double filter_min_;
    double filter_max_;

};

#endif // CONE_DETECTION_NODE_HPP_