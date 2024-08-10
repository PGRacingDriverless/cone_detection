#include "cone_detection/cone_detection.hpp"

ConeDetector::ConeDetector(const rclcpp::NodeOptions &node_options) 
: Node("cone_detection", node_options)
{
    lidar_points_topic_ = this->declare_parameter<std::string>("lidar_points_topic");
    frame_id_ = this->declare_parameter<std::string>("frame_id");
    filter_min_ = this->declare_parameter<double>("filter_min");
    filter_max_ = this->declare_parameter<double>("filter_max");

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_points_topic_, 10, std::bind(&ConeDetector::pointCloudCallback, this, _1));

    filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_points", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker_array", 10);
    pose_array_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("clustered_points", 10);
}


void ConeDetector::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    // RANSAC for ground plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    // the ground points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false); 
    extract.filter(*ground_cloud);

    // non-ground points
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(true); 
    extract.filter(*non_ground_cloud);

    sensor_msgs::msg::PointCloud2 filtered_msg;
    pcl::toROSMsg(*non_ground_cloud, filtered_msg);
    filtered_msg.header = msg->header;
    filtered_pub_->publish(filtered_msg);
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(non_ground_cloud);

    // DBSCAN clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.3);
    ec.setMinClusterSize(5);
    ec.setSearchMethod(tree);
    ec.setInputCloud(non_ground_cloud);
    ec.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray marker_array;
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp = this->get_clock()->now();
    pose_array.header.frame_id = frame_id_;

    int id = 0;
    for (const auto& indices : cluster_indices) {
        if (indices.indices.empty()) continue;

        // compute centroid of the cluster
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*non_ground_cloud, indices.indices, centroid);

        geometry_msgs::msg::Pose pose;
        pose.position.x = centroid[0];
        pose.position.y = centroid[1];
        pose.position.z = centroid[2];
        pose_array.poses.push_back(pose);

        // marker
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "cone_markers";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose = pose;
        marker.scale.x = 0.15;
        marker.scale.y = 0.15;
        marker.scale.z = 0.25;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker_array.markers.push_back(marker);
    }

    marker_pub_->publish(marker_array);
    pose_array_pub_->publish(pose_array);
}


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetector)
