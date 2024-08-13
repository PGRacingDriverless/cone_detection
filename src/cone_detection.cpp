#include "cone_detection/cone_detection.hpp"
#include <cmath>

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
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 10);
    cone_pair_array_publisher_ = this->create_publisher<pathplanner_msgs::msg::ConePairArray>("cone_pair_array", 10);
}

void ConeDetector::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false); 
    extract.filter(*ground_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(true); 
    extract.filter(*non_ground_cloud);

    sensor_msgs::msg::PointCloud2 filtered_msg;
    pcl::toROSMsg(*non_ground_cloud, filtered_msg);
    filtered_msg.header = msg->header;
    filtered_pub_->publish(filtered_msg);
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(non_ground_cloud);

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
    
    pathplanner_msgs::msg::ConePairArray pairs_message;
    ConePairArray m_cone_pair_array;

    std::vector<geometry_msgs::msg::Pose> cone_poses;
   
    for (const auto& indices : cluster_indices) {
        if (indices.indices.empty()) continue;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*non_ground_cloud, indices.indices, centroid);
        geometry_msgs::msg::Pose pose;
        pose.position.x = centroid[0];
        pose.position.y = centroid[1];  
        pose.position.z = centroid[2];
        cone_poses.push_back(pose);
    }

    for (size_t i = 0; i < cone_poses.size(); i += 2) {
        if (i + 1 < cone_poses.size()) {
            pathplanner_msgs::msg::ConePair cone_pair;

            const auto& cone1 = cone_poses[i];
            const auto& cone2 = cone_poses[i + 1];

            double mid_x = (cone1.position.x + cone2.position.x) / 2.0;
            double mid_y = (cone1.position.y + cone2.position.y) / 2.0;

            double vec1_x = cone1.position.x - mid_x;
            double vec1_y = cone1.position.y - mid_y;
            double vec2_x = cone2.position.x - mid_x;
            double vec2_y = cone2.position.y - mid_y;

            double angle1 = std::atan2(vec1_y, vec1_x);
            double angle2 = std::atan2(vec2_y, vec2_x);

            if (angle1 < angle2) {
                cone_pair.cone_inner.x = cone1.position.x;
                cone_pair.cone_inner.y = cone1.position.y;
                cone_pair.cone_inner.side = 1; 

                cone_pair.cone_outer.x = cone2.position.x;
                cone_pair.cone_outer.y = cone2.position.y;
                cone_pair.cone_outer.side = 0; 
            } else {
                cone_pair.cone_inner.x = cone2.position.x;
                cone_pair.cone_inner.y = cone2.position.y;
                cone_pair.cone_inner.side = 1; 

                cone_pair.cone_outer.x = cone1.position.x;
                cone_pair.cone_outer.y = cone1.position.y;
                cone_pair.cone_outer.side = 0; 
            }

            pairs_message.cone_pair_array.push_back(cone_pair);
                        
            visualization_msgs::msg::Marker outer_marker;
            outer_marker.header.frame_id = frame_id_;
            outer_marker.header.stamp = this->get_clock()->now();
            outer_marker.ns = "cone_pairs";
            outer_marker.id = i; 
            outer_marker.type = visualization_msgs::msg::Marker::CYLINDER;
            outer_marker.action = visualization_msgs::msg::Marker::ADD;

            outer_marker.pose.position.x = cone_pair.cone_outer.x;
            outer_marker.pose.position.y = cone_pair.cone_outer.y;
            outer_marker.pose.position.z = 0.0; 
            outer_marker.pose.orientation.x = 0.0;
            outer_marker.pose.orientation.y = 0.0;
            outer_marker.pose.orientation.z = 0.0;
            outer_marker.pose.orientation.w = 1.0; 

            outer_marker.scale.x = 0.15; 
            outer_marker.scale.y = 0.15; 
            outer_marker.scale.z = 0.25; 
            outer_marker.color.a = 1.0; 
            outer_marker.color.r = 0.0; 
            outer_marker.color.g = 0.0;
            outer_marker.color.b = 1.0;
            marker_array.markers.push_back(outer_marker);

            visualization_msgs::msg::Marker inner_marker;
            inner_marker.header.frame_id = frame_id_;
            inner_marker.header.stamp = this->get_clock()->now();
            inner_marker.ns = "cone_pairs";
            inner_marker.id = i + 1; 
            inner_marker.type = visualization_msgs::msg::Marker::CYLINDER;
            inner_marker.action = visualization_msgs::msg::Marker::ADD;

            inner_marker.pose.position.x = cone_pair.cone_inner.x;
            inner_marker.pose.position.y = cone_pair.cone_inner.y;
            inner_marker.pose.position.z = 0.0; 
            inner_marker.pose.orientation.x = 0.0;
            inner_marker.pose.orientation.y = 0.0;
            inner_marker.pose.orientation.z = 0.0;
            inner_marker.pose.orientation.w = 1.0; 

            inner_marker.scale.x = 0.15; 
            inner_marker.scale.y = 0.15; 
            inner_marker.scale.z = 0.25; 
            inner_marker.color.a = 1.0; 
            inner_marker.color.r = 0.0; 
            inner_marker.color.g = 1.0;
            inner_marker.color.b = 0.0;
            marker_array.markers.push_back(inner_marker);
        }
    }

    marker_pub_->publish(marker_array);
    pose_array_pub_->publish(pose_array);
    publishPose();
    cone_pair_array_publisher_->publish(pairs_message);
}

void ConeDetector::publishPose() {
    geometry_msgs::msg::PoseStamped pose_;

    pose_.header.stamp = this->now();
    pose_.header.frame_id = frame_id_;

    pose_.pose.position.x = 0.0;
    pose_.pose.position.y = 0.0;
    pose_.pose.position.z = 0.0;

    pose_.pose.orientation.x = 0.0;
    pose_.pose.orientation.y = 0.0;
    pose_.pose.orientation.z = 0.0;
    pose_.pose.orientation.w = 1.0;

    pose_pub_->publish(pose_);
}


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetector)
