#include "cone_detection/cone_detection.hpp"

using namespace message_filters;
using namespace std::chrono_literals;

ConeDetection::ConeDetection(const rclcpp::NodeOptions &node_options) 
: Node("cone_detection", node_options)
{
    lidar_points_topic_ = this->declare_parameter<std::string>("lidar_points_topic");
    camera_image_topic_ = this->declare_parameter<std::string>("camera_image_topic");

    maxlen_ = this->declare_parameter<float>("maxlen");
    minlen_ = this->declare_parameter<float>("minlen");
    max_FOV_ = this->declare_parameter<float>("max_FOV");
    min_FOV_ = this->declare_parameter<float>("min_FOV");
    angular_resolution_x_ = this->declare_parameter<float>("angular_resolution_x");
    angular_resolution_y_ = this->declare_parameter<float>("angular_resolution_y");
    max_angle_width_ = this->declare_parameter<float>("max_angle_width");
    max_angle_height_ = this->declare_parameter<float>("max_angle_height");
    max_var_ = this->declare_parameter<float>("max_var");
    interpol_value_ = this->declare_parameter<float>("interpol_value");
    filter_pc_ = this->declare_parameter<bool>("filter_pc");

    frame_id_ = this->declare_parameter<std::string>("frame_id");
    range_image_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("range_image", 10);

    // ---- SYNC ----
    // Custom qos profile and options for subscrubers
    rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    custom_qos.history=RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    //custom_qos.history=RMW_QOS_POLICY_HISTORY_KEEP_ALL;
    custom_qos.depth=10;
    //custom_qos.reliability=RMW_QOS_POLICY_RELIABILITY_RELIABLE;
    custom_qos.reliability=RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    custom_qos.durability=RMW_QOS_POLICY_DURABILITY_VOLATILE;
    //custom_qos.durability=RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL;
    rclcpp::SubscriptionOptions sub_options;
    //sub_options.callback_group=callback_group_subs_;
    sub_options.callback_group=nullptr;
    sub_options.use_default_callbacks=false;
    sub_options.ignore_local_publications=false;

    pc2_sub_ = std::make_shared<Subscriber<sensor_msgs::msg::PointCloud2>>(this, lidar_points_topic_, custom_qos, sub_options);
    image_sub_ = std::make_shared<Subscriber<sensor_msgs::msg::Image>>(this, camera_image_topic_, custom_qos, sub_options);

    sync_ = std::make_shared<Synchronizer<sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>>>(
        sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>(30), // Queue size
        *pc2_sub_, *image_sub_
    );

    sync_->registerCallback(std::bind(&ConeDetection::cone_detection_callback, this, std::placeholders::_1, std::placeholders::_2));
    // ----


    detected_cones_pub_ =
        this->create_publisher<pathplanner_msgs::msg::ConeArray>(detected_cones_topic_, 5);

    ModelParams params;
    // Read params for the model from the .yaml config file
    const std::string model_path =
        this->declare_parameter<std::string>("model_path");
    const std::vector<std::string> classes =
        this->declare_parameter<std::vector<std::string>>("classes");
    const int width = this->declare_parameter<int>("width");
    const int height = this->declare_parameter<int>("height");
    const float iou_threshold =
        this->declare_parameter<float>("iou_threshold");
    const float rect_confidence_threshold =
        this->declare_parameter<float>("rect_confidence_threshold");
    // Set all params
    params.model_path = model_path;
    params.classes = classes;
    params.img_size = { width, height };
    params.iou_threshold = iou_threshold;
    params.rect_confidence_threshold = rect_confidence_threshold;

    SessionOptions options;
    // Read options for the session from the .yaml config file
    const bool cuda_enable = this->declare_parameter<bool>("cuda_enable");
    const int intra_op_num_threads =
        this->declare_parameter<int>("intra_op_num_threads");
    const int log_severity_level =
        this->declare_parameter<int>("log_severity_level");
    // Set all options

// For the case: "cudaEnable: true" in the config, but CUDA is not used.
#ifdef USE_CUDA
    options.cuda_enable = cuda_enable;
#else
    options.cuda_enable = false;
#endif
    options.intra_op_num_threads = intra_op_num_threads;
    options.log_severity_level = log_severity_level;

    // Create model instance and pointer to it
    model_ = std::make_shared<Model>();
    // Create a session with specified options
    model_->init(options, params);

    // Camera-lidar fusion matrixes read
    Tlc.resize(3,1);
    Rlc.resize(3,3);
    Mc.resize(3,4);
    std::vector<double> param =
        this->declare_parameter<std::vector<double>>("camera_matrix");
    Mc  <<  (double)param[0] ,(double)param[1] ,(double)param[2] ,(double)param[3]
            ,(double)param[4] ,(double)param[5] ,(double)param[6] ,(double)param[7]
            ,(double)param[8] ,(double)param[9] ,(double)param[10],(double)param[11];
    param = this->declare_parameter<std::vector<double>>("rlc");
    Rlc <<  (double)param[0] ,(double)param[1] ,(double)param[2]
            ,(double)param[3] ,(double)param[4] ,(double)param[5]
            ,(double)param[6] ,(double)param[7] ,(double)param[8];
    param = this->declare_parameter<std::vector<double>>("tlc");
    Tlc <<  (double)param[0]
            ,(double)param[1]
            ,(double)param[2];


    // DEGUB MODE
    debug_mode_ = this->declare_parameter<bool>("debug_mode");
    if(debug_mode_)
    {
        detection_frames_publisher_ =
            this->create_publisher<sensor_msgs::msg::Image>(detection_frames_topic_, 5);
        markers_cones_publisher_ = 
            this->create_publisher<visualization_msgs::msg::MarkerArray>(markers_cones_topic_, 5);
        marker_id_ = 0;

        pcOnimg_pub = this->create_publisher<sensor_msgs::msg::Image>("pc_on_image_topic", 10);
        pc_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud_topic", 10);
    }
}

void ConeDetection::cone_detection_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg)
{
    // Convert ROS image_msg to CV image
    cv_bridge::CvImagePtr cv_image_ptr;
    try
    {
        /**
         * toCvCopy and toCvShare difference, encodings, etc.:
         * @see https://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
        */
        cv_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    // Detect cones
    std::vector<std::pair<std::string, cv::Rect>> detected_cones = camera_cones_detect(cv_image_ptr);

    std::vector<std::pair<std::string, pcl::PointXYZRGB>> cone_poses = camera_lidar_fusion(point_cloud_msg, image_msg, detected_cones);

    marker_id_ = 0;
    // Markers for publish in debug mode
    auto cones_markers_array = visualization_msgs::msg::MarkerArray();
    // Detected cones to publish for path_planning
    auto cones_message = pathplanner_msgs::msg::ConeArray();

    for (const auto& pair : detected_cones) {
        int center_x = pair.second.x + pair.second.width / 2;
        int center_y = pair.second.y + pair.second.height / 2;
        cv::circle(cv_image_ptr->image, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 0), -1);
    }

    for (const auto& pair : cone_poses) {

        pcl::PointXYZRGB closest_point = pair.second;

        auto cone_detail = pathplanner_msgs::msg::Cone();
        cone_detail.x = closest_point.x;
        cone_detail.y = closest_point.y;

        // YELLOW - INNER = 1 
        // BLUE - OUTER = 0
        if(pair.first == "yellow_cone")
            cone_detail.side = 1;
        else if(pair.first == "blue_cone")
            cone_detail.side = 0;
        else
            continue;
          
        cones_message.cone_array.push_back(cone_detail);

        // DEBUG MODE
        if(debug_mode_) {

            // Create and add a marker for each unique cone
            auto marker = visualization_msgs::msg::Marker();
            marker.header.frame_id = frame_id_;
            marker.header.stamp = this->now();
            marker.ns = "basic_shapes";
            marker.id = marker_id_++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            geometry_msgs::msg::Point point;
            point.x = cone_detail.x;
            point.y = cone_detail.y;
            point.z = 0.2;

            marker.pose.position = point;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;

            if(cone_detail.side == 1) {
                marker.color.r = 0.51;
                marker.color.g = 1.0;
                marker.color.b = 0.56;
            } else if(cone_detail.side == 0) {
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
            }

            marker.color.a = 1.0;
            marker.lifetime = rclcpp::Duration(0s);
            cones_markers_array.markers.push_back(marker);
        }
    }

    // Publish detected cones
    detected_cones_pub_->publish(cones_message);

    
    // DEGUB MODE
    if(debug_mode_) { 
        // Convert CV image to ROS msg
        std::shared_ptr<sensor_msgs::msg::Image> edited_image_msg
            = cv_image_ptr->toImageMsg();
        // Create output msg and assigning values from edited_image_msg pointer to it 
        sensor_msgs::msg::Image output_image_msg;
        output_image_msg.header = edited_image_msg->header;
        output_image_msg.height = edited_image_msg->height;
        output_image_msg.width = edited_image_msg->width;
        output_image_msg.encoding = edited_image_msg->encoding;
        output_image_msg.is_bigendian = edited_image_msg->is_bigendian;
        output_image_msg.step = edited_image_msg->step;
        output_image_msg.data = edited_image_msg->data;
        // Publish message with detected cones
        detection_frames_publisher_->publish(output_image_msg);
        
        markers_cones_publisher_->publish(cones_markers_array);
        //RCLCPP_INFO(this->get_logger(), "%d markers published", marker_id_);
        //RCLCPP_INFO(this->get_logger(), "%lu detected cones", detected_cones.size());
    }
}

std::vector<std::pair<std::string, cv::Rect>> ConeDetection::camera_cones_detect(
    cv_bridge::CvImagePtr cv_image_ptr)
{
    std::vector<std::pair<std::string, cv::Rect>> detected_cones;
    std::vector<ModelResult> res = model_->detect(cv_image_ptr->image);

    for (auto& r : res)
    {
        cv::Scalar color(0, 256, 0);

        detected_cones.push_back(std::make_pair(std::string(model_->get_class_by_id(r.class_id)), r.box));
        cv::rectangle(cv_image_ptr->image, r.box, color, 3);

        float confidence = floor(100 * r.confidence) / 100;
        std::string label = model_->get_class_by_id(r.class_id) + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        cv::rectangle(
            cv_image_ptr->image,
            cv::Point(r.box.x, r.box.y - 25),
            cv::Point(r.box.x + label.length() * 10, r.box.y),
            color,
            cv::FILLED
        );

        cv::putText(
            cv_image_ptr->image,
            label,
            cv::Point(r.box.x, r.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            2
        );
    }

    return detected_cones;
}

std::vector<std::pair<std::string, pcl::PointXYZRGB>> ConeDetection::camera_lidar_fusion(        
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
    const std::vector<std::pair<std::string, cv::Rect>> &detected_cones)
{
    std::vector<std::pair<std::string, pcl::PointXYZRGB>> closest_points;  // Список ближайших точек для каждого объекта

    // Convert image to OpenCV format
    cv_bridge::CvImagePtr cv_ptr, color_pcl;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        color_pcl = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return closest_points;  // Вернем пустой список в случае ошибки
    }

    // Convert PointCloud2 message to pcl::PointCloud
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*point_cloud_msg, pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

    // Filter out invalid points from the point cloud
    if (msg_pointCloud->empty()) {
        RCLCPP_ERROR(this->get_logger(), "Input point cloud is empty!");
        return closest_points;
    }

    // Pre-process the point cloud
    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    // Filter points based on distance
    for (const auto& point : cloud_in->points) {
        double distance = std::sqrt(point.x * point.x + point.y * point.y);
        if (distance < minlen_ || distance > maxlen_)
            continue;
        cloud_out->push_back(point);
    }

    if (cloud_out->empty()) {
        RCLCPP_ERROR(this->get_logger(), "Filtered point cloud is empty!");
        return closest_points;
    }

    // Prepare transformation matrix (Lidar to Camera)
    Eigen::MatrixXf RTlc(4, 4);
    RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
            Rlc(1), Rlc(4), Rlc(7), Tlc(1),
            Rlc(2), Rlc(5), Rlc(8), Tlc(2),
            0, 0, 0, 1;

    Eigen::MatrixXf Lidar_cam(3, 1);
    Eigen::MatrixXf pc_matrix(4, 1);
    uint px_data, py_data;

    // Iterate over each detected cone and find the closest point
    for (const auto& pair : detected_cones) {
        // Get the center of the detected cone (cv::Rect)
        int center_x = pair.second.x + pair.second.width / 2;
        int center_y = pair.second.y + pair.second.height / 2;

        // Transform center pixel to 3D point
        pcl::PointXYZRGB closest_point;
        double min_distance = std::numeric_limits<double>::max();

        // Loop over all points in the filtered point cloud
        for (const auto& point : cloud_out->points) {
            // Transform point from LiDAR to Camera
            pc_matrix << -point.y, -point.z, point.x, 1.0;
            Lidar_cam = Mc * (RTlc * pc_matrix);

            // Project 3D point to image plane
            px_data = static_cast<int>(Lidar_cam(0, 0) / Lidar_cam(2, 0));
            py_data = static_cast<int>(Lidar_cam(1, 0) / Lidar_cam(2, 0));

            // Skip points outside of image bounds
            if (px_data >= image_msg->width || py_data >= image_msg->height)
                continue;

            // Calculate the distance to the center of the cone (center_x, center_y)
            double distance_to_center = std::sqrt(
                std::pow(px_data - center_x, 2) + std::pow(py_data - center_y, 2));

            // Find the closest point
            if (distance_to_center < min_distance) {
                min_distance = distance_to_center;
                closest_point.x = point.x;
                closest_point.y = point.y;
                closest_point.z = point.z;
                closest_point.r = color_pcl->image.at<cv::Vec3b>(py_data, px_data)[2];
                closest_point.g = color_pcl->image.at<cv::Vec3b>(py_data, px_data)[1];
                closest_point.b = color_pcl->image.at<cv::Vec3b>(py_data, px_data)[0];
            }
        }

        double distance = std::sqrt(closest_point.x * closest_point.x + closest_point.y * closest_point.y);
        if (distance < 25.0)
            closest_points.push_back(std::make_pair(pair.first, closest_point));
    }

    // Return the list of closest points for each detected cone
    return closest_points;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetection)
