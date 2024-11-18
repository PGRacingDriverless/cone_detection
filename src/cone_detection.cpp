#include "cone_detection/cone_detection.hpp"

ConeDetection::ConeDetection(const rclcpp::NodeOptions &node_options) 
    : Node("cone_detection", node_options)
{
    // Read and set topic names from the .yaml config file
    lidar_points_topic_ = 
        this->declare_parameter<std::string>("lidar_points_topic");
    camera_image_topic_ = 
        this->declare_parameter<std::string>("camera_image_topic");
    

    // Synchronization of messages from lidar_points_topic_ and
    // camera_image_topic_
    
    // Define custom QoS profile for subscribers
    rmw_qos_profile_t lidar_camera_qos = rmw_qos_profile_default;
    // Only keep the last message in history
    lidar_camera_qos.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    // Maximum number of messages to store
    lidar_camera_qos.depth = 10;
    // Best effort reliability (i.e. no guarantees to deliver samples)
    lidar_camera_qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    // Messages are lost if the node crashes
    lidar_camera_qos.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;

    // Define subscription options for subscribers
    rclcpp::SubscriptionOptions lidar_camera_options;
    // No specific callback group
    lidar_camera_options.callback_group = nullptr;
    // Use custom callbacks instead of default ones
    lidar_camera_options.use_default_callbacks = false;
    // Don't ignore local publications (i.e. messages published by this node)
    lidar_camera_options.ignore_local_publications = false;

    // Create subscribers for lidar_points_topic_ and camera_image_topic_
    // with our custom profile and custom options
    lidar_points_subscriber_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
        this, lidar_points_topic_, lidar_camera_qos, lidar_camera_options
    );
    camera_image_subscriber_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        this, camera_image_topic_, lidar_camera_qos, lidar_camera_options
    );

    // Create a synchronizer to handle messages from both subscribers with
    // the specified queue size
    lidar_camera_synchronizer_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>>>(
        message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>(30),
        *lidar_points_subscriber_,
        *camera_image_subscriber_
    );

    // Register a callback with the synchronizer to process synchronized messages
    lidar_camera_synchronizer_->registerCallback(std::bind(&ConeDetection::cone_detection_callback, this, std::placeholders::_1, std::placeholders::_2));


    ModelParams params;
    // Read and set params for the model from the .yaml config file
    params.model_path = this->declare_parameter<std::string>("model_path");
    params.classes = 
        this->declare_parameter<std::vector<std::string>>("classes");
    params.img_size = {
        this->declare_parameter<int>("width"),
        this->declare_parameter<int>("height")
    };
    params.iou_threshold = this->declare_parameter<float>("iou_threshold");
    params.rect_confidence_threshold = 
        this->declare_parameter<float>("rect_confidence_threshold");

    SessionOptions options;
    // Read and set options for the session from the .yaml config file
// For the case: "cudaEnable: true" in the config, but CUDA is not used.
#ifdef USE_CUDA
    options.cuda_enable = this->declare_parameter<bool>("cuda_enable");
#else
    options.cuda_enable = false;
#endif
    options.intra_op_num_threads = 
        this->declare_parameter<int>("intra_op_num_threads");
    options.log_severity_level = 
        this->declare_parameter<int>("log_severity_level");

    // Create model instance and pointer to it
    model_ = std::make_shared<Model>();
    // Create a session with specified options
    model_->init(options, params);

    
    // Read and set params for the lidar-camera fusion from the .yaml config
    params_.max_len = this->declare_parameter<float>("max_len");
    params_.min_len = this->declare_parameter<float>("min_len");
    params_.max_fov = this->declare_parameter<float>("max_fov");
    params_.min_fov = this->declare_parameter<float>("min_fov");
    params_.ang_res_x = this->declare_parameter<float>("ang_res_x");
    params_.ang_res_y = this->declare_parameter<float>("ang_res_y");


    // Read and set matrices for lidar-camera fusion
    std::vector<double> temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("camera_matrix");
    camera_matrix_.resize(3,4);
    camera_matrix_ << (double)temp_matrix_vec[0], (double)temp_matrix_vec[1], (double)temp_matrix_vec[2], (double)temp_matrix_vec[3],
                      (double)temp_matrix_vec[4], (double)temp_matrix_vec[5], (double)temp_matrix_vec[6], (double)temp_matrix_vec[7],
                      (double)temp_matrix_vec[8], (double)temp_matrix_vec[9], (double)temp_matrix_vec[10], (double)temp_matrix_vec[11];
    // camera_matrix_(Eigen::Map<const double*>(temp_matrix_vec.data(), 3, 4));
    // camera_matrix_ = Eigen::Map<Eigen::Matrix<double, 3, 4> >(temp_matrix_vec.data());
    temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("rotation_matrix");
    rotation_matrix_.resize(3,3);
    rotation_matrix_ << (double)temp_matrix_vec[0], (double)temp_matrix_vec[1], (double)temp_matrix_vec[2],
                        (double)temp_matrix_vec[3], (double)temp_matrix_vec[4], (double)temp_matrix_vec[5],
                        (double)temp_matrix_vec[6], (double)temp_matrix_vec[7], (double)temp_matrix_vec[8];
    // rotation_matrix_(Eigen::Map<const double*>(temp_matrix_vec.data(), 3, 3));
    // rotation_matrix_ = Eigen::Map<Eigen::Matrix<double, 3, 3> >(temp_matrix_vec.data());
    temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("translation_matrix");
    translation_matrix_.resize(3,1);
    translation_matrix_ << (double)temp_matrix_vec[0],
                           (double)temp_matrix_vec[1],
                           (double)temp_matrix_vec[2];
    // translation_matrix_(Eigen::Map<const double*>(temp_matrix_vec.data(), 3, 1));
    // translation_matrix_ = Eigen::Map<Eigen::Matrix<double, 3, 1> >(temp_matrix_vec.data());
    // Create transformation matrix
    transformation_matrix_.resize(4,4);
    transformation_matrix_ <<
        rotation_matrix_(0), rotation_matrix_(3), rotation_matrix_(6), translation_matrix_(0),
        rotation_matrix_(1), rotation_matrix_(4), rotation_matrix_(7), translation_matrix_(1),
        rotation_matrix_(2), rotation_matrix_(5), rotation_matrix_(8), translation_matrix_(2),
        0                  , 0                  , 0                  , 1;


    // Create publisher for detected_cones_topic_ with queue size of 5
    detected_cones_publisher_ = 
        this->create_publisher<pathplanner_msgs::msg::ConeArray>(detected_cones_topic_, 5);


#ifndef NDEBUG
    // Create publishers for debug
    detection_frames_publisher_ = 
        this->create_publisher<sensor_msgs::msg::Image>(detection_frames_topic_, 5);
    cone_markers_publisher_ = 
        this->create_publisher<visualization_msgs::msg::MarkerArray>(cone_markers_topic_, 5);
    fusion_point_cloud_publisher_ = 
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_detection/fusion_point_cloud", 5);
    range_img_publisher_ = 
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_detection/range_image", 5);
    point_cloud_on_img_publisher_ = 
        this->create_publisher<sensor_msgs::msg::Image>("cone_detection/point_cloud_on_img", 5);
#endif
}

void ConeDetection::cone_detection_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg
) {
    // Convert ROS image_msg to CV image
    cv_bridge::CvImagePtr cv_image_ptr;
    try {
        // toCvCopy and toCvShare difference, encodings, etc.:
        // https://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
        cv_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Detect cones
    std::vector<std::pair<std::string, cv::Rect>> detected_cones = 
        camera_cones_detect(cv_image_ptr);
    // Merge camera and lidar data and return closest points for each cone
    std::vector<std::pair<std::string, pcl::PointXYZ>> cone_positions = 
        lidar_camera_fusion(point_cloud_msg, image_msg, detected_cones);
    
#ifndef NDEBUG
    int cone_marker_id_ = 0;
    auto cone_markers_array_msg = visualization_msgs::msg::MarkerArray();
#endif
    // Creating message with detected cones for path planner
    auto cone_array_msg = pathplanner_msgs::msg::ConeArray();

    for (const auto& pair : cone_positions) {
        pcl::PointXYZ cone_closest_point = pair.second;

        // Creating cone msg
        auto cone_msg = pathplanner_msgs::msg::Cone();
        cone_msg.x = cone_closest_point.x;
        cone_msg.y = cone_closest_point.y;
        /**
         * Blue - Outer - 0
         * Yellow - Inner - 1
         * @todo Change the logic for detecting color. Yellow is not
         * always inner. Look at the Skid Pad.
         */
        if (pair.first == "blue_cone") {
            cone_msg.side = 0;
        } else if(pair.first == "yellow_cone") {
            cone_msg.side = 1;
        } else {
            continue;
        }

        // Add a cone to the cone_array in the message
        cone_array_msg.cone_array.push_back(cone_msg);

#ifndef NDEBUG
        // https://github.com/autowarefoundation/autoware.universe/blob/1e829e2948222e8b9c60a2af3bf3008ffbe2ec6d/common/tier4_autoware_utils/include/tier4_autoware_utils/geometry/geometry.hpp#L42
        geometry_msgs::msg::Point point;
        point.x = cone_msg.x;
        point.y = cone_msg.y;
        point.z = -0.3;
        geometry_msgs::msg::Quaternion quaternion;
        quaternion.x = 0.0;
        quaternion.y = 0.0;
        quaternion.z = 0.0;
        quaternion.w = 1.0;
        geometry_msgs::msg::Pose pose;
        pose.position = point;
        pose.orientation = quaternion;
        geometry_msgs::msg::Vector3 vector3;
        vector3.x = 0.2;
        vector3.y = 0.2;
        vector3.z = 0.2;

        // Create cone marker message
        auto cone_marker_msg = visualization_msgs::msg::Marker();
        cone_marker_msg.header.stamp = this->now();
        cone_marker_msg.header.frame_id = "/fsds/Lidar1";
        cone_marker_msg.ns = "basic_shapes";
        cone_marker_msg.id = cone_marker_id_++;
        cone_marker_msg.type = visualization_msgs::msg::Marker::CUBE;
        cone_marker_msg.action = visualization_msgs::msg::Marker::ADD;
        cone_marker_msg.pose = pose;
        cone_marker_msg.scale = vector3;
        if (cone_msg.side == 0) {
            cone_marker_msg.color.r = 0.0;
            cone_marker_msg.color.g = 0.0;
            cone_marker_msg.color.b = 1.0;
        } else if (cone_msg.side == 1) {
            cone_marker_msg.color.r = 0.96;
            cone_marker_msg.color.g = 0.94;
            cone_marker_msg.color.b = 0.06;
        }
        cone_marker_msg.color.a = 1.0;
        cone_marker_msg.lifetime = 
            rclcpp::Duration(std::chrono::milliseconds(500));
        cone_marker_msg.frame_locked = false;
        // Push cone marker to cone markers array
        cone_markers_array_msg.markers.push_back(cone_marker_msg);
#endif
    }

#ifndef NDEBUG
    // Convert CV image to ROS msg
    std::shared_ptr<sensor_msgs::msg::Image> edited_img_msg = 
        cv_image_ptr->toImageMsg();
    // Create output msg and assigning values from edited msg pointer to it 
    sensor_msgs::msg::Image output_img_msg;
    output_img_msg.header = edited_img_msg->header;
    output_img_msg.height = edited_img_msg->height;
    output_img_msg.width = edited_img_msg->width;
    output_img_msg.encoding = edited_img_msg->encoding;
    output_img_msg.is_bigendian = edited_img_msg->is_bigendian;
    output_img_msg.step = edited_img_msg->step;
    output_img_msg.data = edited_img_msg->data;

    // Publish image message with detected cones (labels and boxes)
    detection_frames_publisher_->publish(output_img_msg);
    // Publish cone markers for visualization
    cone_markers_publisher_->publish(cone_markers_array_msg);
#endif
    // Publish message with detected cones (positions and sides)
    detected_cones_publisher_->publish(cone_array_msg);
}

std::vector<std::pair<std::string, cv::Rect>> ConeDetection::camera_cones_detect(
    cv_bridge::CvImagePtr cv_image_ptr
) {
    std::vector<std::pair<std::string, cv::Rect>> detected_cones;
    // Detecting cones with model
    std::vector<ModelResult> results = model_->detect(cv_image_ptr->image);

    // Creating pairs with cone class name and cone box in the image
    for (const auto& result : results) {
        detected_cones.push_back(std::make_pair(
            std::string(model_->get_class_by_id(result.class_id)),
            result.box
        ));
#ifndef NDEBUG
        // Choose the color
        cv::Scalar color(0, 256, 0);

        // Highlight the box with the color
        cv::rectangle(cv_image_ptr->image, result.box, color, 3);

        // Calculating probability from confidence
        float confidence = floor(100 * result.confidence) / 100;
        // Creating label with class name and probability
        std::string label = model_->get_class_by_id(result.class_id) + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        // Creating field for the label on the image
        cv::rectangle(
            cv_image_ptr->image,
            cv::Point(result.box.x, result.box.y - 25),
            cv::Point(result.box.x + label.length() * 10, result.box.y),
            color,
            cv::FILLED
        );

        // Adding label to the image
        cv::putText(
            cv_image_ptr->image,
            label,
            cv::Point(result.box.x, result.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            2
        );

        // Mark the "center" of each cone on the CV image
        int center_x = result.box.x + result.box.width / 2;
        int center_y = result.box.y + result.box.height / 2;
        cv::circle(
            cv_image_ptr->image,
            cv::Point(center_x, center_y),
            5,
            color,
            -1
        );
#endif
    }

    return detected_cones;
}

std::vector<std::pair<std::string, pcl::PointXYZ>> ConeDetection::lidar_camera_fusion(        
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
    const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
) {
    // The closest point to each cone
    std::vector<std::pair<std::string, pcl::PointXYZ>> closest_points;


    // Convert ROS image_msg to CV image
    cv_bridge::CvImagePtr cv_image_ptr;
    try {
        // toCvCopy and toCvShare difference, encodings, etc.:
        // https://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
        cv_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        // Return empty vector
        return closest_points;
    }


    // Convert ROS point_cloud_msg to PCL-compatible format
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>
    );
    // From deprecated pcl::fromROSMsg
    // https://docs.ros.org/en/hydro/api/pcl_conversions/html/pcl__conversions_8h_source.html#l00517
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*point_cloud_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, *point_cloud);

    // Check if the point cloud empty
    if (point_cloud->empty()) {
        RCLCPP_ERROR(this->get_logger(), "lidar_camera_fusion: Input point cloud is empty!");
        // Return empty vector
        return closest_points;
    }

    // Pre-process the point cloud
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*point_cloud, *point_cloud, indices);

    // Filter points based on distance
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZ>
    );
    for (const auto& point : point_cloud->points) {
        double distance = std::sqrt(point.x * point.x + point.y * point.y);

        if (distance >= params_.min_len && distance <= params_.max_len)
            point_cloud_filtered->push_back(point);
    }

    // Check if the filtered point cloud empty
    if (point_cloud_filtered->empty()) {
        RCLCPP_ERROR(this->get_logger(), "lidar_camera_fusion: Filtered point cloud is empty!");
        // Return empty vector
        return closest_points;
    }


    Eigen::MatrixXf lidar_camera(3, 1);
    Eigen::MatrixXf point_cloud_matrix(4, 1);
    uint px_data, py_data;

    //auto start = std::chrono::high_resolution_clock::now();

    // Store closest point and min distance for each cone
    std::vector<std::pair<pcl::PointXYZ, double>> cone_closest_points(detected_cones.size(),
        {pcl::PointXYZ(), std::numeric_limits<double>::max()});

    // Loop over all the points in the filtered point cloud
    for (const auto& point : point_cloud_filtered->points) {
        // Transform point from LiDAR to Camera
        point_cloud_matrix << -point.y, -point.z, point.x, 1.0;
        lidar_camera = camera_matrix_ * (transformation_matrix_ * point_cloud_matrix);

        // Project 3D point to image plane
        px_data = static_cast<int>(lidar_camera(0, 0) / lidar_camera(2, 0));
        py_data = static_cast<int>(lidar_camera(1, 0) / lidar_camera(2, 0));

        // Skip points outside of image bounds
        if (px_data >= image_msg->width || py_data >= image_msg->height)
            continue;
    
        // Iterate over each detected cone and find the closest point
        for (size_t idx = 0; idx < detected_cones.size(); ++idx) {
            const auto& pair = detected_cones[idx];
            // Get the center of each cone box (cv::Rect)
            int center_x = pair.second.x + pair.second.width / 2;
            int center_y = pair.second.y + pair.second.height / 2;

            // Calculate the distance to the center of the cone
            double distance_to_center = std::sqrt(
                std::pow(px_data - center_x, 2) + std::pow(py_data - center_y, 2)
            );

            // Find the closest point
            if (distance_to_center < cone_closest_points[idx].second) {
                cone_closest_points[idx] = {point, distance_to_center};
            }
        }
    }

    // Write closest points for return
    for (size_t i = 0; i < cone_closest_points.size(); ++i) {
        const auto& closest_point = cone_closest_points[i].first;
        double distance = std::sqrt(closest_point.x * closest_point.x + closest_point.y * closest_point.y);

        if (distance < (params_.max_len - 4.0)) {
            std::string cone_id = detected_cones[i].first;
            closest_points.push_back(std::make_pair(cone_id, closest_point));
        }
    }


    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Execution time: " << duration.count() << " microseconds  size: " << closest_points.size() << std::endl;

    return closest_points;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetection)
