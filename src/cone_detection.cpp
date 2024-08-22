#include "cone_detection/cone_detection.hpp"

using namespace message_filters;
using namespace std::chrono_literals;

ConeDetection::ConeDetection(const rclcpp::NodeOptions &node_options) 
: Node("cone_detection", node_options)
{
    lidar_points_topic_ = this->declare_parameter<std::string>("lidar_points_topic");
    camera_image_topic_ = this->declare_parameter<std::string>("camera_image_topic");

    frame_id_ = this->declare_parameter<std::string>("frame_id");

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
    const double iou_threshold =
        this->declare_parameter<double>("iou_threshold");
    const double rect_confidence_threshold =
        this->declare_parameter<double>("rect_confidence_threshold");
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


    // DEGUB MODE
    debug_mode_ = this->declare_parameter<bool>("debug_mode");
    if(debug_mode_)
    {
        detection_frames_publisher_ =
            this->create_publisher<sensor_msgs::msg::Image>(detection_frames_topic_, 5);
        markers_cones_publisher_ = 
            this->create_publisher<visualization_msgs::msg::MarkerArray>(markers_cones_topic_, 5);
        marker_id_ = 0;
    }

    // Calibration matrix
    camera_to_lidar_ << 0.999, 0.001, 0.012, 0.1,
                        -0.001, 0.999, -0.005, 0.2,
                        -0.012, 0.005, 0.999, 0.3,
                        0.0, 0.0, 0.0, 1.0;
}


void ConeDetection::cone_detection_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg)
{
    // Markers for publish in debug mode
    auto cones_markers_array = visualization_msgs::msg::MarkerArray();

    // Detected cones to publish for path_planning
    auto cones_message = pathplanner_msgs::msg::ConeArray();

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

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*point_cloud_msg, *cloud);

    int img_width = 1280;//cv_image_ptr->image.cols;
    int img_height = 1280;//cv_image_ptr->image.rows;

    // Matrix calculated based on camera and lidar data using calculate_p_matrix.py
    cv::Mat projection_matrix = (cv::Mat_<double>(3, 4) << 
        640, 0, 640, -704,
        0, 640, 640, 256,
        0, 0, 1, 0.4);

    cv::Mat projection_matrix_inv;
    cv::invert(projection_matrix, projection_matrix_inv, cv::DECOMP_SVD);

    for (const auto& pair : detected_cones) {
        int center_x = pair.second.x + pair.second.width / 2;
        int center_y = pair.second.y + pair.second.height / 2;

        // Normalize coordinates
        float normalized_x = static_cast<float>(center_x) / img_width;
        float normalized_y = static_cast<float>(center_y) / img_height;

        // Convert normalized coordinates to homogeneous coordinates
        cv::Mat image_point = (cv::Mat_<double>(3, 1) << normalized_x * img_width, normalized_y * img_height, 1.0);

        // Project point into 3D space using the inverse of the projection matrix
        cv::Mat camera_point = projection_matrix_inv * image_point;

        // Convert the resulting point to 3D coordinates
        pcl::PointXYZ target_point(camera_point.at<double>(0), camera_point.at<double>(1), camera_point.at<double>(2));

        pcl::PointXYZ closest_point;
        float min_dist = std::numeric_limits<float>::max();

        for (const auto& point : cloud->points) {
            float dist = std::sqrt(std::pow(point.x - target_point.x, 2) +
                                   std::pow(point.y - target_point.y, 2) +
                                   std::pow(point.z - target_point.z, 2));

            if (dist < min_dist) {
                min_dist = dist;
                closest_point = point;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Target Point: (%f, %f, %f)", target_point.x, target_point.y, target_point.z);
        RCLCPP_INFO(this->get_logger(), "Closest Point: (%f, %f, %f)", closest_point.x, closest_point.y, closest_point.z);

        auto cone_detail = pathplanner_msgs::msg::Cone();
        cone_detail.x = closest_point.x;
        cone_detail.y = closest_point.y;

        // YELLOW - INNER = 1 
        // BLUE - OUTER = 0
        if(pair.first == "yellow_cone")
            cone_detail.side = 1;
        else if(pair.first == "blue_cone")
            cone_detail.side = 0;

        cones_message.cone_array.push_back(cone_detail);

        // DEBUG MODE
        if(debug_mode_) {
            cv::circle(cv_image_ptr->image, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 0), -1);

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
        RCLCPP_INFO(this->get_logger(), "%d markers published", marker_id_);
        RCLCPP_INFO(this->get_logger(), "%d detected cones", detected_cones.size());
    }
}

std::vector<std::pair<std::string, cv::Rect>> ConeDetection::camera_cones_detect(cv_bridge::CvImagePtr cv_image_ptr)
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

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetection)
