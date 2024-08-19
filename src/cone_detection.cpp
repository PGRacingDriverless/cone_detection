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


    // TESTING ONLY VISUALIZATION
    test_visualization_ = this->declare_parameter<bool>("test_visualization");
    if(test_visualization_)
    {
        detection_frames_publisher_ =
            this->create_publisher<sensor_msgs::msg::Image>(detection_frames_topic_, 5);
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
    std::vector<cv::Rect> detected_cones = camera_cones_detect(cv_image_ptr);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*point_cloud_msg, *cloud);

    int img_width = cv_image_ptr->image.cols;
    int img_height = cv_image_ptr->image.rows;

    for (const auto& rect : detected_cones) {
        std::shared_ptr<pathplanner_msgs::msg::Cone> cone;
        int center_x = rect.x + rect.width / 2;
        int center_y = rect.y + rect.height / 2;

        float normalized_x = static_cast<float>(center_x) / img_width;
        float normalized_y = static_cast<float>(center_y) / img_height;

        pcl::PointXYZ closest_point;
        float min_dist = std::numeric_limits<float>::max();

        for (const auto& point : cloud->points) {
            float dist = std::sqrt(std::pow(point.x - normalized_x * img_width, 2) +
                                   std::pow(point.y - normalized_y * img_height, 2));

            if (dist < min_dist) {
                min_dist = dist;
                closest_point = point;

                auto cone_detail = pathplanner_msgs::msg::Cone();
                cone_detail.x = closest_point.x;
                cone_detail.y = closest_point.y;
                cone_detail.side = 0;
                cones_message.cone_array.push_back(cone_detail);
                // YELLOW - INNER = 1 
                // BLUE - OUTER = 0
            }
        }

        if(test_visualization_)
            cv::circle(cv_image_ptr->image, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 0), -1);
    }

    // Publish detected cones
    detected_cones_pub_->publish(cones_message);


    if(test_visualization_)
    { 
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
    }
}

std::vector<cv::Rect> ConeDetection::camera_cones_detect(cv_bridge::CvImagePtr cv_image_ptr)
{
    std::vector<cv::Rect> detected_cones;
    std::vector<ModelResult> res = model_->detect(cv_image_ptr->image);

    for (auto& r : res)
    {
        cv::Scalar color(0, 256, 0);

        detected_cones.push_back(r.box);
        cv::rectangle(cv_image_ptr->image, r.box, color, 3);

        float confidence = floor(100 * r.confidence) / 100;
        std::string label = model_->get_class_by_id(r.class_id) + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        cv::rectangle(
            cv_image_ptr->image,
            cv::Point(r.box.x, r.box.y - 25),
            cv::Point(r.box.x + label.length() * 15, r.box.y),
            color,
            cv::FILLED
        );

        cv::putText(
            cv_image_ptr->image,
            label,
            cv::Point(r.box.x, r.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );
    }

    return detected_cones;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetection)
