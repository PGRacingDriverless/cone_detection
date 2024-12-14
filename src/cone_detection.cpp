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
    lidar_camera_qos.depth = 2;
    // Best effort reliability (i.e. no guarantees to deliver samples)
    lidar_camera_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
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
        message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>(3),
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
        (int)this->declare_parameter<int>("width"),
        (int)this->declare_parameter<int>("height")
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
    params_.max_ang_w = this->declare_parameter<float>("max_ang_w");
    params_.max_ang_h = this->declare_parameter<float>("max_ang_h");
    params_.interp_value = this->declare_parameter<float>("interp_value");
    params_.max_var = this->declare_parameter<float>("max_var");

    // Read and set matrices for lidar-camera fusion
    std::vector<double> temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("camera_matrix");
    camera_matrix_.resize(3,4);
    camera_matrix_ << (double)temp_matrix_vec[0], (double)temp_matrix_vec[1], (double)temp_matrix_vec[2], (double)temp_matrix_vec[3],
                      (double)temp_matrix_vec[4], (double)temp_matrix_vec[5], (double)temp_matrix_vec[6], (double)temp_matrix_vec[7],
                      (double)temp_matrix_vec[8], (double)temp_matrix_vec[9], (double)temp_matrix_vec[10], (double)temp_matrix_vec[11];
    temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("rotation_matrix");
    rotation_matrix_.resize(3,3);
    rotation_matrix_ << (double)temp_matrix_vec[0], (double)temp_matrix_vec[1], (double)temp_matrix_vec[2],
                        (double)temp_matrix_vec[3], (double)temp_matrix_vec[4], (double)temp_matrix_vec[5],
                        (double)temp_matrix_vec[6], (double)temp_matrix_vec[7], (double)temp_matrix_vec[8];
    temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("translation_matrix");
    translation_matrix_.resize(3,1);
    translation_matrix_ << (double)temp_matrix_vec[0],
                           (double)temp_matrix_vec[1],
                           (double)temp_matrix_vec[2];
    
    // Create transformation matrix
    transformation_matrix_.resize(4,4);
    transformation_matrix_ <<
        rotation_matrix_(0), rotation_matrix_(3), rotation_matrix_(6), translation_matrix_(0),
        rotation_matrix_(1), rotation_matrix_(4), rotation_matrix_(7), translation_matrix_(1),
        rotation_matrix_(2), rotation_matrix_(5), rotation_matrix_(8), translation_matrix_(2),
        0                  , 0                  , 0                  , 1;


    // Calculate height threshold in pixels for distance filtering
    // distance = height_real * focal_length / height_pixel
    float cone_height = 0.35;
    height_in_pixels = static_cast<int>(std::ceil(
        (cone_height * camera_matrix_.coeff(1, 1)) / params_.max_len
    ));


    // Create publisher for detected cones with queue size of 5
    detected_cones_publisher_ = 
        this->create_publisher<common_msgs::msg::ConeArray>("cone_array", 5);

#ifndef NDEBUG
    // Create publishers for debug
    detection_frames_publisher_ = 
        this->create_publisher<sensor_msgs::msg::Image>("cone_detection/image_detected", 5);
    cone_markers_publisher_ = 
        this->create_publisher<visualization_msgs::msg::MarkerArray>("cone_detection/cone_markers", 5);
    filtered_point_cloud_publisher_ = 
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_detection/filtered_point_cloud", 5);
    interp_point_cloud_publisher_ = 
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_detection/interp_point_cloud", 5);
    range_img_publisher_ = 
        this->create_publisher<sensor_msgs::msg::Image>("cone_detection/range_image", 5);
    range_img_cloud_publisher_ = 
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_detection/range_image_cloud", 5);
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
    // Filter detected cones
    filter_by_px_height(detected_cones);
    // Merge camera and lidar data and return closest points for each cone
    std::vector<std::pair<std::string, pcl::PointXYZ>> cone_positions = 
        lidar_camera_fusion(point_cloud_msg, image_msg, detected_cones);

#ifndef NDEBUG
    int cone_marker_id_ = 0;
    visualization_msgs::msg::MarkerArray cone_markers_array_msg;
#endif
    // Creating message with detected cones for path planner
    common_msgs::msg::ConeArray cone_array_msg;

    for (const auto& pair : cone_positions) {
        pcl::PointXYZ cone_closest_point = pair.second;

        // Creating cone msg
        common_msgs::msg::Cone cone_msg;
        cone_msg.x = cone_closest_point.x;
        cone_msg.y = cone_closest_point.y;

        /**
         * @todo Change the logic for detecting color. Yellow is not
         * always inner. Look at the Skid Pad.
         */
        if (pair.first == "blue_cone") {
            cone_msg.side = 1;
        } else if(pair.first == "yellow_cone") {
            cone_msg.side = 0;
        } else {
            continue;
        }

        // Add a cone to the cone_array in the message
        cone_array_msg.cone_array.push_back(cone_msg);

#ifndef NDEBUG
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
        visualization_msgs::msg::Marker cone_marker_msg;
        cone_marker_msg.header.stamp = this->now();
        cone_marker_msg.header.frame_id = "/fsds/Lidar1";
        cone_marker_msg.ns = "basic_shapes";
        cone_marker_msg.id = cone_marker_id_++;
        cone_marker_msg.type = visualization_msgs::msg::Marker::CUBE;
        cone_marker_msg.action = visualization_msgs::msg::Marker::ADD;
        cone_marker_msg.pose = pose;
        cone_marker_msg.scale = vector3;
        if (cone_msg.side == 1) {
            cone_marker_msg.color.r = 0.0;
            cone_marker_msg.color.g = 0.0;
            cone_marker_msg.color.b = 1.0;
        } else if (cone_msg.side == 0) {
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
    // Convert CV image to ROS msg and publish image message with
    // detected cones (labels and boxes)
    detection_frames_publisher_->publish(*cv_image_ptr->toImageMsg());
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

void ConeDetection::filter_by_px_height(
    std::vector<std::pair<std::string, cv::Rect>> &detected_cones
) {
    std::vector<std::pair<std::string, cv::Rect>> filtered_cones;
    
    for(const auto& cone : detected_cones) {
        if(cone.second.height >= height_in_pixels) {
            filtered_cones.push_back(cone);
        }
    }

    detected_cones = filtered_cones;
}

std::vector<std::pair<std::string, pcl::PointXYZ>> ConeDetection::lidar_camera_fusion(        
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
    const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
) {
    // The closest point to each cone
    std::vector<std::pair<std::string, pcl::PointXYZ>> closest_points;

    std::vector<ConeInfo> cone_infos;
    for (const auto& cone : detected_cones) {
        ConeInfo info;
        info.id = cone.first;
        info.bbox = cone.second;
        cone_infos.push_back(info);
    }

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
    pcl::fromROSMsg(*point_cloud_msg, *point_cloud);

    // Check if the point cloud empty
    if (point_cloud->empty()) {
        RCLCPP_ERROR(this->get_logger(), "lidar_camera_fusion: Input point cloud is empty!");
        // Return empty vector
        return closest_points;
    }

    // Filtering
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud = distance_filter(point_cloud);
    // filtered_point_cloud = ground_removal_filter(filtered_point_cloud);

    // Check if the filtered point cloud empty
    if (filtered_point_cloud->empty()) {
        RCLCPP_ERROR(this->get_logger(), "lidar_camera_fusion: Filtered point cloud is empty!");
        // Return empty vector
        return closest_points;
    }


    pcl::RangeImageSpherical::Ptr range_img(new pcl::RangeImageSpherical);
    // Range image will be represented in the lidar coordinate system.
    pcl::RangeImage::CoordinateFrame coordinate_frame = 
        pcl::RangeImage::LASER_FRAME;

    Eigen::Affine3f sensor_pose = 
        (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    range_img->pcl::RangeImage::createFromPointCloud(
        *filtered_point_cloud,
        pcl::deg2rad(params_.ang_res_x),
        pcl::deg2rad(params_.ang_res_y),
        pcl::deg2rad(params_.max_ang_w),
        pcl::deg2rad(params_.max_ang_h),
        sensor_pose,
        coordinate_frame,
        0.0f,
        0.0f,
        0
    );


    // Interpolation
    arma::mat range_matrix_interp, height_matrix_interp;
    interp_range_img(range_img, range_matrix_interp, height_matrix_interp);

    // Filtering
    arma::mat range_matrix_out = variance_filter(range_matrix_interp);


    // Point cloud reconstruction from range image
    pcl::PointCloud<pcl::PointXYZ>::Ptr interp_point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>
    );
    interp_point_cloud->width = range_matrix_interp.n_cols;
    interp_point_cloud->height = range_matrix_interp.n_rows;
    interp_point_cloud->is_dense = false;
    interp_point_cloud->points.resize(
        interp_point_cloud->width * interp_point_cloud->height
    );
    // Range image to point cloud
    int num_pc = 0;

    for(int i = 0; i < range_matrix_interp.n_rows - params_.interp_value; ++i) {
        for(int j = 0; j < (int)range_matrix_interp.n_cols; ++j) {
            float ang = params_.min_fov + 
                ((params_.max_fov - params_.min_fov) * j) / (range_matrix_interp.n_cols);

            if(!(range_matrix_out(i, j) == 0)) {
                float pc_modulo = range_matrix_out(i, j);

                if (pow(pc_modulo, 2) < pow(height_matrix_interp(i, j), 2)) 
                    continue;
                
                float sqrt_component = sqrt(pow(pc_modulo, 2) - pow(height_matrix_interp(i, j), 2));
                float pc_x = sqrt_component * sin(ang);
                float pc_y = sqrt_component * cos(ang);

                float ang_x_lidar = 0.6 * M_PI / 180.0;
                //matrix  transformation between lidar and range image. It rotates the angles that it has of error with respect to the ground
                Eigen::MatrixXf Lidar_matrix(3,3);
                Eigen::MatrixXf result(3,1);
                Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                                0               , 1, 0,
                               -sin(ang_x_lidar), 0, cos(ang_x_lidar);

                result <<   pc_x,
                            pc_y,
                            height_matrix_interp(i, j);

                // range image to point cloud X-axis rotation for correction
                result = Lidar_matrix * result;

                interp_point_cloud->points[num_pc].x = result(0);
                interp_point_cloud->points[num_pc].y = result(1);
                interp_point_cloud->points[num_pc].z = result(2);

                num_pc++;
            }
        }
    }

    /** @todo temp */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_no_ground = ground_removal_filter(interp_point_cloud);

    // cloud_no_ground = sor_filter(cloud_no_ground);


    Eigen::MatrixXf lidar_camera(3, 1);
    Eigen::MatrixXf point_cloud_matrix(4, 1);
    uint px_data, py_data;

    // auto start = std::chrono::high_resolution_clock::now();

    // Store closest point and min distance for each cone
    std::vector<std::pair<pcl::PointXYZ, double>> cone_closest_points(
        detected_cones.size(),
        { pcl::PointXYZ(), std::numeric_limits<double>::max() }
    );

    // Loop over all the points in the interpolated point cloud
    for (const auto& point : cloud_no_ground->points) {
        // Transform point from LiDAR to Camera
        point_cloud_matrix << -point.y, -point.z, point.x, 1.0;
        lidar_camera = camera_matrix_ * (transformation_matrix_ * point_cloud_matrix);

        // Project 3D point to image plane
        px_data = static_cast<int>(lidar_camera(0, 0) / lidar_camera(2, 0));
        py_data = static_cast<int>(lidar_camera(1, 0) / lidar_camera(2, 0));

        // Skip points outside of image bounds
        if (px_data >= image_msg->width || py_data >= image_msg->height)
            continue;
        
#ifndef NDEBUG
        int x_color = (int)(255 * ((point.x) / params_.max_len));
        int z_color = (int)(255 * ((point.x) / 10.0));

        if(z_color > 255) {
            z_color = 255;
        }

        cv::circle(
            cv_image_ptr->image,
            cv::Point(px_data, py_data),
            1,
            CV_RGB(255 - x_color, (int)(z_color), x_color),
            cv::FILLED
        );
#endif
    
        for (auto& cone : cone_infos) {
            cv::Rect search_area = cone.bbox;
            search_area.x -= search_area.width / 2;
            search_area.y -= search_area.height / 2;
            search_area.width *= 2;
            search_area.height *= 2;

            if (search_area.contains(cv::Point(px_data, py_data))) {
                cone.associated_points.push_back(point);
            }
        }
    }

    // for (auto& cone : cone_infos) {
    //     if (cone.associated_points.empty()) continue;

    //     pcl::PointXYZ sum(0, 0, 0);
    //     for (const auto& p : cone.associated_points) {
    //         sum.x += p.x;
    //         sum.y += p.y;
    //         sum.z += p.z;
    //     }
    //     cone.average_point.x = sum.x / cone.associated_points.size();
    //     cone.average_point.y = sum.y / cone.associated_points.size();
    //     cone.average_point.z = sum.z / cone.associated_points.size();
    //     cone.confidence = std::min(1.0, cone.associated_points.size() / 100.0);

    //     if (cone.confidence > 0.5) {
    //         closest_points.push_back(std::make_pair(cone.id, cone.average_point));
    //     }
    // }

    for (auto& cone : cone_infos) {
        if (cone.associated_points.empty()) continue;

        pcl::PointXYZ sum(0, 0, 0);
        double min_distance = std::numeric_limits<double>::max();
        pcl::PointXYZ closest_point;

        for (const auto& p : cone.associated_points) {
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;

            double distance = std::sqrt(
                std::pow(p.x - cone.average_point.x, 2) +
                std::pow(p.y - cone.average_point.y, 2) +
                std::pow(p.z - cone.average_point.z, 2)
            );

            if (distance < min_distance) {
                min_distance = distance;
                closest_point = p;
            }
        }

        cone.average_point.x = sum.x / cone.associated_points.size();
        cone.average_point.y = sum.y / cone.associated_points.size();
        cone.average_point.z = sum.z / cone.associated_points.size();

        closest_points.push_back(std::make_pair(cone.id, closest_point));
    }

    for (size_t i = 0; i < closest_points.size(); ++i) {
        for (size_t j = i + 1; j < closest_points.size(); ++j) {
            double distance = std::sqrt(
                std::pow(closest_points[i].second.x - closest_points[j].second.x, 2) +
                std::pow(closest_points[i].second.y - closest_points[j].second.y, 2) +
                std::pow(closest_points[i].second.z - closest_points[j].second.z, 2)
            );

            if (distance < 2.0) {
                closest_points.erase(closest_points.begin() + j);
                --i;
                break;
            }
        }
    }


    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Execution time: " << duration.count() << " microseconds  size: " << closest_points.size() << std::endl;

#ifndef NDEBUG
    sensor_msgs::msg::PointCloud2 temp_cloud;

    pcl::toROSMsg(*filtered_point_cloud, temp_cloud);
    temp_cloud.header = point_cloud_msg->header;
    filtered_point_cloud_publisher_->publish(temp_cloud);

    temp_cloud.data.clear();
    pcl::toROSMsg(*cloud_no_ground, temp_cloud);
    temp_cloud.header = point_cloud_msg->header;
    interp_point_cloud_publisher_->publish(temp_cloud);

    // cv::Mat range_img_cv(range_img->height, range_img->width, CV_8UC1);
    // for(int i = 0; i < range_img->height; ++i) {
    //     for (int j = 0; j < range_img->width; ++j) {
    //         float range = range_img->getPoint(i, j).range;
    //         range_img_cv.at<float>(i, j) = range;
    //         // if(std::isfinite(range)) {
    //         //     range_img_cv.at<float>(y, x) = range;
    //         // } else {
    //         //     range_img_cv.at<float>(y, x) = 0.0f;
    //         // }
    //     }
    // }
    // cv::normalize(range_img_cv, range_img_cv, 0, 255, cv::NORM_MINMAX);
    // range_img_cv.convertTo(range_img_cv, CV_8UC1);
    // sensor_msgs::msg::Image::SharedPtr range_img_msg = 
    //     cv_bridge::CvImage(image_msg->header, "mono8", range_img_cv).toImageMsg();
    // range_img_publisher_->publish(*range_img_msg);

    temp_cloud.data.clear();
    // range_img_cloud_publisher_->publish(temp_cloud);

    point_cloud_on_img_publisher_->publish(*cv_image_ptr->toImageMsg());
#endif

    return closest_points;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ConeDetection::distance_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
) {
    // Create a new cloud to store the filtered points
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Remove NaN points from the input point cloud
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*point_cloud, *filtered_point_cloud, indices);

    // Filter points based on distance (keeping points within a specified range)
    pcl::PointCloud<pcl::PointXYZ>::Ptr distance_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : filtered_point_cloud->points) {
        // Calculate the distance from the origin (x, y plane)
        double distance = std::sqrt(point.x * point.x + point.y * point.y);
        // Keep points that are within the specified distance range
        if (distance >= params_.min_len && distance <= params_.max_len) {
            distance_filtered->push_back(point);
        }
    }

    return distance_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ConeDetection::ground_removal_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);

    // Create the SAC segmentation object for ground plane extraction
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    // Max distance from the plane for a point to be considered part of it
    seg.setDistanceThreshold(0.05);

    pcl::PointIndices::Ptr inlier_indices(new pcl::PointIndices);
    // Coefficients of the plane model
    pcl::ModelCoefficients::Ptr model_coefficients(new pcl::ModelCoefficients);

    seg.setInputCloud(point_cloud);
    // Apply segmentation to obtain the indices of the ground points (inliers)
    seg.segment(*inlier_indices, *model_coefficients);

    if (inlier_indices->indices.size() == 0) {
        PCL_ERROR("Ground plane not found.\n");
        return point_cloud;
    }

    // Extract the points that are not part of the ground (remove inliers)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(point_cloud);
    extract.setIndices(inlier_indices);
    // True to remove the ground points
    extract.setNegative(true);
    extract.filter(*cloud_no_ground);
    
    return cloud_no_ground;
}

void ConeDetection::interp_range_img(
    const pcl::RangeImageSpherical::Ptr &range_img,
    arma::mat &range_matrix_interp,
    arma::mat &height_matrix_interp
) {
    // Get the dimensions of the image
    int img_cols = range_img->width;
    int img_rows = range_img->height;

    // Range image (height values mapped onto the range image)
    arma::mat range_matrix, height_matrix;
    range_matrix.zeros(img_rows, img_cols);
    height_matrix.zeros(img_rows, img_cols);

    for (int i = 0; i < img_cols; ++i) {
        for (int j = 0; j < img_rows; ++j) {
            float range = range_img->getPoint(i, j).range;
            float height = range_img->getPoint(i, j).z;

            if (
                std::isinf(range) ||
                range < params_.min_len ||
                range > params_.max_len ||
                std::isnan(height)
            ) {
                continue;
            }

            range_matrix.at(j, i) = range;   
            height_matrix.at(j, i) = height;
        }
    }

    // Horizontal spacing of the range image (i.e., the column indices)
    arma::vec x_spacing = arma::regspace(1, range_matrix.n_cols);
    // Vertical spacing of the range image (i.e., the row indices)
    arma::vec y_spacing = arma::regspace(1, range_matrix.n_rows);

    // Magnify by approximately 2
    arma::vec x_spacing_scaled = 
        arma::regspace(x_spacing.min(), 1.0, x_spacing.max());
    //
    arma::vec y_spacing_scaled = arma::regspace(
        y_spacing.min(), 1.0 / params_.interp_value, y_spacing.max()
    );

    arma::interp2(
        x_spacing, y_spacing,
        range_matrix,
        x_spacing_scaled, y_spacing_scaled,
        range_matrix_interp, "lineal"
    );
    arma::interp2(
        x_spacing, y_spacing,
        height_matrix,
        x_spacing_scaled, y_spacing_scaled,
        height_matrix_interp, "lineal"
    );
}

arma::mat ConeDetection::variance_filter(
    const arma::mat &range_matrix_interp
) {
    arma::mat range_matrix_out = range_matrix_interp;

    // Filtering of elements interpolated with the background
    for(int i = 0; i < (int)range_matrix_interp.n_rows; ++i) {
        for(int j = 0; j < (int)range_matrix_interp.n_cols ; ++j) {
            if(range_matrix_interp(i, j) == 0) {
                if(i + params_.interp_value < range_matrix_interp.n_rows) {
                    for(int k = 1; k <= params_.interp_value; ++k) {
                        range_matrix_out(i + k, j) = 0;
                    }
                }
                if(i > params_.interp_value) {
                    for (int k = 1; k <= params_.interp_value; ++k) {
                        range_matrix_out(i - k, j) = 0;
                    }
                }
            }
        }
    }

    // Variance filtering
    for(int i = 0; i < ((range_matrix_interp.n_rows - 1) / params_.interp_value); ++i) {
        for(int j = 0; j < (int)range_matrix_interp.n_cols; ++j) {
            double avg = 0;
            double variance = 0;

            for(int k = 0; k < params_.interp_value ; ++k) {
                avg = avg + range_matrix_interp((i * params_.interp_value) + k, j);
            }
            
            avg = avg / params_.interp_value;

            for(int k = 0; k < params_.interp_value; ++k) {
                variance = variance + pow(
                    range_matrix_interp((i * params_.interp_value) + k, j) - avg,
                    2.0
                );
            }

            if(variance > params_.max_var) {
                for(int k = 0; k < params_.interp_value; ++k) {
                    range_matrix_out((i * params_.interp_value) + k, j) = 0;
                }
            }
        }
    }

    return range_matrix_out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ConeDetection::sor_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_outliers(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    // Number of neighboring points for outlier detection, neighborhood size for each point
    sor.setMeanK(32);
    // Standard deviation threshold, points farther away from the value will be removed
    sor.setStddevMulThresh(1.0);
    sor.setInputCloud(point_cloud);
    sor.filter(*cloud_no_outliers);

    return cloud_no_outliers;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetection)
