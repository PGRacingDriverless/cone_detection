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
    camera_lidar_fusion(point_cloud_msg, image_msg);

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

    int img_width = cv_image_ptr->image.cols;
    int img_height = cv_image_ptr->image.rows;

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
        //RCLCPP_INFO(this->get_logger(), "Target Point: (%f, %f, %f)", target_point.x, target_point.y, target_point.z);
        //RCLCPP_INFO(this->get_logger(), "Closest Point: (%f, %f, %f)", closest_point.x, closest_point.y, closest_point.z);

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
            //cones_markers_array.markers.push_back(marker);
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

void ConeDetection::camera_lidar_fusion(        
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg)
{
    float maxlen =100.0;       //maxima distancia del lidar
    float minlen = 0.2;     //minima distancia del lidar
    float max_FOV = 3.0;    
    float min_FOV = 0.4;    
    // parameters convertation from pc to image
    float angular_resolution_x = 0.069;
    float angular_resolution_y = 0.069;
    float max_angle_width= 100.0f;
    float max_angle_height = 100.0f;
    double max_var = 50.0; 
    bool f_pc = true; 
    float interpol_value = 20.0;
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
    pcl::RangeImage::Ptr rangeImage(new pcl::RangeImage);

    cv_bridge::CvImagePtr cv_ptr , color_pcl;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        color_pcl = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    //Conversion from sensor_msgs::PointCloud2 to pcl::PointCloud<T>
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*point_cloud_msg,pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2,*msg_pointCloud);

    // filter point cloud 
    if (msg_pointCloud == NULL) return;

    PointCloud::Ptr cloud_in (new PointCloud);
    //PointCloud::Ptr cloud_filter (new PointCloud);
    PointCloud::Ptr cloud_out (new PointCloud);


    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    for (int i = 0; i < (int) cloud_in->points.size(); i++)
    {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);     
        if(distance<minlen || distance>maxlen)
        continue;
        cloud_out->push_back(cloud_in->points[i]);
    }

    if (cloud_out->points.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Filtered point cloud is empty!");
        return;
    }

    // point cloud to image 
    Eigen::Affine3f sensorPose = Eigen::Affine3f::Identity();

    rangeImage->pcl::RangeImage::createFromPointCloud(
        *cloud_out, 
        pcl::deg2rad(angular_resolution_x), 
        pcl::deg2rad(angular_resolution_y),
        pcl::deg2rad(max_angle_width), 
        pcl::deg2rad(max_angle_height),
        sensorPose, 
        coordinate_frame, 
        0.0f, 
        0.0f, 
        0
    );
    
    if (rangeImage->points.empty()) {
        RCLCPP_ERROR(this->get_logger(), "RangeImage empty");
        return; 
    }

    cv::Mat range_image_cv(rangeImage->height, rangeImage->width, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < range_image_cv.rows; ++y) {
        for (int x = 0; x < range_image_cv.cols; ++x) {
            float range = rangeImage->getPoint(x, y).range;
            if (std::isfinite(range)) {
                range_image_cv.at<float>(y, x) = range;
            } else {
                range_image_cv.at<float>(y, x) = 0.0f;
            }
        }
    }

    cv::Mat range_image_normalized;
    cv::normalize(range_image_cv, range_image_normalized, 0, 255, cv::NORM_MINMAX);
    range_image_normalized.convertTo(range_image_normalized, CV_8UC1);

    sensor_msgs::msg::Image range_image_msg;
    range_image_msg.height = range_image_normalized.rows;
    range_image_msg.width = range_image_normalized.cols;
    range_image_msg.encoding = "mono8";
    range_image_msg.is_bigendian = false;
    range_image_msg.step = range_image_normalized.cols;
    range_image_msg.data.assign(range_image_normalized.datastart, range_image_normalized.dataend);

    pcOnimg_pub->publish(range_image_msg);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;


    arma::mat Z; 
    arma::mat Zz;

    Z.zeros(rows_img,cols_img);         
    Zz.zeros(rows_img,cols_img);       

    Eigen::MatrixXf ZZei (rows_img,cols_img);

    for (int i=0; i< cols_img; ++i) {
        for (int j=0; j<rows_img ; ++j) {
            float r =  rangeImage->getPoint(i, j).range;     
            float zz = rangeImage->getPoint(i, j).z; 

            if(std::isinf(r) || r<minlen || r>maxlen || std::isnan(zz)){
            continue;
            }             
            Z.at(j,i) = r;   
            Zz.at(j,i) = zz;
        }
    }

    // interpolation
    arma::vec X = arma::regspace(1, Z.n_cols);  // X = horizontal spacing
    arma::vec Y = arma::regspace(1, Z.n_rows);  // Y = vertical spacing 

    arma::vec XI = arma:: regspace(X.min(), 1.0, X.max()); // magnify by approx 2
    arma::vec YI = arma::regspace(Y.min(), 1.0/interpol_value, Y.max()); // 

    arma::mat ZI_near;  
    arma::mat ZI;
    arma::mat ZzI;

    arma::interp2(X, Y, Z, XI, YI, ZI,"lineal");  
    arma::interp2(X, Y, Zz, XI, YI, ZzI,"lineal");  
    // image reconstruction to 3D cloud
    PointCloud::Ptr point_cloud (new PointCloud);
    PointCloud::Ptr cloud (new PointCloud);
    point_cloud->width = ZI.n_cols; 
    point_cloud->height = ZI.n_rows;
    point_cloud->is_dense = false;
    point_cloud->points.resize (point_cloud->width * point_cloud->height);

    arma::mat Zout = ZI;

    // filtering of elements interpolated with the background
    for (uint i=0; i< ZI.n_rows; i+=1) {       
        for (uint j=0; j<ZI.n_cols ; j+=1) {             
            if((ZI(i,j)== 0 ))
            {
            if(i+interpol_value<ZI.n_rows)
                for (int k=1; k<= interpol_value; k+=1) 
                Zout(i+k,j)=0;
            if(i>interpol_value)
                for (int k=1; k<= interpol_value; k+=1) 
                Zout(i-k,j)=0;
            }
        }      
    }

    if (f_pc){    
    /// filtering by variance
        for (uint i=0; i< ((ZI.n_rows-1)/interpol_value); i+=1)      
            for (uint j=0; j<ZI.n_cols-5 ; j+=1) {
                double promedio = 0;
                double varianza = 0;
                for (uint k=0; k<interpol_value ; k+=1)
                    promedio = promedio+ZI((i*interpol_value)+k,j);

                promedio = promedio / interpol_value;    

                for (uint l = 0; l < interpol_value; l++) 
                    varianza = varianza + pow((ZI((i*interpol_value)+l,j) - promedio), 2.0);  
                

                if(varianza>max_var)
                    for (uint m = 0; m < interpol_value; m++) 
                    Zout((i*interpol_value)+m,j) = 0;                 
            }   
        ZI = Zout;
    }

    // range image to point cloud 
    int num_pc = 0; 
    for (uint i=0; i< ZI.n_rows - interpol_value; i+=1) {       
      for (uint j=0; j<ZI.n_cols ; j+=1) {
        float ang = M_PI-((2.0 * M_PI * j )/(ZI.n_cols));

        if (ang < min_FOV-M_PI/2.0|| ang > max_FOV - M_PI/2.0) 
          continue;

        if(!(Zout(i,j)== 0 )) {  
          float pc_modulo = Zout(i,j);
          float pc_x = sqrt(pow(pc_modulo,2)- pow(ZzI(i,j),2)) * cos(ang);
          float pc_y = sqrt(pow(pc_modulo,2)- pow(ZzI(i,j),2)) * sin(ang);

          float ang_x_lidar = 0.1*M_PI/180.0;  

          Eigen::MatrixXf Lidar_matrix(3,3); 
          Eigen::MatrixXf result(3,1);
          Lidar_matrix <<   cos(ang_x_lidar) ,0                ,sin(ang_x_lidar),
                            0                ,1                ,0,
                            -sin(ang_x_lidar),0                ,cos(ang_x_lidar) ;

          result << pc_x,
                    pc_y,
                    ZzI(i,j);
          
          result = Lidar_matrix*result;

          point_cloud->points[num_pc].x = result(0);
          point_cloud->points[num_pc].y = result(1);
          point_cloud->points[num_pc].z = result(2);

          cloud->push_back(point_cloud->points[num_pc]); 

          num_pc++;
        }
      }
    }

    PointCloud::Ptr P_out (new PointCloud);
    P_out = cloud;

    Eigen::MatrixXf RTlc(4,4); // translation matrix lidar-camera
    RTlc<<   Rlc(0), Rlc(3) , Rlc(6) ,Tlc(0)
            ,Rlc(1), Rlc(4) , Rlc(7) ,Tlc(1)
            ,Rlc(2), Rlc(5) , Rlc(8) ,Tlc(2)
            ,0       , 0        , 0  , 1    ;
    int size_inter_Lidar = (int) P_out->points.size();

    Eigen::MatrixXf Lidar_camera(3,size_inter_Lidar);
    Eigen::MatrixXf Lidar_cam(3,1);
    Eigen::MatrixXf pc_matrix(4,1);
    Eigen::MatrixXf pointCloud_matrix(4,size_inter_Lidar);

    unsigned int cols = image_msg->width;
    unsigned int rows = image_msg->height;

    uint px_data = 0; uint py_data = 0;
    pcl::PointXYZRGB point;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int i = 0; i < size_inter_Lidar; i++)
    {
        pc_matrix(0,0) = -P_out->points[i].y;   
        pc_matrix(1,0) = -P_out->points[i].z;   
        pc_matrix(2,0) =  P_out->points[i].x;  
        pc_matrix(3,0) = 1.0;

        Lidar_cam = Mc * (RTlc * pc_matrix);

        px_data = (int)(Lidar_cam(0,0)/Lidar_cam(2,0));
        py_data = (int)(Lidar_cam(1,0)/Lidar_cam(2,0));
        
        if(px_data<0.0 || px_data>=cols || py_data<0.0 || py_data>=rows)
            continue;

        int color_dis_x = (int)(255*((P_out->points[i].x)/maxlen));
        int color_dis_z = (int)(255*((P_out->points[i].x)/10.0));
        if(color_dis_z>255)
            color_dis_z = 255;


        //point cloud con color
        cv::Vec3b & color = color_pcl->image.at<cv::Vec3b>(py_data,px_data);

        point.x = P_out->points[i].x;
        point.y = P_out->points[i].y;
        point.z = P_out->points[i].z;
        

        point.r = (int)color[2]; 
        point.g = (int)color[1]; 
        point.b = (int)color[0];

        
        pc_color->points.push_back(point);   
        
        cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, CV_RGB(255-color_dis_x,(int)(color_dis_z),color_dis_x),cv::FILLED);
        
    }
    pc_color->is_dense = true;
    pc_color->width = (int) pc_color->points.size();
    pc_color->height = 1;
    pc_color->header.frame_id = frame_id_;

    // publish
    std::shared_ptr<sensor_msgs::msg::Image> edited_image_msg
        = cv_ptr->toImageMsg();
    sensor_msgs::msg::Image output_image_msg;
    output_image_msg.header = edited_image_msg->header;
    output_image_msg.height = edited_image_msg->height;
    output_image_msg.width = edited_image_msg->width;
    output_image_msg.encoding = edited_image_msg->encoding;
    output_image_msg.is_bigendian = edited_image_msg->is_bigendian;
    output_image_msg.step = edited_image_msg->step;
    output_image_msg.data = edited_image_msg->data;
    //pcOnimg_pub->publish(output_image_msg);

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*pc_color, output);

    pc_pub->publish(output);
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetection)
