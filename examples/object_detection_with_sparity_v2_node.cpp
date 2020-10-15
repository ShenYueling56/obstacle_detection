//
// Created by shenyl on 2020/7/20.
//

#include "sparity_2d_detection.h"
#include <thread>
#include "yolo_opencv.h"

#define ros_result_show

//#define video_save
//#define image_save

extern ros::Publisher obj_points_pub_;
extern ros::Publisher obj_object_points_pub_;
extern ros::Publisher obj_box_pub_;
extern ros::Publisher rectified_left_image_pub_;
extern ros::Publisher rectified_right_image_pub_;
extern ros::Publisher obstacle_mask_pub_;
extern ros::Publisher yolo_pub_;
extern ros::Publisher yolo_mask_pub_;

String root_path;
Mat rectified_l, rectified_r;
Mat disparity_l, disparity_r;
Mat filtered_disparity;
Mat cameraMatrix_L = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 相机的内参数
Mat cameraMatrix_R = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 初始化相机的内参数
Mat distCoeffs_L = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 相机的畸变系数
Mat distCoeffs_R = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 初始化相机的畸变系数
Mat R, T;
Size imageSize;
Rect validRoi[2];//双目矫正有效区域
// 立体校正参数
Mat R1(3,3,CV_64FC1,Scalar::all(0));
Mat R2(3,3,CV_64FC1,Scalar::all(0));
Mat P1(3,4,CV_64FC1,Scalar::all(0));
Mat P2(3,4,CV_64FC1,Scalar::all(0));
Mat Q(4,4,CV_64FC1,Scalar::all(0));
Mat mapl1, mapl2, mapr1, mapr2; // 图像重投影映射表
Mat Tcw(4,4,CV_64FC1,Scalar::all(0));

std::vector<std::string> classes;
cv::dnn::Net net;

ofstream fResult;
ofstream fTime;


// ros object box marker
vector<vector<double>> color_lists(5,{0,0,0});
vector<visualization_msgs::Marker> box_lines_vector(5);

//save to video
#ifdef video_save
VideoWriter videoWriter;
#endif

static int frame;

static int print_help()
{
    cout <<
         " Given the root dir of the exp dir, the calib method and the match algorithm \n"
         " output the 3d object detection result\n"<< endl;
    cout << "Usage:\n // usage：rosrun object_detection_v2_node -d=<dir default=/home/shenyl/Documents/sweeper/data/> -c=<calibration method> -a=<stereo match algorithm default = sgbm>\n" << endl;
}

bool callback_fusion(const sensor_msgs::ImageConstPtr & img0_msg, const sensor_msgs::ImageConstPtr& img1_msg)
{
    auto start = std::chrono::steady_clock::now();
    cout<<"///////////////////frame"<<frame<<"///////////////////"<<endl;
    cv_bridge::CvImagePtr img0_ptr = cv_bridge::toCvCopy(img0_msg, sensor_msgs::image_encodings::BGR8);
    Mat img1 = img0_ptr -> image;
    cv_bridge::CvImagePtr img1_ptr = cv_bridge::toCvCopy(img1_msg, sensor_msgs::image_encodings::BGR8);
    Mat img2 = img1_ptr -> image;
    // Step3 rectified images
    Rectification(root_path, img1, img2, rectified_l, rectified_r, mapl1, mapl2, mapr1, mapr2, validRoi, frame);
    // yolo_det thread
    vector<bbox> yolo_bbox_list;
    Mat yolo_mask = Mat(rectified_l.rows, rectified_l.cols, CV_8UC1, Scalar::all(0));
//    thread thread1(yolo_det, root_path, rectified_l, frame, classes, net, std::ref(yolo_bbox_list), (int)(cameraMatrix_L.at<double>(1, 2)), std::ref(yolo_mask));

    vector<AABB_box3d> box_3d_list;
    vector<bbox> obs_bbox_list;
    //障碍物检测
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    obstacle_points(root_path, rectified_l, rectified_r, Q, P1, frame,  Tcw.at<double>(1,3), T.at<double>(0), (int)(cameraMatrix_L.at<double>(1, 2)), pc_object_ptr);
//    thread1.join();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start; // std::micro 表示以微秒为时间单位
    std::cout<< "time for two thread "<< elapsed.count()/1000000<< "s" << std::endl;
    //四种类型物体识别
//    obj_det_3d_with_yolo_sparity(root_path, yolo_bbox_list, box_3d_list, frame, imageSize, P1);
    auto end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed2 = end2 - end; // std::micro 表示以微秒为时间单位
    std::cout<< "time for fusion "<< elapsed2.count()/1000000<< "s" << std::endl;

    // show result
//    cout<<"box3d size: "<<box_3d_list.size()<<endl;
//    for(int i = 0; i<box_3d_list.size();i++)
//    {
//        AABB_box3d b3d=box_3d_list[i];
//        AABB_box3d b3d_w;
//        transAABB2w(Tcw, b3d, b3d_w);
//        fResult<<frame<<" "<<yolo_bbox_list.size()<<" "<<b3d_w._c<<" "<<b3d_w._position_x <<" "<<b3d_w._position_y/1000<<" "<<b3d_w._position_z<<endl;
//        int box_class = b3d._c;
//        cout<<"b3d: "<<box_class<<" "<<b3d._position_x<<" "<<b3d._position_y<<" "<<b3d._position_z\
//        <<" "<<b3d._width<<" "<<b3d._height<<" "<<b3d._length<<endl;
//
//#ifdef ros_result_show
//        if (b3d._c>=0)
//        {
//            // 发布障碍物marker消息用于可视化
//            //publish by ros
//            // 8 points for each cube
//            vector<geometry_msgs::Point> points8;
//            geometry_msgs::Point p1;
//            geometry_msgs::Point p8;
//            p1.x = (b3d._position_x - b3d._width/2)*scale_offset; p1.y = (b3d._position_y - b3d._height/2)*scale_offset; p1.z = (b3d._position_z - b3d._length/2)*scale_offset;
//            p8.x = (b3d._position_x + b3d._width/2)*scale_offset; p8.y = (b3d._position_y + b3d._height/2)*scale_offset; p8.z = (b3d._position_z + b3d._length/2)*scale_offset;
//            geometry_msgs::Point p2 = p1;
//            geometry_msgs::Point p3 = p1;
//            geometry_msgs::Point p4 = p1;
//            geometry_msgs::Point p5 = p8;
//            geometry_msgs::Point p6 = p8;
//            geometry_msgs::Point p7 = p8;
//            p2.x = p8.x;
//            p3.z = p8.z;
//            p4.x = p8.x;
//            p4.z = p8.z;
//            p5.x = p1.x;
//            p5.z = p1.z;
//            p6.z = p1.z;
//            p7.x = p1.x;
//            points8.push_back(p1);
//            points8.push_back(p2);
//            points8.push_back(p3);
//            points8.push_back(p4);
//            points8.push_back(p5);
//            points8.push_back(p6);
//            points8.push_back(p7);
//            points8.push_back(p8);
//
//            // 12 lines with 24 points
//            vector<int> line_points = {0,1,2,3,0,2,1,3,4,5,6,7,4,6,5,7,0,4,1,5,2,6,3,7};
//            box_lines_vector[box_class].header.frame_id = "map";
//            box_lines_vector[box_class].header.stamp = ros::Time::now();
//            for (int i=0;i<24;i++)
//            {
//                box_lines_vector[box_class].points.push_back(points8[line_points[i]]);
//            }
//        }
//#endif
//    }
//
//#ifdef ros_result_show
//    for (int i=0; i<5; i++)
//    {
//        if (box_lines_vector[i].points.size() == 0)
//        {
//            visualization_msgs::Marker p0_marker;
//            p0_marker.header.frame_id = "map";
//            p0_marker.header.stamp = ros::Time::now();
//            geometry_msgs::Point p0;
//            p0.x = p0.y = p0.z = 0;
//            for (int j=0; j<24; j++)
//            {
//                p0_marker.points.push_back(p0);
//                box_lines_vector[i].points.push_back(p0);
//            }
//        }
//        obj_box_pub_.publish(box_lines_vector[i]);
//        box_lines_vector[i].points.clear(); //清空各类需要发布的框
//    }
//
//#endif
//    auto end_all = std::chrono::steady_clock::now();
//    std::chrono::duration<double, std::micro> elapsed_all = end_all - start; // std::micro 表示以微秒为时间单位
//    std::cout<< "time for one frame : "<< elapsed_all.count()/1000000<< "s" << std::endl;
//    fTime<< elapsed_all.count()/1000000<<endl;
//#ifdef video_save
//    videoWriter.write(rectified_l);
//#endif
    frame = frame + 1;
}


int main(int argc,char *argv[])
{
    cv::CommandLineParser parser(argc, argv, "{d|/home/shenyl/Documents/sweeper/data/|}{c|matlab|}{a|sgbm|}{help||}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    root_path = parser.get<String>("d");
    String algorithm = parser.get<String>("a");
    string calib_method = parser.get<String>("c");


#ifdef image_save
    //dir for save result images
    string left_rectified = root_path + "rectified/left/";
    string right_rectified = root_path + "rectified/right/";
    string pairs_rectified = root_path + "rectified/pairs/";
    string disparities = root_path + "disparities/";
    string filtered_disparities_path = root_path + "filtered_disparities/";

    // object det by sparity
    string obstacle_disp_path = root_path + "obstacle_mask/";
    string masked_disparity_path = root_path + "masked_disparity/";

    string command = "mkdir -p " + left_rectified;
    system(command.c_str());
    command = "mkdir -p " + right_rectified;
    system(command.c_str());
    command = "mkdir -p " + pairs_rectified;
    system(command.c_str());
    command = "mkdir -p " + disparities;
    system(command.c_str());
    command = "mkdir -p " + filtered_disparities_path;
    system(command.c_str());
    command = "mkdir -p " + obstacle_disp_path;
    system(command.c_str());
    command = "mkdir -p " + masked_disparity_path;
    system(command.c_str());
#endif

    //dir for object det result
    string object_det_result_path = root_path + "object_det_result/";
    string command = "mkdir -p " + object_det_result_path;
    system(command.c_str());

    //step1 进行双目矫正、匹配得到深度图
    string calib_opencv_file =  root_path + "calib_img/stereocalibrateresult_L.yaml";
    string calib_matlab_file = root_path + "calib_img/stereocalibrateresult_matlab.yaml";
    string calib_file;
    string rectified_parameters = root_path + "calib_img/stereoRectifyParams.yaml";

    string Tcr_path = root_path + "calib_img/Tcr.yaml";
    cv::FileStorage fs_read(Tcr_path, cv::FileStorage::READ);
    fs_read["Tcw"] >> Tcw;
    fs_read.release();

    // init ROS
    ros::init(argc, argv, "object_detection");
    ros::NodeHandle n;

    // Publish Init
    std::string rectified_left_img_topic;
    n.param<std::string>("rectified_left_img_topic", rectified_left_img_topic, "/rectified_left_image");
    ROS_INFO("Rectified Left Image: %s", rectified_left_img_topic.c_str());

    std::string rectified_right_img_topic;
    n.param<std::string>("rectified_right_img_topic", rectified_right_img_topic, "/rectified_right_image");
    ROS_INFO("Rectified Right Image: %s", rectified_right_img_topic.c_str());

    std::string obstacle_img_topic;
    n.param<std::string>("obstacle_mask_topic", obstacle_img_topic, "/obstacle_mask");
    ROS_INFO("obstacle_mask_topic: %s", obstacle_img_topic.c_str());

    std::string obj_points_topic;
    n.param<std::string>("obj_point_topic", obj_points_topic, "/points_obstacle");
    ROS_INFO("Object Output Point Cloud: %s", obj_points_topic.c_str());

    std::string obj_object_points_topic;
    n.param<std::string>("obj_object_points_topic", obj_object_points_topic, "/points_object");
    ROS_INFO("obj_object_points_topic: %s", obj_object_points_topic.c_str());

    std::string visualization_marker_topic;
    n.param<std::string>("visualization_marker_topic", visualization_marker_topic, "/object_box");
    ROS_INFO("visualization_marker_topic: %s", visualization_marker_topic.c_str());

    std::string yolo_result_topic;
    n.param<std::string>("yolo_result_topic", yolo_result_topic, "/2d_obj_det");
    ROS_INFO("yolo_result_topic: %s", yolo_result_topic.c_str());

    std::string yolo_mask_topic;
    n.param<std::string>("yolo_mask_topic", yolo_mask_topic, "/yolo_mask");
    ROS_INFO("yolo_mask_topic: %s", yolo_mask_topic.c_str());

    rectified_left_image_pub_     = n.advertise<sensor_msgs::Image>(rectified_left_img_topic, 2);
    rectified_right_image_pub_    = n.advertise<sensor_msgs::Image>(rectified_right_img_topic, 2);
    obstacle_mask_pub_            = n.advertise<sensor_msgs::Image>(obstacle_img_topic, 2);
    obj_points_pub_               = n.advertise<sensor_msgs::PointCloud2>(obj_points_topic, 2);              //2d框对应的3d椎体
    obj_object_points_pub_        = n.advertise<sensor_msgs::PointCloud2>(obj_object_points_topic, 2);
    obj_box_pub_                  = n.advertise<visualization_msgs::Marker>( visualization_marker_topic, 100);//3d maker
    yolo_pub_                     = n.advertise<sensor_msgs::Image>( yolo_result_topic, 2 );//yolo_result
    yolo_mask_pub_                = n.advertise<sensor_msgs::Image>( yolo_mask_topic, 2 );//yolo_mask
    // SUB INIT
    std::string left_img_topic;
    n.param<std::string>("left_img_topic", left_img_topic, "/cam0/image_raw");
    ROS_INFO("left Image: %s", left_img_topic.c_str());

    std::string right_img_topic;
    n.param<std::string>("right_img_topic", right_img_topic, "/cam1/image_raw");
    ROS_INFO("right Image: %s", right_img_topic.c_str());

    message_filters::Subscriber<sensor_msgs::Image> img0_sub(n, left_img_topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> img1_sub(n, right_img_topic, 1);

    typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), img0_sub, img1_sub);
    sync.registerCallback(boost::bind(&callback_fusion, _1, _2));


    //Step1 read camera parameters
    if (calib_method == "matlab")
    {
        calib_file = calib_matlab_file;
    }
    if (calib_method == "opencv")
    {
        calib_file = calib_opencv_file;
    }
    read_calib_parameters(calib_file, cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R,
                          R, T, imageSize);
    cout<< R << endl<< T << endl << cameraMatrix_L << endl << distCoeffs_L << endl << cameraMatrix_R << endl<< distCoeffs_R<<endl;

    //Step2 get stereo rectified parameters
    validRoi[0], validRoi[1] = stereoRectification(cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R,
                                                   imageSize, R, T, R1, R2, P1, P2, Q, mapl1, mapl2, mapr1, mapr2);
    char object_det_result_file[30];
    sprintf(object_det_result_file, "object_det_");
    fResult.open(object_det_result_path + object_det_result_file+calib_method+"_"+algorithm+".txt", ios::out); //save det result
    fTime.open(object_det_result_path + "time"+calib_method+"_"+algorithm+".txt", ios::out);

    //Step3 load yolo net parameters
    // load classes
    string classesFile = root_path + "yolo_parameters/sweeper.names";
//    string classesFile = root_path + "yolo_parameters/five_classes/sweeper.names";
    std::ifstream classNamesFile(classesFile.c_str());
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (std::getline(classNamesFile, className))
        {
            classes.push_back(className);
        }

    }
    else{
        std::cout<<"can not open classNamesFile"<<std::endl;
    }

    // load weights and cfg
    // Give the configuration and weight files for the model
    string modelConfiguration = root_path + "yolo_parameters/prune_0.8_keep_0.01_12_shortcut_yolov3_sweeper.cfg";
    cv::String modelWeights = root_path + "yolo_parameters/converted(tune_160_0.8_0.01_12).weights";
//    string modelConfiguration = root_path + "yolo_parameters/five_classes/yolov3_sweeper.cfg";
//    cv::String modelWeights = root_path + "yolo_parameters/five_classes/yolov3_sweeper.backup";
    // Load the network
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

#ifdef ros_result_show
    // box marker for different classes
    vector<double> red = {1.0,0.0,0.0};  // badminton
    vector<double> green = {0.0,1.0,0.0}; //wire
    vector<double> blue = {0.0,0.0,1.0}; //mahjong
    vector<double> pink = {153.0/255.0,50.0/255.0,204.0/255.0 };//dogshit
    vector<double> yellow = {128.0/255.0, 128.0/255.0, 0.0};//carpet
    color_lists[0] = red;
    color_lists[1] = green;
    color_lists[2] = blue;
    color_lists[3] = pink;
    color_lists[4] = yellow;
//    cout<<"color_lists:"<<endl;
//    for(int i=0; i<6; i++)
//    {
//        cout<<color_lists[i][0]<<" "<<color_lists[i][1]<<" "<<color_lists[i][1]<<endl;
//    }

    for(int i=0; i<5; i++)
    {
        box_lines_vector[i].type = visualization_msgs::Marker::LINE_LIST;
        box_lines_vector[i].action = visualization_msgs::Marker::ADD;
        box_lines_vector[i].scale.x = 0.005;
        box_lines_vector[i].pose.orientation.x = 0.0;
        box_lines_vector[i].pose.orientation.y = 0.0;
        box_lines_vector[i].pose.orientation.z = 0.0;
        box_lines_vector[i].pose.orientation.w = 1.0;
        box_lines_vector[i].color.a = 1.0; // Don't forget to set the alpha!
        box_lines_vector[i].color.r = color_lists[i][0];
        box_lines_vector[i].color.g = color_lists[i][1];
        box_lines_vector[i].color.b = color_lists[i][2];
    }
#endif
#ifdef video_save
    //save to video
    videoWriter.open(root_path + "result_2d.avi", CV_FOURCC('M', 'J', 'P', 'G'), 5, imageSize, true);
#endif
    //运行ros
    ros::spin();
    fResult.close();
    fTime.close();
    return 0;
}