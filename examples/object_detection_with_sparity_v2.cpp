//
// Created by shenyl on 2020/7/20.
//


//#include "bbox2box3d.h"
#include "sparity_2d_detection.h"
#include <thread>
#include "yolo_opencv.h"


static int print_help()
{
    cout <<
         " Given the root dir of the exp dir and the match algorithm \n"
         " output the 3d object detection result\n"<< endl;
    cout << "Usage:\n // usage：./object_detection_v2 -d=<dir default=/home/shenyl/Documents/sweeper/data/> -a=<stereo match algorithm default = sgbm>\n" << endl;
}



int main(int argc,char *argv[])
{
    cv::CommandLineParser parser(argc, argv, "{d|/home/shenyl/Documents/sweeper/data/|}{c|matlab|}{a|sgbm|}{help||}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String root_path = parser.get<String>("d");
    String algorithm = parser.get<String>("a");
    string calib_method = parser.get<String>("c");

    Mat rectified_l, rectified_r;
    srand(time(NULL));//设置随机数种子，使每次获取的随机序列不同。
    //dir for save rectificated images
    string left_rectified = root_path + "rectified/left/";
    string right_rectified = root_path + "rectified/right/";
    string pairs_rectified = root_path + "rectified/pairs/";
    string disparities = root_path + "disparities/";
    string filtered_disparities_path = root_path + "filtered_disparities/";
    //dir for save sparity img with bbox
    string disparity_with_bbox_path = root_path + "disparity_with_bbox/";
    //dir for save pcd result
    string pcd_path = root_path + "pcd/";
    string frame_pc_path = pcd_path + "frame_pc/";
    string object_pc_before_cluster_path = pcd_path + "object_pc_before_cluster/";
    string object_pc_after_cluster_path = pcd_path + "object_pc_after_cluster/";
    string object_pc_path = pcd_path + "object_pc/";
    string object_pc_frame_path = pcd_path + "object_pc_frame/";
    //dir for object det result
    string object_det_result_path = root_path + "object_det_result/";

    // object det by sparity
    string img_binary_path = root_path + "img_binary/";
    string obstacle_disp_path = root_path + "obstacle_mask/";
    string masked_disparity_path = root_path + "masked_disparity/";
    string cluster_path = root_path + "pcd/cluster/";

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
    command = "mkdir -p " + disparity_with_bbox_path;
    system(command.c_str());
    command = "mkdir -p " + frame_pc_path;
    system(command.c_str());
    command = "mkdir -p " + object_pc_before_cluster_path;
    system(command.c_str());
    command = "mkdir -p " + object_pc_after_cluster_path;
    system(command.c_str());
    command = "mkdir -p " + object_pc_path;
    system(command.c_str());
    command = "mkdir -p " + object_pc_frame_path;
    system(command.c_str());
    command = "mkdir -p " + object_det_result_path;
    system(command.c_str());
    command = "mkdir -p " + img_binary_path;
    system(command.c_str());
    command = "mkdir -p " + obstacle_disp_path;
    system(command.c_str());
    command = "mkdir -p " + masked_disparity_path;
    system(command.c_str());
    command = "mkdir -p " + cluster_path;
    system(command.c_str());

    //step1 进行双目矫正、匹配得到深度图

    string imageList_L = root_path + "img/file_left.txt";
    string imageList_R = root_path + "img/file_right.txt";
//    string calib_opencv_file =  root_path + "calib_img/stereocalibrateresult_L.txt";
//    string calib_matlab_file = root_path + "calib_img/stereocalibrateresult_matlab.txt";
    string calib_opencv_file =  root_path + "calib_img/stereocalibrateresult_L.yaml";
    string calib_matlab_file = root_path + "calib_img/stereocalibrateresult_matlab.yaml";
    string calib_file;
//    string rectified_parameters = root_path + "calib_img/stereoRectifyParams.txt";
    string rectified_parameters = root_path + "calib_img/stereoRectifyParams.yaml";

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
    // todo: read the Twc from file
    // 左相机外参
//    [0.9998613930260787, 0.004426922294121998, 0.01604983161753834;
//    -0.004747457841199377, 0.9997889387204514, 0.01998848809388965;
//    -0.01595795663597199, -0.02006191344900429, 0.9996713776280529]
//    [0.0397648383105771;
//    0.1799994361370056;
//    -0.01578391077428604]
    Mat Tcw(4,4,CV_64FC1,Scalar::all(0));
    string Tcr_path = root_path + "calib_img/Tcr.yaml";
    cv::FileStorage fs_read(Tcr_path, cv::FileStorage::READ);
    fs_read["Tcw"] >> Tcw;
    fs_read.release();
//    Tcw.at<double>(0,0)= 0.9998613930260787;
//    Tcw.at<double>(0,1)= 0.004426922294121998;
//    Tcw.at<double>(0,2)= 0.01604983161753834;
//    Tcw.at<double>(1,0)= -0.004747457841199377;
//    Tcw.at<double>(1,1)= 0.9997889387204514;
//    Tcw.at<double>(1,2)= 0.01998848809388965;
//    Tcw.at<double>(2,0)= -0.01595795663597199;
//    Tcw.at<double>(2,1)= -0.02006191344900429;
//    Tcw.at<double>(2,2)= 0.9996713776280529;
//    Tcw.at<double>(0,3)= 0.0397648383105771;
//    Tcw.at<double>(1,3)= 0.1799994361370056;
//    Tcw.at<double>(2,3)= -0.01578391077428604;
//    Tcw.at<double>(3,3)= 1.0;


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

    VideoWriter videoWriter;
    videoWriter.open(root_path + "result_2d.avi", CV_FOURCC('M', 'J', 'P', 'G'), 5, imageSize, true);

    ifstream imageStore_L(imageList_L); // 打开存放测试图片名称的txt
    ifstream imageStore_R(imageList_R); // 打开存放测试图片名称的txt
    string imageName_L; // 读取的测试图片的名称
    string imageName_R; // 读取的测试图片的名称
    int count = 0 ;
    char object_det_result_file[30];
    sprintf(object_det_result_file, "object_det_");
    ofstream fResult(object_det_result_path + object_det_result_file+calib_method+"_"+algorithm+".txt", ios::out); //save det result
    ofstream fTime(object_det_result_path + "time"+calib_method+"_"+algorithm+".txt", ios::out);

    //load yolo net parameters
    // load classes
    std::vector<std::string> classes;
//    string classesFile = "/media/shenyl/KIOXIA/yolo3/mydata/all_dataset/sweeper.names";
    string classesFile = root_path + "yolo_parameters/sweeper.names";
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
//    string modelConfiguration = root_path + "yolo_parameters/five_classes/yolov3_sweeper.cfg";
//    cv::String modelWeights = root_path + "yolo_parameters/five_classes/yolov3_sweeper.backup";
    string modelConfiguration = root_path + "yolo_parameters/prune_0.8_keep_0.01_12_shortcut_yolov3_sweeper.cfg";
    cv::String modelWeights = root_path + "yolo_parameters/converted(tune_160_0.8_0.01_12).weights";


    // Load the network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    while (getline(imageStore_L, imageName_L))
    {
        auto start = std::chrono::steady_clock::now();
        getline(imageStore_R, imageName_R);
        cout<<"*************FRAME"<<count<<"*****************"<<endl;
        // Step3 rectified images
        Rectification(root_path, imageName_L, imageName_R, rectified_l, rectified_r, mapl1, mapl2, mapr1, mapr2, validRoi, count);
        count = count + 1;
        continue;
        // yolo_det thread
        vector<bbox> yolo_bbox_list;
        Mat yolo_mask = Mat(rectified_l.rows, rectified_l.cols, CV_8UC1, Scalar::all(0));
        thread thread1(yolo_det, root_path, rectified_l, count, classes, net, std::ref(yolo_bbox_list), (int)(cameraMatrix_L.at<double>(1, 2))-30, std::ref(yolo_mask));
//        yolo_det(root_path, rectified_l, count, classes, net, bbox_list);

        vector<AABB_box3d> box_3d_list;
        vector<bbox> obs_bbox_list;
//        thread thread2(obstacle_det, root_path, rectified_l, rectified_r, Q, count, std::ref(box_3d_list));
        obstacle_det(root_path, rectified_l, rectified_r, Q, P1, count, box_3d_list, obs_bbox_list, Tcw.at<double>(1,3)-50, T.at<double>(0), (int)(cameraMatrix_L.at<double>(1, 2))-30);
        thread1.join();
//        thread2.join();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start; // std::micro 表示以微秒为时间单位
//        std::cout<< "time for two thread "<< elapsed.count()/1000000<< "s" << std::endl;
        if(box_3d_list.size()>0)
        {
//            cout<<"finish 3d object detection"<<endl;
//            cout<<"3d detection result"<<endl;
            for(int i = 0; i<box_3d_list.size();i++)
            {
                AABB_box3d b3d=box_3d_list[i];
                AABB_box3d b3d_w;
                bbox b = obs_bbox_list[i];
                //为每个检测结果赋予类别
                //将3d框投影到2d
//                proj3d2d(b3d, b, P1, imageSize);
                // show obstacle bbox on image
                rectangle(rectified_l, cv::Point(b._xmin, b._ymin), cv::Point(b._xmax, b._ymax), cv::Scalar(0, 0, 255)); //red

                //计算和各个2d框的iou
                cv::Rect rectA(b._xmin, b._ymin, b._xmax-b._xmin, b._ymax-b._ymin);
                double max_iou = 0;
                int iou_index = -1;
                for(int j=0; j<yolo_bbox_list.size(); j++)
                {
                    bbox bbox_yolo = yolo_bbox_list[j];
                    // show yolo detection bbox on image
                    rectangle(rectified_l, cv::Point(bbox_yolo._xmin, bbox_yolo._ymin), cv::Point(bbox_yolo._xmax, bbox_yolo._ymax), cv::Scalar(255, 0, 0)); // blue
                    cv::Rect rectB(bbox_yolo._xmin, bbox_yolo._ymin, bbox_yolo._xmax-bbox_yolo._xmin, bbox_yolo._ymax-bbox_yolo._ymin);
                    cv::Rect rect_intersect;
                    double iou = intersectRect(rectA, rectB, rect_intersect);
                    if (max_iou<iou)
                    {
                        iou_index = j;
                        max_iou = iou;
                    }
                }
                if(iou_index>=0)
                {
                    b3d._c = yolo_bbox_list[iou_index]._c;
                }
                //如果iou大于阈值，将该2d框类别赋予3d框
                transAABB2w(Tcw, b3d, b3d_w);
                fResult<<count<<" "<<yolo_bbox_list.size()<<" "<<b3d_w._c<<" "<<b3d_w._position_x <<" "<<b3d_w._position_y<<" "<<b3d_w._position_z-b3d_w._length/2<<endl;
            }
        }

        count = count + 1;
        auto end_all = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed_all = end_all - start; // std::micro 表示以微秒为时间单位
        std::cout<< "time for one frame : "<< elapsed_all.count()/1000000<< "s" << std::endl;
        fTime<< elapsed_all.count()/1000000<<endl;

        // show
//        {
//            char rectified_file[20];
//            sprintf(rectified_file, "%06d.jpg", count);
//            imwrite(left_rectified + rectified_file, rectified_l);
//            imshow("det result", rectified_l);
//            if (waitKey(500))
//            {
//                destroyAllWindows();
//            }
//        }
        //save as  avi
        {
            videoWriter.write(rectified_l);
        }
    }
    fResult.close();
    return 0;
}