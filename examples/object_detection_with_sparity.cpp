//
// Created by shenyl on 2020/7/20.
//


#include "bbox2box3d.h"
#include "sparity_2d_detection.h"


static int print_help()
{
    cout <<
         " Given the root dir of the exp dir and the match algorithm \n"
         " output the rectified images and sparities\n"<< endl;
    cout << "Usage:\n // usage：./stereo_match -d=<dir default=/home/shenyl/Documents/sweeper/data/> -a=<stereo match algorithm default = sgbm>\n" << endl;
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
    Mat disparity_l, disparity_r;
    Mat filtered_disparity;
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
    string calib_opencv_file =  root_path + "calib_img/stereocalibrateresult_L.txt";
    string calib_matlab_file = root_path + "calib_img/stereocalibrateresult_matlab.txt";
    string calib_file;
    string rectified_parameters = root_path + "calib_img/stereoRectifyParams.txt";

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
    Tcw.at<double>(0,0)= 0.9998613930260787;
    Tcw.at<double>(0,1)= 0.004426922294121998;
    Tcw.at<double>(0,2)= 0.01604983161753834;
    Tcw.at<double>(1,0)= -0.004747457841199377;
    Tcw.at<double>(1,1)= 0.9997889387204514;
    Tcw.at<double>(1,2)= 0.01998848809388965;
    Tcw.at<double>(2,0)= -0.01595795663597199;
    Tcw.at<double>(2,1)= -0.02006191344900429;
    Tcw.at<double>(2,2)= 0.9996713776280529;
    Tcw.at<double>(0,3)= 0.0397648383105771;
    Tcw.at<double>(1,3)= 0.1799994361370056;
    Tcw.at<double>(2,3)= -0.01578391077428604;
    Tcw.at<double>(3,3)= 1.0;

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

    ifstream imageStore_L(imageList_L); // 打开存放测试图片名称的txt
    ifstream imageStore_R(imageList_R); // 打开存放测试图片名称的txt
    string imageName_L; // 读取的测试图片的名称
    string imageName_R; // 读取的测试图片的名称
    int count = 0 ;
    char object_det_result_file[30];
    // for each pair rectified images and generate sparity
    sprintf(object_det_result_file, "object_det_");
    ofstream fResult(object_det_result_path + object_det_result_file+calib_method+"_"+algorithm+".txt", ios::out); //save det result
    ofstream fTime(object_det_result_path + "time"+calib_method+"_"+algorithm+".txt", ios::out);
    while (getline(imageStore_L, imageName_L))
    {
        auto start = std::chrono::steady_clock::now();
        getline(imageStore_R, imageName_R);
        cout<<"*************FRAME"<<count<<"*****************"<<endl;
        // Step3 rectified images
        Rectification(root_path, imageName_L, imageName_R, rectified_l, rectified_r, mapl1, mapl2, mapr1, mapr2, validRoi, count);
//        auto end = std::chrono::steady_clock::now();
//        std::chrono::duration<double, std::micro> elapsed = end - start; // std::micro 表示以微秒为时间单位
//        cout<<"time for rectificate: "<<elapsed.count()/1000000<<endl;
        //        cout<<"finish rectification"<<endl;
        // Step4 generate sparities
//        computeDisparityImage(root_path, rectified_l, rectified_r, disparity_l, disparity_r, count, algorithm);
//        cout<<"finish compute disparity image"<<endl;
        // Step4 计算视差图,对视差图进行加权最小二乘滤波
        int v0=245;
        auto start1 = std::chrono::steady_clock::now();
        filterDisparityImage(root_path, rectified_l, rectified_r, disparity_l, disparity_r, filtered_disparity, v0, count);
        auto end1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed1 = end1 - start1; // std::micro 表示以微秒为时间单位
        cout<<"time for filterDisparityImage: "<<elapsed1.count()/1000000<<endl;
        // Step5 得到障碍物mask
        Mat obstacle_mask(rectified_l.rows-v0, rectified_l.cols, CV_8UC1, Scalar::all(0));
        get_obstacle_mask(root_path, filtered_disparity, obstacle_mask, v0, count);
        // Step6 将障碍物mask和视差图相与
        Mat masked_disparity(rectified_l.rows-v0, rectified_l.cols, CV_8UC1, Scalar::all(0));
        get_masked_disp(root_path, obstacle_mask, filtered_disparity, masked_disparity, count);
        // Step7 利用视差图和投影参数Q得到三维图像
//        cout<<"Q"<<endl<<Q<<endl;
        reprojectImageTo3D(masked_disparity, result3DImage, Q);
//        imshow("masked_disparity", masked_disparity);
//        setMouseCallback("masked_disparity", onMouse);
//        if (waitKey(0) == 27) {
//            destroyAllWindows();
//        }
//        cout<<"finish reproject image to 3d"<<endl;
        //Step8 三维物体检测, 对每一帧的障碍物进行聚类
        vector<box3d> bbox_3d_list;
        if(obj_det_3d_with_sparity(root_path, result3DImage, bbox_3d_list, count))
        {
//            cout<<"finish 3d object detection"<<endl;
//            cout<<"3d detection result"<<endl;
            for(int i = 0; i<bbox_3d_list.size();i++)
            {
                box3d b3d=bbox_3d_list[i];
                box3d b3d_w;
                trans2w(Tcw, b3d, b3d_w);
//                cout <<"p_x: "<<b3d_w._position_x <<", p_y: "<<b3d_w._position_y<<", p_z: "<<b3d_w._position_z<<endl;
                fResult<<count<<" "<<b3d_w._position_x <<" "<<b3d_w._position_y<<" "<<b3d_w._position_z<<endl;
            }
        }
        // Step8 找到障碍物和地面连接点的三维坐标
        count = count + 1;
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start; // std::micro 表示以微秒为时间单位
//        std::cout<< "time for one frame : "<< elapsed.count()/1000000<< "s" << std::endl;
        fTime<<elapsed.count()/1000000<<"\t"<<elapsed1.count()/1000000<<endl;
    }
    fResult.close();
    return 0;
}