//
// Created by shenyl on 2020/7/6.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <map>
#include <stdlib.h>
#include <cstdlib>
#include <string>

using namespace cv;
using namespace std;

Mat img1;
Mat img2;
Mat img3;
Mat img4;
static vector<Point2f> points;

static int print_help()
{
    cout <<
         " Given a list of chessboard images, the number of corners (nx, ny)\n"
         " on the chessboards, the size of the square and the root dir of the calib image and the flag of if show the rectified results \n"
         " Calibrate and rectified the stereo camera, and save the result to txt \n"<< endl;
    cout << "Usage:\n // usage：./Rt_calib -d=<dir default=/home/shenyl/Documents/sweeper/data/> \n" << endl;
    return 0;
}

void onMouse(int event, int x, int y, int flags, void *param)
{
    Point point;
    point.x = x;
    point.y = y;
    if(event == EVENT_LBUTTONDOWN)
    {
        cout << x << " " << y << endl;
        points.push_back(point);
    }
}

bool read_3d_points(string points3d_path, vector<Point3f>& points3d)
{
    ifstream points3d_file(points3d_path);
    if (!points3d_file)
    {
        cout<<points3d_path<<"is not exist"<<endl;
    }
    Point3f p_3d;
    float x, y, z;
    while (points3d_file >> x >> y >> z)
    {
        p_3d.x=x*1000;
        p_3d.y=y*1000;
        p_3d.z=z*1000;
//        cout<<x<<" "<<y<<" "<<z<<endl;
        points3d.push_back(p_3d);
    }
    points3d_file.close();
    return true;
}

bool read_camera_para(string camera_para_path, Mat& CameraMatrix, Mat& distCoeffs, Mat& RobotInWorld)
{
    cv::FileStorage fs_read(camera_para_path, cv::FileStorage::READ);
    fs_read["cameraMatrixL"] >> CameraMatrix;
    fs_read["distCoeffsL"] >> distCoeffs;
    fs_read["RobotInWorld"] >> RobotInWorld;
    fs_read.release();
}

bool get_img_points(string img_path)
{
    int img_num = 8;
    for (int i=0; i<img_num; i ++)
    {
        char img_file_name[30];
        sprintf(img_file_name, "%06d.jpg", i);
        cv::Mat img = imread(img_path + img_file_name);
        imshow("img", img);
        setMouseCallback("img", onMouse);
        if (waitKey(0) == 27) {
            destroyAllWindows();
        }
    }
    return true;
}

int main(int argc,char *argv[]) {
    cv::CommandLineParser parser(argc, argv, "{p1||}{p2||}{p3||}{p4||}{p5||}{help||}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string img_path = parser.get<String>("p1");
    string points3d_path = parser.get<String>("p2");
    string camera_para_path = parser.get<String>("p3");
    string Tcr_path = parser.get<String>("p4");
    string pts_path = parser.get<String>("p5");


    //get 2d img points
    get_img_points(img_path);
    // save 2d points to file
    ofstream pts_f(pts_path, ios::out);
    for (int i=0;i<points.size();i++)
    {
        cout<<points[i].x << " "<<points[i].y <<endl;
        pts_f<<points[i].x << " "<<points[i].y <<endl;
    }
    pts_f.close();

    //read camera camera parameters
    Mat CameraMatrix, distcoeffs;
    Mat Twr = Mat(4, 4, CV_64FC1, Scalar::all(1));
    read_camera_para(camera_para_path, CameraMatrix, distcoeffs, Twr);

    //read 3d coordinates from txt, in world coordinates
    vector<Point3f> points3d_w, points3d_r;
    read_3d_points(points3d_path, points3d_w);
    Mat points3d_w_m = Mat(4, points3d_w.size(), CV_64FC1, Scalar::all(1));
    Mat points3d_r_m = Mat(4, points3d_w.size(), CV_64FC1, Scalar::all(1));

    // change 3d points from world coordinates to robot coordinates
    for(int i =0;i<points3d_w.size();i++)
    {
        cout<<points3d_w[i].x<<" "<<points3d_w[i].y<<" "<<points3d_w[i].z<<endl;
        points3d_w_m.at<double>(0, i) = points3d_w[i].x;
        points3d_w_m.at<double>(1, i) = points3d_w[i].y;
        points3d_w_m.at<double>(2, i) = points3d_w[i].z;
    }
    cout<<"points3d_w_m"<<endl<<points3d_w_m<<endl;
    //T from robot coordinate to world coordinate
    cout<<"Twr: "<<endl<<Twr<<endl;
    //T from world coordinate to robot coordinate
    Mat Trw = Mat(4, 4, CV_64FC1, Scalar::all(1));
    cv::invert(Twr, Trw);
    cout<<"Trw: "<<endl<<Trw<<endl;
    points3d_r_m = Trw * points3d_w_m;
    cout<<"points3d_r_m"<<endl<<points3d_r_m<<endl;
    cout<<"points3d_r"<<endl;
    for (int i=0;i<points3d_r_m.cols; i++)
    {
        Point3f p3d(points3d_r_m.at<double>(0, i), points3d_r_m.at<double>(1, i), points3d_r_m.at<double>(2, i));
        points3d_r.push_back(p3d);
        cout<<p3d.x<<" "<<p3d.y<<" "<<p3d.z<<endl;
    }

    // calculate R and t
    Mat rvec, tvec;
    // RANSAC parameters
    int iterationsCount = 300;      // number of Ransac iterations.
    float reprojectionError = 5.991;  // maximum allowed distance to consider it an inlier.
    double confidence = 0.95;        // ransac successful confidence.
    Mat inliers;

    cout<<"2d points num"<<points.size()<<endl;
    cout<<"3d points num"<<points3d_w.size()<<endl;
    solvePnPRansac(points3d_r, points, CameraMatrix, distcoeffs, rvec, tvec, false, iterationsCount, reprojectionError, confidence, inliers, SOLVEPNP_ITERATIVE);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    // robot在相机坐标系下的位姿
    Mat Tcr = Mat(4, 4, CV_64FC1, Scalar::all(0));
    // 相机在robot坐标系下的位姿
    Mat Trc = Mat(4, 4, CV_64FC1, Scalar::all(0));
    // 世界坐标系在相机坐标系下的位姿
    Mat Tcw = Mat(4, 4, CV_64FC1, Scalar::all(0));
    Tcr.at<double>(0,0) = R.at<double>(0,0);
    Tcr.at<double>(0,1) = R.at<double>(0,1);
    Tcr.at<double>(0,2) = R.at<double>(0,2);
    Tcr.at<double>(1,0) = R.at<double>(1,0);
    Tcr.at<double>(1,1) = R.at<double>(1,1);
    Tcr.at<double>(1,2) = R.at<double>(1,2);
    Tcr.at<double>(2,0) = R.at<double>(2,0);
    Tcr.at<double>(2,1) = R.at<double>(2,1);
    Tcr.at<double>(2,2) = R.at<double>(2,2);

    Tcr.at<double>(0,3) = tvec.at<double>(0);
    Tcr.at<double>(1,3) = tvec.at<double>(1);
    Tcr.at<double>(2,3) = tvec.at<double>(2);

    Tcr.at<double>(3,3) = 1;

    Trw.at<double>(0,3)=0;
    Trw.at<double>(1,3)=0;
    cout<<"Tcr"<<endl<<Tcr<<endl;
    cout<<"Trc"<<endl<<Trc<<endl;
    cout<<"Twr"<<endl<<Twr<<endl;
    cout<<"Trw"<<endl<<Trw<<endl;


    Tcw = Tcr * Trw;

    cout<<"Tcw"<<endl<<Tcw<<endl;
    cv::FileStorage fs_write(Tcr_path, cv::FileStorage::WRITE);
    fs_write << "Tcr" << Tcr;
    fs_write << "Tcw" << Tcw;
    fs_write.release();

}

