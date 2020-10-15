//
// Created by shenyl on 2020/7/22.
//

#ifndef OBJECT_DETECTION_SPARITY_2D_DETECTION_H
#define OBJECT_DETECTION_SPARITY_2D_DETECTION_H

#include <iostream>
#include <string>
#include <ctime>
#include <chrono>

//#include "bbox.h"
//#include "box3d.h"
//#include "stereo_match.h"
#include "bbox2box3d.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
Mat result3DImage;


ros::Publisher obj_points_pub_;
ros::Publisher obj_object_points_pub_;
ros::Publisher obj_box_pub_;
ros::Publisher rectified_left_image_pub_;
ros::Publisher rectified_right_image_pub_;
ros::Publisher obstacle_mask_pub_;


void onMouse(int event, int x, int y, int flags, void *param);
//bool obj_det_3d(string root_path, vector<bbox> bbox_list_l, Mat rectified_l, Mat result3DImage, vector<box3d>& pclist, int count, Size imageSize, Mat P1);
//bool obj_det_3d_with_sparity(string root_path, Mat result3DImage, vector<box3d>& bbox_3d_list, int count);
bool obj_det_3d_with_sparity(string root_path, Mat result3DImage, vector<AABB_box3d>& AABB_list, vector<bbox>& bbox_list, int count, Mat P1, Size imageSize);
bool getBboxFromSparity(string root_path, Mat filtered_disparity, vector<bbox> bbox_list_l, int count);
bool get_obstacle_mask(string root_path, Mat disparity_l, Mat& obstacle_mask, int v0, int count, double h, double b, int iou_offset, double h0=0.0);
bool get_masked_disp(string root_path, Mat obstacle_mask, Mat disparity_l, Mat& masked_disparity, int count);
bool obstacle_det(string root_path, Mat rectified_l, Mat rectified_r, Mat Q, Mat P1, int count, vector<AABB_box3d>& aabb_list, vector<bbox>& bbox_list, double h, double b, int v0);

bool obstacle_points(string root_path, Mat rectified_l, Mat rectified_r, Mat Q, Mat P1, int count,  double h, double b, int v0, pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr);
bool obstacle_with_sparity(string root_path, Mat result3DImage, int count, Mat P1, Size imageSize, pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr);
bool yolo_sparity_obj_det(string root_path, Mat yolo_mask, vector<AABB_box3d>& box_3d_list, vector<bbox>& bbox_list, int frame, Mat P1);
bool obj_det_3d_with_yolo_sparity(string root_path, vector<bbox> bbox_list_l, vector<AABB_box3d>& AABB_list, int count, Size imageSize, Mat P1);
#endif //OBJECT_DETECTION_SPARITY_2D_DETECTION_H

