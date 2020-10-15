//
// Created by shenyl on 2020/7/2.
//

#ifndef OBJECT_DETECTION_BBOX2BOX3D_H
#define OBJECT_DETECTION_BBOX2BOX3D_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <map>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <typeinfo>
#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/visualization/cloud_viewer.h>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
//#include <image_transport/image_transport.h>

// message include
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>


#include "bbox.h"
#include "box3d.h"
#include "stereo_match.h"
//#include "sparity_2d_detection.h"

using namespace cv;
using namespace std;
using namespace ros;
using namespace message_filters;


// 和随机数相关
#define N 999 //随机数精确到三位小数。
#define dif_cy_threadshold 20 //the biggest y dif for bbox pairs
#define w1 0.3 //weight for dist
#define w2 2 //weight for IoU
#define offset_x 0//0.05
#define offset_y 0
#define offset_z 0
#define scale_offset 1.05//1.05

bool read_2d_object(string root_path, vector<bbox>& bbox_list_l, vector<bbox>& bbox_list_r, int count);
bool screen_2d_object(vector<bbox> bbox_list_l, vector<bbox> bbox_list_r, vector<bbox>& bbox_list_after_screen_l, vector<bbox>& bbox_list_after_screen_r);
bool show_bbox_on_sparity(string root_path, vector<bbox> bbox_list_l, Mat disparity, int count);
bool cluster_pc(pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object, vector<pcl::PointCloud<pcl::PointXYZRGB>>& pc_clusters, int frame_index, int object_index, string object_pc_before_cluster_path);
bool trans2w(Mat Twc, box3d b3d, box3d& b3d_w);
bool transAABB2w(Mat Tcw, AABB_box3d b3d, AABB_box3d& b3d_w);
cv::Mat Cal3D_2D(pcl::PointXYZ point3D, Mat projectionMatrix, Size imageSize);
cv::Mat Cal3D_2D(pcl::PointXYZRGB point3D, Mat projectionMatrix, Size imageSize);
float intersectRect(const cv::Rect rectA, const cv::Rect rectB, cv::Rect& intersectRect);
bool pick_cluster(bbox b, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pc_clusters, pcl::PointCloud<pcl::PointXYZ>::Ptr& object_cluster, double object_point_num, Size imageSize, Mat projectionMatrix, box3d& b3d);
bool proj3d2d(AABB_box3d b3d_w, bbox& b, Mat projectionMatrix, Size imageSize);
//bool cal_ious(bbox b, vector<bbox> bbox_list, Mat& ious);

#endif //OBJECT_DETECTION_BBOX2BOX3D_H