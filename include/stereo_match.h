//
// Created by shenyl on 2020/6/27.
//

#ifndef OBJECT_DETECTION_STEREO_MATCH_H
#define OBJECT_DETECTION_STEREO_MATCH_H


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "elas.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <map>
#include <string>

using namespace cv;
using namespace std;

void split(string &s, vector<string> &list1);
void read_calib_parameters(string calib_parameters, Mat& cameraMatrix_L, Mat& distCoeffs_L, Mat& cameraMatrix_R, Mat& distCoeffs_R,
                           Mat& R, Mat& T, Size& imageSize);
Rect stereoRectification(Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2,
                         Size& imageSize, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2);
bool Rectification(string root_path, string imageName_L, string imageName_R, Mat& img1_rectified,
                   Mat& img2_rectified, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2, Rect validRoi[2], int count);
bool Rectification(string root_path, Mat img1, Mat img2, Mat& img1_rectified,
                   Mat& img2_rectified, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2, Rect validRoi[2], int count);
bool computeDisparityImage(string root_path, Mat& img1_rectified,
                           Mat& img2_rectified, Mat& disparity_l, Mat& disparity_r, int count, string algorithm);
bool filterDisparityImage(string root_path, Mat rectified_l, Mat rectified_r, Mat& disparity_l, Mat& disparity_r, Mat& filtered_disparities, int v0, int count, int iou_offset);
bool filterDisparityImage_bm(string root_path, Mat rectified_l, Mat rectified_r, Mat& disparity_l, Mat& disparity_r, Mat& filtered_disparities, int count);

#endif //OBJECT_DETECTION_STEREO_MATCH_H