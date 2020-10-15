//
// Created by shenyl on 2020/8/23.
//

#ifndef OBJECT_DETECTION_YOLO_OPENCV_H
#define OBJECT_DETECTION_YOLO_OPENCV_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include "bbox.h"
using namespace std;
using namespace cv;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, vector<Mat>& outs, vector<bbox>& bbox_list, vector<string> classes);
// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, vector<string> classes);

void yolo_det(string root_path, Mat frame, int count, vector<string> classes, cv::dnn::Net net, vector<bbox>& bbox_list, int v0, Mat& object_mask);

#endif //OBJECT_DETECTION_YOLO_OPENCV_H
