//
// Created by shenyl on 2020/6/26.
//

#include "stereo_match.h"

static int print_help()
{
    cout <<
         " Given the root dir of the exp dir and the match algorithm \n"
         " output the rectified images and sparities\n"<< endl;
    cout << "Usage:\n // usage：./stereo_match -d=<dir default=/home/shenyl/Documents/sweeper/data/> -a=<stereo match algorithm default = sgbm>\n" << endl;
}

//attention: data must be read as double to be the same with the parameter definition of cv::stereoRectify function
void read_calib_parameters(string calib_parameters, Mat& cameraMatrix_L, Mat& distCoeffs_L, Mat& cameraMatrix_R, Mat& distCoeffs_R,
                           Mat& R, Mat& T, Size& imageSize) {
    cv::FileStorage fs_read(calib_parameters, cv::FileStorage::READ);
    fs_read["cameraMatrixL"] >> cameraMatrix_L;
    fs_read["distCoeffsL"] >> distCoeffs_L;
    fs_read["cameraMatrixR"] >> cameraMatrix_R;
    fs_read["distCoeffsR"] >> distCoeffs_R;
    fs_read["R"] >> R;
    fs_read["T"] >> T;
    fs_read["imageSize"] >> imageSize;
    fs_read.release();

}

Rect stereoRectification(Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2,
                         Size& imageSize, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2)
{
    Rect validRoi[2];
//    cout<<cameraMatrix1<<endl<<distCoeffs1<<endl<<cameraMatrix2<<endl<<distCoeffs2<<endl<<imageSize<<endl<<R <<endl<<T<<endl;
//    flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大
//    alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。如果设置为0，那么校正后图像只有有效的部分会被显示（没有黑色的部分），如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。
//    alpha 参数必须设置为0，否则图像可能为倒像？？？
    stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
                  R, T, R1, R2, P1, P2, Q, 0, 0, imageSize, &validRoi[0], &validRoi[1]);
    cout << "R1:" << endl;
    cout << R1 << endl;
    cout << "R2:" << endl;
    cout << R2 << endl;
    cout << "P1:" << endl;
    cout << P1 << endl;
    cout << "P2:" << endl;
    cout << P2 << endl;
    cout << "Q:" << endl;
    cout << Q << endl;
    initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_32FC1, mapl1, mapl2);
    initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_32FC1, mapr1, mapr2);
    return validRoi[0], validRoi[1];
}

bool Rectification(string root_path, string imageName_L, string imageName_R, Mat& img1_rectified,
                               Mat& img2_rectified, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2, Rect validRoi[2], int count) {
    Size imageSize;
    string left_rectified = root_path + "rectified/left/";
    string right_rectified = root_path + "rectified/right/";
    string pairs_rectified = root_path + "rectified/pairs/";
    Mat img1 = imread(imageName_L);
    Mat img2 = imread(imageName_R);
    if (img1.empty() | img2.empty()) {
        cout << imageName_L << " , " <<imageName_R<<" is not exist" << endl;
    }
    Mat gray_img1, gray_img2;
    cvtColor(img1, gray_img1, COLOR_BGR2GRAY);
    cvtColor(img2, gray_img2, COLOR_BGR2GRAY);
    imageSize.width = img1.cols; // 获取图片的宽度
    imageSize.height = img1.rows; // 获取图片的高度
    Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC1); // 注意数据类型
    Mat canLeft = canvas(Rect(0, 0, imageSize.width, imageSize.height));
    Mat canRight = canvas(Rect(imageSize.width, 0, imageSize.width, imageSize.height));

    Mat img1_rectified_grey, img2_rectified_grey;
    remap(gray_img1, img1_rectified_grey, mapl1, mapl2, INTER_LINEAR);
    remap(gray_img2, img2_rectified_grey, mapr1, mapr2, INTER_LINEAR);
    remap(img1, img1_rectified, mapl1, mapl2, INTER_LINEAR);
    remap(img2, img2_rectified, mapr1, mapr2, INTER_LINEAR);

//    imshow("img1_rectified", img1_rectified);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }

//    char left_file[200];
//    sprintf(left_file, "%06d.jpg", count*2);
//    imwrite(left_rectified + left_file, img1_rectified);
//    char right_file[200];
//    sprintf(right_file, "%06d.jpg", count*2+1);
//    imwrite(right_rectified + right_file, img2_rectified);

    img1_rectified_grey.copyTo(canLeft);
    img2_rectified_grey.copyTo(canRight);

    rectangle(canLeft, validRoi[0], Scalar(255, 255, 255), 5, 8);
    rectangle(canRight, validRoi[1], Scalar(255, 255, 255), 5, 8);
    for (int j = 0; j <= canvas.rows; j += 16)
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

    char pairs_file[200];
    sprintf(pairs_file, "%06d.jpg", count);
    imwrite(pairs_rectified + pairs_file, canvas);
//    imshow("rectified", canvas);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }
}

bool Rectification(string root_path, Mat img1, Mat img2, Mat& img1_rectified,
                   Mat& img2_rectified, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2, Rect validRoi[2], int count) {
    Size imageSize;
    string left_rectified = root_path + "rectified/left/";
    string right_rectified = root_path + "rectified/right/";
    string pairs_rectified = root_path + "rectified/pairs/";

    if (img1.empty() | img2.empty()) {
        cout << img1 << " , " <<img2<<" is not exist" << endl;
    }
//    Mat gray_img1, gray_img2;
    Mat gray_img1, gray_img2;
    cvtColor(img1, gray_img1, COLOR_BGR2GRAY);
    cvtColor(img2, gray_img2, COLOR_BGR2GRAY);
    imageSize.width = img1.cols; // 获取图片的宽度
    imageSize.height = img1.rows; // 获取图片的高度
    Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC1); // 注意数据类型
    Mat canLeft = canvas(Rect(0, 0, imageSize.width, imageSize.height));
    Mat canRight = canvas(Rect(imageSize.width, 0, imageSize.width, imageSize.height));


//    remap(gray_img1, img1_rectified, mapl1, mapl2, INTER_LINEAR);
//    remap(gray_img2, img2_rectified, mapr1, mapr2, INTER_LINEAR);
    remap(img1, img1_rectified, mapl1, mapl2, INTER_LINEAR);
    remap(img2, img2_rectified, mapr1, mapr2, INTER_LINEAR);

//      imshow("img1_rectified", img1_rectified);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }

//    char left_file[200];
//    sprintf(left_file, "%06d.jpg", count*2);
//    imwrite(left_rectified + left_file, img1_rectified);
//    char right_file[200];
//    sprintf(right_file, "%06d.jpg", count*2+1);
//    imwrite(right_rectified + right_file, img2_rectified);

//    img1_rectified.copyTo(canLeft);
//    img2_rectified.copyTo(canRight);
//
//    rectangle(canLeft, validRoi[0], Scalar(255, 255, 255), 5, 8);
//    rectangle(canRight, validRoi[1], Scalar(255, 255, 255), 5, 8);
//    for (int j = 0; j <= canvas.rows; j += 16)
//        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
//
//    char pairs_file[200];
//    sprintf(pairs_file, "%06d.jpg", count);
//    imwrite(pairs_rectified + pairs_file, canvas);
//    imshow("rectified", canvas);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }
}

bool computeDisparityImage(string root_path, Mat& img1_rectified,
                           Mat& img2_rectified, Mat& disparity_l, Mat& disparity_r, int count, string algorithm)
{
    enum StringValue { evNotDefined,
        evStringValue1,
        evStringValue2,
        evStringValue3,
        evStringValue4,
        evEnd };
    map<std::string, StringValue> s_mapStringValues;
    s_mapStringValues["bm"] = evStringValue1;
    s_mapStringValues["sgbm"] = evStringValue2;
    s_mapStringValues["elas"] = evStringValue3;
    s_mapStringValues["filtered_sgbm"] = evStringValue4;
    s_mapStringValues["end"] = evEnd;

    switch(s_mapStringValues[algorithm]){
        case evStringValue1:
        {
            // 进行立体匹配 bm
            Ptr<StereoBM> bm;
            bm = StereoBM::create(16, 9); // Ptr<>是一个智能指针
            Ptr<StereoMatcher> right_bm = ximgproc::createRightMatcher(bm);
            bm->compute(img1_rectified, img2_rectified, disparity_l); // 计算视差图 input rectified must be one-channel image
            right_bm->compute(img2_rectified, img1_rectified, disparity_r); // 计算视差图 input rectified must be one-channel image
            disparity_l.convertTo(disparity_l, CV_32F, 1.0 / 16);
            disparity_r.convertTo(disparity_r, CV_32F, 1.0 / 16);
            // 归一化视差映射
            break;
        }
        case evStringValue2:
        {
            // 进行立体匹配 sgbm
            Ptr<StereoSGBM> sgbm;
//            cout<<"image type: "<<img1_rectified.type()<<endl;
//            cout<<"image shape"<<img1_rectified.cols << "," << img1_rectified.rows<<endl;
            //todo: the numDisparities determine the nearest distance for det, try to modify the numDisparities according to the possible distance
            sgbm = cv::StereoSGBM::create(
                    0, 160, 8, 8*8*8, 32*8*8, 1, 1, 10, 200, 200, cv::StereoSGBM::MODE_SGBM);
            Ptr<StereoMatcher> right_sgbm = ximgproc::createRightMatcher(sgbm);
            sgbm->compute(img1_rectified, img2_rectified, disparity_l); // 计算视差图
            right_sgbm->compute(img2_rectified, img1_rectified, disparity_r);
//            cout<<"type before: "<<disparity_l.type()<<endl;
//            disparity_l.convertTo(disparity_l, CV_8U);
//            disparity_r.convertTo(disparity_r, CV_8U);
//            cout<<"type after: "<<disparity_l.type()<<endl;
            disparity_l.convertTo(disparity_l, CV_32F, 1.0 / 16);
            disparity_r.convertTo(disparity_r, CV_32F, 1.0 / 16);
            // 归一化视差映射
//            normalize(disparity, disparity, 0, 256, NORM_MINMAX, CV_8U);
            break;
        }
        case evStringValue3:
        {
            // generate disparity image using LIBELAS
            cv::Mat disp_l,disp_r,disp8u_l,disp8u_r;
            double minVal; double maxVal; //视差图的极值
            int bd = 0;
            const int32_t dims[3] = {img1_rectified.cols,img1_rectified.rows,img1_rectified.cols};
            cv::Mat leftdpf = cv::Mat::zeros(cv::Size(img1_rectified.cols,img1_rectified.rows), CV_32F);
            cv::Mat rightdpf = cv::Mat::zeros(cv::Size(img1_rectified.cols,img1_rectified.rows), CV_32F);
            Elas::parameters param;
            param.postprocess_only_left = false;
            Elas elas(param);
            elas.process(img1_rectified.data,img2_rectified.data,leftdpf.ptr<float>(0),rightdpf.ptr<float>(0),dims);

            cv::Mat(leftdpf(cv::Rect(bd,0,img1_rectified.cols,img1_rectified.rows))).copyTo(disp_l);
            cv::Mat(rightdpf(cv::Rect(bd,0,img2_rectified.cols,img2_rectified.rows))).copyTo(disp_r);
//            disp_l.convertTo(disp8u_l, CV_8U);
            disp_l.convertTo(disparity_l, CV_32F, 1.0 / 16);
            disp_r.convertTo(disparity_r, CV_32F, 1.0 / 16);
            break;
        }
        case evStringValue4:
        {
            // 进行立体匹配 sgbm
//            Mat m_matDisp16;
//            Mat filtered_Disp;
            Ptr<StereoSGBM> sgbm;
//            cout<<"image type: "<<img1_rectified.type()<<endl;
//            cout<<"image shape"<<img1_rectified.cols << "," << img1_rectified.rows<<endl;
            //todo: the numDisparities determine the nearest distance for det, try to modify the numDisparities according to the possible distance
            sgbm = cv::StereoSGBM::create(
                    0, 80, 8, 8*8*8, 32*8*8, 1, 1, 10, 200, 200, cv::StereoSGBM::MODE_SGBM);
            Ptr<StereoMatcher> right_sgbm = ximgproc::createRightMatcher(sgbm);
            sgbm->compute(img1_rectified, img2_rectified, disparity_l); // 计算视差图
            right_sgbm->compute(img2_rectified, img1_rectified, disparity_r);

            // 归一化视差映射
//            normalize(disparity, disparity, 0, 256, NORM_MINMAX, CV_8U);
            break;
        }

    }
    // write sparities to files
    string disparities = root_path + "disparities/";
//    imshow("disparity_l", disparity_l);
//    imshow("disparity_r", disparity_r);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }
    char disparities_file[200];
    sprintf(disparities_file, "%06d.jpg", count);
    imwrite(disparities + disparities_file, disparity_l);
    return true;
}

bool filterDisparityImage(string root_path, Mat rectified_l, Mat rectified_r, Mat& disparity_l_roi, Mat& disparity_r_roi, Mat& filtered_disparities, int v0, int count, int iou_offset)
{
//    resize(rectified_l,rectified_l,Size(0,0),0.8,0.8);
//    resize(rectified_r,rectified_r,Size(0,0),0.8,0.8);
    cvtColor(rectified_l, rectified_l, COLOR_BGR2GRAY);
    cvtColor(rectified_r, rectified_r, COLOR_BGR2GRAY);
    Mat disp;
    Mat filtered_Disp;
    Mat rectified_l_roi(rectified_l, Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset));
    Mat rectified_r_roi(rectified_r, Rect(0, v0-iou_offset, rectified_r.cols, rectified_r.rows-v0+iou_offset));

    Ptr<StereoSGBM> sgbm;
    sgbm = cv::StereoSGBM::create(
            0, 160, 8, 8*8*8, 32*8*8, 1, 1, 10, 200, 200, cv::StereoSGBM::MODE_SGBM);
    Ptr<StereoMatcher> right_sgbm = ximgproc::createRightMatcher(sgbm);
    sgbm->compute(rectified_l_roi, rectified_r_roi, disparity_l_roi); // 计算视差图
    right_sgbm->compute(rectified_r_roi, rectified_l_roi, disparity_r_roi);

    //filter the disparities
    Ptr<ximgproc::DisparityWLSFilter> wls_filter = ximgproc::createDisparityWLSFilter(sgbm);
    wls_filter->setLambda(8000.);
    wls_filter->setSigmaColor(0.5);
//            wls_filter->filter(matDispLeft16, img1_rectified, m_matDisp16, matDispRight16);
    wls_filter->filter(disparity_l_roi, rectified_l_roi, disp, disparity_r_roi);

//    resize(disparity_l,disparity_l,Size(0,0),1.25,1.25);
//    resize(rectified_r,rectified_r,Size(0,0),1.25,1.25);
//    resize(disp,disp,Size(0,0),1.25,1.25);

    disp.convertTo(filtered_disparities, CV_32F, 1.0 / 16);
    disparity_l_roi.convertTo(disparity_l_roi, CV_32F, 1.0 / 16);
    disparity_r_roi.convertTo(disparity_r_roi, CV_32F, 1.0 / 16);
    // 归一化视差映射,用于可视化
//    normalize(filtered_disparities, filtered_disparities, 0, 256, NORM_MINMAX, CV_8U);

//    imshow("rectified_r_roi", rectified_r_roi);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }
//    imshow("filtered_Disp", filtered_disparities);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }
//    string filtered_disparity_path = root_path + "filtered_disparities/";
//    char filtered_disparities_file[30];
//    sprintf(filtered_disparities_file, "%06d.jpg", count);
//    imwrite(filtered_disparity_path + filtered_disparities_file, filtered_disparities);
//    string disparities = root_path + "disparities/";
//    char disparities_file[30];
//    sprintf(disparities_file, "%06d.jpg", count);
//    imwrite(disparities + disparities_file, disparity_l_roi);
    return true;
}

bool filterDisparityImage_bm(string root_path, Mat rectified_l, Mat rectified_r, Mat& disparity_l, Mat& disparity_r, Mat& filtered_disparities, int count)
{
    Mat disp;
    Mat filtered_Disp;
    //filter the disparities
    Ptr<StereoBM> bm;
    bm = StereoBM::create(160, 9); // Ptr<>是一个智能指针
    Ptr<StereoMatcher> right_bm = ximgproc::createRightMatcher(bm);
    bm->compute(rectified_l, rectified_r, disparity_l); // 计算视差图 input rectified must be one-channel image
    right_bm->compute(rectified_r, rectified_l, disparity_r);

    Ptr<ximgproc::DisparityWLSFilter> wls_filter = ximgproc::createDisparityWLSFilter(bm);
    wls_filter->setLambda(1000.);//1000
    wls_filter->setSigmaColor(2.0);
//            wls_filter->filter(matDispLeft16, img1_rectified, m_matDisp16, matDispRight16);
    wls_filter->filter(disparity_l, rectified_l, disp, disparity_r);
    disp.convertTo(filtered_disparities, CV_32F, 1.0 / 16);
    disparity_l.convertTo(disparity_l, CV_32F, 1.0 / 16);
    disparity_r.convertTo(disparity_r, CV_32F, 1.0 / 16);
    // 归一化视差映射,用于可视化
//    normalize(filtered_disparities, filtered_disparities, 0, 256, NORM_MINMAX, CV_8U);
    string filtered_disparity_path = root_path + "filtered_disparities/";
//    imshow("filtered_Disp", filtered_Disp);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }
    char filtered_disparities_file[30];
    sprintf(filtered_disparities_file, "%06d.jpg", count);
    imwrite(filtered_disparity_path + filtered_disparities_file, filtered_disparities);
    string disparities = root_path + "disparities/";
    char disparities_file[30];
    sprintf(disparities_file, "%06d.jpg", count);
    imwrite(disparities + disparities_file, disparity_r);
    return true;
}





