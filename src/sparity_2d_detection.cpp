//
// Created by shenyl on 2020/7/21.
//
#include "sparity_2d_detection.h"
//#define ros_box_pub
#define ros_points_pub
#define ros_object_points_pub
//#define pcl_show
//#define write_pcd
//#define image save
//#define image show


bool get_obstacle_mask(string root_path, Mat disparity_l, Mat& obstacle_mask, int v0, int count, double h, double b, int iou_offset, double h0)
{
    b = -b/1000;
    h = h/1000;
    for(int v = 0 ;v<disparity_l.rows;v++)
    {
        double gd_disparity = b*(v - iou_offset)/(h-h0);//比地面高5cm作为障碍物
        for(int u = 0; u<disparity_l.cols;u++)
        {
            Point point;
            point.x = u;
            point.y = v;
            float d = disparity_l.at<float>(point);
            if(d>gd_disparity)
            {
                obstacle_mask.at<unsigned char>(v,u)=255;
            }
        }
    }
    Mat element = getStructuringElement(MORPH_CROSS, Size( 3, 3), Point(-1,-1));

    erode(obstacle_mask, obstacle_mask, element);
    dilate( obstacle_mask, obstacle_mask, element );
#ifdef image_save
//    string obstacle_disp_path = root_path + "obstacle_mask/";
//    char obstacle_disp_file[20];
//    sprintf(obstacle_disp_file, "%06d.jpg", count);
//    imwrite(obstacle_disp_path+obstacle_disp_file, obstacle_mask);
#endif
    return true;
}

bool get_masked_disp(string root_path, Mat obstacle_mask, Mat disparity_l, Mat& masked_disparity, int count)
{
    bitwise_and(disparity_l, disparity_l, masked_disparity, obstacle_mask);
#ifdef image_show
//    imshow("masked_disparity", masked_disparity);
//    if (waitKey(0) == 27) {
//        destroyAllWindows();
//    }
#endif
#ifdef image_save
//    string masked_disparity_path = root_path + "masked_disparity/";
//    char masked_disparity_file[20];
//    sprintf(masked_disparity_file, "%06d.jpg", count);
//    imwrite(masked_disparity_path+masked_disparity_file, masked_disparity);
#endif
    return true;
}

// 鼠标回调函数，点击视差图显示深度
void onMouse(int event, int x, int y, int flags, void *param)
{
    int v0 = 245;
    Point point;
    point.x = x;
    point.y = y+v0;
    if(event == EVENT_LBUTTONDOWN)
    {
        cout <<result3DImage.at<Vec3f>(point) << endl;
    }
}

// 对masked 3d image提取点云，并聚类，得到3d检测框
bool obj_det_3d_with_sparity(string root_path, Mat result3DImage, vector<AABB_box3d>& AABB_list, vector<bbox>& bbox_list, int count, Mat P1, Size imageSize) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    int index=0;
    pc_object_ptr->width = result3DImage.cols*result3DImage.rows;
    pc_object_ptr->height = 1;
    pc_object_ptr->is_dense = false;  //不是稠密型的
    pc_object_ptr->points.resize(pc_object_ptr->width * pc_object_ptr->height);  //点云总数大小
    for (int u = 0; u < result3DImage.cols; u++) {
        for (int v = 0; v < result3DImage.rows; v++) {
            Point point;
            point.x = u;
            point.y = v;
            if (result3DImage.at<Vec3f>(point)[2] < 450 | result3DImage.at<Vec3f>(point)[2] > 2000)
                continue;
            pc_object_ptr->points[index].x = result3DImage.at<Vec3f>(point)[0]/1000;
            pc_object_ptr->points[index].y = result3DImage.at<Vec3f>(point)[1]/1000;
            pc_object_ptr->points[index++].z = result3DImage.at<Vec3f>(point)[2]/1000;
        }
    }

    //voxel downsample //!!!!voxel之后会出现一些在零点的点，直接进行聚类会有错误
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    if(pc_object_ptr->points.size()>1000)
    {
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        vox.setInputCloud (pc_object_ptr);
        vox.setLeafSize (0.002f, 0.002f, 0.010f);
        vox.filter (*pc_object_filtered_ptr);
    }
    else
    {
        pc_object_filtered_ptr = pc_object_ptr;
    }


    if(pc_object_filtered_ptr->points.size ()<20) return false;

    // filtering cloud by passthrough
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (pc_object_filtered_ptr);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.50, 2.0);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*pc_object_filtered_ptr);
    if(pc_object_filtered_ptr->points.size ()<20) return false;

    // cluster
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (pc_object_filtered_ptr);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.025); // 2.5cm
    ec.setMinClusterSize (20);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pc_object_filtered_ptr);
    ec.extract (cluster_indices);

    int j = 0;
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pc_clusters ;
#ifdef ros_points_pub
    pcl::PointCloud<pcl::PointXYZ>::Ptr clusters (new pcl::PointCloud<pcl::PointXYZ>);
    clusters->width    = pc_object_filtered_ptr->points.size ();
    clusters->height   = 1;
    clusters->is_dense = false;  //不是稠密型的
    clusters->points.resize (clusters->width * clusters->height);  //点云总数大小
#endif


    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        bbox b;
        cv::Mat point2D_this(2, 1, CV_16UC1, cv::Scalar(0));
        cv::Mat point2D_min(2, 1, CV_16UC1, cv::Scalar(0));
        point2D_min.at<int>(0) = 10000;
        point2D_min.at<int>(1) = 10000;
        cv::Mat point2D_max(2, 1, CV_16UC1, cv::Scalar(0));
        point2D_max.at<int>(0) = 0;
        point2D_max.at<int>(1) = 0;

        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        pc_cluster->width    = pc_object_filtered_ptr->points.size ();
        pc_cluster->height   = 1;
        pc_cluster->is_dense = false;  //不是稠密型的
        pc_cluster->points.resize (pc_cluster->width * pc_cluster->height);  //点云总数大小
        int i=0;
        // cluster parameters
        double pointnum = 0;
        double min_z = 10000;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
            pc_cluster->points[i].x = pc_object_filtered_ptr->points[*pit].x;
            pc_cluster->points[i].y = pc_object_filtered_ptr->points[*pit].y;
            pc_cluster->points[i].z = pc_object_filtered_ptr->points[*pit].z;

            point2D_this = Cal3D_2D(pc_object_filtered_ptr->points[*pit], P1, imageSize);
//            cout<<"Point2d: "<<point2D_this.at<int>(0)<<","<<point2D_this.at<int>(1)<<endl;
            if(point2D_this.at<int>(0)<=point2D_min.at<int>(0))
                point2D_min.at<int>(0) = point2D_this.at<int>(0);
            if(point2D_this.at<int>(1)<=point2D_min.at<int>(1))
                point2D_min.at<int>(1) = point2D_this.at<int>(1);
            if(point2D_this.at<int>(0)>=point2D_max.at<int>(0))
                point2D_max.at<int>(0) = point2D_this.at<int>(0);
            if(point2D_this.at<int>(1)>=point2D_max.at<int>(1))
                point2D_max.at<int>(1) = point2D_this.at<int>(1);

            pointnum = pointnum +1;
            if(pc_object_filtered_ptr->points[i].z<min_z)
            {
                min_z = pc_cluster->points[i].z;
            }
            i=i+1;
        }
        min_z = min_z / scale_offset;

        //利用点数和距离的关系对聚类进行筛选
        if ( pointnum * min_z>50 )
        {
            pc_clusters.push_back(pc_cluster);
            b._xmin = point2D_min.at<int>(0);
            b._ymin = point2D_min.at<int>(1);
            b._xmax  = point2D_max.at<int>(0);
            b._ymax = point2D_max.at<int>(1);
            bbox_list.push_back(b);
#ifdef ros_points_pub
            *clusters = *clusters + *pc_cluster;
#endif
        }

        j++;
    }

#ifdef ros_points_pub
    sensor_msgs::PointCloud2 Obj_Point_msg;
    pcl::toROSMsg(*clusters, Obj_Point_msg);
    Obj_Point_msg.header.stamp = ros::Time::now();
    Obj_Point_msg.header.frame_id = "map";
    obj_points_pub_.publish(Obj_Point_msg);
#endif

    {
        //对每一帧的所有聚类，寻找最小包络框，并显示
#ifdef pcl_show
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (0, 0, 0);
        viewer->addCoordinateSystem (1.0);
        viewer->initCameraParameters ();
#endif

#ifdef ros_box_pub
        visualization_msgs::Marker box_lines;
        box_lines.header.frame_id = "map";
        box_lines.header.stamp = ros::Time::now();
        box_lines.type = visualization_msgs::Marker::LINE_LIST;
        box_lines.action = visualization_msgs::Marker::ADD;
        box_lines.scale.x = 0.005;
        box_lines.pose.orientation.x = 0.0;
        box_lines.pose.orientation.y = 0.0;
        box_lines.pose.orientation.z = 0.0;
        box_lines.pose.orientation.w = 1.0;
        box_lines.color.a = 1.0; // Don't forget to set the alpha!
        box_lines.color.r = 1.0;
        box_lines.color.g = 0.0;
        box_lines.color.b = 0.0;

#endif
        for(int i=0;i<pc_clusters.size();i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster = pc_clusters[i];
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);;

            //原始点云存在很多零点，通过滤波器去除
            pcl::PassThrough<pcl::PointXYZ> pass;
            pass.setInputCloud (cluster);
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (0.5, 2.0);
            //pass.setFilterLimitsNegative (true);
            pass.filter (*cluster_filtered_ptr);

            pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
            feature_extractor.setInputCloud (cluster_filtered_ptr);
            feature_extractor.compute ();

            std::vector <float> moment_of_inertia;
            std::vector <float> eccentricity;
            pcl::PointXYZ min_point_AABB;
            pcl::PointXYZ max_point_AABB;

            feature_extractor.getAABB (min_point_AABB, max_point_AABB);
            double p_x = (min_point_AABB.x + max_point_AABB.x)/2/scale_offset;
            double p_y = (min_point_AABB.y + max_point_AABB.y)/2/scale_offset;
            double p_z = (min_point_AABB.z + max_point_AABB.z)/2/scale_offset;
            double width = (max_point_AABB.x - min_point_AABB.x)/scale_offset;
            double height = (max_point_AABB.y - min_point_AABB.y)/scale_offset;
            double length = (max_point_AABB.z - min_point_AABB.z)/scale_offset;
            AABB_box3d aabb_box3d(p_x, p_y, p_z, width, length, height);
            AABB_list.push_back(aabb_box3d);
#ifdef pcl_show
//             show by pcl visualization
            char cloud_id[20];
            char AABB_id[20];
            sprintf(cloud_id, "cloud%i", i);
            sprintf(AABB_id, "AABB%i", i);
            viewer->addPointCloud<pcl::PointXYZ> (cluster_filtered_ptr, cloud_id);
            viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, AABB_id);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, AABB_id);
#endif
#ifdef ros_box_pub
            //publish by ros
            // 8 points for each cube
            vector<geometry_msgs::Point> points8;
            geometry_msgs::Point p1;
            geometry_msgs::Point p8;
            p1.x = min_point_AABB.x; p1.y = min_point_AABB.y; p1.z = min_point_AABB.z;
            p8.x = max_point_AABB.x; p8.y = max_point_AABB.y; p8.z = max_point_AABB.z;
            geometry_msgs::Point p2 = p1;
            geometry_msgs::Point p3 = p1;
            geometry_msgs::Point p4 = p1;
            geometry_msgs::Point p5 = p8;
            geometry_msgs::Point p6 = p8;
            geometry_msgs::Point p7 = p8;
            p2.x = p8.x;
            p3.z = p8.z;
            p4.x = p8.x;
            p4.z = p8.z;
            p5.x = p1.x;
            p5.z = p1.z;
            p6.z = p1.z;
            p7.x = p1.x;
            points8.push_back(p1);
            points8.push_back(p2);
            points8.push_back(p3);
            points8.push_back(p4);
            points8.push_back(p5);
            points8.push_back(p6);
            points8.push_back(p7);
            points8.push_back(p8);

            // 12 lines with 24 points
            vector<int> line_points = {0,1,2,3,0,2,1,3,4,5,6,7,4,6,5,7,0,4,1,5,2,6,3,7};
            for (int i=0;i<24;i++)
            {
                box_lines.points.push_back(points8[line_points[i]]);
            }
#endif
        }
#ifdef pcl_show
        while(!viewer->wasStopped())
        {
            viewer->spinOnce (100);
        }
#endif
#ifdef ros_box_pub
        obj_box_pub_.publish(box_lines);
#endif
    }
    return true;
}

//仅利用视差图进行三维物体检测：得到视差图、得到障碍物mask并作用于3d image, 对masked 3d image直接聚类，得到3d检测框
bool obstacle_det(string root_path, Mat rectified_l, Mat rectified_r, Mat Q, Mat P1, int count, vector<AABB_box3d>& aabb_list, vector<bbox>& bbox_list, double h, double b, int v0)
{
    Mat disparity_l, disparity_r;
    Mat filtered_disparity;

    //publish rectified images
    sensor_msgs::ImagePtr rectified_img_left = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rectified_l).toImageMsg();
    rectified_left_image_pub_.publish(rectified_img_left);
    sensor_msgs::ImagePtr rectified_img_right = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rectified_r).toImageMsg();
    rectified_right_image_pub_.publish(rectified_img_right);

    auto t0 = std::chrono::steady_clock::now();
    int iou_offset = 30;
    // Step1 计算视差图,对视差图进行加权最小二乘滤波
    filterDisparityImage(root_path, rectified_l, rectified_r, disparity_l, disparity_r, filtered_disparity, v0, count, iou_offset);
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed1 = t1 - t0; // std::micro 表示以微秒为时间单位
    cout<<"time for filterDisparityImage: "<<elapsed1.count()/1000000<<endl;

    // Step2 得到障碍物mask
    Mat obstacle_mask(rectified_l.rows-v0+iou_offset, rectified_l.cols, CV_8UC1, Scalar::all(0));
    get_obstacle_mask(root_path, filtered_disparity, obstacle_mask, v0, count, h, b, iou_offset);
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed2 = t2 - t1; // std::micro 表示以微秒为时间单位
//    cout<<"time for get_obstacle_mask: "<<elapsed2.count()/1000000<<endl;
    // publish obstacle mask
    sensor_msgs::ImagePtr obstacle_mask_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", obstacle_mask).toImageMsg();
    obstacle_mask_pub_.publish(obstacle_mask_msg);

    // Step3 将障碍物mask和视差图相与
    Mat masked_disparity(rectified_l.rows, rectified_l.cols, CV_8UC1, Scalar::all(0));
    Mat masked_disparity_roi(masked_disparity, Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset));
    get_masked_disp(root_path, obstacle_mask, filtered_disparity, masked_disparity_roi, count);
    masked_disparity_roi.copyTo(masked_disparity(Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset)));

    // Step4 利用视差图和投影参数Q得到三维图像
    reprojectImageTo3D(masked_disparity, result3DImage, Q);
    Mat result3DImage_iou(result3DImage, Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset));
#ifdef image_show
    imshow("masked_disparity", masked_disparity_roi);
    setMouseCallback("masked_disparity", onMouse);
    if (waitKey(0) == 27) {
        destroyAllWindows();
    }
#endif
    //Step5 三维物体检测, 对每一帧的障碍物进行聚类
    Size imageSize(rectified_l.cols, rectified_l.rows);
    if(obj_det_3d_with_sparity(root_path, result3DImage_iou, aabb_list, bbox_list, count, P1, imageSize))
    {
        auto t3 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed3 = t3 - t2; // std::micro 表示以微秒为时间单位
        cout<<"time for obj_det_3d_with_sparity: "<<elapsed3.count()/1000000<<endl;
        return true;
    }
    else
    {
        auto t3 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed3 = t3 - t2; // std::micro 表示以微秒为时间单位
        cout<<"time for obj_det_3d_with_sparity: "<<elapsed3.count()/1000000<<endl;
        return false;
    }

}

// 从3d image得到障碍物点云
bool obstacle_with_sparity(string root_path, Mat result3DImage, int count, Mat P1, Size imageSize, pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr)
{
    int index=0;
    pc_object_ptr->width = result3DImage.cols*result3DImage.rows;
    pc_object_ptr->height = 1;
    pc_object_ptr->is_dense = false;  //不是稠密型的
    pc_object_ptr->points.resize(pc_object_ptr->width * pc_object_ptr->height);  //点云总数大小
    for (int u = 0; u < result3DImage.cols; u++) {
        for (int v = 0; v < result3DImage.rows; v++) {
            Point point;
            point.x = u;
            point.y = v;
            if (result3DImage.at<Vec3f>(point)[2] < 450 | result3DImage.at<Vec3f>(point)[2] > 2000)
                continue;
            pc_object_ptr->points[index].x = result3DImage.at<Vec3f>(point)[0]/1000;
            pc_object_ptr->points[index].y = result3DImage.at<Vec3f>(point)[1]/1000;
            pc_object_ptr->points[index++].z = result3DImage.at<Vec3f>(point)[2]/1000;
        }
    }

    //voxel downsample //!!!!voxel之后会出现一些在零点的点，直接进行聚类会有错误
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    if(pc_object_ptr->points.size()>1000)
    {
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        vox.setInputCloud (pc_object_ptr);
        vox.setLeafSize (0.002f, 0.002f, 0.005f);
        vox.filter (*pc_object_filtered_ptr);
    }
    else
    {
        pc_object_filtered_ptr = pc_object_ptr;
    }


    if(pc_object_filtered_ptr->points.size ()<20) return false;

    // filtering cloud by passthrough
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (pc_object_filtered_ptr);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.50, 2.0);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*pc_object_filtered_ptr);
    if(pc_object_filtered_ptr->points.size ()<20) return false;


#ifdef ros_points_pub
    sensor_msgs::PointCloud2 Obj_Point_msg;
    pcl::toROSMsg(*pc_object_filtered_ptr, Obj_Point_msg);
    Obj_Point_msg.header.stamp = ros::Time::now();
    Obj_Point_msg.header.frame_id = "map";
    obj_points_pub_.publish(Obj_Point_msg);
#endif
    return true;
}

// 得到视差图、得到障碍物mask,发布障碍物部分的点云
bool obstacle_points(string root_path, Mat rectified_l, Mat rectified_r, Mat Q, Mat P1, int count,  double h, double b, int v0, pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr)
{
    Mat disparity_l, disparity_r;
    Mat filtered_disparity;

    //publish rectified images
    sensor_msgs::ImagePtr rectified_img_left = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rectified_l).toImageMsg();
    rectified_left_image_pub_.publish(rectified_img_left);
    sensor_msgs::ImagePtr rectified_img_right = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rectified_r).toImageMsg();
    rectified_right_image_pub_.publish(rectified_img_right);

    auto t0 = std::chrono::steady_clock::now();
    int iou_offset=30;
    // Step4 计算视差图,对视差图进行加权最小二乘滤波
    filterDisparityImage(root_path, rectified_l, rectified_r, disparity_l, disparity_r, filtered_disparity, v0, count, iou_offset);

    // Step5 得到障碍物mask
    Mat obstacle_mask_for_carpet(rectified_l.rows-v0+iou_offset, rectified_l.cols, CV_8UC1, Scalar::all(0));
    get_obstacle_mask(root_path, filtered_disparity, obstacle_mask_for_carpet, v0, count, h, b, iou_offset, 0.03);
    Mat obstacle_mask(rectified_l.rows-v0+iou_offset, rectified_l.cols, CV_8UC1, Scalar::all(0));
    get_obstacle_mask(root_path, filtered_disparity, obstacle_mask, v0, count, h, b, iou_offset, 0.02);

    // publish obstacle mask
    sensor_msgs::ImagePtr obstacle_mask_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", obstacle_mask_for_carpet).toImageMsg();
    obstacle_mask_pub_.publish(obstacle_mask_msg);

    // Step6 将障碍物mask和视差图相与
    Mat masked_disparity_obs(rectified_l.rows, rectified_l.cols, CV_8UC1, Scalar::all(0));
    Mat masked_disparity_obj(rectified_l.rows, rectified_l.cols, CV_8UC1, Scalar::all(0));
    Mat masked_disparity_roi_obs(masked_disparity_obs, Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset));
    Mat masked_disparity_roi_obj(masked_disparity_obj, Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset));

    // 将障碍物淹没作用于masked_disparity_roi_obs：得到障碍物部分
    get_masked_disp(root_path, obstacle_mask, filtered_disparity, masked_disparity_roi_obs, count);
    masked_disparity_roi_obs.copyTo(masked_disparity_obs(Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset)));
    // 直接从filtered视差图得到masked_disparity_roi_obj,不和mask想与
    filtered_disparity.copyTo(masked_disparity_obj(Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset)));
    // Step7 利用视差图和投影参数Q得到三维图像
    //得到用于障碍物检测的3d image
    Mat result3DImage_obs(rectified_l.rows, rectified_l.cols, CV_8UC1, Scalar::all(0));
    reprojectImageTo3D(masked_disparity_obs, result3DImage_obs, Q);
    Mat result3DImage_iou(result3DImage_obs, Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset));

    // 得到用于物体识别的3d image
    reprojectImageTo3D(masked_disparity_obj, result3DImage, Q);

#ifdef image_show
    imshow("masked_disparity", masked_disparity_roi);
    setMouseCallback("masked_disparity", onMouse);
    if (waitKey(0) == 27) {
        destroyAllWindows();
    }
        cout<<"finish reproject image to 3d"<<endl;
#endif

    //Step8 通过3d img 得到障碍物点云
    Size imageSize(rectified_l.cols, rectified_l.rows);
    if(obstacle_with_sparity(root_path, result3DImage_iou, count, P1, imageSize, pc_object_ptr))
    {
        auto t3 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed3 = t3 - t0; // std::micro 表示以微秒为时间单位
        cout<<"time for obstacle_with_sparity: "<<elapsed3.count()/1000000<<endl;
        return true;
    }
    else
    {
        auto t3 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed3 = t3 - t0; // std::micro 表示以微秒为时间单位
        cout<<"time for obstacle_with_sparity: "<<elapsed3.count()/1000000<<endl;
        return false;
    }
}


// 对2d检测部分的3d image，每个2d检测框内进行聚类并选择最好的聚类，得到每个2d检测对应的3d box
bool obj_det_3d_with_yolo_sparity(string root_path, vector<bbox> bbox_list_l, vector<AABB_box3d>& AABB_list, int count, Size imageSize, Mat P1)
{
    bool has_obj = false;
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> object_clusters;
    vector<box3d> b3d_list;
    for (int i=0;i<bbox_list_l.size();i++)
    {
        cout<<"/////////////object"<<i<<"/////////////////"<<endl;
        bbox b= bbox_list_l[i];

        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        pc_object_ptr->width    = (b._xmax - b._xmin)*(b._ymax-b._ymin);
        pc_object_ptr->height   = 1;
        pc_object_ptr->is_dense = false;  //不是稠密型的
        pc_object_ptr->points.resize (pc_object_ptr->width * pc_object_ptr->height);  //点云总数大小

        int p_index_object=0;
        for (int x = b._xmin;  x < b._xmax; x++) {
            for (int y = b._ymin; y < b._ymax; y++) {
                Point point;
                point.x = x;
                point.y = y;
                if (result3DImage.at<Vec3f>(point)[2] < 450 | result3DImage.at<Vec3f>(point)[2] > 2000)
                    continue;
                pc_object_ptr->points[p_index_object].x = result3DImage.at<Vec3f>(point)[0]/1000;
                pc_object_ptr->points[p_index_object].y = result3DImage.at<Vec3f>(point)[1]/1000;
                pc_object_ptr->points[p_index_object++].z = result3DImage.at<Vec3f>(point)[2]/1000;
            }
        }

        //voxel downsample //!!!!voxel之后会出现一些在零点的点，直接进行聚类会有错误
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        if(pc_object_ptr->points.size()>5000)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vox;
            vox.setInputCloud (pc_object_ptr);
            vox.setLeafSize (0.001f, 0.001f, 0.001f);
            vox.filter (*pc_object_filtered_ptr);
        }
        else
        {
            pc_object_filtered_ptr = pc_object_ptr;
        }
        if(pc_object_filtered_ptr->points.size()<20)
            continue;

        // filtering cloud by passthrough
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (pc_object_filtered_ptr);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.45, 2.0);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*pc_object_filtered_ptr);


        if(pc_object_filtered_ptr->points.size()<20)
            continue;
        // cluster the pc_object
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pc_clusters;
        auto start = std::chrono::steady_clock::now();
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (pc_object_filtered_ptr);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance (0.025); // 2.5cm
        ec.setMinClusterSize (20);
        ec.setMaxClusterSize (25000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (pc_object_filtered_ptr);
        ec.extract (cluster_indices);
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr pc_cluster (new pcl::PointCloud<pcl::PointXYZ>);
            pc_cluster->width    = 20000;
            pc_cluster->height   = 1;
            pc_cluster->is_dense = false;  //不是稠密型的
            pc_cluster->points.resize (pc_cluster->width * pc_cluster->height);  //点云总数大小
            int i=0;
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            {
//            cout<<pc_object_ptr->points[*pit].x<<" "<<pc_object_ptr->points[*pit].y<<" "<<pc_object_ptr->points[*pit].z<<endl;
                pc_cluster->points[i].x = pc_object_ptr->points[*pit].x;
                pc_cluster->points[i].y = pc_object_ptr->points[*pit].y;
                pc_cluster->points[i++].z = pc_object_ptr->points[*pit].z;
            }
            pc_clusters.push_back(pc_cluster);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start; // std::micro 表示以微秒为时间单位

        // find the best cluster for the object
        int object_point_num = pc_object_ptr->points.size();
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        box3d b3d;
        if(pick_cluster(b, pc_clusters, object_cluster, object_point_num, imageSize, P1, b3d))
        {
            object_clusters.push_back(object_cluster);
            b3d_list.push_back(b3d);
            has_obj = true;
        }
        else
        {
        }
    }
    cout<<"clusters num: "<<object_clusters.size()<<endl;
#ifdef  ros_object_points_pub
    if (object_clusters.size() == 0)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZ>);
        sensor_msgs::PointCloud2 Obj_Object_msg;
        pcl::toROSMsg(*cluster, Obj_Object_msg);
        Obj_Object_msg.header.stamp = ros::Time::now();
        Obj_Object_msg.header.frame_id = "map";
        obj_object_points_pub_.publish(Obj_Object_msg);
    }
#endif
#ifdef  ros_object_points_pub
    pcl::PointCloud<pcl::PointXYZ>::Ptr object_clusters_ros (new pcl::PointCloud<pcl::PointXYZ>);
#endif
    for(int i=0;i<object_clusters.size();i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster = object_clusters[i];
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_cluster_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ> ());
        box3d b3d = b3d_list[i];

        //原始点云存在很多零点，通过滤波器去除
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (cluster);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.5, 2.0);
        //pass.setFilterLimitsNegative (true);
        //ect_cluster_filtered_ptr
        pass.filter (*object_cluster_filtered_ptr);

        *object_clusters_ros = *object_clusters_ros + *object_cluster_filtered_ptr;
//        cout<<"!!!!!!!!!!!!!!!!!!!"<<object_cluster_filtered_ptr->points.size()<<endl;

        pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud (object_cluster_filtered_ptr);
        feature_extractor.compute ();

        std::vector <float> moment_of_inertia;
        std::vector <float> eccentricity;
        pcl::PointXYZ min_point_AABB;
        pcl::PointXYZ max_point_AABB;

        feature_extractor.getAABB (min_point_AABB, max_point_AABB);
        double p_x = (min_point_AABB.x + max_point_AABB.x)/2/scale_offset;
        double p_y = (min_point_AABB.y + max_point_AABB.y)/2/scale_offset;
        double p_z = (min_point_AABB.z + max_point_AABB.z)/2/scale_offset;
        double width = (max_point_AABB.x - min_point_AABB.x)/scale_offset;
        double height = (max_point_AABB.y - min_point_AABB.y)/scale_offset;
        double length = (max_point_AABB.z - min_point_AABB.z)/scale_offset;
//            todo: fix the bug for MomentOfInertiaEstimation ptr
        AABB_box3d aabb_box3d(p_x, p_y, p_z, width, length, height, b3d._c);
//        AABB_box3d aabb_box3d(b3d._position_x, b3d._position_y, b3d._position_z, 0, 0, 0, b3d._c);
        AABB_list.push_back(aabb_box3d);

#ifdef pcl_show
        //show by pcl visualization
            char cloud_id[20];
            char AABB_id[20];
            sprintf(cloud_id, "cloud%i", i);
            sprintf(AABB_id, "AABB%i", i);
            viewer->addPointCloud<pcl::PointXYZ> (cluster_filtered_ptr, cloud_id);
            viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, AABB_id);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, AABB_id);
#endif
#ifdef ros_box_pub
        //publish by ros
            // 8 points for each cube
            vector<geometry_msgs::Point> points8;
            geometry_msgs::Point p1;
            geometry_msgs::Point p8;
            p1.x = min_point_AABB.x; p1.y = min_point_AABB.y; p1.z = min_point_AABB.z;
            p8.x = max_point_AABB.x; p8.y = max_point_AABB.y; p8.z = max_point_AABB.z;
            geometry_msgs::Point p2 = p1;
            geometry_msgs::Point p3 = p1;
            geometry_msgs::Point p4 = p1;
            geometry_msgs::Point p5 = p8;
            geometry_msgs::Point p6 = p8;
            geometry_msgs::Point p7 = p8;
            p2.x = p8.x;
            p3.z = p8.z;
            p4.x = p8.x;
            p4.z = p8.z;
            p5.x = p1.x;
            p5.z = p1.z;
            p6.z = p1.z;
            p7.x = p1.x;
            points8.push_back(p1);
            points8.push_back(p2);
            points8.push_back(p3);
            points8.push_back(p4);
            points8.push_back(p5);
            points8.push_back(p6);
            points8.push_back(p7);
            points8.push_back(p8);

            // 12 lines with 24 points
            vector<int> line_points = {0,1,2,3,0,2,1,3,4,5,6,7,4,6,5,7,0,4,1,5,2,6,3,7};
            for (int i=0;i<24;i++)
            {
                box_lines.points.push_back(points8[line_points[i]]);
            }
#endif

    }
#ifdef  ros_object_points_pub
    sensor_msgs::PointCloud2 Obj_Object_msg;
    pcl::toROSMsg(*object_clusters_ros, Obj_Object_msg);
    Obj_Object_msg.header.stamp = ros::Time::now();
    Obj_Object_msg.header.frame_id = "map";
    obj_object_points_pub_.publish(Obj_Object_msg);
#endif

#ifdef pcl_show
    while(!viewer->wasStopped())
        {
            viewer->spinOnce (100);
        }
#endif
#ifdef ros_box_pub
    obj_box_pub_.publish(box_lines);
#endif

    return has_obj;
}

// 对既是障碍物又是物体区域的3d image部分进行聚类，得到四类物体的3d box
bool yolo_sparity_obj_det(string root_path, Mat yolo_mask, vector<AABB_box3d>& box_3d_list, vector<bbox>& bbox_list, int frame, Mat P1)
{
    // fuse yolo mask and obstacle masked result3DImage, get object mask
    Mat obj_3DImage;
    bitwise_and(result3DImage, result3DImage, obj_3DImage, yolo_mask);

    // get 3d object box from masked obj_3DImage
    Size imageSize(yolo_mask.cols, yolo_mask.rows);
    if(obj_det_3d_with_sparity(root_path, obj_3DImage, box_3d_list, bbox_list, frame, P1, imageSize))
    {
        return true;
    }
    else
    {
        return false;
    }

}