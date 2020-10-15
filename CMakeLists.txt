cmake_minimum_required(VERSION 2.8)
project(object_detection)

# 设置编译器编译模式
set( CMAKE_CXX_FLAGS "-std=c++11" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set( CMAKE_BUILD_TYPE "Release" )


# 设置动态链接库输出目录
SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

#ros
find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        image_transport
        pcl_ros
        pcl_conversions
        velodyne_msgs
        velodyne_pointcloud
        roscpp
        rospy
        std_msgs
        )

# 寻找OpenCV库
#find_package( OpenCV 3 REQUIRED PATHS /usr/local/opencv3.4.5)
find_package( OpenCV 3 REQUIRED )
## 寻找PCL
#find_package(PCL 1.7 REQUIRED )
find_package(PCL REQUIRED )
# find_package(PCL 1.11 REQUIRED )
# set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.11)
add_definitions(${PCL_DEFINITIONS})

# ELAS
set (LIBELAS_DIR "third_party/libelas")

# 添加头文件
catkin_package(
        CATKIN_DEPENDS
)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${LIBELAS_DIR}/include
)


MESSAGE("catkin_LIBRARIES: "${catkin_LIBRARIES})
MESSAGE("catkin_INCLUDE_DIRS: "${catkin_INCLUDE_DIRS})

aux_source_directory(${LIBELAS_DIR}/src LIBELAS_SRC_DIR)
add_library(ELAS_LIBS SHARED ${LIBELAS_SRC_DIR})

aux_source_directory(${PROJECT_SOURCE_DIR}/src MYLIB_SRC_DIR)
add_library(MY_LIBS SHARED ${MYLIB_SRC_DIR})
target_link_libraries(MY_LIBS
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
        ELAS_LIBS
        )

# stereo_calib_with_opencv
add_executable(stereo_calib
        examples/stereo_calib.cpp
        )

target_link_libraries(
        stereo_calib
        ${OpenCV_LIBS}
)

# camera R t calibration wrt. word coordinates
add_executable(Rt_calib
        examples/Rt_calib.cpp
        )

target_link_libraries(
        Rt_calib
        ${OpenCV_LIBS}
)
# gen_sparities
add_executable(gen_sparities
        examples/gen_sparities.cpp
)

target_link_libraries(
        gen_sparities
        MY_LIBS
)

# object_detection node with stereo and yolo
add_executable(object_detection_v2_node
        examples/object_detection_with_sparity_v2_node.cpp
        )

target_link_libraries(
        object_detection_v2_node
        MY_LIBS
)



