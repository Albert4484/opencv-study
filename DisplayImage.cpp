#include <stdio.h>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "image_pro.hpp"

using namespace cv;

int main(int argc, char** argv )
{

    std::string inputVideoPath = "Video/video2.mp4"; // 输入视频路径
    std::string outputVideoPath = "Video/output_video.mp4"; // 输出 MP4 视频路径
    if (argc == 2) {
        inputVideoPath = argv[1]; // 输入视频路径
        std::cout << "视频路径：" << argv[1] << std::endl;
    }

    // 打开输入视频
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件!" << std::endl;
        return -1;
    }

    // 获取视频属性
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 创建视频写入对象，使用适合 MP4 的编码
    cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(frameWidth, frameHeight));

    cv::Mat frame;
    while (true) {
        cap >> frame; // 读取一帧
        if (frame.empty()) break; // 检查是否到达视频末尾

        // 在这里处理每一帧，例如转换为灰度图像
        cv::Mat processedFrame = ipro::Get_Presons(frame);
        
        // 写入处理后的帧
        if (processedFrame.data)
            writer.write(processedFrame);
    }

    // 释放资源
    cap.release();
    writer.release();
    std::cout << "视频处理完成，输出视频已保存。" << std::endl;

    return 0;

// 图处理    
    Mat image;
    if (argc != 2)
        image = imread( "Images/Va4.jpg", 1 );
    else image = imread(argv[1], 1);

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    // 定义颜色范围（这里以红色为例）
    // 红色的 HSV 范围
    cv::Scalar lower_red1(0, 100, 100); // 红色下界
    cv::Scalar upper_red1(10, 255, 255); // 红色上界
    cv::Scalar lower_red2(160, 100, 100); // 红色下界
    cv::Scalar upper_red2(180, 255, 255); // 红色上界

    //黄绿色, 
    cv::Scalar lower_yellow_green(30, 100, 100); // 黄绿色下界30
    cv::Scalar upper_yellow_green(40, 255, 255); // 黄绿色上界40

    cv::Mat result = ipro::Image_Mask(image, lower_yellow_green, upper_yellow_green);

    // 转换为灰度图
    Mat gray;
    // 提取V通道作为灰度图
    std::vector<cv::Mat> hsvChannels;
    cv::split(result, hsvChannels);
    gray = hsvChannels[2]; // V通道
    //展示灰度图
    imshow("Gray", gray);

    // 使用阈值化将灰度图转换为二值图
    Mat binaryImage;
    double thresholdValue = 165; // 阈值，可以根据需要调整
    double maxValue = 255; // 最大值
    cv::threshold(gray, binaryImage, thresholdValue, maxValue, cv::THRESH_BINARY);
    imshow("binaryImage", binaryImage);

    // 边缘检测
    Mat edges;
    cv::Canny(binaryImage, edges, 100, 200);
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 在原图上绘制轮廓
    Mat contoursImg = image.clone();
    cv::drawContours(contoursImg, contours, -1, cv::Scalar(0, 0, 255), 2);
    double minArea = 2.0; // 最小轮廓面积
    double maxArea = 1000.0; // 最小轮廓面积


    std::vector<cv::Point2f> Box_centers;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > minArea && area < maxArea) {
            cv::Rect boundingBox = cv::boundingRect(contour);
            cv::rectangle(contoursImg, boundingBox, cv::Scalar(0, 255, 0), 2);
            Box_centers.push_back(cv::Point2f(boundingBox.x + boundingBox.width / 2.0, boundingBox.y + boundingBox.height / 2.0));
        }
    }
    
    cv::imshow("Contours", contoursImg);

    /* 集群 */

     // 计算轮廓矩形的中心
    std::vector<cv::Point2f> centers;
    for (const auto& contour : contours) {
        cv::Moments m = cv::moments(contour);
        if (m.m00 > 10) { // 确保面积大于最小面积
            centers.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
        }
    }

    // 使用DBSCAN进行集群
    
    // 使用轮廓中心(错误)
    /*
    cv::Mat centersMat(centers.size(), 1, CV_32FC2, centers.data());
        
    cv::ml::DBSCAN dbscan(65, 5); // 设置epsilon和minPoints
    dbscan.fit(centers);
*/

   // 使用轮廓外接矩形中心
    cv::Mat centersMat(Box_centers.size(), 1, CV_32FC2, Box_centers.data());
    
    cv::ml::DBSCAN dbscan(65, 3); // 设置epsilon和minPoints,最低35 2
    dbscan.fit(centersMat);
 

    // 获取标签
    std::vector<int> labels = dbscan.getLabels();
    int max_labels = -1;
    
    // 绘制每个聚类的外接矩形
    std::unordered_map<int, std::vector<cv::Point2f>> clusters;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] != -1) {
            clusters[labels[i]].push_back(Box_centers[i]);
            if (labels[i] > max_labels) max_labels = labels[i];
        }
    }

    std::cout << "labels类数量：" << max_labels << std::endl;
    std::cout << "labels点数量：" << labels.size() << std::endl;

    for (const auto& cluster : clusters) {
        const auto& points = cluster.second;
        if (!points.empty()) {
            cv::Rect boundingBox = cv::boundingRect(points);
            cv::rectangle(image, boundingBox, cv::Scalar(255, 0, 0), 2); // 绘制外接矩形
            std::cout << "矩形中心位置x,y：" << boundingBox.x + boundingBox.width / 2 << "," << boundingBox.y +  boundingBox.height / 2 << std::endl;
            std::cout << "矩形宽，高：" << boundingBox.width << "," << boundingBox.height << std::endl << std::endl;
        }
    }


    // 显示结果
    cv::imshow("Original Image with Clusters", image);
    
    cv::waitKey(0);
    cv::waitKey(0);//方便按截屏
    return 0;
}