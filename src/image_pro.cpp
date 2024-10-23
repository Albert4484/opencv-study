#include "image_pro.hpp"

namespace ipro{
cv::Mat Image_Mask(cv::Mat image, cv::Scalar lower, cv::Scalar upper)
{
    // 定义颜色范围（这里以红色为例）
    // 红色的 HSV 范围
    /*
    cv::Scalar lower_red1(0, 100, 100); // 红色下界
    cv::Scalar upper_red1(10, 255, 255); // 红色上界
    cv::Scalar lower_red2(160, 100, 100); // 红色下界
    cv::Scalar upper_red2(180, 255, 255); // 红色上界

    //黄绿色
    cv::Scalar lower_yellow_green(15, 100, 100); // 黄绿色下界
    cv::Scalar upper_yellow_green(30, 255, 255); // 黄绿色上界
    */

    // 创建掩膜
    //cv::Mat mask1, mask2, mask3, mask;
    //cv::inRange(hsv, lower_red1, upper_red1, mask1);
    //cv::inRange(hsv, lower_red2, upper_red2, mask2);
    //cv::inRange(hsv, lower_yellow_green, upper_yellow_green, mask3);
    // 合并掩膜
    //mask = mask3;

    // 转换到 HSV 颜色空间
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv_image, lower, upper, mask);

    // 提取红色区域
    cv::Mat result;
    cv::bitwise_and(image, image, result, mask);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Mask", mask);
    cv::imshow("Extracted Color", result);

    return result;
}



}