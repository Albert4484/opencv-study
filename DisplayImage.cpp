#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

namespace cv::ml{
class DBSCAN {
public:
    DBSCAN(double epsilon, int minPoints) : epsilon(epsilon), minPoints(minPoints) {}

    void fit(const std::vector<cv::Point2f>& points) {
        int n = points.size();
        labels.resize(n, -1);
        visited.resize(n, false);
        int clusterId = 0;

        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;
            visited[i] = true;
            std::vector<int> neighbors = regionQuery(points, i);

            if (neighbors.size() < minPoints) {
                labels[i] = -1; // 标记为噪声
            } else {
                clusterId++;
                expandCluster(points, i, neighbors, clusterId);
            }
        }
    }

    const std::vector<int>& getLabels() const {
        return labels;
    }

private:
    double epsilon;
    int minPoints;
    std::vector<int> labels;
    std::vector<bool> visited;

    std::vector<int> regionQuery(const std::vector<cv::Point2f>& points, int idx) {
        std::vector<int> neighbors;
        for (int i = 0; i < points.size(); ++i) {
            if (cv::norm(points[i] - points[idx]) <= epsilon) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }

    void expandCluster(const std::vector<cv::Point2f>& points, int idx, std::vector<int>& neighbors, int clusterId) {
        labels[idx] = clusterId;

        for (size_t i = 0; i < neighbors.size(); ++i) {
            int neighborIdx = neighbors[i];

            if (!visited[neighborIdx]) {
                visited[neighborIdx] = true;
                std::vector<int> newNeighbors = regionQuery(points, neighborIdx);

                if (newNeighbors.size() >= minPoints) {
                    neighbors.insert(neighbors.end(), newNeighbors.begin(), newNeighbors.end());
                }
            }

            if (labels[neighborIdx] == -1) {
                labels[neighborIdx] = clusterId; // 从噪声变为边界点
            }
        }
    }
};
}
using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;

    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    // 转换到 HSV 颜色空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 定义颜色范围（这里以红色为例）
    // 红色的 HSV 范围
    cv::Scalar lower_red1(0, 100, 100); // 红色下界
    cv::Scalar upper_red1(10, 255, 255); // 红色上界
    cv::Scalar lower_red2(160, 100, 100); // 红色下界
    cv::Scalar upper_red2(180, 255, 255); // 红色上界

    //黄绿色
    cv::Scalar lower_yellow_green(25, 100, 100); // 黄绿色下界
    cv::Scalar upper_yellow_green(80, 255, 255); // 黄绿色上界


    // 创建掩膜
    cv::Mat mask1, mask2, mask3, mask;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::inRange(hsv, lower_yellow_green, upper_yellow_green, mask3);

    // 合并掩膜
    mask = mask3;

    // 提取红色区域
    cv::Mat result;
    cv::bitwise_and(image, image, result, mask);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Mask", mask);
    cv::imshow("Extracted Color", result);


    // 转换为灰度图
    Mat gray;
    // 提取V通道作为灰度图
    std::vector<cv::Mat> hsvChannels;
    cv::split(result, hsvChannels);
    gray = hsvChannels[2]; // V通道
    //展示灰度图
    imshow("Gray", gray);

    // 边缘检测
    Mat edges;
    cv::Canny(gray, edges, 100, 200);
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 在原图上绘制轮廓
    Mat contoursImg = image.clone();
    //cv::drawContours(contoursImg, contours, -1, cv::Scalar(0, 255, 0), 2);
    double minArea = 100.0; // 最小轮廓面积
    double maxArea = 1000.0; // 最小轮廓面积

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > minArea && area < maxArea) {
            cv::Rect boundingBox = cv::boundingRect(contour);
            cv::rectangle(contoursImg, boundingBox, cv::Scalar(0, 255, 0), 2);
        }
    }
    
    cv::imshow("Contours", contoursImg);


    /* 集群 */

     // 计算轮廓中心
    std::vector<cv::Point2f> centers;
    for (const auto& contour : contours) {
        cv::Moments m = cv::moments(contour);
        if (m.m00 > 0) { // 确保面积不为零
            centers.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
        }
    }

    // 使用DBSCAN进行集群
    cv::Mat centersMat(centers.size(), 1, CV_32FC2, centers.data());
    std::vector<int> labels;
    cv::ml::DBSCAN dbscan(30, 5); // 设置epsilon和minPoints
    dbscan.fit(centersMat, labels);

    // 在图像上绘制每个群体的包围矩形
    for (int i = 0; i < centers.size(); i++) {
        if (labels[i] != -1) { // 只绘制属于某个集群的中心
            cv::Rect boundingBox = cv::boundingRect(std::vector<cv::Point>(centers.begin(), centers.end()));
            cv::rectangle(image, boundingBox, cv::Scalar(0, 255, 0), 2);
        }
    }

    // 显示结果
    cv::imshow("Original Image with Clusters", image);
    

    cv::waitKey(0);
    return 0;
}