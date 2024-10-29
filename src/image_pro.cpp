#include "image_pro.hpp"

/***** config ******/
//color
#define    Mask_YellowGreen 1
#define    Mask_Red 2
#define    Mask_Pruple 3


#define Mask_Color Mask_YellowGreen

#if (Mask_Color == Mask_YellowGreen)
#define Lower_Mask cv::Scalar(20, 124, 144)
#define Upper_Mask cv::Scalar(22, 155, 254)
#elif (Mask_Color == Mask_Red)
#define Lower_Mask cv::Scalar(173, 117, 91)   // 示例值
#define Upper_Mask cv::Scalar(3, 119, 245)  // 示例值
#elif (Mask_Color == Mask_Pruple)
#define Lower_Mask cv::Scalar(127, 83, 94) // 示例值
#define Upper_Mask cv::Scalar(154, 146, 184) // 示例值
#else
#define Lower_Mask cv::Scalar(0, 0, 0) // 默认值
#define Upper_Mask cv::Scalar(180, 255, 255) // 默认值
#endif

// oc_size
#define OC_SIZE 1
#define OC_CHOISE cv::MORPH_OPEN


namespace ipro{
cv::Mat Image_Mask(cv::Mat image, cv::Scalar lower, cv::Scalar upper)
{
    // 转换到 HSV 颜色空间
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    if (lower[0] > upper[0]) {
        cv::Scalar l1(lower[0], lower[1], lower[2]);
        cv::Scalar u1(180, upper[1], upper[2]);
        cv::Scalar l2(0, lower[1], lower[2]);
        cv::Scalar u2(upper[0], upper[1], upper[2]);
        cv::Mat mask1, mask2;

        cv::inRange(hsv_image, l1, u1, mask1);
        cv::inRange(hsv_image, l2, u2, mask2);

        mask = mask1 | mask2;
        
    }
    else cv::inRange(hsv_image, lower, upper, mask);

    // 提取红色区域
    cv::Mat result;
    cv::bitwise_and(image, image, result, mask);

    // 显示结果
    //cv::imshow("Original Image", image);
    //cv::imshow("Mask", mask);
    cv::imshow("Mask open", mask);
    cv::imshow("Extracted Color", result);

    return result;
}

cv::Mat Get_Presons(cv::Mat fram)
{
    if (!fram.data)//无数据
    {
        printf("Warning: Frame cannot be used! \n");
        return cv::Mat();
    }

    cv::Mat image = fram.clone();
    int width = image.cols;
    int height = image.rows;
    //去掉右上角
   // 定义要去除的矩形区域
    cv::Rect removeRegion(width * 3 / 4, 0, width / 4, height / 4); // 从右上角开始

    // 检查区域是否在图像范围内
    if (removeRegion.x + removeRegion.width <= width && removeRegion.y + removeRegion.height <= height) {
        // 用黑色填充矩形区域
        image(removeRegion) = cv::Scalar(0, 0, 0); // BGR 格式，黑色
    } else {
        std::cerr << "去除区域超出图像边界！" << std::endl;
    }

    //黄绿色, 
    cv::Scalar lower_yellow_green(27,  37, 160); // 黄绿色下界30
    cv::Scalar upper_yellow_green(31, 238, 255); // 黄绿色上界40

    //cv::Mat result = ipro::Image_Mask(image, lower_yellow_green, upper_yellow_green);
    cv::Mat result = ipro::Image_Mask(image, Lower_Mask, Upper_Mask);

    // 转换为灰度图
    cv::Mat gray;
    // 提取V通道作为灰度图
    std::vector<cv::Mat> hsvChannels;
    cv::split(result, hsvChannels);
    gray = hsvChannels[2]; // V通道    

    // 梯度算法
    cv::Mat gradient_img = gradient_sobel(gray);


    // 这里执行开闭运算
    int size_MORPH = OC_SIZE;
    cv::Mat res_ol;
    // 选择结构元素
    if (size_MORPH != 0) {
        cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(size_MORPH, size_MORPH)); // 矩形结构元素

        // 执行开/闭运算
        cv::morphologyEx(gradient_img, res_ol, OC_CHOISE, element);
    }
    else res_ol = gradient_img;


    // 边缘检测
    cv::Mat edges;
    cv::Canny(res_ol, edges, 100, 200);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 在原图上绘制轮廓
    cv::Mat contoursImg = image.clone();
    cv::drawContours(contoursImg, contours, -1, cv::Scalar(0, 0, 255), 2);
    double minArea = 0.0; // 最小轮廓面积
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

    /* 集群 */

     // 计算轮廓矩形的中心
    std::vector<cv::Point2f> centers;
    for (const auto& contour : contours) {
        cv::Moments m = cv::moments(contour);
        if (m.m00 > minArea) { // 确保面积大于最小面积
            centers.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
        }
    }

    // 使用DBSCAN进行集群

   // 使用轮廓外接矩形中心
    cv::Mat centersMat(Box_centers.size(), 1, CV_32FC2, Box_centers.data());
    
    cv::ml::DBSCAN dbscan(50, 2); // 设置epsilon和minPoints,最低35 2
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

    std::cout << "labels类数量：" << max_labels << ",labels点数量：" << labels.size()<< std::endl;

    for (const auto& cluster : clusters) {
        const auto& points = cluster.second;
        if (!points.empty()) {
            cv::Rect boundingBox = cv::boundingRect(points);
            cv::rectangle(image, boundingBox, cv::Scalar(255, 0, 0), 2); // 绘制外接矩形
            std::cout << "矩形中心位置x,y：" << boundingBox.x + boundingBox.width / 2 << "," << boundingBox.y +  boundingBox.height / 2 << std::endl;
            std::cout << "矩形宽，高：" << boundingBox.width << "," << boundingBox.height << std::endl;
        }
    }
    std::cout << std::endl;

    // 显示结果
    // cv::imshow("Original Image with Clusters", image);

    return image;
}


cv::Mat gradient_sobel(cv::Mat image) {
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return cv::Mat();
    }

    // 计算 Sobel 梯度
    cv::Mat grad_x, grad_y;
    cv::Sobel(image, grad_x, CV_64F, 1, 0, 5); // 水平方向
    cv::Sobel(image, grad_y, CV_64F, 0, 1, 5); // 垂直方向

    // 计算梯度幅值
    cv::Mat gradient_magnitude;
    cv::magnitude(grad_x, grad_y, gradient_magnitude);
    gradient_magnitude.convertTo(gradient_magnitude, CV_8U);

    // 显示结果
    //cv::imshow("Gradient Magnitude", gradient_magnitude);
    //cv::waitKey(0);
    //cv::destroyAllWindows();

    return gradient_magnitude;
}


}