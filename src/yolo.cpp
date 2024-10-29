#include "yolo.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>
#include "iostream"
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

YOLO::YOLO(const string& modelPath) {
    // 检查 ONNX 文件是否存在
    if (!fileExists(modelPath)) {
        throw runtime_error("ONNX model file not found: " + modelPath);
    }
    cout << "模型路径：" << modelPath << endl;
    net = readNetFromONNX(modelPath);

    if (cudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    
    cout << "模型载入完成！" << endl;
}

YOLO::YOLO(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda) {
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    //loadOnnxNetwork
    net = cv::dnn::readNetFromONNX(modelPath);
    if (cudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

Mat YOLO::inferAndGetResult(const Mat& frame) {
    //将图片补位正方形
    Mat image = formatToSquare(frame);
    cout << image.size << endl;
    //image.resize(640);
    imshow("Sq image", image);
    
    
    Mat blob = blobFromImage(image, 1.0/255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // 前向传播
    auto tt1 = cv::getTickCount();
    std::vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    auto tt2 = cv::getTickCount();

    // 检查输出
    if (outs.empty()) {
        std::cerr << "Error: No outputs from the network!" << std::endl;
        return frame.clone(); // 返回原始图像
    }

    // 创建一个新图像以保存结果
    //Mat resultFrame = frame.clone(); // 克隆输入图像以不改变原图
    Mat resultFrame = image;

    // 处理检测结果
    postprocess(resultFrame, outs);

    std::string label = format("Inference time: %.2f ms", (tt2 - tt1) / cv::getTickFrequency() * 1000);
    cv::putText(resultFrame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));


    return resultFrame; // 返回结果图像
}

void YOLO::infer(Mat& frame) {
    // 图象预处理 - 格式化操作
     int w = frame.cols;
     int h = frame.rows;
     int _max = std::max(h, w);
     cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
     cv::Rect roi(0, 0, w, h);
     frame.copyTo(image(roi));


    Mat blob = blobFromImage(frame, 1/255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // 前向传播
    //std::vector<Mat> outs;
    Mat outs;
    net.forward(outs);
    cv::Mat det_output(outs.size[1], outs.size[2], CV_32F, outs.ptr<float>());

    // 处理检测结果
    postprocess(frame, outs);
}

void YOLO::showResult(const Mat& frame) {
    imshow("Detections", frame);
    waitKey(0);
}


void postprocess(Mat& frame, const std::vector<Mat>& outs) {
    int confidence_threshold = 0.5;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;

    float x_factor = frame.cols / 640.0f;
    float y_factor = frame.rows / 640.0f;

    cout << "x,y_factor: " << x_factor << ", " << y_factor << endl;    
    
    for (const Mat& output : outs) {
        // 检查输出的有效性
        if (output.empty()) {
            std::cerr << "Output is empty." << std::endl;
            continue;
        }
        else cout << output.size() << endl;

        for (int i = 0; i < output.size[1]; i++) {
            //float* data = (float*)output.data + i * output.size[2];
            const float* data = output.ptr<float>(0) + i * output.size[2]; // 获取每个框的数据

            //cout << "data:" << data[0] << ',' << data[1] << ',' << data[2] << ',' << data[3] << ',' << data[4] << ',' << data[5] << endl; 
            // 解析边界框的坐标和置信度
            float confidence = data[4];
            
            if (confidence >= confidence_threshold) { // 阈值可以调整
                //cout << confidence << endl;
                cout << "data:" << data[0] << ',' << data[1] << ',' << data[2] << ',' << data[3] << ',' << data[4] << ',' << data[5] << endl; 
                int left   = static_cast<int>(data[0] * x_factor);
                int top    = static_cast<int>(data[1] * y_factor);
                int right  = static_cast<int>(data[2] * x_factor);
                int bottom = static_cast<int>(data[3] * y_factor);

                int x = left;
                int y = top;
                int width  = abs(right - left);
                int height = abs(bottom - top);

/*
                float cx = output.at<float>(i, 0);
                float cy = output.at<float>(i, 1);
                float ow = output.at<float>(i, 2);
                float oh = output.at<float>(i, 3);

                int x = static_cast<int>(output.at<float>(i, 0) * x_factor);
                int y = static_cast<int>(output.at<float>(i, 1) * y_factor);
                int x2 = static_cast<int>(output.at<float>(i, 2) * x_factor);
                int y2 = static_cast<int>(output.at<float>(i, 3) * y_factor);
                int width = abs(x2 - x);
                int height = abs(y2 - y);
  */              

                //int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                //int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                //int width = static_cast<int>(ow * x_factor);
                //int height = static_cast<int>(oh * y_factor);
                
/*
                int x1 = static_cast<int>(data[0] * frame.cols / 640.0);
                int y1 = static_cast<int>(data[1] * frame.rows / 640.0);
                int x2 = static_cast<int>(data[2] * frame.cols / 640.0);
                int y2 = static_cast<int>(data[3] * frame.rows / 640.0);

                int x1 = static_cast<int>(data[0] * frame.rows / 660.0);
                int y1 = static_cast<int>(data[1] * frame.cols / 660.0);
                int x2 = static_cast<int>(data[2] * frame.rows / 660.0);
                int y2 = static_cast<int>(data[3] * frame.cols / 660.0);
*/
                //cout << "l,t,w,h:" << left << ',' << top << ',' << width << ',' << height << endl;
                //cout << "x1,y1,x2,y2:" << x1 << ',' << y1 << ',' << x2 << ',' << y2 << endl;

                // 使用类 ID
                int classId = static_cast<int>(data[5]);
                cout << "classId" << classId << endl;
                cout << "l,t,r,b:" << left << ',' << top << ',' << right << ',' << bottom << endl;
                classIds.push_back(classId);
                confidences.push_back(confidence);
                //boxes.push_back(Rect(left, top, width, height));
                //boxes.push_back(Rect(x1, y1, abs(x2 - x1), abs(y2 - y1)));
                boxes.push_back(Rect(x, y, width, height));

                //cout << output.at<float>(i, 0) << ',' << output.at<float>(i, 1) << ',' << output.at<float>(i, 2) << ',' << output.at<float>(i, 3) << ',' << output.at<float>(i, 4) << ',' << output.at<float>(i, 5) << endl;
            }
        }
    }

    // 非极大值抑制
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, 0.25, 0.4, indices);

    cout << indices.size() << '/' << boxes.size() << endl;
    
    // 绘制检测框
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        
        rectangle(frame, box, Scalar(0, 255, 0), 2);
        string label = format("Class: %d, Conf: %.2f", classIds[idx], confidences[idx]);

        // 计算文本位置
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int labelTop = max(box.y, labelSize.height);

        rectangle(frame, Point(box.x, labelTop - labelSize.height), 
                  Point(box.x + labelSize.width, labelTop + baseLine), Scalar(0, 255, 0), FILLED);
        

        // 绘制标签文本
        putText(frame, label, Point(box.x, labelTop), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
   
    }
}




void YOLO::postprocess(Mat& frame, const std::vector<Mat>& outs) {

    std::vector<int> classIds;//类型id
    std::vector<float> confidences;//置信度
    std::vector<Rect> boxes;//位置

    float x_factor = frame.cols / 640.0f;
    float y_factor = frame.rows / 640.0f;

    cout << "x,y_factor: " << x_factor << ", " << y_factor << endl;

    for (const Mat& output : outs) {
        // 检查输出的有效性
        if (output.empty()) {
            std::cerr << "Output is empty." << std::endl;
            continue;
        }

        // [1, 300, 6] -> [300,6]
        int rows = output.size[1];
        int out_data_num = output.size[2];

        auto tmp = output.reshape(1, rows);
        //cv::transpose(tmp, tmp);
        std::cout << tmp.size << endl;

        // 数据正确性验证(Debug)：
        for (int j = 0 ; j < rows; j++)
            for (int i = 0; i < 6; i++) {
                if (output.at<float>(0,j,i) != tmp.at<float>(j,i)) {
                    cout << "Error: In ("<< j << ',' << i << ") output data is " << output.at<float>(1,j,i) << " ,but tmp data is " << tmp.at<float>(j,i) << endl;
                    throw runtime_error("data error in: " + to_string(j) + ',' + to_string(i));
                }
            }
        


        // End 数据验证区间结束
        //float *data = (float *)tmp.data;

        for (int i = 0; i < rows; i++) {
            // 逐行读取
            //const float* rowData = tmp.ptr<float>(i);
            // 从每一行中获取的参数（x,y,w,h,...）
            int x = static_cast<int>(tmp.at<float>(i, 0) * x_factor);
            int y = static_cast<int>(tmp.at<float>(i, 1) * y_factor);

            int width  = static_cast<int>(tmp.at<float>(i, 2) * x_factor);
            int height = static_cast<int>(tmp.at<float>(i, 3) * y_factor);

            float confidence = tmp.at<float>(i, 4);
            int classId = static_cast<int>(tmp.at<float>(i, 5));

            // 输出当前框的信息
            std::cout << "Box " << i << ": "
              << "x=" << x << ", "
              << "y=" << y << ", "
              << "Width=" << width << ", "
              << "Height=" << height << ", "
              << "Confidence=" << confidence << ", "
              << "Class ID=" << classId << std::endl;

            if (width < 0 || height < 0) continue;
        /*
            // 从每一行中获取框的参数(x1,y1,x2,y2,...)
            float left = tmp.at<float>(i, 0);
            float top = tmp.at<float>(i, 1);
            float right = tmp.at<float>(i, 2);
            float bottom = tmp.at<float>(i, 3);
            float confidence = tmp.at<float>(i, 4);
            int classId = static_cast<int>(tmp.at<float>(i, 5));

            // 计算框的宽度和高度
            int x = max(0, static_cast<int>(left * x_factor));
            int y = max(0, static_cast<int>(top * y_factor));

            int width = static_cast<int>((right - left) * x_factor);
            int height = static_cast<int>((bottom - top) * y_factor);

            // 输出当前框的信息
            std::cout << "Box " << i << ": "
              << "Left=" << left << ", "
              << "Top=" << top << ", "
              << "Right=" << right << ", "
              << "Bottom=" << bottom << ", "
              << "x=" << x << ", "
              << "y=" << y << ", "
              << "Width=" << width << ", "
              << "Height=" << height << ", "
              << "Confidence=" << confidence << ", "
              << "Class ID=" << classId << std::endl;

            if (width < 0 || height < 0) continue;
        */

            // 从每一行中获取框的参数(xc,yc,w,h,...)

            /*
            float xc = output.at<float>(0, i, 0);
            float yc = output.at<float>(0, i, 1);
            float w = output.at<float>(0, i, 2);
            float h = output.at<float>(0, i, 3);
            float confidence = output.at<float>(0, i, 4);
            int classId = static_cast<int>(output.at<float>(0, i, 5));

            

            // 计算框的位置
            int x = static_cast<int>((xc - w/2.0) * x_factor);
            int y = static_cast<int>((yc - h/2.0) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            

            std::cout << "Box " << i << ": "
              << "xc=" << xc << ", "
              << "yc=" << yc << ", "
              << "w=" << w << ", "
              << "h=" << h << ", "
              << "x=" << x << ", "
              << "y=" << y << ", "
              << "Width=" << width << ", "
              << "Height=" << height << ", "
              << "Confidence=" << confidence << ", "
              << "Class ID=" << classId << std::endl;

            if (xc < 0 || yc < 0 || w < 0 || h < 0) continue;
            if (width < 0 || height < 0 || x < 0 || y < 0) continue;

*/
            if (confidence >= confidence_threshold) { // 阈值

                classIds.push_back(classId);
                confidences.push_back(confidence);
                boxes.push_back(Rect(x, y, width, height));
            }

        }

    }

    // 非极大值抑制
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, 0.25, 0.4, indices);

    cout << indices.size() << '/' << boxes.size() << endl;

    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        
        // 选择颜色
        Scalar color;
        switch (classIds[idx]%3)
        {
            case 0:
                color = Scalar(255,0,0);
                break;
            case 1:
                color = Scalar(0,255,0);
                break;
            case 2:
                color = Scalar(0,0,255);
                break;
        }
        // 绘制目标位置框
        rectangle(frame, box, color, 2);
        

        string label = format("Class: %d, Conf: %.2f", classIds[idx], confidences[idx]);

        // 计算文本位置
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int labelTop = max(box.y, labelSize.height);

        rectangle(frame, Point(box.x, labelTop - labelSize.height), 
                  Point(box.x + labelSize.width, labelTop + baseLine), color, FILLED);
        

        // 绘制标签文本
        putText(frame, label, Point(box.x, labelTop), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        cout << "ClassId=" << classIds[idx] << ", "
             << "confidences=" << confidences[idx] << ", "
             << "x,y=" <<  box.x << ", " << box.y << endl;

   
    }
}



//Mat YOLO::formatToSquare(const Mat& inputImage, double tar_sz = 640.0) {
    /*
    // 获取原始图像的尺寸
    int originalWidth = inputImage.cols;
    int originalHeight = inputImage.rows;

    // 计算缩放比例
    float scale = tar_sz / std::max(originalWidth, originalHeight);
    int newWidth = static_cast<int>(originalWidth * scale);
    int newHeight = static_cast<int>(originalHeight * scale);

    // 缩放图像
    Mat resizedImage;
    resize(inputImage, resizedImage, Size(newWidth, newHeight));

    // 创建一个640x640的黑色图像
    Mat paddedImage = Mat::zeros(640, 640, resizedImage.type());

    // 将缩放后的图像放置到黑色图像的左上角
    resizedImage.copyTo(paddedImage(Rect(0, 0, newWidth, newHeight)));

    return paddedImage;
}
*/

void YOLO::runInference(const cv::Mat &input) {
    cv::Mat modelInput = input;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];
}

cv::Mat YOLO::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
