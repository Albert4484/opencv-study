#pragma ones
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

class YOLO {
public:
    YOLO(const string& modelPath);
    YOLO(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda);
    void infer(Mat& frame);//直接在图上修改
    Mat inferAndGetResult(const Mat& frame);//返回结果
    void showResult(const Mat& frame);
    void set_confidence(double confidence) {
        if (confidence > 0 && confidence < 1.0) confidence_threshold = confidence;
        cout << "当前置信度：" << confidence_threshold << endl;
    }

    double get_confidence(void) {return confidence_threshold;}

    void runInference(const cv::Mat &input);

private:
    Net net;

    std::string modelPath{};
    std::string classesPath{};
    //bool cudaEnabled{};

    cv::Size2f modelShape{};

    std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};


    bool cudaEnabled = false;
    double confidence_threshold = 0.5;//置信度阈值
    void postprocess(Mat& frame, const std::vector<Mat>& outs);
    bool fileExists(const string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
    
    bool letterBoxForSquare = true;

    //Mat formatToSquare(const Mat& inputImage, double tar_sz = 640.0);
    cv::Mat formatToSquare(const cv::Mat &source);


};