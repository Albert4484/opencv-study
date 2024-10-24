#pragma ones
#include <stdio.h>
#include <string>
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

namespace ipro{
    //
cv::Mat Image_Mask(cv::Mat hsv_image, cv::Scalar lower, cv::Scalar upper);
cv::Mat Get_Presons(cv::Mat fram);
cv::Mat gradient_sobel(cv::Mat image);


}


