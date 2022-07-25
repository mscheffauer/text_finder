#ifndef CGCV_HELPER_H
#define CGCV_HELPER_H

#include "opencv2/opencv.hpp"

class helper
{
   public:
    static std::vector<std::vector<int>> connected_letters(int n, std::vector<std::vector<int>>& edges);
    static void find_letter_groups(const cv::Mat &input_image, const cv::Mat &swt_image, const cv::Mat &text_labels,
                                   const std::vector<std::vector<cv::Point2i>> &text_components,
                                   const std::vector<cv::Rect2i> &bounding_boxes, const float height_ratio_threshold,
                                   const float width_ratio_threshold, const float median_ratio_threshold,
                                   const float distance_ratio, const float color_distance_threshold,
                                   std::vector<cv::Rect2i> &group_bounding_boxes,
                                   std::vector<cv::Rect2i> &letter_bounding_boxes);
};

#endif  // CGCV_HELPER_H
