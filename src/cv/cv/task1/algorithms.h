#ifndef CGCV_ALGORITHMS_H
#define CGCV_ALGORITHMS_H

#include "helper.h"
#include <opencv2/opencv.hpp>

class algorithms
{
   public:
    static void compute_grayscale(const cv::Mat &input_image, cv::Mat &grayscale_image);

    static void compute_gradient(const cv::Mat &grayscale_image, cv::Mat &gradient_x, cv::Mat &gradient_y,
                                 cv::Mat &gradient_abs);

    static void compute_directions(const cv::Mat &gradient_x, const cv::Mat &gradient_y, const cv::Mat &gradient_abs,
                                   cv::Mat &direction_x, cv::Mat &direction_y);

    static void swt_compute_stroke_width(const cv::Mat &edges, const cv::Mat &direction_x, const cv::Mat &direction_y,
                                         bool black_on_white, std::vector<std::vector<cv::Point2i>> &rays,
                                         cv::Mat &swt_stroke_width_image);

    static void swt_postprocessing(const cv::Mat &swt_stroke_width_image,
                                   const std::vector<std::vector<cv::Point2i>> &rays, cv::Mat &swt_final_image);

    static void get_connected_components(const cv::Mat &swt_image, const float stroke_width_ratio_threshold,
                                         const int neighbor_offset, cv::Mat &labels,
                                         std::vector<std::vector<cv::Point2i>> &components);

    static void compute_bounding_boxes(const std::vector<std::vector<cv::Point2i>> &components,
                                       std::vector<cv::Rect2i> &bounding_boxes);

    static void discard_non_text(const cv::Mat &swt_image, const std::vector<cv::Rect2i> &bounding_boxes,
                                 const std::vector<std::vector<cv::Point2i>> &components, const cv::Mat &labels,
                                 const float variance_ratio, const float aspect_ratio_threshold,
                                 const float diameter_ratio_threshold, const int min_height, const int max_height,
                                 std::vector<cv::Rect2i> &text_bounding_boxes,
                                 std::vector<std::vector<cv::Point2i>> &text_components, cv::Mat &text_labels);

    struct PosStrokeWidth
    {
        int col;
        int row;
        float stroke_width;
    };

    // Bonus
    static void non_maxima_suppression(const cv::Mat &gradient_image, const cv::Mat &gradient_x,
                                       const cv::Mat &gradient_y, cv::Mat &non_maxima);

    static void hysteresis(const cv::Mat &non_max_sup, const uchar threshold_min, const uchar threshold_max,
                           cv::Mat &output_image);

    static void canny_own(const cv::Mat &grayscale_image, const uchar threshold_min, const uchar threshold_max,
                          cv::Mat &output_image);
};

#endif  // CGCV_ALGORITHMS_H
