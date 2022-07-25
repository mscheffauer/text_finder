#include "helper.h"
#include "algorithms.h"

#include "opencv2/opencv.hpp"

//===============================================================================
// merge()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
// source: https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
int merge(int* parent, int x)
{
    if (parent[x] == x)
        return x;
    return merge(parent, parent[x]);
}

//===============================================================================
// connected_letters()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
std::vector<std::vector<int>> helper::connected_letters(int n, std::vector<std::vector<int>>& edges)
{
    int parent[n];
    for (int i = 0; i < n; i++)
    {
        parent[i] = i;
    }
    for (auto x : edges)
    {
        parent[merge(parent, x[0])] = merge(parent, x[1]);
    }
    int ans = 0;
    for (int i = 0; i < n; i++)
    {
        ans += (parent[i] == i);
    }
    for (int i = 0; i < n; i++)
    {
        parent[i] = merge(parent, parent[i]);
    }
    std::map<int, std::list<int>> m;
    for (int i = 0; i < n; i++)
    {
        m[parent[i]].push_back(i);
    }

    std::vector<std::vector<int>> groups;

    for (auto it = m.begin(); it != m.end(); it++)
    {
        std::list<int> l = it->second;
        std::vector<int> group;
        for (auto x : l)
        {
            group.push_back(x);
        }
        groups.push_back(group);
    }
    return groups;
}

//===============================================================================
// find_letter_groups()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
// Groups letters that likely belong together
//       - Loops over the component combinations
//       - Checks each combination whether the components belong together
//         (with the help of its bounding boxes)
//       - Keeps the criteria (ratios and thresholds) of the task description in mind
//       - Adds the combination {index1, index2} to the combinations vector
//       - Uses helper::connected_letters(<number_of_components>, <combinations>) to
//         receive a vector with all indices of the components that belong to one group
//       - Loops over the letter groups and choose only groups which contain more than one component
//       - Combines all components of a group to one group component and use your implemented
//         function algorithms::compute_bounding_boxes(<group_components>, <group_bounding_boxes>)
//
// parameters:
//  - input_image: [CV_32FC1] matrix with the input image
//  - swt_image: [CV_32FC1] matrix with the stroke widths
//  - text_labels: [CV_16UC1] matrix with labels of recognized text components
//  - text_components: vector of vector of points with text components
//  - bounding_boxes: vector of bounding boxes of all text components
//  - height_ratio_threshold: ratio of the vertical distance between two bounding boxes
//  - width_ratio_threshold: max width ratio between two bounding boxes
//  - median_ratio_threshold: max stroke median ratio between two bounding boxes
//  - distance_ratio: ratio of the horizontal distance between two bounding boxes
//  - color_distance_threshold: max ratio of the mean between two bounding boxes
//  - group_bounding_boxes: vector of bounding boxes of "words"
//  - letter_bounding_boxes: vector of bounding boxes of recognized letters forming the "words"
// return: void
//===============================================================================
void helper::find_letter_groups(const cv::Mat &input_image, const cv::Mat &swt_image, const cv::Mat &text_labels,
                                const std::vector<std::vector<cv::Point2i>> &text_components,
                                const std::vector<cv::Rect2i> &bounding_boxes, const float height_ratio_threshold,
                                const float width_ratio_threshold, const float median_ratio_threshold,
                                const float distance_ratio, const float color_distance_threshold,
                                std::vector<cv::Rect2i> &group_bounding_boxes,
                                std::vector<cv::Rect2i> &letter_bounding_boxes)
{
    cv::Mat mask = cv::Mat::zeros(input_image.size(), CV_8U);
    cv::threshold(text_labels, mask, 0, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    std::vector<std::vector<int>> combinations;
    for (int comp_i = 0; comp_i < text_components.size(); comp_i++)
    {
        int offset = comp_i + 1;
        for (int comp_j = offset; comp_j < text_components.size(); comp_j++)
        {
            cv::Rect2i box1 = bounding_boxes.at(comp_i);
            cv::Rect2i box2 = bounding_boxes.at(comp_j);

            // check height ratio between bounding boxes
            // paper: 1/2 < ratio < 2
            float height_ratio = box1.height / (float)box2.height;
            if (height_ratio > height_ratio_threshold || height_ratio < 1 / height_ratio_threshold)
                continue;

            // check width ratio between bounding boxes
            // paper: no threshold
            float width_ratio = box1.width / (float)box2.width;
            if (width_ratio > width_ratio_threshold || width_ratio < 1 / width_ratio_threshold)
                continue;

            // check if bounding boxes are on the same line
            int pos1 = box1.y + box1.height / 2.f;
            int pos2 = box2.y + box2.height / 2.f;
            if (pos1 < box2.y || pos2 < box1.y)
                continue;

            // check the distance between the bounding boxes
            // paper: 3 * max_width
            float max_width = std::max(box1.width, box2.width);
            int distance;
            if (box1.x < box2.x)
                distance = std::abs(box1.x + box1.width - box2.x);
            else
                distance = std::abs(box2.x + box2.width - box1.x);

            if (distance > distance_ratio * max_width)
                continue;

            // compare median stroke width
            std::vector<float> stroke_widths_comp1;
            for (const cv::Point2i &p : text_components.at(comp_i))
            {
                float stroke_width = swt_image.at<float>(p);
                stroke_widths_comp1.push_back(stroke_width);
            }
            float median1 = 0.0f;
            std::sort(stroke_widths_comp1.begin(), stroke_widths_comp1.end());
            int vector_length1 = stroke_widths_comp1.size();
            if (vector_length1 % 2)
                median1 = stroke_widths_comp1.at(vector_length1 / 2);
            else
                median1 = 0.5f *
                          (stroke_widths_comp1.at(vector_length1 / 2) + stroke_widths_comp1.at(vector_length1 / 2 - 1));

            std::vector<float> stroke_widths_comp2;
            for (const cv::Point2i &p : text_components.at(comp_j))
            {
                float stroke_width = swt_image.at<float>(p);
                stroke_widths_comp2.push_back(stroke_width);
            }
            float median2 = 0.0f;
            std::sort(stroke_widths_comp2.begin(), stroke_widths_comp2.end());
            int vector_length2 = stroke_widths_comp2.size();
            if (vector_length2 % 2)
                median2 = stroke_widths_comp2.at(vector_length2 / 2);
            else
                median2 = 0.5f *
                          (stroke_widths_comp2.at(vector_length2 / 2) + stroke_widths_comp2.at(vector_length2 / 2 - 1));

            // check the ratio of the median stroke widths
            // paper: 1/2 < ratio < 2.0
            if (median1 / median2 > median_ratio_threshold || median2 / median1 > median_ratio_threshold)
                continue;

            // compare average color distance
            // paper: color_distance < 40
            cv::Mat mask_roi1 = mask(box1);
            cv::Mat mask_roi2 = mask(box2);

            cv::Mat roi1 = input_image(box1);
            cv::Mat roi2 = input_image(box2);

            cv::Scalar average_color1 = cv::mean(roi1, mask_roi1);
            cv::Scalar average_color2 = cv::mean(roi2, mask_roi2);

            float color_distance = std::sqrt(std::pow(average_color1[0] - average_color2[0], 2) +
                                             std::pow(average_color1[1] - average_color2[1], 2) +
                                             std::pow(average_color1[2] - average_color2[2], 2));

            if (color_distance > color_distance_threshold)
                continue;

            combinations.push_back({comp_i, comp_j});
        }
    }

    // merge letters to create groups
    // use helper function to find connected components
    std::vector<std::vector<int>> letter_groups = helper::connected_letters(text_components.size(), combinations);
    std::vector<std::vector<cv::Point2i>> group_components;

    for (std::vector<int> &letter_group : letter_groups)
    {
        // include only groups with multiple letters (more than 1)
        if (letter_group.size() <= 1)
            continue;

        std::vector<cv::Point2i> group_component;
        for (int &letter_index : letter_group)
        {
            letter_bounding_boxes.push_back(bounding_boxes.at(letter_index));
            std::vector<cv::Point2i> component = text_components.at(letter_index);
            group_component.insert(group_component.begin(), component.begin(), component.end());
        }
        group_components.push_back(group_component);
    }
    algorithms::compute_bounding_boxes(group_components, group_bounding_boxes);
}
