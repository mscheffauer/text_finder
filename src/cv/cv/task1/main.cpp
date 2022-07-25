#include <dirent.h>
#include <sys/stat.h>

#include <iostream>
#include <vector>

#include "algorithms.h"
#include "helper.h"
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

//#define FULL_VERSION 1
//#define FINAL_RUN 0
//#define GENERATE_REF 0

#define RST "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST

#define BOLD(x) "\x1B[1m" x RST


//===============================================================================
// Configuration
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
struct Config
{
    // edge detection
    int edge_threshold_min = 0;
    int edge_threshold_max = 0;
    bool black_on_white = true;

    // connected components
    float stroke_width_ratio_threshold = 0.f;
    int neighbor_offset = 0;

    // discard non text
    float variance_ratio = 0.f;
    float aspect_ratio_threshold = 0.f;
    float diameter_ratio_threshold = 0.f;
    int min_height = 0;
    int max_height = 0;

    // find letter groups
    float height_ratio_threshold = 0.f;
    float width_ratio_threshold = 0.f;
    float distance_ratio = 0.f;
    float median_ratio_threshold = 0.f;
    float color_distance_threshold = 0.f;
};

//===============================================================================
// make_directory()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void make_directory(const char *path)
{
#if defined(_WIN32)
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

//===============================================================================
// is_path_existing()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
bool is_path_existing(const char *path)
{
    struct stat buffer {};
    return (stat(path, &buffer)) == 0;
}



//===============================================================================
// save_image()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void save_image(const std::string& out_directory, const std::string& name, size_t number, const cv::Mat &image)
{
    std::stringstream number_stringstream;
    number_stringstream << std::setfill('0') << std::setw(2) << number;
    std::string path = out_directory + number_stringstream.str() + "_" + name + ".png";
    cv::imwrite(path, image);
    std::cout << "saving image: " << path << std::endl;
}

//===============================================================================
// build_colormap()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
std::vector<cv::Vec3b> build_colormap()
{
    std::vector<cv::Vec3b> colors {
        cv::Vec3b(255, 179, 0),   cv::Vec3b(128, 62, 117), cv::Vec3b(255, 104, 0),   cv::Vec3b(166, 189, 215),
        cv::Vec3b(193, 0, 32),    cv::Vec3b(206, 162, 98), cv::Vec3b(129, 112, 102), cv::Vec3b(0, 125, 52),
        cv::Vec3b(246, 118, 142), cv::Vec3b(0, 83, 138),   cv::Vec3b(255, 122, 92),  cv::Vec3b(83, 55, 122),
        cv::Vec3b(255, 142, 0),   cv::Vec3b(179, 40, 81),  cv::Vec3b(244, 200, 0),   cv::Vec3b(127, 24, 13),
        cv::Vec3b(147, 170, 0),   cv::Vec3b(89, 51, 21),   cv::Vec3b(241, 58, 19),   cv::Vec3b(35, 44, 22)
    };

    std::vector<cv::Vec3b> lut;
    lut.push_back(cv::Vec3b(255, 255, 255));
    for (int i = 1; i < 256; i++)
    {
        lut.push_back(colors.at(i % colors.size()));
    }
    return lut;
}

//===============================================================================
// create_ray_image()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
cv::Mat create_ray_image(const std::vector<std::vector<cv::Point2i>> &rays, cv::Size image_dims)
{
    // display rays
    cv::Mat ray_image = cv::Mat::zeros(image_dims, CV_8UC3);

    for (const std::vector<cv::Point2i>& ray : rays)
    {
        for (const cv::Point2i& point : ray)
        {
            ray_image.at<cv::Vec3b>(point) = cv::Vec3b{255, 0, 0};
        }
    }
    return ray_image;
}

//===============================================================================
// run()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void run(const cv::Mat& input_image, const std::string& out_directory, const std::string& ref_directory, Config config)
{
    size_t image_counter = 0;
    //=============================================================================
    // Grayscale image
    //=============================================================================
    std::cout << "Step 1 - calculating grayscale image... " << std::endl;
    cv::Mat grayscale = cv::Mat::zeros(input_image.size(), CV_8UC1);
    cv::Mat blurred_image;
    cv::GaussianBlur(input_image, blurred_image, cv::Size(3, 3), 0.0);
    algorithms::compute_grayscale(blurred_image, grayscale);
    save_image(out_directory, "grayscale", ++image_counter, grayscale);

    //=============================================================================
    // Gradient image
    //=============================================================================
    std::cout << "Step 2 - calculating gradient image... " << std::endl;
    cv::Mat gradient_x = cv::Mat::zeros(input_image.size(), CV_32FC1);
    cv::Mat gradient_y = cv::Mat::zeros(input_image.size(), CV_32FC1);
    cv::Mat gradient_abs = cv::Mat::zeros(input_image.size(), CV_32FC1);
    algorithms::compute_gradient(grayscale, gradient_x, gradient_y, gradient_abs);
    save_image(out_directory, "gradient_x", ++image_counter, gradient_x);
    save_image(out_directory, "gradient_y", ++image_counter, gradient_y);
    save_image(out_directory, "gradient_abs", ++image_counter, gradient_abs);
    //=============================================================================
    // Compute Directions
    //=============================================================================
    std::cout << "Step 3 - calculating directions image... " << std::endl;
    cv::Mat direction_x = cv::Mat::zeros(input_image.size(), CV_32FC1);
    cv::Mat direction_y = cv::Mat::zeros(input_image.size(), CV_32FC1);
    algorithms::compute_directions(gradient_x, gradient_y, gradient_abs, direction_x, direction_y);

    // display directions
    cv::Mat display_dir_x = cv::Mat::zeros(input_image.size(), CV_8UC1);
    cv::normalize(direction_x, display_dir_x, 0, 255, cv::NORM_MINMAX);
    cv::Mat display_dir_y = cv::Mat::zeros(input_image.size(), CV_8UC1);
    cv::normalize(direction_y, display_dir_y, 0, 255, cv::NORM_MINMAX);
    save_image(out_directory, "direction_x", ++image_counter, display_dir_x);
    save_image(out_directory, "direction_y", ++image_counter, display_dir_y);

    //=============================================================================
    // Canny Edges - cv-function (to get edges for calc)
    //=============================================================================
    cv::Mat canny_edges = cv::Mat::zeros(input_image.size(), CV_8UC1);
    cv::Canny(grayscale, canny_edges, config.edge_threshold_min, config.edge_threshold_max, 3);
    save_image(out_directory, "canny_edges", ++image_counter, canny_edges);

    //=============================================================================
    // SWT - SWT Estimate Stroke Width
    //=============================================================================
    std::cout << "Step 4 - calculating swt image... " << std::endl;
    cv::Mat swt_stroke_width_image = cv::Mat::zeros(input_image.size(), CV_32FC1);
    std::vector<std::vector<cv::Point2i>> rays;
    algorithms::swt_compute_stroke_width(canny_edges, direction_x, direction_y, config.black_on_white, rays,
                                         swt_stroke_width_image);

    cv::Mat ray_image = create_ray_image(rays, input_image.size());
    save_image(out_directory, "swt_rays", ++image_counter, ray_image);
    //=============================================================================
    // SWT - SWT Postprocessing
    //=============================================================================
    cv::Mat swt_final_image = cv::Mat(canny_edges.size(), CV_32FC1, cv::Scalar(FLT_MAX));
    algorithms::swt_postprocessing(swt_stroke_width_image, rays, swt_final_image);

    cv::Mat display_swt_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
    cv::normalize(swt_final_image, display_swt_image, 0, 255, cv::NORM_MINMAX);
    save_image(out_directory, "swt", ++image_counter, display_swt_image);
    //=============================================================================
    // Connected components
    //=============================================================================
    std::cout << "Step 5 - calculating connected components... " << std::endl;
    cv::Mat labels = cv::Mat::zeros(swt_final_image.size(), CV_16UC1);
    std::vector<std::vector<cv::Point2i>> components;
    algorithms::get_connected_components(swt_final_image, config.stroke_width_ratio_threshold, config.neighbor_offset, labels,
                                         components);

    // normalize labels
    double min_label, max_label;
    cv::minMaxLoc(labels, &min_label, &max_label);
    cv::Mat display_labels = cv::Mat::zeros(input_image.size(), CV_8UC1);
    cv::normalize(labels, display_labels, 0, 255, cv::NORM_MINMAX);
    display_labels.convertTo(display_labels, CV_8UC1);

    // display labels
    cv::cvtColor(display_labels, display_labels, cv::COLOR_GRAY2RGB);
    cv::LUT(display_labels, build_colormap(), display_labels);
    save_image(out_directory, "connected_components", ++image_counter, display_labels);
    //=============================================================================
    // Bounding box
    //=============================================================================
    std::cout << "Step 6 - calculating bounding boxes... " << std::endl;
    std::vector<cv::Rect2i> bounding_boxes;
    algorithms::compute_bounding_boxes(components, bounding_boxes);

    // display bounding boxes
    cv::Mat display_bounding_boxes;
    display_labels.copyTo(display_bounding_boxes);
    for (cv::Rect2i & bounding_box : bounding_boxes)
    {
        cv::rectangle(display_bounding_boxes, bounding_box, cv::Scalar(0, 255, 0));
    }
    save_image(out_directory, "bounding_boxes", ++image_counter, display_bounding_boxes);
    //=============================================================================
    // Discard non-text
    //=============================================================================
    std::cout << "Step 7 - discard non-text... " << std::endl;
    std::vector<std::vector<cv::Point2i>> text_components;
    cv::Mat text_labels = cv::Mat::zeros(swt_final_image.size(), CV_16UC1);
    std::vector<cv::Rect2i> text_bounding_boxes;
    algorithms::discard_non_text(swt_final_image, bounding_boxes, components, labels, config.variance_ratio,
                                 config.aspect_ratio_threshold, config.diameter_ratio_threshold, config.min_height, config.max_height,
                                 text_bounding_boxes, text_components, text_labels);

    // normalize labels with max_label and min_label to generate same color coding
    cv::Mat display_text_labels = cv::Mat::zeros(input_image.size(), CV_8UC1);
    text_labels.copyTo(display_text_labels);
    display_text_labels.convertTo(display_text_labels, CV_8UC1, 255.0 / (max_label - min_label),
                                  -min_label * 255.0 / (max_label - min_label));
    // display labels
    cv::cvtColor(display_text_labels, display_text_labels, cv::COLOR_GRAY2RGB);
    cv::LUT(display_text_labels, build_colormap(), display_text_labels);

    // display bounding boxes
    cv::Mat display_letter_bounding_boxes;
    display_text_labels.copyTo(display_letter_bounding_boxes);
    for (cv::Rect2i & text_bounding_box : text_bounding_boxes)
    {
        cv::rectangle(display_letter_bounding_boxes, text_bounding_box, cv::Scalar(0, 255, 0));
    }

    save_image(out_directory, "discard_non_text", ++image_counter, display_letter_bounding_boxes);
    //=============================================================================
    // Find letter groups
    //=============================================================================
    std::cout << "Step 8 - find letter groups... " << std::endl;
    std::vector<cv::Rect2i> group_bounding_boxes;
    std::vector<cv::Rect2i> letter_bounding_boxes;
    helper::find_letter_groups(input_image, swt_final_image, text_labels, text_components, text_bounding_boxes,
                               config.height_ratio_threshold, config.width_ratio_threshold, config.median_ratio_threshold,
                               config.distance_ratio, config.color_distance_threshold, group_bounding_boxes,
                               letter_bounding_boxes);
    // display bounding boxes
    cv::Mat display_group_bounding_boxes;
    display_text_labels.copyTo(display_group_bounding_boxes);
    for (cv::Rect2i & group_bounding_box : group_bounding_boxes)
    {
        cv::rectangle(display_group_bounding_boxes, group_bounding_box, cv::Scalar(0, 255, 0));
    }
    for (cv::Rect2i & letter_bounding_box : letter_bounding_boxes)
    {
        cv::rectangle(display_group_bounding_boxes, letter_bounding_box, cv::Scalar(0, 0, 255));
    }

    save_image(out_directory, "letter_groups", ++image_counter, display_group_bounding_boxes);

    //=============================================================================
    // Display bounding boxes in input image
    //=============================================================================
    std::cout << "Step 9 - calculating final output... " << std::endl;
    cv::Mat final_image = cv::Mat::zeros(input_image.size(), CV_8UC3);
    input_image.copyTo(final_image);
    // Display bounding boxes
    for (cv::Rect2i & group_bounding_box : group_bounding_boxes)
    {
        cv::rectangle(final_image, group_bounding_box, cv::Scalar(0, 255, 0));
    }
    for (cv::Rect2i & letter_bounding_box : letter_bounding_boxes)
    {
        cv::rectangle(final_image, letter_bounding_box, cv::Scalar(0, 0, 255));
    }

    save_image(out_directory, "final", ++image_counter, final_image);

    //=============================================================================
    // BONUS
    //=============================================================================
    //=============================================================================
    // Non-Maxima Suppression
    //=============================================================================
    std::cout << "Bonus 1 - compute non maxima suppression... " << std::endl;
    cv::Mat non_maxima = cv::Mat::zeros(input_image.size(), CV_32FC1);
    algorithms::non_maxima_suppression(gradient_abs, gradient_x, gradient_y, non_maxima);
    save_image(out_directory + "bonus/", "bonus_non_maxima", ++image_counter, non_maxima);

    //=============================================================================
    // Hysteresis
    //=============================================================================
    std::cout << "Bonus 2 - compute hysteresis... " << std::endl;
    cv::Mat hysteresis = cv::Mat::zeros(input_image.size(), CV_8UC1);
    non_maxima.convertTo(non_maxima, CV_8UC1);
    algorithms::hysteresis(non_maxima, config.edge_threshold_min, config.edge_threshold_max, hysteresis);
    save_image(out_directory + "bonus/", "bonus_hysteresis", ++image_counter, hysteresis);

    //=============================================================================
    // Own Canny
    //=============================================================================
    std::cout << "Bonus 3 - calculate edges... " << std::endl;
    cv::Mat edges = cv::Mat::zeros(input_image.size(), CV_8UC1);
    algorithms::canny_own(grayscale, config.edge_threshold_min, config.edge_threshold_max, edges);
    save_image(out_directory + "bonus/", "bonus_edges", ++image_counter, edges);
}


//===============================================================================
// execute_testcase()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void execute_testcase(const rapidjson::Value &config_data)
{
    //=============================================================================
    // Parse input data
    //=============================================================================
    std::string name = config_data["name"].GetString();
    std::string image_path = config_data["image_path"].GetString();

    Config config;

    // edge detection
    config.edge_threshold_min = (int) config_data["edge_threshold_min"].GetUint();
    config.edge_threshold_max = (int) config_data["edge_threshold_max"].GetUint();
    config.black_on_white = config_data["black_on_white"].GetBool();

    // connected components
    config.stroke_width_ratio_threshold = (float) config_data["stroke_width_ratio_threshold"].GetDouble();
    config.neighbor_offset = (int) config_data["neighbor_offset"].GetUint();

    // discard non text
    config.variance_ratio = (float) config_data["variance_ratio"].GetDouble();
    config.aspect_ratio_threshold = (float) config_data["aspect_ratio_threshold"].GetDouble();
    config.diameter_ratio_threshold = (float) config_data["diameter_ratio_threshold"].GetDouble();
    config.min_height = (int) config_data["min_height"].GetUint();
    config.max_height = (int) config_data["max_height"].GetUint();

    // find letter groups
    config.height_ratio_threshold = (float) config_data["height_ratio_threshold"].GetDouble();
    config.width_ratio_threshold = (float) config_data["width_ratio_threshold"].GetDouble();
    config.distance_ratio = (float) config_data["distance_ratio"].GetDouble();
    config.median_ratio_threshold = (float) config_data["median_ratio_threshold"].GetDouble();
    config.color_distance_threshold = (float) config_data["color_distance_threshold"].GetDouble();

    //=============================================================================
    // Load input images
    //=============================================================================
    std::cout << BOLD(FGRN("[INFO]")) << " Input image: " << image_path << std::endl;

    cv::Mat img = cv::imread(image_path);

    if (!img.data)
    {
        std::cout << BOLD(FRED("[ERROR]")) << " Could not load image (" << name + ".png"
                  << ")" << std::endl;
        throw std::runtime_error("Could not load file");
    }

    //=============================================================================
    // Create output directory
    //=============================================================================
    std::string output_directory = "output/" + name + "/";

    std::cout << BOLD(FGRN("[INFO]")) << " Output path: " << output_directory << std::endl;

    make_directory("output/");
    make_directory(output_directory.c_str());
    // create bonus directory
    make_directory((output_directory + "/bonus/").c_str());

    std::string ref_path = "data/ref_x64/json/";
    std::string ref_directory = ref_path + name + "/";



    //=============================================================================
    // Starting default task
    //=============================================================================
    std::cout << "Starting MAIN Task..." << std::endl;
    run(img, output_directory, ref_directory, config);
}

//===============================================================================
// main()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
int main(int argc, char *argv[])
{
    std::cout << "CV/task1 framework version 1.0" << std::endl;  // DO NOT REMOVE THIS LINE!!!
    std::cout << "===================================" << std::endl;
    std::cout << "               CV Task 1           " << std::endl;
    std::cout << "===================================" << std::endl;

    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <config-file>" << std::endl;
        return 1;
    }

    std::string path = std::string(argv[1]);
    std::ifstream fs(path);
    if (!fs)
    {
        std::cout << "Error: Failed to open file " << path << std::endl;
        return 2;
    }
    std::stringstream buffer;
    buffer << fs.rdbuf();

    rapidjson::Document doc;
    rapidjson::ParseResult check;
    check = doc.Parse<0>(buffer.str().c_str());

    if (check)
    {
        if (doc.HasMember("testcases"))
        {
            rapidjson::Value &testcases = doc["testcases"];
            for (rapidjson::SizeType i = 0; i < testcases.Size(); i++)
            {
                rapidjson::Value &testcase = testcases[i];
                try
                {
                    execute_testcase(testcase);
                }
                catch (const std::exception &e)
                {
                    std::cout << e.what() << std::endl;
                    std::cout << BOLD(FRED("[ERROR]")) << " Program exited with errors!" << std::endl;
                    return -1;
                }
            }
        }
        std::cout << "Program exited normally!" << std::endl;
    }
    else
    {
        std::cout << "Error: Failed to parse file " << argv[1] << ":" << check.Offset() << std::endl;
        return 3;
    }
    return 0;
}
