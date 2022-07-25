#include "algorithms.h"

//===============================================================================
// compute_grayscale()
//-------------------------------------------------------------------------------
// TODO: Compute the grayscale image of the input
// hints: - use cv::Vec3b type to access the color values of a 3-channel image
//        - be aware that OpenCV treats matrix accesses in row-major order!
//          (iterate through rows then columns)
//
// parameters:
//  - input_image: [CV_8UC3] the image for the grayscale calculation
//  - grayscale_image: [CV_8UC1] grayscaled image
// return: void
//===============================================================================
void algorithms::compute_grayscale(const cv::Mat &input_image, cv::Mat &grayscale_image)
{

    cv::MatIterator_<unsigned char> mat_out = grayscale_image.begin<unsigned char>();

    for (auto mat_beg = input_image.begin<cv::Vec3b>(); mat_beg != input_image.end<cv::Vec3b>(); ++mat_beg,++mat_out) {
        *mat_out = (unsigned char)((*mat_beg)[2]*0.2989+(*mat_beg)[1]*0.5870+(*mat_beg)[0]*0.1140);
    }

}

//===============================================================================
// compute_gradient()
//-------------------------------------------------------------------------------
// TODO: Compute the 1st Scharr derivative once in x and once in y direction
//       and combine these two.
// hint: use the arithmetic operations of OpenCV
//
// parameters:
//  - grayscale_image: [CV_8UC1] the grayscale image for the gradient calculation
//  - gradient_x: [CV_32FC1] output matrix for the gradient in x direction
//  - gradient_y: [CV_32FC1] output matrix for the gradient in y direction
//  - gradient_abs: [CV_32FC1] output matrix for the gradient image
// return: void
//===============================================================================
void algorithms::compute_gradient(const cv::Mat &grayscale_image, cv::Mat &gradient_x, cv::Mat &gradient_y,
                                  cv::Mat &gradient_abs)
{
    //calculate x derivative
    cv::Scharr(grayscale_image,gradient_x,CV_32F,1,0);
    //calculate y derivative
    cv::Scharr(grayscale_image,gradient_y,CV_32F,0,1);

    //now combine
    cv::MatConstIterator_<float> mat_y = gradient_y.begin<float>();
    cv::MatIterator_<float> mat_abs = gradient_abs.begin<float>();

    for (auto mat_x = gradient_x.begin<float>(); mat_x != gradient_x.end<float>(); ++mat_x,++mat_abs,++mat_y) {
        *mat_abs = (float)(sqrt(pow((*mat_x),2)+pow((*mat_y),2)));
    }


}

//===============================================================================
// compute_directions()
//-------------------------------------------------------------------------------
// TODO: Compute the directions of the x/y-gradients.
// hint: be aware of the access type (float)
//
// parameters:
//  - gradient_x: [CV_32FC1] matrix with the gradient in x direction
//  - gradient_y: [CV_32FC1] matrix with the gradient in y direction
//  - gradient_abs: [CV_32FC1] matrix with the gradient image
//  - direction_x: [CV_32FC1] output matrix for the gradient direction in x direction
//  - direction_y: [CV_32FC1] output matrix for the gradient direction in y direction
// return: void
//===============================================================================
void algorithms::compute_directions(const cv::Mat &gradient_x, const cv::Mat &gradient_y, const cv::Mat &gradient_abs,
                                    cv::Mat &direction_x, cv::Mat &direction_y)
{


    cv::MatConstIterator_<float> mat_y = gradient_y.begin<float>();
    cv::MatConstIterator_<float> mat_abs = gradient_abs.begin<float>();
    cv::MatIterator_<float> mat_dir_x = direction_x.begin<float>();
    cv::MatIterator_<float> mat_dir_y = direction_y.begin<float>();

    for (auto mat_x = gradient_x.begin<float>(); mat_x != gradient_x.end<float>(); ++mat_x,++mat_abs,++mat_y,++mat_dir_x,++mat_dir_y) {
        if ((unsigned char)((*mat_abs) == 0)){
            *mat_dir_x = 0;
            *mat_dir_y = 0;
        }else{
            *mat_dir_x = (*mat_x)/(*mat_abs);
            *mat_dir_y = (*mat_y)/(*mat_abs);
        }
    }
}

//===============================================================================
// swt_estimate_stroke_width()
//-------------------------------------------------------------------------------
// TODO: Calculate the stroke width for each ray. A ray starts on an edge point.
//       - Add the appropriate points cv::Point2i(col, row) to a ray vector
//       - Store the stroke width of a point in swt_estimation_image
// hint: - in OpenCV cv::Point(x, y) is declared as x=column and y=row
//       - you can use either mat.at<type>(row, col) or mat.at<type>(cv::Point(col, row))
//         to access the same point
//       - use the the mathematical functions provided by the standard library
//         (example: std::floor, std::sqrt, std::pow, etc.)
//
// parameters:
//  - edges: [CV_8UC1] matrix filled with the Canny-edges
//  - direction_x: [CV_32FC1] matrix of the gradient direction in x direction
//  - direction_y: [CV_32FC1] matrix of the gradient direction in y direction
//  - black_on_white: bool parameter to decide the direction of the rays
//  - rays: vector of vectors of points (x = col, y = row)
//  - swt_estimation_image: [CV_32FC1] output matrix for the stroke widths, initialize with FLT_MAX
// return: void
//===============================================================================
void algorithms::swt_compute_stroke_width(const cv::Mat &edges, const cv::Mat &direction_x, const cv::Mat &direction_y,
                                          const bool black_on_white, std::vector<std::vector<cv::Point2i>> &rays,
                                          cv::Mat &swt_stroke_width_image) {
    //init SWT image
    swt_stroke_width_image.setTo(cv::Scalar(FLT_MAX));

    int8_t direction = 1;
    int step_size = 0;
    int current_row = 0;
    int current_col = 0;
    std::vector<cv::Point2i> temporary_ray_pixels;

    if (black_on_white == true) {
        direction = -1;
    } else {
        direction = 1;
    }

    //LOOP OVER THE IMAGE
    for (int i = 0; i < edges.rows; i++) {
        for (int j = 0; j < edges.cols; j++) {
            //a edge pixel is found

           if (edges.at<unsigned char>(i, j)==255){

                auto ray_dir_x = direction_x.at<float>(i, j);
                auto ray_dir_y = direction_y.at<float>(i, j);
                //first clear the temp ray array
                step_size = 0;
                temporary_ray_pixels.clear();
                //put in the first pixel
                temporary_ray_pixels.push_back(cv::Point2i(j, i));


                //RAY EMIT FOR EVERY EDGE PIXEL check if iin boundary
                while (true){
                    step_size++;
                    current_col = j;
                    current_row = i;
                    auto old_row = current_row;
                    auto old_col = current_col;
                    current_row = std::floor(i + (ray_dir_y *step_size * direction));
                    current_col = std::floor(j + (ray_dir_x *step_size * direction));
                    //has not moved do nothing
                    if  (((current_row == old_row) && (current_col == old_col))){
                        continue;
                    }
                    //check if new current_row or col is outside boundary
                    if((current_col < 0 ) || (current_row < 0) || (current_row == edges.rows) || (current_col == edges.cols)){
                        break;
                    }


                    //another edge pixel found, check if valid
                    if (edges.at<unsigned char>(current_row, current_col)==255) {
                        //now check if indeed ray is valid
                        auto curr_dir_x = direction_x.at<float>(current_row, current_col);
                        auto curr_dir_y = direction_y.at<float>(current_row, current_col);

                        double dotp = ((ray_dir_x * curr_dir_x) +(ray_dir_y * curr_dir_y)) *(double)(-1);
                        //if yes copy over to ray array
                        if (dotp >= cos(CV_PI / 6)) {
                            if (temporary_ray_pixels.back() != cv::Point2i(current_col, current_row)) {
                                temporary_ray_pixels.push_back(cv::Point2i(current_col, current_row));
                            }

                            rays.push_back(std::vector<cv::Point2i>(temporary_ray_pixels));
                            //calculate stroke width of ray

                            auto sumx = temporary_ray_pixels.front().x - temporary_ray_pixels.back().x;
                            auto sumy = temporary_ray_pixels.front().y - temporary_ray_pixels.back().y;
                            auto wid = sqrt(pow(sumx,2)+pow(sumy,2));
                            //assign stroke width to every pixel of the ray
                            for (auto point :  temporary_ray_pixels) {

                                 if  (wid < swt_stroke_width_image.at<float>(point)){
                                    swt_stroke_width_image.at<float>(point) = wid;
                                 }
                            }

                        }

                       //in any way discard temp ray after
                        temporary_ray_pixels.clear();
                        break;
                    }


                    //still marching
                    else {
                        //store ray pixels on the way
                        if (temporary_ray_pixels.back() != cv::Point2i(current_col, current_row)) {
                            temporary_ray_pixels.push_back(cv::Point2i(current_col, current_row));
                        }
                        continue;
                    }

                }


            }


        }
    }

}

//===============================================================================
// swt_postprocessing()
//-------------------------------------------------------------------------------
// TODO: Compute the median for each ray and choose the appropriate stroke width.
//       - Loop over the rays
//       - Compute the median stroke width of each ray
//       - Choose the minimum of the median and the pixels on the ray from first run
// hint: use the the mathematical functions provided by the standard library
//
// parameters:
//  - swt_stroke_width_image: [CV_32FC1] matrix with stroke widths from first run
//  - rays: vector of vectors of points (x = col, y = row)
//  - swt_final_image: [CV_32FC1] output matrix with the postprocessed stroke widths, initialized with FLT_MAX
// return: void
//===============================================================================
void algorithms::swt_postprocessing(const cv::Mat &swt_stroke_width_image,
                                    const std::vector<std::vector<cv::Point2i>> &rays, cv::Mat &swt_final_image)
{

    //init SWT image
    swt_final_image.setTo(cv::Scalar(0));

    std::vector<float> temporary_ray_width;
    std::vector<cv::Point2i> temporary_ray_pos;

    float median;
    //for each ray
    for (auto ray : rays) {

        //copy over width value of each ray and pixel positions of each ray
        for (auto point : ray) {
            temporary_ray_width.push_back(swt_stroke_width_image.at<float>(point));
            temporary_ray_pos.push_back(point);
        }

        auto size_of_ray = temporary_ray_width.size();
        int size_half = floor(size_of_ray /2 );
        if  (size_of_ray > 0) {
            //sort each ray
            std::sort(temporary_ray_width.begin(),temporary_ray_width.end());

            if (size_of_ray % 2 == 0){ // even
               median = (temporary_ray_width[size_half]+temporary_ray_width[size_half-1])/2;

            }else{ //odd
                median = temporary_ray_width[size_half];
            }

        }

        //set final image   //clamp all to mean
        for (auto point : temporary_ray_pos) {
            auto width = swt_stroke_width_image.at<float>(point);
            if  (width > median){
                swt_final_image.at<float>(point) = median;
            }else{
                swt_final_image.at<float>(point) = swt_stroke_width_image.at<float>(point);
            }

        }
        temporary_ray_width.clear();
        temporary_ray_pos.clear();


    }

}

//===============================================================================
// get_connected_components()
//-------------------------------------------------------------------------------
// TODO: Find connected components on the basis of the stroke widths in the neighborhood
//       - Loop over the rows and the columns of the swt_image
//       - Check if the neighboring pixels belong to the same component
//         (you can use the provided struct PosStrokeWidth{col, row, stroke_width})
//       - Add the point at the position (cv::Point2i(col, row)) to the component
// hints: - each component has a label, component labels start at 1
//        - assign label 0 for pixels which do not belong to a component
//        - you can use the provided struct PosStrokeWidth{col, row, stroke_width}
//          to store the necessary values of a neighbor
//
// parameters:
//  - swt_image: [CV_32FC1] matrix with the stroke widths
//  - stroke_width_ratio_threshold: ratio of the stroke widths between two neighboring pixels
//  - neighbor_offset: maximum offset for the neighborhood pixels
//  - labels: [CV_16UC1] output matrix with component labels for each position
//  - components: vector of vectors of points (x = col, y = row)
// return: void
//===============================================================================

void algorithms::get_connected_components(const cv::Mat &swt_image, const float stroke_width_ratio_threshold,
                                          const int neighbor_offset, cv::Mat &labels,
                                          std::vector<std::vector<cv::Point2i>> &components)
{

    PosStrokeWidth temp;
    unsigned int curr_label = 1;
    labels.setTo(cv::Scalar(0));
    std::vector<PosStrokeWidth> neighbors;
    std::vector<cv::Point2i> temp_component;

    //LOOP OVER THE IMAGE
    for (int i = 0; i < swt_image.rows; i++) {
        for (int j = 0; j < swt_image.cols; j++) {
            //first skip all stroke width zeros and already labeled pixels
            if ((swt_image.at<float>(i,j) != 0) && (labels.at<unsigned short>(i,j) == 0)){

                //now add the first pixel to a component, then a recursion finds all connected pixels, then a new component gets a new label_cnt
                temp.col = j;
                temp.row = i;
                temp.stroke_width = swt_image.at<float>(i,j);
                neighbors.clear();
                neighbors.push_back(temp);
                // the first element can only be valid.

                //iterate over neighbors
                while (neighbors.size() > 0){
                    //take out one element from the neighborhood list and perform all checks

                    auto neighbor_to_check = neighbors.back();
                    neighbors.pop_back();
                   //now find neighbors of candidate and also put them on the neigbor list
                            for (int k = (neighbor_to_check.row - neighbor_offset); k < (neighbor_to_check.row + neighbor_offset +1); ++k) {
                                for (int l = (neighbor_to_check.col - neighbor_offset); l < (neighbor_to_check.col + neighbor_offset +1); ++l) {
                                    //add all neighbours to queue
                                    if ((k >= 0) && (l>= 0) && (k< swt_image.rows) && (l< swt_image.cols)) {
                                        if ((labels.at<unsigned short>(k, l) == 0) && (swt_image.at<float>(k, l) != 0)) {

                                            temp.col = l;
                                            temp.row = k;
                                            temp.stroke_width = swt_image.at<float>(k, l);

                                            double stroke_ratio = neighbor_to_check.stroke_width / temp.stroke_width;
                                            if ((stroke_ratio > (1/stroke_width_ratio_threshold)) && (stroke_ratio < stroke_width_ratio_threshold)) {
                                                labels.at<unsigned short>(temp.row,temp.col) = curr_label;
                                                temp_component.push_back(cv::Point2i(temp.col, temp.row));
                                                neighbors.push_back(temp);
                                            }
                                        }
                                    }
                                }

                    }


               }


               //all done: add component to components and find next component
               components.push_back(temp_component);
               temp_component.clear();

               curr_label ++;


            }

        }
    }
}

//===============================================================================
// compute_bounding_boxes()
//-------------------------------------------------------------------------------
// TODO: Compute the bounding box for each component
// hint: - save a rectangle in order (x = col, y = row, width, height)
//       - use the the mathematical functions provided by the standard library
//
// parameters:
//  - components: vector of vectors of points (x = col, y = row)
//  - bounding_boxes: output vector of rectangles (x = col, y = row, width, height)
// return: void
//===============================================================================
//compare functions for cv::point
bool sortby_x(const cv::Point2i& a, const cv::Point2i& b)
{
    return (a.x < b.x);
}
bool sortby_y(const cv::Point2i& a, const cv::Point2i& b)
{
    return (a.y < b.y);
}

void algorithms::compute_bounding_boxes(const std::vector<std::vector<cv::Point2i>> &components,
                                        std::vector<cv::Rect2i> &bounding_boxes)
{


    for (auto component : components) {

        auto min_col = std::min_element(component.begin(),component.end(),sortby_x);
        auto max_col = std::max_element(component.begin(),component.end(),sortby_x);
        auto min_row = std::min_element(component.begin(),component.end(),sortby_y);
        auto max_row = std::max_element(component.begin(),component.end(),sortby_y);


        bounding_boxes.push_back(cv::Rect2i((*min_col).x,(*min_row).y,((*max_col).x-(*min_col).x)+1,((*max_row).y-(*min_row).y+1)));
    }
}

//===============================================================================
// discard_non_text()
//-------------------------------------------------------------------------------
// TODO: Discard components that are likely no text
//       - Loop over the components and its bounding boxes
//       - Discard non-text components
//       - Keep the criteria (ratios and thresholds) of the task description in mind
// hint: - use the the mathematical functions provided by the standard library
//       - already assigned label numbers should not be changed, e.g. third connected component
//         will keep label number 3, even though the first and second may have been discarded
//
// parameters:
//  -  swt_image: [CV_32FC1] matrix with the stroke widths
//  -  bounding_boxes: vector with rectangles of the bounding boxes
//  -  components: vector of vectors of points (x = col, y = row)
//  -  labels: [CV_16UC1] matrix with component labels for each position
//  -  text_bounding_boxes: subset of "bounding_boxes" of recognized text components
//  -  text_components: subset of "components" of recognized text
//  -  text_labels: [CV_16UC1] output matrix with a subset of "labels" of recognized text components
// return: void
//===============================================================================
void algorithms::discard_non_text(const cv::Mat &swt_image, const std::vector<cv::Rect2i> &bounding_boxes,
                                  const std::vector<std::vector<cv::Point2i>> &components, const cv::Mat &labels,
                                  const float variance_ratio, const float aspect_ratio_threshold,
                                  const float diameter_ratio_threshold, const int min_height, const int max_height,
                                  std::vector<cv::Rect2i> &text_bounding_boxes,
                                  std::vector<std::vector<cv::Point2i>> &text_components, cv::Mat &text_labels)
{
    //iterate over boxes and components together
    auto box = bounding_boxes.begin();

    for (auto component= components.begin() ; component != components.end();++component,++box){

        /////////////////////aspect ratio threshold calculation
        float aspect;
        if (box->height > 0){
            aspect = (float)box->width / (float)box->height;
        }

        bool aspect_correct = ((aspect <= aspect_ratio_threshold) && (aspect >= (float)(1/ aspect_ratio_threshold)));
        /////////////check height
        bool height_correct = ((box->height <= max_height) && (box->height >= min_height));

        //////////////calculate median of stroke widths for later
        //copy over stroke widths
        std::vector<float> temporary_ray_width;
        for (auto point : (*component)) {
            temporary_ray_width.push_back(swt_image.at<float>(point));
        }

        float median;
        std::sort(temporary_ray_width.begin(),temporary_ray_width.end());
        unsigned long size_of_ray = temporary_ray_width.size();
        unsigned long size_half = floor(size_of_ray /2 );

        if (size_of_ray % 2 == 0){ // even
            median = (temporary_ray_width[size_half]+temporary_ray_width[size_half-1])/2;
        }else{ //odd
            median = temporary_ray_width[size_half];
        }

        ////////////////////////////diameter ratio threshold:
        bool diameter_ratio_correct;
        if (median > 0) {
             diameter_ratio_correct = ((float) (std::sqrt(std::pow(box->height, 2) + std::pow(box->width, 2)) / median) <= diameter_ratio_threshold);
        }
        //////////////////////////variance ratio threshold:
        //mean calculation
        double sum = 0;
        for(auto val : temporary_ray_width){
            sum += val;
        }

        double mean;
        if ( temporary_ray_width.size() > 0) {
            mean = sum / temporary_ray_width.size();
        }
        //variance calc

        double variance = 0;
        for(auto val : temporary_ray_width){
            variance += std::pow(val - mean, 2);
        }
        if ( temporary_ray_width.size() > 0) {
            variance = variance / temporary_ray_width.size();
        }

        bool variance_ratio_correct = (variance <= (variance_ratio * median));

        /////////////check if letter is valid & store into letters
        if  ((diameter_ratio_correct && variance_ratio_correct && height_correct && aspect_correct)){
            text_components.push_back(*component);
            text_bounding_boxes.push_back(*box);

            for (auto pt : (*component)){
                text_labels.at<unsigned short>(pt) = labels.at<unsigned short>(pt);
            }

        }

    }

}

//================================================================================
// BONUS
//================================================================================

//================================================================================
// nonMaximaSuppression()
//--------------------------------------------------------------------------------
// TODO:
//  - Depending on the gradient direction of the pixel classify each pixel P in one
//    of the following classes:
//    ____________________________________________________________________________
//    | class |direction                | corresponding pixels Q, R               |
//    |-------|-------------------------|-----------------------------------------|
//    | I     | β <= 22.5 or β > 157.5  | Q: same row (y), left column (x−1)      |
//    |       |                         | R: same row (y), right column (x+1)     |
//    |-------|-------------------------|-----------------------------------------|
//    | II    | 22.5 < β <= 67.5        | Q: row below (y+1), left column (x−1)   |
//    |       |                         | R: row above (y-1), le bft column (x−1) |
//    |-------|-------------------------|-----------------------------------------|
//    | III   | 67.5 < β <= 112.5       | Q: row above (y-1), same column (x)     |
//    |       |                         | R: row below (y+1), same column (x)     |
//    |-------|-------------------------|-----------------------------------------|
//    | IV    | 112.5 < β <= 157.5      | Q: row below (y+1), left column (x−1)   |
//    |       |                         | R: row above (y-1), left column (x−1)   |
//    |_______|_________________________|_________________________________________|
//  - Compare the value of P with the values of Q and R:
//    If Q or R are greater than P -> set P to 0
//
// parameters:
//  - gradient_image: [CV_32FC1] matrix with the gradient image
//  - gradient_x: [CV_32FC1] matrix with the gradient in x direction
//  - gradient_y: [CV_32FC1] matrix with the gradient in y direction
//  - non_max_sup: [CV_32FC1] output matrix for the non maxima suppression
// return: void
//================================================================================
void algorithms::non_maxima_suppression(const cv::Mat &gradient_image, const cv::Mat &gradient_x,
                                        const cv::Mat &gradient_y, cv::Mat &non_max_sup)
{



}

//================================================================================
// hysteresis()
//--------------------------------------------------------------------------------
// TODO:
//  - Set all pixels under the lower threshold to 0
//  - Set all pixels over the high threshold to 255
//  - Classify all weak edges (thres_min <= weak edge < thres_max)
//    - If one of the the 8 surrounding pixel values is higher than thres_max,
//      also the weak pixel is a strong pixel
//    - Check this recursively to be sure not to miss one
//  - Set all remaining, not classified pixels to 0
//
// parameters:
//  - non_max_sup: [CV_8UC1] matrix containing the result of the non-maxima suppression
//  - threshold_min: the lower threshold
//  - threshold_min: the upper threshold
//  - output_image: [CV_8UC1] output matrix holding the results of the hysteresis calculation
// return: void
//================================================================================
void algorithms::hysteresis(const cv::Mat &non_max_sup, const uint8_t threshold_min, const uint8_t threshold_max,
                            cv::Mat &output_image)
{
}
//================================================================================
// cannyOwn()
//--------------------------------------------------------------------------------
// TODO:
//  - Calculate the 1st Sobel derivative once in x and once in y direction
//    and combine these two.
//  - Use these results for the non-maxima suppression
//  - Apply the hysteresis
//
// parameters:
//  - grayscale_image: [CV_8UC1] matrix containing the grayscaled image
//  - threshold_min: the lower threshold
//  - threshold_min: the upper threshold
//  - output_image: [CV_8UC1] output matrix holding canny edges
// return: void
//================================================================================
void algorithms::canny_own(const cv::Mat &grayscale_image, const uint8_t threshold_min, const uint8_t threshold_max,
                           cv::Mat &output_image)
{


}
