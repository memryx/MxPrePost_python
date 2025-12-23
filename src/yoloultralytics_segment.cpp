#include "yoloultralytics_segment.h"

#include "utils.h"

using namespace MX::Pipe;

#define mxutil_prepost_sigmoid(_x_)                                                               \
    (1.0 / (1.0 + expf(-1.0 * (_x_))))  // sigmoid: f(x) = 1 / (1 + e^(-x))
#define mxutil_prepost_sigmoid_fast_sigmoid(_x_)                                                  \
    ((_x_) /                                                                                      \
     (((_x_) < 0) ? (1.0 - (_x_)) : (1.0 + (_x_))))  // fast-sigmoid: f(x) = x / (1 + abs(x))

YoloUltralyticsSegment::YoloUltralyticsSegment(const YoloConfig& config) {
    class_labels_ = COCO_NAMES;
    class_count_ = COCO_CLASS_NUMBER;
    int color_size = COCO_TEXT_COLORS.size();

    // init settings from config
    conf_thres_ = config.conf;
    iou_thres_ = config.iou;
    valid_classes_ = config.valid_classes;

    // compute padding
    // TODO: support vertical images as well
    if (!MX::Pipe::Util::is_horizontal_input(config.ori_width, config.ori_height))
        return;

    ori_w_ = config.ori_width;
    ori_h_ = config.ori_height;

    // letterbox params
    letterbox_ratio_ = (float)model_w_ / ori_w_;
    letterbox_w_ = ori_w_ * letterbox_ratio_;
    letterbox_h_ = ori_h_ * letterbox_ratio_;

    pad_w_ = (model_w_ - letterbox_w_) / 2;
    pad_h_ = (model_h_ - letterbox_h_) / 2;

    // convert conf thres to fast-sigmoid input value
    conf_thres_fastSigmoid_ = _conf_to_fastSigmoid_inputVal(conf_thres_);

    yolo_post_layers_[0] = {
            .coord_port = 0,
            .conf_port = 1,
            .mask_coef_port = 3,
            .width = model_w_ / 8,   // L0_HW, 640 / 8 = 80
            .height = model_h_ / 8,  // L0_HW, 640 / 8 = 80
            .ratio = 8,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[1] = {
            .coord_port = 4,
            .conf_port = 5,
            .mask_coef_port = 6,
            .width = model_w_ / 16,   // L1_HW, 640 / 16 = 40
            .height = model_h_ / 16,  // L1_HW, 640 / 16 = 40
            .ratio = 16,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[2] = {
            .coord_port = 7,
            .conf_port = 8,
            .mask_coef_port = 9,
            .width = model_w_ / 32,   // L2_HW, 640 / 32 = 20
            .height = model_h_ / 32,  // L2_HW, 640 / 32 = 20
            .ratio = 32,
            .coord_fmap_size = 64,
    };
}

cv::Mat YoloUltralyticsSegment::preprocess(const cv::Mat& image) {
    return MX::Pipe::Util::preprocess(image, letterbox_w_, letterbox_h_, pad_w_, pad_h_);
}

void YoloUltralyticsSegment::draw(cv::Mat& image, const Result& result) {
    for (const Mask& mask : result.masks) {
        MX::Pipe::Util::draw_mask(image, mask);
    }
}

float YoloUltralyticsSegment::_conf_to_fastSigmoid_inputVal(float conf) {
    // Converts a conf value in [0,1] to the corresponding input for the fast-sigmoid
    // function. The fast-sigmoid function: f(x) = x / (1 + |x|), which maps [-inf, +inf] -> [-1,
    // 1]. Steps:
    //   1. Map conf [0,1] -> x [-1,1]
    //   2. Return x as input for fast-sigmoid

    float x = conf * 2.0f - 1.0f;     // map [0,1] -> [-1,1]
    return x / (1.0f - std::abs(x));  // map [-1, 1] -> [-inf, inf]
}

void YoloUltralyticsSegment::_get_detection(std::vector<BBox>& boxes,
                                            std::vector<float*>& all_mask_coefs,
                                            int layer_id,
                                            float* conf_cell_buf,
                                            float* coord_cell_buf,
                                            float* mask_row_buf,
                                            int row,
                                            int col) {
    // process conf score
    float best_label_score = conf_cell_buf[0] - 1.f;  // arbitrary small number
    int best_label = -1;

    // find best label
    auto try_update = [&](int label) {
        float score = conf_cell_buf[label];
        if (score < conf_thres_fastSigmoid_)
            return;
        if (score > best_label_score) {
            best_label_score = score;
            best_label = label;
        }
    };

    if (valid_classes_.empty()) {
        // loop through all classes
        for (int label = 0; label < class_count_; ++label)
            try_update(label);
    } else {
        // loop through valid classes only
        for (int label : valid_classes_)
            try_update(label);
    }

    // No score of detection over conf threshold
    if (best_label == -1)
        return;

    // NOTE: Be aware of the range of fast_signoid: (-1, 1).
    best_label_score = mxutil_prepost_sigmoid_fast_sigmoid(best_label_score);

    // range (-1, 1) -> (0, 1), need to convert, because conf_thresh is based on range(0, 1)
    best_label_score = (best_label_score + 1.0f) * 0.5f;

    std::vector<float> feature_value;

    for (int channel = 0; channel < 4; channel++) {  // split 64 into 4*16
        float value = 0.0;
        float* feature_buf = coord_cell_buf + channel * 16;
        float softmax_sum = 0.0;
        float local_max = feature_buf[0];

        // apply softmax and weighted sum
        for (int i = 1; i < 16; i++) {
            if (feature_buf[i] > local_max)
                local_max = feature_buf[i];
        }

// more SIMD hints
#pragma omp simd reduction(+ : softmax_sum)
        for (int i = 0; i < 16; i++) {
            softmax_sum += expf(feature_buf[i] - local_max);
        }

// more SIMD hints
#pragma omp simd reduction(+ : value)
        for (int i = 0; i < 16; i++) {
            value += ((float)i * (float)(expf(feature_buf[i] - local_max) / softmax_sum));
        }
        feature_value.push_back(value);
    }

    // decode bbox
    float center_x, center_y, w, h;
    center_x = (feature_value[2] - feature_value[0] + 2 * (0.5 + ((float)col))) * 0.5 *
               yolo_post_layers_[layer_id].ratio;
    center_y = (feature_value[3] - feature_value[1] + 2 * (0.5 + ((float)row))) * 0.5 *
               yolo_post_layers_[layer_id].ratio;
    w = (feature_value[2] + feature_value[0]) * yolo_post_layers_[layer_id].ratio;
    h = (feature_value[3] + feature_value[1]) * yolo_post_layers_[layer_id].ratio;

    // coord on padded image
    float min_x = std::max(center_x - 0.5f * w, 0.0f);
    float min_y = std::max(center_y - 0.5f * h, 0.0f);
    float max_x = std::min(center_x + 0.5f * w, (float)model_w_);
    float max_y = std::min(center_y + 0.5f * h, (float)model_h_);

    // convert to raw bbox coords
    min_x = static_cast<int>((min_x - pad_w_) / letterbox_ratio_);
    min_y = static_cast<int>((min_y - pad_h_) / letterbox_ratio_);
    max_x = static_cast<int>((max_x - pad_w_) / letterbox_ratio_);
    max_y = static_cast<int>((max_y - pad_h_) / letterbox_ratio_);

    BBox bbox(min_x, min_y, max_x, max_y, best_label_score, best_label, COCO_NAMES[best_label]);
    boxes.push_back(bbox);

    // add mask coef pointer
    const int mask_fmap_size = 32;
    all_mask_coefs.push_back(mask_row_buf + col * mask_fmap_size);
}

void YoloUltralyticsSegment::postprocess(const std::vector<float*>& outputs, Result& result) {

    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    const int mask_fmap_size = 32;
    std::vector<BBox> all_boxes;
    std::vector<float*> all_mask_coefs;
    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {

        // get layer params
        const auto& layer = yolo_post_layers_[layer_id];
        const int conf_per_row = (layer.width * class_count_);            // 80 x 80
        const int coord_per_row = (layer.width * layer.coord_fmap_size);  // 80 x 64
        const int mask_coef_per_row = (layer.width * mask_fmap_size);     // 80 x 32

        const int conf_id = layer.conf_port;
        const int coord_id = layer.coord_port;
        const int mask_coef_id = layer.mask_coef_port;

        float* conf_base = outputs.at(conf_id);
        float* coord_base = outputs.at(coord_id);
        float* mask_coef_base = outputs.at(mask_coef_id);

        if (!conf_base || !coord_base) {
            throw std::invalid_argument("One or more output buffers are null.");
        }

        // iterate each cell
        for (size_t row = 0; row < layer.height; row++) {
            float* conf_row_buf = conf_base + row * conf_per_row;
            float* coord_row_buf = coord_base + row * coord_per_row;
            float* mask_coef_row_buf = mask_coef_base + row * mask_coef_per_row;

            for (size_t col = 0; col < layer.width; col++) {
                float* conf_cell_buf = conf_row_buf + col * class_count_;
                float* coord_cell_buf = coord_row_buf + col * layer.coord_fmap_size;
                _get_detection(all_boxes,
                               all_mask_coefs,
                               layer_id,
                               conf_cell_buf,
                               coord_cell_buf,
                               mask_coef_row_buf,
                               row,
                               col);
            }
        }
    }

    // apply NMS
    std::vector<int> keep_indices = MX::Pipe::Util::nms(all_boxes, iou_thres_);

    // early stop if no boxes kept
    if (keep_indices.empty())
        return;

    int num_filtered = static_cast<int>(keep_indices.size());

    // gather final result
    result.boxes.reserve(keep_indices.size());

    std::vector<float*> filtered_mask_coefs;
    filtered_mask_coefs.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
        filtered_mask_coefs.push_back(all_mask_coefs[idx]);
    }

    // (160 * 160, 32)
    const int mask_w = 160;
    const int mask_h = 160;
    Eigen::Map<const RowMatrix> mask_proto(outputs[2], mask_h * mask_w, mask_fmap_size);

    /* --------------------------------------------------
     * 3. Convert filtered_mask_coefs → Eigen (32, N)
     * -------------------------------------------------- */
    using ColVector = Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>;
    RowMatrix mask_coef(mask_fmap_size, filtered_mask_coefs.size());  // (32, N)

    for (int i = 0; i < num_filtered; ++i) {
        // Map the raw float* as a column
        Eigen::Map<const ColVector> coef_col(filtered_mask_coefs[i], mask_fmap_size);
        mask_coef.col(i) = coef_col;
    }

    // (160*160, 32) @ (32, N) -> (160*160, N)
    RowMatrix masks_eigen = mask_proto * mask_coef;

    // 1. Wrap the raw data into a 3D-aware shape (H, W, Channels)
    // Note: num_filtered must match the number of masks in masks_eigen
    cv::Mat mask_stack(160, 160, CV_32FC(num_filtered), (void*)masks_eigen.data());

    // 2. Resize the entire stack at once
    // OpenCV supports multi-channel resize up to 512 channels
    cv::Mat resized_stack;
    cv::resize(mask_stack, resized_stack, cv::Size(model_w_, model_h_), 0, 0, cv::INTER_LINEAR);

    // 3. Define the ROI (Region of Interest) to remove letterbox padding
    cv::Rect roi(pad_w_, pad_h_, letterbox_w_, letterbox_h_);

    // 4. Crop the stack and split into individual mask matrices
    cv::Mat cropped_stack = resized_stack(roi);

    cv::Mat final_mask;
    cv::resize(cropped_stack, final_mask, cv::Size(ori_w_, ori_h_), 0, 0, cv::INTER_LINEAR);

    std::vector<cv::Mat> mask_vec;
    cv::split(final_mask, mask_vec);

    // Assign to result.masks
    for (int i = 0; i < mask_vec.size(); ++i) {
        const cv::Mat& mask = mask_vec[i];
        const BBox& box = result.boxes[i];
        int cls_id = box.cls_id;

        // 1. Threshold the float mask to binary (0 or 255)
        cv::Mat binary_mask;
        cv::threshold(mask, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);

        // 2. Create a Rect from your BBox
        // Ensure coordinates are within image bounds to prevent crashes
        int x = std::max(0, (int)box.x_min);
        int y = std::max(0, (int)box.y_min);
        int w = std::min(binary_mask.cols - x, (int)(box.x_max - box.x_min));
        int h = std::min(binary_mask.rows - y, (int)(box.y_max - box.y_min));
        cv::Rect roi_rect(x, y, w, h);

        // 3. Create a global mask and only keep the ROI area
        cv::Mat roi_only_mask = cv::Mat::zeros(binary_mask.size(), binary_mask.type());
        binary_mask(roi_rect).copyTo(roi_only_mask(roi_rect));

        // 4. Find contours on the clipped mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(roi_only_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 3. Convert cv::Point to your custom Point struct
        for (const auto& contour : contours) {

            Mask mask_struct;
            mask_struct.cls_id = cls_id;  // Assign class ID

            // assign points
            for (const auto& p : contour) {
                mask_struct.points.push_back({p.x, p.y});
            }

            // Add to your results
            if (!mask_struct.points.empty()) {
                result.masks.push_back(mask_struct);
            }
        }
    }
}
