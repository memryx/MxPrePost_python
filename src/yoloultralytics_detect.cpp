#include "yoloultralytics_detect.h"

#include "utils.h"

using namespace MX::Pipe;

#define mxutil_prepost_sigmoid(_x_)                                                               \
    (1.0 / (1.0 + expf(-1.0 * (_x_))))  // sigmoid: f(x) = 1 / (1 + e^(-x))
#define mxutil_prepost_sigmoid_fast_sigmoid(_x_)                                                  \
    ((_x_) /                                                                                      \
     (((_x_) < 0) ? (1.0 - (_x_)) : (1.0 + (_x_))))  // fast-sigmoid: f(x) = x / (1 + abs(x))

YoloUltralyticsDetect::YoloUltralyticsDetect(const YoloConfig& config) {
    class_labels_ = COCO_NAMES;
    class_count_ = COCO_CLASS_NUMBER;
    int color_size = COCO_TEXT_COLORS.size();

    // init settings from config
    conf_thres_ = config.conf;
    iou_thres_ = config.iou;

    if (config.valid_classes.empty()) {
        // use all classes
        for (int i = 0; i < class_count_; ++i) {
            valid_classes_.push_back(i);
        }
    } else {
        // use specified classes
        for (int cls : config.valid_classes) {
            valid_classes_.push_back(cls);
        }
    }

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
            .width = model_w_ / 8,   // L0_HW, 640 / 8 = 80
            .height = model_h_ / 8,  // L0_HW, 640 / 8 = 80
            .ratio = 8,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[1] = {
            .coord_port = 2,
            .conf_port = 3,
            .width = model_w_ / 16,   // L1_HW, 640 / 16 = 40
            .height = model_h_ / 16,  // L1_HW, 640 / 16 = 40
            .ratio = 16,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[2] = {
            .coord_port = 4,
            .conf_port = 5,
            .width = model_w_ / 32,   // L2_HW, 640 / 32 = 20
            .height = model_h_ / 32,  // L2_HW, 640 / 32 = 20
            .ratio = 32,
            .coord_fmap_size = 64,
    };
}

cv::Mat YoloUltralyticsDetect::preprocess(const cv::Mat& image) {
    return MX::Pipe::Util::preprocess(image, letterbox_w_, letterbox_h_, pad_w_, pad_h_);
}

void YoloUltralyticsDetect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Pipe::Util::draw_bbox(image, bbox);
    }
}

float YoloUltralyticsDetect::_conf_to_fastSigmoid_inputVal(float conf) {
    // Converts a conf value in [0,1] to the corresponding input for the fast-sigmoid
    // function. The fast-sigmoid function: f(x) = x / (1 + |x|), which maps [-inf, +inf] -> [-1,
    // 1]. Steps:
    //   1. Map conf [0,1] -> x [-1,1]
    //   2. Return x as input for fast-sigmoid

    float x = conf * 2.0f - 1.0f;     // map [0,1] -> [-1,1]
    return x / (1.0f - std::abs(x));  // map [-1, 1] -> [-inf, inf]
}

void YoloUltralyticsDetect::_gather_candidate(std::vector<BBox>& boxes,
                                              int layer_id,
                                              float* conf_cell_buf,
                                              float* coord_cell_buf,
                                              int row,
                                              int col) {

    float best_score;
    int best_label = MX::Pipe::Util::get_best_label(
            best_score, conf_cell_buf, valid_classes_, conf_thres_fastSigmoid_);

    if (best_label == -1)
        return;

    // NOTE: Be aware of the range of fast_signoid: (-1, 1).
    best_score = mxutil_prepost_sigmoid_fast_sigmoid(best_score);

    // range (-1, 1) -> (0, 1), need to convert, because conf_thresh is based on range(0, 1)
    best_score = (best_score + 1.0f) * 0.5f;

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

    BBox bbox(min_x, min_y, max_x, max_y, best_score, best_label, COCO_NAMES[best_label]);
    boxes.push_back(bbox);
}

void YoloUltralyticsDetect::postprocess(const std::vector<float*>& outputs, Result& result) {

    // Candidate Gathering
    std::vector<BBox> all_boxes;
    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {
            _gather_candidate(all_boxes,
                              layer_id,
                              conf_base + i * class_count_,
                              coord_base + i * layer.coord_fmap_size,
                              i / layer.width /* row */,
                              i % layer.width /* col */);
        }
    }

    // apply NMS
    std::vector<int> keep_indices = MX::Pipe::Util::nms(all_boxes, iou_thres_);

    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }
}
