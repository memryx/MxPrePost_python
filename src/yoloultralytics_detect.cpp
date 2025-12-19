#include "yoloultralytics_detect.h"

#define FONT (cv::FONT_ITALIC)
#define COCO_CLASS_NUMBER (80)

static const std::vector<cv::Scalar> COCO_TEXT_COLORS = {
        {0, 0, 0},
        {255, 255, 255},
        {255, 255, 255},
        {255, 255, 255},
        {255, 215, 0},
};

static const std::vector<cv::Scalar> COCO_BOX_COLORS = {
        {255, 255, 0, 0.6},
        {26, 35, 126, 0.6},
        {255, 50, 50, 0.6},
        {0, 0, 0, 0.6},
        {51, 51, 51, 0.6},
};

/**
 * @brief Labels of COCO dataset, COCO 2014 and 2017 uses the same images but
 * different train/val/test splits. Also, COCO defines 91 classes but the data
 * only uses 80 classes.
 */
static const char* COCO_NAMES[COCO_CLASS_NUMBER] = {
        "person",        "bicycle",       "car",           "motorbike",
        "aeroplane",     "bus",           "train",         "truck",
        "boat",          "traffic light", "fire hydrant",  "stop sign",
        "parking meter", "bench",         "bird",          "cat",
        "dog",           "horse",         "sheep",         "cow",
        "elephant",      "bear",          "zebra",         "giraffe",
        "backpack",      "umbrella",      "handbag",       "tie",
        "suitcase",      "frisbee",       "skis",          "snowboard",
        "sports ball",   "kite",          "baseball bat",  "baseball glove",
        "skateboard",    "surfboard",     "tennis racket", "bottle",
        "wine glass",    "cup",           "fork",          "knife",
        "spoon",         "bowl",          "banana",        "apple",
        "sandwich",      "orange",        "broccoli",      "carrot",
        "hot dog",       "pizza",         "donut",         "cake",
        "chair",         "sofa",          "pottedplant",   "bed",
        "diningtable",   "toilet",        "tvmonitor",     "laptop",
        "mouse",         "remote",        "keyboard",      "cell phone",
        "microwave",     "oven",          "toaster",       "sink",
        "refrigerator",  "book",          "clock",         "vase",
        "scissors",      "teddy bear",    "hair drier",    "toothbrush",
};

float YoloUltralyticsDetect::_calc_iou(const BBox& bbox_0, const BBox& bbox_1) {
    float y_min = mxutil_max(bbox_0.y_min, bbox_1.y_min);
    float x_min = mxutil_max(bbox_0.x_min, bbox_1.x_min);
    float y_max = mxutil_min(bbox_0.y_max, bbox_1.y_max);
    float x_max = mxutil_min(bbox_0.x_max, bbox_1.x_max);
    float intersection_area = mxutil_max(0, (y_max - y_min)) * mxutil_max(0, (x_max - x_min));
    float bbox_0_area = (bbox_0.y_max - bbox_0.y_min) * (bbox_0.x_max - bbox_0.x_min);
    float bbox_1_area = (bbox_1.y_max - bbox_1.y_min) * (bbox_1.x_max - bbox_1.x_min);
    float union_area = bbox_0_area + bbox_1_area - intersection_area;
    return intersection_area / union_area;
}

void YoloUltralyticsDetect::_nms(std::list<BBox>& boxes, const BBox& candidate, float iou_thresh) {
    bool candidate_survives = true;

    for (auto it = boxes.begin(); it != boxes.end();) {
        float iou = _calc_iou(*it, candidate);

        if (iou > iou_thresh) {
            if (it->conf >= candidate.conf) {
                // Existing box suppresses candidate → stop early
                candidate_survives = false;
                break;
            } else {
                // Candidate suppresses existing box
                it = boxes.erase(it);  // safe: returns next iterator
                continue;
            }
        }

        ++it;
    }

    // Add candidate if it wasn't suppressed
    if (candidate_survives) {
        boxes.push_back(candidate);
    }
}

void YoloUltralyticsDetect::_draw_bbox(cv::Mat& image, const BBox& bbox) {

    int x_min = (int)bbox.x_min;
    int y_min = (int)bbox.y_min;
    int x_max = (int)bbox.x_max;
    int y_max = (int)bbox.y_max;
    int cls_id = bbox.cls_id;
    float conf = bbox.conf;
    cv::Scalar box_color = bounding_box_colors_[cls_id];
    cv::Scalar text_color = class_label_colors_[cls_id];

    double font_scale = ((double)image.rows / 640.0);
    double bbox_thickness = font_scale * 3;
    double font_thickness = font_scale * 2;
    cv::Size text_size;
    int baseline;
    char text[64];

    /* bounding box rectangle line */
    cv::rectangle(image,
                  cv::Point(x_min, y_min) /*top left*/,
                  cv::Point(x_max, y_max) /*bottom right*/,
                  box_color,
                  bbox_thickness,
                  cv::LINE_4);

    sprintf(text, "%s(%.f%%)", bbox.cls_name.c_str(), 100 * conf);

    text_size = cv::getTextSize(text, FONT, 2 * font_scale, bbox_thickness, &baseline);

    /* label background rectangle */
    cv::rectangle(image,
                  cv::Rect(x_min,
                           mxutil_max(0, y_min - text_size.height),
                           text_size.width * 0.5,
                           text_size.height),  // top left, width, height
                  box_color,
                  cv::FILLED);

    /* label text */
    cv::putText(image,
                text,
                cv::Point(x_min,
                          mxutil_max(0, y_min - text_size.height) == 0
                                  ? text_size.height - 5
                                  : y_min - 10 * font_scale),  // bottom left
                FONT,
                font_scale,
                text_color,
                font_thickness,
                cv::LINE_AA);
}

YoloUltralyticsDetect::YoloUltralyticsDetect(const YoloDetectConfig& config) {
    model_w_ = 640;
    model_h_ = 640;
    model_ch_ = 3;
    class_labels_ = COCO_NAMES;
    class_count_ = COCO_CLASS_NUMBER;
    int color_size = COCO_TEXT_COLORS.size();

    // init settings from config
    conf_thres_ = config.conf_thres;
    iou_thres_ = config.iou_thres;

    // compute padding
    // TODO: support vertical images as well
    if (!_is_horizontal_input(config.ori_width, config.ori_height))
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

    // set label and bbox color
    for (size_t i = 0; i < class_count_; i++) {
        cv::Scalar label_color = COCO_TEXT_COLORS[i % color_size];
        cv::Scalar bbox_color = COCO_BOX_COLORS[i % color_size];
        class_label_colors_.push_back(label_color);
        bounding_box_colors_.push_back(bbox_color);
    }

    yolo_post_layers_[0] = {
            .coord_ofmap_flow_id = 0,
            .conf_ofmap_flow_id = 1,
            .width = model_w_ / 8,   // L0_HW, 640 / 8 = 80
            .height = model_h_ / 8,  // L0_HW, 640 / 8 = 80
            .ratio = 8,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[1] = {
            .coord_ofmap_flow_id = 2,
            .conf_ofmap_flow_id = 3,
            .width = model_w_ / 16,   // L1_HW, 640 / 16 = 40
            .height = model_h_ / 16,  // L1_HW, 640 / 16 = 40
            .ratio = 16,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[2] = {
            .coord_ofmap_flow_id = 4,
            .conf_ofmap_flow_id = 5,
            .width = model_w_ / 32,   // L2_HW, 640 / 32 = 20
            .height = model_h_ / 32,  // L2_HW, 640 / 32 = 20
            .ratio = 32,
            .coord_fmap_size = 64,
    };
}

bool YoloUltralyticsDetect::_is_horizontal_input(int ori_w, int ori_h) {
    if (ori_h > ori_w) {
        printf("Invalid display image: only horizontal images are supported.\n");
        return false;
    }
    return true;
}

cv::Mat YoloUltralyticsDetect::preprocess(const cv::Mat& image) {

    // Resize keeping aspect ratio
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(letterbox_w_, letterbox_h_), 0, 0, cv::INTER_LINEAR);

    // Apply letterbox pad (black border)
    cv::Mat padded;
    cv::copyMakeBorder(resized,
                       padded,
                       pad_h_,  // top,
                       pad_h_,  // bottom
                       pad_w_,  // left
                       pad_w_,  // right
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

    // Convert to float and normalize (0–1)
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    return padded;  // shape: (640, 640, 3), range [0,1]
}

void YoloUltralyticsDetect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        _draw_bbox(image, bbox);
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

void YoloUltralyticsDetect::_get_detection(std::list<BBox>& boxes,
                                           int layer_id,
                                           float* conf_cell_buf,
                                           float* coord_cell_buf,
                                           int row,
                                           int col) {
    // process conf score
    float best_label_score = conf_cell_buf[0] - 1.f;  // arbitrary small number
    int best_label = -1;

    for (size_t label = 0; label < class_count_; label++) {

        if (conf_cell_buf[label] < conf_thres_fastSigmoid_)
            continue;

        if (conf_cell_buf[label] > best_label_score) {
            best_label_score = conf_cell_buf[label];
            best_label = label;
        }
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
    float min_x = mxutil_max(center_x - 0.5 * w, .0);
    float min_y = mxutil_max(center_y - 0.5 * h, .0);
    float max_x = mxutil_min(center_x + 0.5 * w, model_w_);
    float max_y = mxutil_min(center_y + 0.5 * h, model_h_);

    // convert to raw bbox coords
    min_x = static_cast<int>((min_x - pad_w_) / letterbox_ratio_);
    min_y = static_cast<int>((min_y - pad_h_) / letterbox_ratio_);
    max_x = static_cast<int>((max_x - pad_w_) / letterbox_ratio_);
    max_y = static_cast<int>((max_y - pad_h_) / letterbox_ratio_);

    BBox bbox(min_x, min_y, max_x, max_y, best_label_score, best_label, COCO_NAMES[best_label]);

    // apply NMS
    _nms(boxes, bbox, iou_thres_);
}

void YoloUltralyticsDetect::postprocess(const std::vector<float*>& outputs, Result& result) {

    if (outputs.empty()) {
        throw std::invalid_argument("outputs cannot be null.");
    }

    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {

        // get layer params
        const auto& layer = yolo_post_layers_[layer_id];
        const int conf_per_row = (layer.width * class_count_);            // 80 x 80
        const int coord_per_row = (layer.width * layer.coord_fmap_size);  // 80 x 64

        const int conf_id = layer.conf_ofmap_flow_id;
        const int coord_id = layer.coord_ofmap_flow_id;

        float* conf_base = outputs.at(conf_id);
        float* coord_base = outputs.at(coord_id);

        if (!conf_base || !coord_base) {
            throw std::invalid_argument("One or more output buffers are null.");
        }

        // iterate each cell
        for (size_t row = 0; row < layer.height; row++) {
            float* conf_row_buf = conf_base + row * conf_per_row;
            float* coord_row_buf = coord_base + row * coord_per_row;

            for (size_t col = 0; col < layer.width; col++) {
                float* conf_cell_buf = conf_row_buf + col * class_count_;
                float* coord_cell_buf = coord_row_buf + col * layer.coord_fmap_size;
                _get_detection(result.boxes, layer_id, conf_cell_buf, coord_cell_buf, row, col);
            }
        }
    }
}
