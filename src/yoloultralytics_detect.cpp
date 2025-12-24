#include "yoloultralytics_detect.h"

#include "utils.h"

using namespace MX::Pipe;

YoloUltralyticsDetect::YoloUltralyticsDetect(const YoloConfig& config) {
    class_labels_ = COCO_NAMES;
    class_count_ = COCO_CLASS_NUMBER;
    int color_size = COCO_TEXT_COLORS.size();

    // init settings from config
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
    letterbox_ratio_ = (float)MX::Pipe::model_w / ori_w_;
    letterbox_w_ = ori_w_ * letterbox_ratio_;
    letterbox_h_ = ori_h_ * letterbox_ratio_;

    pad_w_ = (MX::Pipe::model_w - letterbox_w_) / 2;
    pad_h_ = (MX::Pipe::model_h - letterbox_h_) / 2;

    // init score manager
    smgr_ = new MX::Pipe::Util::ScoreManager(config.conf, config.fast_sigmoid);

    yolo_post_layers_[0] = {
            .coord_port = 0,
            .conf_port = 1,
            .width = MX::Pipe::model_w / 8,   // L0_HW, 640 / 8 = 80
            .height = MX::Pipe::model_h / 8,  // L0_HW, 640 / 8 = 80
            .stride = 8,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[1] = {
            .coord_port = 2,
            .conf_port = 3,
            .width = MX::Pipe::model_w / 16,   // L1_HW, 640 / 16 = 40
            .height = MX::Pipe::model_h / 16,  // L1_HW, 640 / 16 = 40
            .stride = 16,
            .coord_fmap_size = 64,
    };

    yolo_post_layers_[2] = {
            .coord_port = 4,
            .conf_port = 5,
            .width = MX::Pipe::model_w / 32,   // L2_HW, 640 / 32 = 20
            .height = MX::Pipe::model_h / 32,  // L2_HW, 640 / 32 = 20
            .stride = 32,
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

void YoloUltralyticsDetect::postprocess(const std::vector<float*>& outputs, Result& result) {

    // Candidate Gathering
    std::vector<BBox> all_boxes;
    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // get best label and score
            float best_score;
            int best_label = MX::Pipe::Util::get_best_label(best_score,
                                                            conf_base + i * class_count_,
                                                            valid_classes_,
                                                            smgr_->thres_before_sigmoid);

            // no label with sufficient score
            if (best_label == -1)
                continue;

            // convert best score in [0,1] (e.g. apply sigmoid)
            best_score = smgr_->convert(best_score);

            // decode bbox (Distribution Focal Loss)
            std::array<float, 4> coord =
                    MX::Pipe::Util::dfl(coord_base + i * layer.coord_fmap_size,
                                        pad_w_,
                                        pad_h_,
                                        i / layer.width /* row */,
                                        i % layer.width /* col */,
                                        layer.stride,
                                        letterbox_ratio_);
            // create bbox
            BBox bbox(coord[0],
                      coord[1],
                      coord[2],
                      coord[3],
                      best_score,
                      best_label,
                      COCO_NAMES[best_label]);

            all_boxes.push_back(bbox);
        }
    }

    // apply NMS
    std::vector<int> keep_indices = MX::Pipe::Util::nms(all_boxes, iou_thres_);

    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }
}
