#include "yoloultralytics_detect.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

YoloUltralyticsDetect::YoloUltralyticsDetect(MX::Runtime::MxAcclBase* accl,
                                             const YoloUserConfig& user_cfg,
                                             const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Load layer configuration from embedded YAML
    yolo_post_layers_ = MX::Prepost::Util::loadYoloLayerConfig(task, cfg_.model_h, cfg_.model_w);
}

cv::Mat YoloUltralyticsDetect::preprocess(const cv::Mat& image) {
    const int ori_w = image.cols;
    const int ori_h = image.rows;

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    return MX::Prepost::Util::preprocess(image,
                                         lb.letterbox_w,
                                         lb.letterbox_h,
                                         lb.pad_left,
                                         lb.pad_top,
                                         lb.pad_right,
                                         lb.pad_bottom);
}

void YoloUltralyticsDetect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void YoloUltralyticsDetect::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void YoloUltralyticsDetect::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        const cv::Mat& original_image) {
    if (original_image.empty()) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void YoloUltralyticsDetect::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        int ori_h,
                                        int ori_w) {
    if (ori_w <= 0 || ori_h <= 0) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void YoloUltralyticsDetect::postprocess_impl(const std::vector<float*>& outputs,
                                             Result& result,
                                             int ori_h,
                                             int ori_w) {

    result.boxes.clear();
    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    // Candidate Gathering
    std::vector<BBox> all_boxes;
    all_boxes.reserve(total_preds_);

    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // get best label and score
            // NOTE: use inv_conf_thres here because score is still raw (not applied sigmoidyet).
            // sigmoid is expensive and we only apply it if needed.

            float best_score;
            int best_label =
                    MX::Prepost::Util::get_best_label(best_score,
                                                      conf_base + i * cfg_.class_labels.size(),
                                                      cfg_.valid_classes,
                                                      smgr_->inv_conf_thres);

            // no label with sufficient score
            if (best_label == -1)
                continue;

            // convert best score in [0,1] (e.g. apply sigmoid)
            best_score = smgr_->convert(best_score);

            // decode bbox (Distribution Focal Loss)
            std::array<float, 4> coord = MX::Prepost::Util::dfl(coord_base + i * COORD_FMAP_SIZE,
                                                                i / layer.width /* row */,
                                                                i % layer.width /* col */,
                                                                layer.stride,
                                                                cfg_.model_w,
                                                                cfg_.model_h);

            // convert to raw bbox coords
            coord[0] = (coord[0] - lb.pad_left) / lb.ratio;
            coord[1] = (coord[1] - lb.pad_top) / lb.ratio;
            coord[2] = (coord[2] - lb.pad_left) / lb.ratio;
            coord[3] = (coord[3] - lb.pad_top) / lb.ratio;

            // Clip to image bounds
            coord[0] = std::max(0.0f, std::min(coord[0], (float)ori_w));
            coord[1] = std::max(0.0f, std::min(coord[1], (float)ori_h));
            coord[2] = std::max(0.0f, std::min(coord[2], (float)ori_w));
            coord[3] = std::max(0.0f, std::min(coord[3], (float)ori_h));

            // Drop invalid/degenerate boxes
            if (coord[2] <= coord[0] || coord[3] <= coord[1]) {
                continue;
            }

            // store bbox
            all_boxes.emplace_back(coord[0],
                                   coord[1],
                                   coord[2],
                                   coord[3],
                                   best_score,
                                   best_label,
                                   cfg_.class_labels[best_label]);
        }
    }

    // apply NMS
    std::vector<int> keep_indices =
            MX::Prepost::Util::nms(all_boxes, cfg_.iou, cfg_.class_agnostic);

    // early exit
    int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }
}
