#include "yolo10_detect.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

Yolo10Detect::Yolo10Detect(MX::Runtime::MxAccl* accl,
                           const YoloUserConfig& user_cfg,
                           const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Load layer configuration from embedded YAML
    yolo_post_layers_ = MX::Prepost::Util::loadYoloLayerConfig(task, cfg_.model_w, cfg_.model_h);
}

cv::Mat Yolo10Detect::preprocess(const cv::Mat& image) {
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

void Yolo10Detect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void Yolo10Detect::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void Yolo10Detect::postprocess(const std::vector<float*>& outputs,
                               Result& result,
                               const cv::Mat& original_image) {
    if (original_image.empty()) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.cols, original_image.rows);
}

void Yolo10Detect::postprocess(const std::vector<float*>& outputs,
                               Result& result,
                               int ori_w,
                               int ori_h) {
    if (ori_w <= 0 || ori_h <= 0) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_w, ori_h);
}

void Yolo10Detect::postprocess_impl(const std::vector<float*>& outputs,
                                    Result& result,
                                    int ori_w,
                                    int ori_h) {

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
            // NOTE: use inv_conf_thres here because score is still raw (not applied sigmoid yet).
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

    // early exit
    int n_bboxes = all_boxes.size();
    if (n_bboxes == 0)
        return;

    // Ensure boxes are sorted based on confidence.
    std::sort(all_boxes.begin(), all_boxes.end(), [](const BBox& a, const BBox& b) {
        return a.conf > b.conf;
    });

    result.boxes.reserve(n_bboxes);
    for (BBox bbox : all_boxes) {
        result.boxes.push_back(bbox);
    }
}
