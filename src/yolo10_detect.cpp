#include "yolo10_detect.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

Yolo10Detect::Yolo10Detect(MX::Runtime::MxAccl* accl, const YoloUserConfig& user_cfg) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // init post-process layer params
    yolo_post_layers_[0] = {
            .coord_port = 0,
            .conf_port = 1,
            .width = cfg_.model_w / 8,   // L0_HW, 640 / 8 = 80
            .height = cfg_.model_h / 8,  // L0_HW, 640 / 8 = 80
            .stride = 8,
    };

    yolo_post_layers_[1] = {
            .coord_port = 2,
            .conf_port = 3,
            .width = cfg_.model_w / 16,   // L1_HW, 640 / 16 = 40
            .height = cfg_.model_h / 16,  // L1_HW, 640 / 16 = 40
            .stride = 16,
    };

    yolo_post_layers_[2] = {
            .coord_port = 4,
            .conf_port = 5,
            .width = cfg_.model_w / 32,   // L2_HW, 640 / 32 = 20
            .height = cfg_.model_h / 32,  // L2_HW, 640 / 32 = 20
            .stride = 32,
    };

    std::vector<MX::Prepost::Util::Grid> grids = {
            {yolo_post_layers_[0].width, yolo_post_layers_[0].height},
            {yolo_post_layers_[1].width, yolo_post_layers_[1].height},
            {yolo_post_layers_[2].width, yolo_post_layers_[2].height},
    };
}

cv::Mat Yolo10Detect::preprocess(const cv::Mat& image) {
    return MX::Prepost::Util::preprocess(
            image, cfg_.letterbox_w, cfg_.letterbox_h, cfg_.pad_w, cfg_.pad_h);
}

void Yolo10Detect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void Yolo10Detect::postprocess(const std::vector<float*>& outputs, Result& result) {

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
                                                      conf_base + i * cfg_.valid_classes.size(),
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
            coord[0] = (coord[0] - cfg_.pad_w) / cfg_.letterbox_ratio;
            coord[1] = (coord[1] - cfg_.pad_h) / cfg_.letterbox_ratio;
            coord[2] = (coord[2] - cfg_.pad_w) / cfg_.letterbox_ratio;
            coord[3] = (coord[3] - cfg_.pad_h) / cfg_.letterbox_ratio;

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
