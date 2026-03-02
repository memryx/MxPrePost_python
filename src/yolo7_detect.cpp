
#include "yolo7_detect.h"

#include "config_finalizer.h"
#include "memx/accl/MxAcclBase.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

Yolo7Detect::Yolo7Detect(MX::Runtime::MxAcclBase* accl,
                         const YoloUserConfig& user_cfg,
                         const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Load layer configuration from embedded YAML
    yolo_post_layers_ = MX::Prepost::Util::loadYoloLayerConfig(task, cfg_.model_h, cfg_.model_w);
}

cv::Mat Yolo7Detect::preprocess(const cv::Mat& image) {
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

void Yolo7Detect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void Yolo7Detect::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void Yolo7Detect::postprocess(const std::vector<float*>& outputs,
                              Result& result,
                              const cv::Mat& original_image) {
    if (original_image.empty()) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void Yolo7Detect::postprocess(const std::vector<float*>& outputs,
                              Result& result,
                              int ori_h,
                              int ori_w) {
    if (ori_w <= 0 || ori_h <= 0) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void Yolo7Detect::postprocess_impl(const std::vector<float*>& outputs,
                                   Result& result,
                                   int ori_h,
                                   int ori_w) {

    result.boxes.clear();
    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    // Candidate Gathering
    std::vector<BBox> all_boxes;
    all_boxes.reserve(total_preds_);

    const int num_classes_total = static_cast<int>(cfg_.class_labels.size());

    const int kNumAnchors = 3;
    const int per_anchor = 5 + num_classes_total;  // [tx,ty,tw,th,obj] + classes
    const int per_cell = kNumAnchors * per_anchor;

    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* out_base = outputs.at(layer.port_out);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            float* cell = out_base + i * per_cell;

            int best_label = -1;
            int best_anchor = -1;
            float best_score = 0.0f;

            for (int a = 0; a < 3; ++a) {
                float* p = cell + a * per_anchor;

                const float obj_logit = p[4];

                float cls_logit;
                int label = MX::Prepost::Util::get_best_label(
                        cls_logit, p + 5, cfg_.valid_classes, -1e9f);

                if (label == -1)
                    continue;

                const float obj_prob = smgr_->convert(obj_logit);
                const float cls_prob = smgr_->convert(cls_logit);
                const float score = obj_prob * cls_prob;

                if (best_label == -1 || score > best_score) {
                    best_label = label;
                    best_anchor = a;
                    best_score = score;
                }
            }

            if (best_label == -1)
                continue;

            if (best_score < cfg_.conf)
                continue;

            // Decode bbox for chosen anchor
            float* p = cell + best_anchor * per_anchor;

            const float tx = p[0];
            const float ty = p[1];
            const float tw = p[2];
            const float th = p[3];

            const float sx = smgr_->convert(tx);
            const float sy = smgr_->convert(ty);
            const float sw = smgr_->convert(tw);
            const float sh = smgr_->convert(th);

            const float stride = static_cast<float>(layer.stride);
            const float row = static_cast<float>(i / layer.width);
            const float col = static_cast<float>(i % layer.width);

            static const float anchors_w[3][3] = {
                    {12.f, 19.f, 40.f},     // stride 8
                    {36.f, 76.f, 72.f},     // stride 16
                    {142.f, 192.f, 459.f},  // stride 32
            };
            static const float anchors_h[3][3] = {
                    {16.f, 36.f, 28.f},
                    {75.f, 55.f, 146.f},
                    {110.f, 243.f, 401.f},
            };

            const float cx = (sx * 2.0f - 0.5f + col) * stride;
            const float cy = (sy * 2.0f - 0.5f + row) * stride;

            float bw = (sw * 2.0f);
            float bh = (sh * 2.0f);
            bw = bw * bw * anchors_w[layer_id][best_anchor];
            bh = bh * bh * anchors_h[layer_id][best_anchor];

            std::array<float, 4> coord = {
                    cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f};

            // Undo letterbox using per-call padding/ratio (IMPORTANT: left for x, top for y)
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

    if (keep_indices.empty())
        return;

    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }
}