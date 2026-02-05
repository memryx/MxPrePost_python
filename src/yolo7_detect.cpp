
#include "yolo7_detect.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

Yolo7Detect::Yolo7Detect(MX::Runtime::MxAccl* accl, const YoloUserConfig& user_cfg) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // init post-process layer params
    yolo_post_layers_[0] = {
            .out_port = 0,
            .width = cfg_.model_w / 8,   // L0_HW, 640 / 8 = 80
            .height = cfg_.model_h / 8,  // L0_HW, 640 / 8 = 80
            .stride = 8,
    };

    yolo_post_layers_[1] = {
            .out_port = 1,
            .width = cfg_.model_w / 16,   // L1_HW, 640 / 16 = 40
            .height = cfg_.model_h / 16,  // L1_HW, 640 / 16 = 40
            .stride = 16,
    };

    yolo_post_layers_[2] = {
            .out_port = 2,
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

cv::Mat Yolo7Detect::preprocess(const cv::Mat& image) {
    return MX::Prepost::Util::preprocess(
            image, cfg_.letterbox_w, cfg_.letterbox_h, cfg_.pad_w, cfg_.pad_h);
}

void Yolo7Detect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void Yolo7Detect::postprocess(const std::vector<float*>& outputs, Result& result) {
    // std::cout << "YOLO7 Post Process";
    // Candidate Gathering
    std::vector<BBox> all_boxes;
    all_boxes.reserve(total_preds_);

    const int num_classes_total = static_cast<int>(cfg_.class_labels.size());

    const int kNumAnchors = 3;
    const int per_anchor = 5 + num_classes_total;  // [tx,ty,tw,th,obj] + classes
    const int per_cell = kNumAnchors * per_anchor;

    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* out_base = outputs.at(layer.out_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // Cell base points to 255 floats: [a0(85) | a1(85) | a2(85)]
            float* cell = out_base + i * per_cell;

            // Pick the best anchor+class using RAW logits (no sigmoid unless needed)
            int best_label = -1;
            float best_cls_logit = 0.0f;
            float best_obj_logit = 0.0f;
            int best_anchor = -1;
            float best_score = 0.0f;

            // We must consider objectness * class; to avoid sigmoids, we:
            // - gate objectness using inv_conf_thres (logit of cfg_.conf) as a cheap early filter
            // - select best class by logit (monotonic w.r.t sigmoid)
            // - only convert (sigmoid) for the winning candidate (same style as Ultralytics code)
            for (int a = 0; a < 3; ++a) {
                float* p = cell + a * per_anchor;  // Kperanchor

                const float obj_logit = p[4];

                // Early objectness gating in RAW space:
                // Using inv_conf_thres is conservative and saves work;
                // you can tune this if you expose a separate obj threshold.
                // if (obj_logit < smgr_->inv_conf_thres)
                //     continue;

                // Best class among valid classes using RAW logits
                // IMPORTANT: do NOT threshold classes with conf here
                float cls_logit;
                int label = MX::Prepost::Util::get_best_label(
                        cls_logit,
                        p + 5,               // class logits start
                        cfg_.valid_classes,  // subset to consider
                        -1e9f                // effectively "no threshold"
                );

                if (label == -1)
                    continue;

                // Select purely by best YOLOv7 score
                const float obj_prob = smgr_->convert(obj_logit);
                const float cls_prob = smgr_->convert(cls_logit);
                const float score = obj_prob * cls_prob;

                if (best_label == -1 || score > best_score) {
                    best_label = label;
                    best_anchor = a;
                    best_obj_logit = obj_logit;
                    best_cls_logit = cls_logit;
                    best_score = score;
                }
            }

            // no sufficient candidate in this cell
            if (best_label == -1)
                continue;

            // YOLOv7-equivalent final threshold
            if (best_score < cfg_.conf)
                continue;

            // NOTE: bbox decoding + storage happens after this point
            // (not shown here because your snippet stops before decode)
            // Decode bbox for the chosen anchor (YOLOv7 decode)
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

            // TODO: there may be an issue with these hard coded anchors.
            // anchors[layer_id][anchor_id] = (w,h) in model-input pixels
            static const float anchors_w[3][3] = {
                    {10.f, 16.f, 33.f},     // P3/8
                    {30.f, 62.f, 59.f},     // P4/16
                    {116.f, 156.f, 373.f},  // P5/32
            };
            static const float anchors_h[3][3] = {
                    {13.f, 30.f, 23.f},    // P3/8
                    {61.f, 45.f, 119.f},   // P4/16
                    {90.f, 198.f, 326.f},  // P5/32
            };

            // center
            const float cx = (sx * 2.0f - 0.5f + col) * stride;
            const float cy = (sy * 2.0f - 0.5f + row) * stride;

            // size
            float bw = (sw * 2.0f);
            float bh = (sh * 2.0f);
            bw = bw * bw * anchors_w[layer_id][best_anchor];
            bh = bh * bh * anchors_h[layer_id][best_anchor];

            std::array<float, 4> coord = {
                    cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f};

            // convert to raw bbox coords (undo letterbox)
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
    // apply NMS
    std::vector<int> keep_indices = MX::Prepost::Util::nms(all_boxes, cfg_.iou);

    // early exit
    int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }
}