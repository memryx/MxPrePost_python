#include "yoloultralytics_pose.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

YoloUltralyticsPose::YoloUltralyticsPose(MX::Runtime::MxAcclBase* accl,
                                         const YoloUserConfig& user_cfg,
                                         const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Load layer configuration from embedded YAML
    yolo_post_layers_ = MX::Prepost::Util::loadYoloLayerConfig(task, cfg_.model_h, cfg_.model_w);

    // Process anchors for each layer (computed after getting yolo_post_layers_)
    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        auto& layer = yolo_post_layers_[layer_id];
        for (size_t y = 0; y < layer.height; ++y) {
            for (size_t x = 0; x < layer.width; ++x) {
                layer.anchors.push_back({x + 0.5f, y + 0.5f});
            }
        }
    }
}

cv::Mat YoloUltralyticsPose::preprocess(const cv::Mat& image) {
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

void YoloUltralyticsPose::draw(cv::Mat& image, const Result& result) {

    // draw bbox
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }

    // draw keypoints and skeleton
    for (const auto& kpts_per_box : result.keypoints) {

        // draw lines (skeletons)
        for (const auto& connection : KEYPOINT_PAIRS) {
            int idx1 = connection.first;
            int idx2 = connection.second;

            if (idx1 < kpts_per_box.size() && idx2 < kpts_per_box.size()) {
                auto kpt1 = kpts_per_box[idx1];
                auto kpt2 = kpts_per_box[idx2];

                if (kpt1.xy.x != -1 && kpt2.xy.x != -1) {
                    cv::line(image,
                             cv::Point(kpt1.xy.x, kpt1.xy.y),
                             cv::Point(kpt2.xy.x, kpt2.xy.y),
                             cv::Scalar(255, 255, 255),
                             3);
                }
            }
        }

        // Draw individual keypoints
        for (int i = 0; i < kpts_per_box.size(); ++i) {
            auto& kpt = kpts_per_box[i];
            if (kpt.xy.x != -1) {
                cv::circle(image,
                           cv::Point(kpt.xy.x, kpt.xy.y),
                           4,
                           KEYPOINT_COLORS[i % KEYPOINT_COLORS.size()],
                           -1);
            }
        }
    }
}

void YoloUltralyticsPose::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void YoloUltralyticsPose::postprocess(const std::vector<float*>& outputs,
                                      Result& result,
                                      const cv::Mat& original_image) {
    if (original_image.empty()) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void YoloUltralyticsPose::postprocess(const std::vector<float*>& outputs,
                                      Result& result,
                                      int ori_h,
                                      int ori_w) {
    if (ori_w <= 0 || ori_h <= 0) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void YoloUltralyticsPose::postprocess_impl(const std::vector<float*>& outputs,
                                           Result& result,
                                           int ori_h,
                                           int ori_w) {

    result.boxes.clear();
    result.keypoints.clear();

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    std::vector<BBox> all_boxes;
    std::vector<std::vector<Keypoint>> all_kpts;
    all_boxes.reserve(total_preds_);
    all_kpts.reserve(total_preds_);
    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);
        float* kpt_base = outputs.at(layer.keypt_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {
            float score = conf_base[i];

            // NOTE: use inv_conf_thres here because score is still raw (not applied sigmoid yet).
            // sigmoid is expensive and we only apply it if needed.
            if (score < smgr_->inv_conf_thres)
                continue;

            // convert score in [0,1] (e.g. apply sigmoid)
            score = smgr_->convert(score);

            // Get the specific anchor for this grid cell
            const Point2f& anchor = layer.anchors[i];

            // 1. Decode BBox (Distribution Focal Loss)
            std::array<float, 4> coord = MX::Prepost::Util::dfl(coord_base + i * COORD_FMAP_SIZE,
                                                                i / layer.width /* row */,
                                                                i % layer.width /* col */,
                                                                layer.stride,
                                                                cfg_.model_w,
                                                                cfg_.model_h);

            // Convert BBox to original image scale
            float x1 = (coord[0] - lb.pad_left) / lb.ratio;
            float y1 = (coord[1] - lb.pad_top) / lb.ratio;
            float x2 = (coord[2] - lb.pad_left) / lb.ratio;
            float y2 = (coord[3] - lb.pad_top) / lb.ratio;

            // Clip to image bounds
            x1 = std::max(0.0f, std::min(x1, (float)ori_w));
            y1 = std::max(0.0f, std::min(y1, (float)ori_h));
            x2 = std::max(0.0f, std::min(x2, (float)ori_w));
            y2 = std::max(0.0f, std::min(y2, (float)ori_h));

            // Drop invalid/degenerate boxes
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            all_boxes.emplace_back(x1, y1, x2, y2, score, 0, "person");

            // 2. Decode Keypoints
            std::vector<Keypoint> kpts_per_box;
            for (size_t k = 0; k < NUM_KEYPOINTS; ++k) {
                // offset: Jump to grid cell 'i', then jump to keypoint 'k', each has (x, y, conf)
                int offset = (i * NUM_KEYPOINTS * 3) + (k * 3);

                float raw_x = kpt_base[offset];
                float raw_y = kpt_base[offset + 1];
                float kpt_conf_raw = kpt_base[offset + 2];

                // Only process if keypoint confidence is high enough
                if (kpt_conf_raw > smgr_->inv_conf_thres) {
                    // YOLOv8 Pose Decoding:
                    // 1. Multiply by 2.0 (Model output range is usually -0.5 to 1.5)
                    // 2. Add the grid center (Shift to absolute feature map position)
                    // 3. Multiply by stride (Scale up to model input size: e.g., 640x640)
                    //
                    // detail:
                    // https://docs.google.com/document/d/1ENBtyOH2IyDi_NlozJ2HvOvattHg9WLydIHEzjMoZY0/edit?usp=sharing
                    float kpt_x = (raw_x * 2.0f + (anchor.x - 0.5f)) * layer.stride;
                    float kpt_y = (raw_y * 2.0f + (anchor.y - 0.5f)) * layer.stride;

                    // 4. Recovery from Letterbox (Scale to original image pixels)
                    kpt_x = (kpt_x - lb.pad_left) / lb.ratio;
                    kpt_y = (kpt_y - lb.pad_top) / lb.ratio;

                    float kpt_conf = smgr_->convert(kpt_conf_raw);
                    kpts_per_box.emplace_back(kpt_x, kpt_y, kpt_conf);
                } else {
                    // Use a sentinel value for hidden/occluded keypoints
                    kpts_per_box.emplace_back(-1, -1, 0.0f);
                }
            }
            all_kpts.push_back(kpts_per_box);
        }
    }

    // apply NMS
    std::vector<int> keep_indices =
            MX::Prepost::Util::nms(all_boxes, cfg_.iou, cfg_.class_agnostic);

    // early exit
    int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    // keep only the selected boxes
    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }

    // keep only the selected keypoints
    result.keypoints.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.keypoints.push_back(all_kpts[idx]);
    }
}
