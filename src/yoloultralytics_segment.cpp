#include "yoloultralytics_segment.h"

#include "utils.h"

using namespace MX::Pipe;
using namespace MX::Pipe::Util;

YoloUltralyticsSegment::YoloUltralyticsSegment(const YoloConfig& config) {

    // init settings from config
    iou_thres_ = config.iou;

    if (config.valid_classes.empty()) {
        // use all classes
        for (int i = 0; i < COCO_CLASS_NUMBER; ++i) {
            valid_classes_.push_back(i);
        }
    } else {
        // use specified classes
        for (int cls : config.valid_classes) {
            valid_classes_.push_back(cls);
        }
    }

    ori_w_ = config.ori_width;
    ori_h_ = config.ori_height;

    // letterbox params
    letterbox_ratio_ = (float)MODEL_W / ori_w_;
    letterbox_w_ = ori_w_ * letterbox_ratio_;
    letterbox_h_ = ori_h_ * letterbox_ratio_;

    pad_w_ = (MODEL_W - letterbox_w_) / 2;
    pad_h_ = (MODEL_H - letterbox_h_) / 2;

    // init score manager
    smgr_ = std::make_unique<MX::Pipe::Util::ScoreManager>(config.conf, config.fast_sigmoid);

    // init post-process layer params
    yolo_post_layers_[0] = {.coord_port = 0,
                            .conf_port = 1,
                            .mask_coef_port = 3,
                            .width = MODEL_W / 8,   // L0_HW, 640 / 8 = 80
                            .height = MODEL_H / 8,  // L0_HW, 640 / 8 = 80
                            .stride = 8};

    yolo_post_layers_[1] = {.coord_port = 4,
                            .conf_port = 5,
                            .mask_coef_port = 6,
                            .width = MODEL_W / 16,   // L1_HW, 640 / 16 = 40
                            .height = MODEL_H / 16,  // L1_HW, 640 / 16 = 40
                            .stride = 16};

    yolo_post_layers_[2] = {.coord_port = 7,
                            .conf_port = 8,
                            .mask_coef_port = 9,
                            .width = MODEL_W / 32,   // L2_HW, 640 / 32 = 20
                            .height = MODEL_H / 32,  // L2_HW, 640 / 32 = 20
                            .stride = 32};
}

cv::Mat YoloUltralyticsSegment::preprocess(const cv::Mat& image) {
    return MX::Pipe::Util::preprocess(image, letterbox_w_, letterbox_h_, pad_w_, pad_h_);
}

void YoloUltralyticsSegment::draw(cv::Mat& image, const Result& result) {
    for (const Mask& mask : result.masks) {
        MX::Pipe::Util::draw_mask(image, mask);
    }
}

void YoloUltralyticsSegment::postprocess(const std::vector<float*>& outputs, Result& result) {

    // Candidate Gathering
    std::vector<BBox> all_boxes;
    std::vector<float*> all_mask_coefs;  // mask coefficient base ptrs
    all_boxes.reserve(TOTAL_ANCHORS);
    all_mask_coefs.reserve(TOTAL_ANCHORS);
    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);
        float* mask_coef_base = outputs.at(layer.mask_coef_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // get best label and score
            // NOTE: use inv_conf_thres here because score is still raw (not applied sigmoid yet).
            // sigmoid is expensive and we only apply it if needed.
            float best_score;
            int best_label = MX::Pipe::Util::get_best_label(best_score,
                                                            conf_base + i * COCO_CLASS_NUMBER,
                                                            valid_classes_,
                                                            smgr_->inv_conf_thres);

            // no label with sufficient score
            if (best_label == -1)
                continue;

            // convert best score in [0,1] (e.g. apply sigmoid)
            best_score = smgr_->convert(best_score);

            // decode bbox (Distribution Focal Loss)
            std::array<float, 4> coord = MX::Pipe::Util::dfl(coord_base + i * COORD_FMAP_SIZE,
                                                             i / layer.width /* row */,
                                                             i % layer.width /* col */,
                                                             layer.stride);

            // convert to raw bbox coords
            coord[0] = (coord[0] - pad_w_) / letterbox_ratio_;
            coord[1] = (coord[1] - pad_h_) / letterbox_ratio_;
            coord[2] = (coord[2] - pad_w_) / letterbox_ratio_;
            coord[3] = (coord[3] - pad_h_) / letterbox_ratio_;

            // store bbox
            all_boxes.emplace_back(coord[0],
                                   coord[1],
                                   coord[2],
                                   coord[3],
                                   best_score,
                                   best_label,
                                   COCO_NAMES[best_label]);

            // store mask coef pointer
            all_mask_coefs.emplace_back(mask_coef_base + i * MASK_FMAP_SIZE);
        }
    }

    // apply NMS
    std::vector<int> keep_indices = MX::Pipe::Util::nms(all_boxes, iou_thres_);

    // early exit
    int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    // keep only the selected boxes
    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }

    // keep only selected mask coefficients
    cv::Mat mask_coefs_mat(MASK_FMAP_SIZE, (int)num_keep, CV_32F);
    for (size_t i = 0; i < num_keep; ++i) {
        // Create a header for the destination column
        cv::Mat col_header = mask_coefs_mat.col((int)i);
        // Wrap the source data in a temporary Mat header and copy it
        cv::Mat src_col(MASK_FMAP_SIZE, 1, CV_32F, all_mask_coefs[keep_indices[i]]);
        src_col.copyTo(col_header);
    }

    // Prepare Proto Masks (160*160, N)
    cv::Mat mask_proto(MASK_PROTO_H * MASK_PROTO_W, MASK_FMAP_SIZE, CV_32F, (void*)outputs[2]);
    cv::Mat raw_masks = mask_proto * mask_coefs_mat;  // (160*160, 32) * (32, N) -> (160*160, N)
    cv::Mat mask_stack = raw_masks.reshape((int)num_keep, MASK_PROTO_H);

    // Factors to move from Model Space (640) to Proto Space (160)
    float model_to_proto_x = (float)MASK_PROTO_W / MODEL_W;
    float model_to_proto_y = (float)MASK_PROTO_H / MODEL_H;

    for (int i = 0; i < num_keep; ++i) {
        const BBox& box = result.boxes[i];

        // --- STEP 1: Get BBox in Model Space (undo the mapping to original image) ---
        // We need the box relative to the 640x640 letterbox to crop the 160x160 proto correctly
        float m_x1 = box.x_min * letterbox_ratio_ + pad_w_;
        float m_y1 = box.y_min * letterbox_ratio_ + pad_h_;
        float m_x2 = box.x_max * letterbox_ratio_ + pad_w_;
        float m_y2 = box.y_max * letterbox_ratio_ + pad_h_;

        // --- STEP 2: Scale BBox to Proto Space (160x160) ---
        int px1 = std::clamp((int)(m_x1 * model_to_proto_x), 0, MASK_PROTO_W - 1);
        int py1 = std::clamp((int)(m_y1 * model_to_proto_y), 0, MASK_PROTO_H - 1);
        int px2 = std::clamp((int)(m_x2 * model_to_proto_x), 0, MASK_PROTO_W - 1);
        int py2 = std::clamp((int)(m_y2 * model_to_proto_y), 0, MASK_PROTO_H - 1);

        if (px2 <= px1 || py2 <= py1)
            continue;

        // --- STEP 3: Crop the 160x160 proto mask ---
        cv::Rect proto_roi(px1, py1, px2 - px1, py2 - py1);
        cv::Mat one_mask;
        cv::extractChannel(mask_stack, one_mask, i);
        cv::Mat mask_crop = one_mask(proto_roi);

        // --- STEP 4: Resize crop to "Original Image" BBox dimensions ---
        int target_w = std::max(1, (int)(box.x_max - box.x_min));
        int target_h = std::max(1, (int)(box.y_max - box.y_min));

        cv::Mat resized_crop;
        cv::resize(mask_crop, resized_crop, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

        // --- STEP 5: Binary Threshold & Contours ---
        cv::Mat binary_mask;
        cv::threshold(resized_crop, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);

        // find contours on the binary mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            // A valid polygon needs at least 3 points
            if (contour.size() < 3)
                continue;

            Mask mask_struct;
            mask_struct.cls_id = box.cls_id;
            mask_struct.xy.reserve(contour.size());

            // --- STEP 7: Map points back to Original Image Space ---
            // Points 'p' are relative to the BBox crop.
            // Add the BBox top-left offset to get global coordinates.
            for (const auto& p : contour) {
                mask_struct.xy.push_back({p.x + box.x_min, p.y + box.y_min});
            }

            if (!mask_struct.xy.empty()) {
                result.masks.push_back(mask_struct);
            }
        }
    }
}
