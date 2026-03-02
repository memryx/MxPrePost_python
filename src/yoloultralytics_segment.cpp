#include "yoloultralytics_segment.h"

#include "config_finalizer.h"
#include "utils.h"
#include <cmath>

#include <memx/accl/utils/macros.h>

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

YoloUltralyticsSegment::YoloUltralyticsSegment(MX::Runtime::MxAccl* accl,
                                               const YoloUserConfig& user_cfg,
                                               const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // autocalculate the layer shapes and corresponding ports based on model output shapes
    // and expected shapes, unless user provided an explicit mapping in the config (override_layer_mapping)
    auto model_info = accl->get_model_info(cfg_.model_id);

    yolo_post_layers_.resize(NUM_LAYERS);

    std::vector<MX::Types::ShapeVector> expected_coord_shapes;
    std::vector<MX::Types::ShapeVector> expected_conf_shapes;
    std::vector<MX::Types::ShapeVector> expected_mask_coef_shapes;
    MX::Types::ShapeVector              expected_mask_proto_shape(cfg_.model_h / MASK_DIV_FACTOR, cfg_.model_w / MASK_DIV_FACTOR, 1, MASK_CHANNELS);
    

    // expected coord/conf are the same
    // mask channels are the same across layers, but height and width depend on the stride
    for(int i=0; i < NUM_LAYERS; ++i) {
        int stride = STRIDES[i];
        expected_coord_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, COORD_FMAP_SIZE});
        expected_conf_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, static_cast<int64_t>(cfg_.class_labels.size())});
        expected_mask_coef_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, MASK_CHANNELS});
        total_preds_ += (cfg_.model_h / stride) * (cfg_.model_w / stride);
    }

    // in case the user is forcing a particular mapping
    if(cfg_.override_layer_mapping.empty()){
        
        // match expected shapes with actual model input shapes to find the correct ports for each layer
        int num_ofmaps = model_info.out_featuremap_shapes.size();

        for (int i=0; i < NUM_LAYERS; ++i){
            int stride = STRIDES[i];
            yolo_post_layers_[i].height = cfg_.model_h / stride;
            yolo_post_layers_[i].width = cfg_.model_w / stride;
            yolo_post_layers_[i].stride = stride;

            bool found_coord = false;
            bool found_conf = false;
            bool found_mask_coef = false;
            bool found_mask_proto = false;
            for (int port = 0; port < num_ofmaps; ++port) {
                const auto& actual_shape = model_info.out_featuremap_shapes[port];
                if (actual_shape == expected_coord_shapes[i]) {
                    yolo_post_layers_[i].coord_port = port;
                    found_coord = true;
                } else if (actual_shape == expected_conf_shapes[i]) {
                    yolo_post_layers_[i].conf_port = port;
                    found_conf = true;
                } else if (actual_shape == expected_mask_coef_shapes[i]) {
                    yolo_post_layers_[i].mask_coef_port = port;
                    found_mask_coef = true;
                } else if (actual_shape == expected_mask_proto_shape) {
                    yolo_post_layers_[i].mask_proto_port = port;
                    found_mask_proto = true;
                    
                    // psa: this will get called multiple times
                    mask_proto_port_ = port;
                }
            }

            if(!found_coord) {
                throw std::runtime_error("Could not find coordinate output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_coord_shapes[i].to_string());
            }
            if(!found_conf) {
                throw std::runtime_error("Could not find confidence output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_conf_shapes[i].to_string());
            }
            if(!found_mask_coef) {
                throw std::runtime_error("Could not find mask coefficient output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_mask_coef_shapes[i].to_string());
            }
            if(!found_mask_proto) {
                throw std::runtime_error("Could not find mask proto output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_mask_proto_shape.to_string());
            }
        }

    }
    else {
        // TODO
    }

}

cv::Mat YoloUltralyticsSegment::preprocess(const cv::Mat& image) {
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

void YoloUltralyticsSegment::draw(cv::Mat& image, const Result& result) {
    for (const Mask& mask : result.masks) {
        MX::Prepost::Util::draw_mask(image, mask);
    }
}

void YoloUltralyticsSegment::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void YoloUltralyticsSegment::postprocess(const std::vector<float*>& outputs,
                                         Result& result,
                                         const cv::Mat& original_image) {
    if (UNLIKELY(original_image.empty())) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.cols, original_image.rows);
}

void YoloUltralyticsSegment::postprocess(const std::vector<float*>& outputs,
                                         Result& result,
                                         int ori_w,
                                         int ori_h) {
    if (UNLIKELY(ori_w <= 0 || ori_h <= 0)) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_w, ori_h);
}

void YoloUltralyticsSegment::postprocess_impl(const std::vector<float*>& outputs,
                                              Result& result,
                                              int ori_w,
                                              int ori_h) {

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    // Candidate Gathering
    std::vector<BBox> all_boxes;
    std::vector<float*> all_mask_coefs;  // mask coefficient base ptrs

    all_boxes.reserve(total_preds_);
    all_mask_coefs.reserve(total_preds_);

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

            // store mask coef pointer
            all_mask_coefs.emplace_back(mask_coef_base + i * MASK_FMAP_SIZE);
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
    cv::Mat mask_proto(
            MASK_PROTO_H * MASK_PROTO_W, MASK_FMAP_SIZE, CV_32F, (void*)outputs[mask_proto_port_]);
    cv::Mat raw_masks = mask_proto * mask_coefs_mat;  // (160*160, 32) * (32, N) -> (160*160, N)
    cv::Mat mask_stack = raw_masks.reshape((int)num_keep, MASK_PROTO_H);

    // Factors to move from Model Space (640) to Proto Space (160)
    float model_to_proto_x = (float)MASK_PROTO_W / cfg_.model_w;
    float model_to_proto_y = (float)MASK_PROTO_H / cfg_.model_h;

    for (int i = 0; i < num_keep; ++i) {
        const BBox& box = result.boxes[i];

        // --- STEP 1: Get BBox in Model Space (undo the mapping to original image) ---
        // We need the box relative to the 640x640 letterbox to crop the 160x160 proto correctly
        float m_x1 = box.x_min * lb.ratio + lb.pad_left;
        float m_y1 = box.y_min * lb.ratio + lb.pad_top;
        float m_x2 = box.x_max * lb.ratio + lb.pad_left;
        float m_y2 = box.y_max * lb.ratio + lb.pad_top;

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

        // Find the largest contour by area (main object region)
        // Concatenating disconnected contours creates invalid polygons that cause glitchy masks
        double max_area = 0.0;
        int largest_contour_idx = -1;
        for (size_t idx = 0; idx < contours.size(); ++idx) {
            if (contours[idx].size() < 3)
                continue;
            double area = cv::contourArea(contours[idx]);
            if (area > max_area) {
                max_area = area;
                largest_contour_idx = static_cast<int>(idx);
            }
        }

        if (largest_contour_idx >= 0) {
            const auto& contour = contours[largest_contour_idx];
            Mask mask_struct;
            mask_struct.cls_id = box.cls_id;
            mask_struct.xys.reserve(contour.size());

            // --- STEP 7: Map points back to Original Image Space ---
            // Points 'p' are relative to the BBox crop.
            // Add the BBox top-left offset to get global coordinates.
            for (const auto& p : contour) {
                mask_struct.xys.push_back({p.x + box.x_min, p.y + box.y_min});
            }

            if (!mask_struct.xys.empty()) {
                result.masks.push_back(mask_struct);
            }
        }
    }
}
