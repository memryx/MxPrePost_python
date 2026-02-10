#include "yoloultralytics_segment.h"

#include "config_finalizer.h"
#include "utils.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <algorithm>

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

YoloUltralyticsSegment::YoloUltralyticsSegment(MX::Runtime::MxAccl* accl,
                                               const YoloUserConfig& user_cfg,
                                               const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Determine model type from task string
    std::string model_type = task;
    if (task == "yolov11_seg") {
        model_type = "yolo11-seg";
    } 
    if (task == "yolov8_seg") {
        model_type = "yolov8-seg";
    }

    // Load port configuration from YAML
    std::string source_file = __FILE__;
    std::string config_path = source_file.substr(0, source_file.find("/src/")) + "/config/model-config.yaml";
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        if (!config[model_type]) {
            throw std::runtime_error("The task for this model is '" + task + "'. Please ensure you selected the correct task.");
        }
        
        YAML::Node model_config = config[model_type];
        
        // Load mask_proto_port (may be in shared or layer_0)
        if (model_config["layers"]["mask_proto_port"]) {
            mask_proto_port_ = model_config["layers"]["mask_proto_port"].as<uint8_t>();
        } else {
            throw std::runtime_error("Mask proto port not found in config");
        }
        
        // Load layer configurations
        for (int layer_idx = 0; layer_idx < 3; ++layer_idx) {
            std::string layer_key = "layer_" + std::to_string(layer_idx);
            YAML::Node layer = model_config["layers"][layer_key];
            
            if (!layer) {
                throw std::runtime_error("Layer " + layer_key + " not found in config");
            }
            
            int stride = (layer_idx == 0) ? 8 : (layer_idx == 1) ? 16 : 32;
            
            yolo_post_layers_[layer_idx] = {
                .coord_port = layer["coord_port"].as<uint8_t>(),
                .conf_port = layer["conf_port"].as<uint8_t>(),
                .mask_coef_port = layer["mask_coef_port"].as<uint8_t>(),
                .width = static_cast<size_t>(cfg_.model_w / stride),
                .height = static_cast<size_t>(cfg_.model_h / stride),
                .stride = static_cast<size_t>(stride)
            };
        }
        
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(
            std::string("Error: ") + e.what() + 
            ". The task for this model is '" + task + "'. Please ensure you selected the correct task."
        );
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("YAML parsing error: ") + e.what() + 
            ". The task for this model is '" + task + "'. Please ensure you selected the correct task."
        );
    }

    std::vector<MX::Prepost::Util::Grid> grids = {
            {yolo_post_layers_[0].width, yolo_post_layers_[0].height},
            {yolo_post_layers_[1].width, yolo_post_layers_[1].height},
            {yolo_post_layers_[2].width, yolo_post_layers_[2].height},
    };
}

cv::Mat YoloUltralyticsSegment::preprocess(const cv::Mat& image) {
    return MX::Prepost::Util::preprocess(
            image, cfg_.letterbox_w, cfg_.letterbox_h, cfg_.pad_w, cfg_.pad_h);
}

void YoloUltralyticsSegment::draw(cv::Mat& image, const Result& result) {
    for (const Mask& mask : result.masks) {
        MX::Prepost::Util::draw_mask(image, mask);
    }
}

void YoloUltralyticsSegment::postprocess(const std::vector<float*>& outputs, Result& result) {
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

            // store mask coef pointer
            all_mask_coefs.emplace_back(mask_coef_base + i * MASK_FMAP_SIZE);
        }
    }

    // apply NMS
    std::vector<int> keep_indices = MX::Prepost::Util::nms(all_boxes, cfg_.iou, cfg_.class_agnostic);

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
    cv::Mat mask_proto(MASK_PROTO_H * MASK_PROTO_W, MASK_FMAP_SIZE, CV_32F, (void*)outputs[mask_proto_port_]);
    cv::Mat raw_masks = mask_proto * mask_coefs_mat;  // (160*160, 32) * (32, N) -> (160*160, N)
    cv::Mat mask_stack = raw_masks.reshape((int)num_keep, MASK_PROTO_H);

    // Factors to move from Model Space (640) to Proto Space (160)
    float model_to_proto_x = (float)MASK_PROTO_W / cfg_.model_w;
    float model_to_proto_y = (float)MASK_PROTO_H / cfg_.model_h;

    for (int i = 0; i < num_keep; ++i) {
        const BBox& box = result.boxes[i];

        // --- STEP 1: Get BBox in Model Space (undo the mapping to original image) ---
        // We need the box relative to the 640x640 letterbox to crop the 160x160 proto correctly
        float m_x1 = box.x_min * cfg_.letterbox_ratio + cfg_.pad_w;
        float m_y1 = box.y_min * cfg_.letterbox_ratio + cfg_.pad_h;
        float m_x2 = box.x_max * cfg_.letterbox_ratio + cfg_.pad_w;
        float m_y2 = box.y_max * cfg_.letterbox_ratio + cfg_.pad_h;

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
