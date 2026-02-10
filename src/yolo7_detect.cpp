
#include "yolo7_detect.h"

#include "config_finalizer.h"
#include "utils.h"
#include "memx/accl/MxAccl.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <algorithm>

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

Yolo7Detect::Yolo7Detect(MX::Runtime::MxAccl* accl, const YoloUserConfig& user_cfg, const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Determine model type from task string
    std::string model_type = "yolo7-det";
    if (task == "yolov7_det") {
        model_type = "yolo7-det";
    }

    // Get output shapes from model
    auto model_info = accl->get_model_info(0);
    output_shapes_.clear();
    for (size_t i = 0; i < model_info.out_featuremap_shapes.size(); ++i) {
        std::vector<int64_t> shape_vec = model_info.out_featuremap_shapes[i].chlast_shape();
        if (shape_vec.size() >= 3) {
            output_shapes_.push_back(std::make_tuple(
                static_cast<int>(shape_vec[0]),
                static_cast<int>(shape_vec[1]),
                static_cast<int>(shape_vec[2])
            ));
        }
    }

    // Load port configuration from YAML
    std::string source_file = __FILE__;
    std::string config_path = source_file.substr(0, source_file.find("/src/")) + "/config/model-config.yaml";
    
    try {
        if (!std::ifstream(config_path).good()) {
            throw std::runtime_error("Config file not found at " + config_path);
        }

        YAML::Node config = YAML::LoadFile(config_path);
        
        if (!config[model_type]) {
            throw std::runtime_error("The task for this model is '" + task + "'. Please ensure you selected the correct task.");
        }
        
        YAML::Node model_config = config[model_type];
        
        // Load layer configurations
        for (int layer_idx = 0; layer_idx < 3; ++layer_idx) {
            std::string layer_key = "layer_" + std::to_string(layer_idx);
            YAML::Node layer = model_config["layers"][layer_key];
            
            if (!layer) {
                throw std::runtime_error("Layer " + layer_key + " not found in config");
            }
            
            int stride = (layer_idx == 0) ? 8 : (layer_idx == 1) ? 16 : 32;
            
            // For YOLOv7, coord_port and conf_port are the same (combined tensor)
            uint8_t port = layer["coord_port"].as<uint8_t>();
            
            yolo_post_layers_[layer_idx] = {
                .out_port = port,
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
    // Print raw logits for debugging
    for (size_t i = 0; i < outputs.size() && i < output_shapes_.size(); ++i) {
        auto [channels, height, width] = output_shapes_[i];
        float* out_data = outputs[i];
        std::cout << "Port " << i << " [" << channels << "," << height << "," << width << "]: ";
        std::cout << "values = [";
        for (int j = 0; j < std::min(5, channels * height * width); ++j) {
            std::cout << out_data[j];
            if (j < std::min(4, channels * height * width - 1)) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

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
                    {12.f, 19.f, 40.f},     // stride 8
                    {36.f, 76.f, 72.f},     // stride 16
                    {142.f, 192.f, 459.f},  // stride 32
            };
            static const float anchors_h[3][3] = {
                    {16.f, 36.f, 28.f},
                    {75.f, 55.f, 146.f},
                    {110.f, 243.f, 401.f},
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