#include "yolo10_detect.h"

#include "config_finalizer.h"
#include "utils.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <algorithm>

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

Yolo10Detect::Yolo10Detect(MX::Runtime::MxAccl* accl, const YoloUserConfig& user_cfg, const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Determine model type from task string
    std::string model_type = "yolo10-det";
    if (task == "yolov10_det") {
        model_type = "yolo10-det";
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
            throw std::runtime_error("Model type '" + model_type + "' not found in config file");
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
            
            yolo_post_layers_[layer_idx] = {
                .coord_port = layer["coord_port"].as<uint8_t>(),
                .conf_port = layer["conf_port"].as<uint8_t>(),
                .width = static_cast<size_t>(cfg_.model_w / stride),
                .height = static_cast<size_t>(cfg_.model_h / stride),
                .stride = static_cast<size_t>(stride)
            };
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load YAML config from " << config_path 
                  << ": " << e.what() << std::endl;
        std::cerr << "Using default yolo10-det configuration." << std::endl;
        
        // Fallback to default yolo10-det ports
        yolo_post_layers_[0] = {
            .coord_port = 0,
            .conf_port = 1,
            .width = static_cast<size_t>(cfg_.model_w / 8),
            .height = static_cast<size_t>(cfg_.model_h / 8),
            .stride = 8
        };
        yolo_post_layers_[1] = {
            .coord_port = 2,
            .conf_port = 3,
            .width = static_cast<size_t>(cfg_.model_w / 16),
            .height = static_cast<size_t>(cfg_.model_h / 16),
            .stride = 16
        };
        yolo_post_layers_[2] = {
            .coord_port = 4,
            .conf_port = 5,
            .width = static_cast<size_t>(cfg_.model_w / 32),
            .height = static_cast<size_t>(cfg_.model_h / 32),
            .stride = 32
        };
    }

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
