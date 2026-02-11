#include "yoloultralytics_pose.h"

#include "config_finalizer.h"
#include "utils.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>

// Forward declaration for embedded config
namespace MX::Prepost::Config {
    const std::string& getModelConfigYaml();
}

using namespace MX::Runtime;

YoloUltralyticsPose::YoloUltralyticsPose(MX::Runtime::MxAccl* accl,
                                         const YoloUserConfig& user_cfg,
                                         const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // Determine model type from task string
    std::string model_type = task;
    if (task == "yolov11_pose") {
        model_type = "yolo11-pose";
    } else if (task == "yolov8_pose") {
        model_type = "yolo8-pose";
    }

    // Load port configuration from embedded YAML
    try {
        YAML::Node config = YAML::Load(MX::Prepost::Config::getModelConfigYaml());
        
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
                .keypt_port = layer["keypt_port"].as<uint8_t>(),
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

    // init anchors for each layer
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
    return MX::Prepost::Util::preprocess(
            image, cfg_.letterbox_w, cfg_.letterbox_h, cfg_.pad_w, cfg_.pad_h);
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

void YoloUltralyticsPose::postprocess(const std::vector<float*>& outputs, Result& result) {    
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
            float x1 = (coord[0] - cfg_.pad_w) / cfg_.letterbox_ratio;
            float y1 = (coord[1] - cfg_.pad_h) / cfg_.letterbox_ratio;
            float x2 = (coord[2] - cfg_.pad_w) / cfg_.letterbox_ratio;
            float y2 = (coord[3] - cfg_.pad_h) / cfg_.letterbox_ratio;

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
                    kpt_x = (kpt_x - cfg_.pad_w) / cfg_.letterbox_ratio;
                    kpt_y = (kpt_y - cfg_.pad_h) / cfg_.letterbox_ratio;

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
