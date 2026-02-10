#pragma once

#include "MxPrepost.h"

// forward declaration
namespace MX::Prepost::Util {
    class ScoreManager;
}

namespace MX {
    namespace Runtime {
        class Yolo7Detect : public MX::Runtime::MxPrepost {

          public:
            Yolo7Detect(MX::Runtime::MxAccl* accl, const YoloUserConfig& config, const std::string& task = "");

            cv::Mat preprocess(const cv::Mat& image) override;
            void postprocess(const std::vector<float*>& outputs, Result& result) override;
            void draw(cv::Mat& image, const Result& result) override;

          private:
            /** @brief Structure representing per-layer information of Yolo7Detect
             * output. */
            struct LayerParams {
                uint8_t out_port;
                size_t width;
                size_t height;
                size_t stride;
            };

            static constexpr size_t kNumPostProcessLayers = 3;
            struct LayerParams yolo_post_layers_[kNumPostProcessLayers];

            size_t total_preds_ = 8400 * kNumPostProcessLayers;

            // misc
            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
        };

    }
}
