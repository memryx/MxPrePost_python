#pragma once

#include "pipeline.h"

// forward declaration
namespace MX::Pipe::Util {
    class ScoreManager;
}

namespace MX {
    namespace Pipe {

        class YoloUltralyticsPose : public MX::Pipe::Pipeline {

          public:
            YoloUltralyticsPose(MX::Runtime::MxAccl* accl, const YoloUserConfig& config);

            cv::Mat preprocess(const cv::Mat& image) override;
            void postprocess(const std::vector<float*>& outputs, Result& result) override;
            void draw(cv::Mat& image, const Result& result) override;

          private:
            /** @brief Structure representing per-layer information of YoloUltralyticsPose
             * output. */
            struct LayerParams {
                uint8_t coord_port;
                uint8_t conf_port;
                uint8_t keypt_port;
                size_t width;
                size_t height;
                size_t stride;
                std::vector<Point2f> anchors;
            };

            static constexpr size_t kNumPostProcessLayers = 3;
            struct LayerParams yolo_post_layers_[kNumPostProcessLayers];

            size_t total_preds_ = 8400;

            std::unique_ptr<MX::Pipe::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
        };

    }
}