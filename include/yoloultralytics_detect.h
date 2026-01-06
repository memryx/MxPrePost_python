#pragma once

#include "pipeline.h"

// forward declaration
namespace MX::Pipe::Util {
    class ScoreManager;
}

namespace MX {
    namespace Pipe {
        class YoloUltralyticsDetect : public MX::Pipe::Pipeline {

          public:
            YoloUltralyticsDetect(const YoloUserConfig& config);

            cv::Mat preprocess(const cv::Mat& image) override;
            void postprocess(const std::vector<float*>& outputs, Result& result) override;
            void draw(cv::Mat& image, const Result& result) override;

          private:
            /** @brief Structure representing per-layer information of YoloUltralyticsDetect
             * output. */
            struct LayerParams {
                uint8_t coord_port;
                uint8_t conf_port;
                size_t width;
                size_t height;
                size_t stride;
            };

            static constexpr size_t kNumPostProcessLayers = 3;
            struct LayerParams yolo_post_layers_[kNumPostProcessLayers];

            // misc
            std::unique_ptr<MX::Pipe::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
        };

    }
}
