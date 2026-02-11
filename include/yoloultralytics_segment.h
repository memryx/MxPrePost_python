#pragma once

#include "MxPrepost.h"
#include "utils.h"

// forward declaration
namespace MX::Prepost::Util {
    class ScoreManager;
}

namespace MX {
    namespace Runtime {
        class YoloUltralyticsSegment : public MX::Runtime::MxPrepost {

          public:
            YoloUltralyticsSegment(MX::Runtime::MxAccl* accl, const YoloUserConfig& config, const std::string& task = "");

            cv::Mat preprocess(const cv::Mat& image) override;
            void postprocess(const std::vector<float*>& outputs, Result& result) override;
            void draw(cv::Mat& image, const Result& result) override;

          private:
            static constexpr size_t kNumPostProcessLayers = 3;
            std::vector<MX::Prepost::Util::LayerParams> yolo_post_layers_;

            size_t total_preds_ = 8400;

            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
            uint8_t mask_proto_port_;

        };

    }
}