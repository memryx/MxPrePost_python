#pragma once

#include "MxPrepost.h"
#include "utils.h"

// forward declaration
namespace MX::Prepost::Util {
    class ScoreManager;
}

namespace MX {
    namespace Runtime {
        class YoloUltralyticsDetect : public MX::Runtime::MxPrepost {

          public:
            YoloUltralyticsDetect(MX::Runtime::MxAccl* accl, const YoloUserConfig& config, const std::string& task = "");

            cv::Mat preprocess(const cv::Mat& image) override;
            void postprocess(const std::vector<float*>& outputs, Result& result) override;
            void draw(cv::Mat& image, const Result& result) override;

          private:
            static constexpr size_t kNumPostProcessLayers = 3;
            std::vector<MX::Prepost::Util::LayerParams> yolo_post_layers_;

            size_t total_preds_ = 8400;

            // misc
            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
        };

    }
}
