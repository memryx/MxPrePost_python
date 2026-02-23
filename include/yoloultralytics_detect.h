#pragma once

#include "MxPrepost.h"
#include "utils.h"

// forward declaration
namespace MX::Prepost::Util {
    class ScoreManager;
}

namespace MX {
    namespace Prepost {
        class YoloUltralyticsDetect : public MX::Runtime::MxPrepost {

          public:
            explicit YoloUltralyticsDetect(MX::Runtime::MxAccl* accl,
                                  const YoloUserConfig& config,
                                  const std::string& task = "");

            cv::Mat preprocess(const cv::Mat& image) override;
            void postprocess(const std::vector<float*>& outputs, Result& result) override;
            void postprocess(const std::vector<float*>& outputs,
                             Result& result,
                             const cv::Mat& original_image);
            void
            postprocess(const std::vector<float*>& outputs, Result& result, int ori_w, int ori_h);
            void draw(cv::Mat& image, const Result& result) override;

          private:
            //-----------------------------
            // model/task-specific constants
            constexpr int    COORD_FMAP_SIZE = 64; // number of channels in the coordinate ofmap
            constexpr const std::vector<int> STRIDES{8, 16, 32}; // fixed by model arch
            constexpr int    NUM_LAYERS = STRIDES.size();
            //-----------------------------
            size_t total_preds_;
            
            void postprocess_impl(const std::vector<float*>& outputs,
                                  Result& result,
                                  int ori_w,
                                  int ori_h);
            std::vector<MX::Prepost::Util::LayerParams> yolo_post_layers_;

            // misc
            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
        };

    }
}
