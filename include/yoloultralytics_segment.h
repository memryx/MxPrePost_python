#pragma once

#include "MxPrepost.h"
#include "utils.h"

// forward declaration
namespace MX::Prepost::Util {
    class ScoreManager;
}

namespace MX {
    namespace Prepost {
        class YoloUltralyticsSegment : public MX::Prepost::MxPrepost {

          public:
            YoloUltralyticsSegment(MX::Runtime::MxAccl* accl,
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
            static constexpr int    COORD_FMAP_SIZE = 64; // number of channels in the coordinate ofmap
            static constexpr size_t kNumPostProcessLayers = 3; // always 3
            static constexpr size_t total_preds_ = 8400; // number of predictions -- fixed by model arch
            static constexpr int MASK_FMAP_SIZE = 32;  // Number of feature map channels for mask coeffs
            static constexpr int MASK_PROTO_H = 160;   // Mask proto height
            static constexpr int MASK_PROTO_W = 160;   // Mask proto width
            //-----------------------------
            
            void postprocess_impl(const std::vector<float*>& outputs,
                                  Result& result,
                                  int ori_w,
                                  int ori_h);
            std::vector<MX::Prepost::Util::LayerParams> yolo_post_layers_;


            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
            uint8_t mask_proto_port_;
        };

    }
}
