#pragma once

#include "MxPrepost.h"
#include "utils.h"

namespace MX {
    namespace Prepost {

        // Color list for drawing keypoints
        static const std::vector<cv::Scalar> KEYPOINT_COLORS = {
                cv::Scalar(128, 255, 0),   cv::Scalar(255, 128, 50),  cv::Scalar(128, 0, 255),
                cv::Scalar(255, 255, 0),   cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255),
                cv::Scalar(51, 153, 255),  cv::Scalar(255, 153, 153), cv::Scalar(255, 51, 51),
                cv::Scalar(153, 255, 153), cv::Scalar(51, 255, 51),   cv::Scalar(0, 255, 0),
                cv::Scalar(255, 0, 51),    cv::Scalar(153, 0, 153),   cv::Scalar(51, 0, 51),
                cv::Scalar(0, 0, 0),       cv::Scalar(0, 102, 255),   cv::Scalar(0, 51, 255),
                cv::Scalar(0, 153, 255),   cv::Scalar(0, 153, 153)};
        
        class YoloUltralyticsPose : public MX::Prepost::MxPrepost {

          public:
            YoloUltralyticsPose(MX::Runtime::MxAccl* accl,
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
            static constexpr std::array<int, 3> STRIDES = {8, 16, 32}; // fixed by model arch
            static constexpr int    NUM_LAYERS = STRIDES.size();
            static constexpr int    NUM_KEYPOINTS = 17;   // Number of keypoints for pose estimation
            //-----------------------------
            // Pairs of keypoints for drawing skeleton
            static constexpr std::array<const std::pair<int, int>, 18> KEYPOINT_PAIRS = {{
                    {0, 1},
                    {0, 2},
                    {1, 3},
                    {2, 4},
                    {0, 5},
                    {0, 6},
                    {5, 7},
                    {7, 9},
                    {6, 8},
                    {8, 10},
                    {5, 6},
                    {5, 11},
                    {6, 12},
                    {11, 12},
                    {11, 13},
                    {13, 15},
                    {12, 14},
                    {14, 16},
            }};
            //-----------------------------
            size_t total_preds_;

            void postprocess_impl(const std::vector<float*>& outputs,
                                  Result& result,
                                  int ori_w,
                                  int ori_h);
            std::vector<MX::Prepost::Util::LayerParams> yolo_post_layers_;

            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
        };

    }
}
