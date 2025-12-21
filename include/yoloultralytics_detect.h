#pragma once

#include "config.h"
#include "pipeline.h"

#include <cstdlib>
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/opencv.hpp>    /* imshow */

namespace MX {
    namespace Pipe {
        class YoloUltralyticsDetect : public MX::Pipe::Pipeline {

          public:
            /** @brief Constructor for using official 80 classes COCO dataset. */
            YoloUltralyticsDetect(const YoloConfig& config);

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
                size_t ratio;
                size_t coord_fmap_size;
            };

            void _get_detection(std::vector<BBox>& boxes,
                                int layer_id,
                                float* conf_buffer,
                                float* coord_buffer,
                                int row,
                                int col);

            float _conf_to_fastSigmoid_inputVal(float conf);

            static constexpr size_t kNumPostProcessLayers = 3;
            struct LayerParams yolo_post_layers_[kNumPostProcessLayers];

            // Model-specific parameters.
            const char** class_labels_;
            int class_count_;
            const int model_w_ = 640;  // Model input width to accelerator
            const int model_h_ = 640;  // Model input height to accelerator
            const int model_ch_ = 3;   // Model input channel to accelerator

            // Colors for labels and bounding boxes.
            std::vector<cv::Scalar> class_label_colors_;
            std::vector<cv::Scalar> bounding_box_colors_;

            // Conf and IOU thresholds.
            float conf_thres_;
            float iou_thres_;
            std::unordered_set<int> valid_classes_;
            float conf_thres_fastSigmoid_;  // Converted conf threshold for fast-sigmoid

            // Letterbox ratio and pad.
            float letterbox_ratio_;
            int letterbox_w_;
            int letterbox_h_;
            int pad_h_;
            int pad_w_;

            int ori_w_;
            int ori_h_;
        };

    }
}