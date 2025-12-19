#pragma once

#include "config.h"
#include "pipeline.h"

#include <cstdlib>
#include <mutex>
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/opencv.hpp>    /* imshow */
#include <queue>

#define mxutil_prepost_sigmoid(_x_)                                                               \
    (1.0 / (1.0 + expf(-1.0 * (_x_))))  // sigmoid: f(x) = 1 / (1 + e^(-x))
#define mxutil_prepost_sigmoid_fast_sigmoid(_x_)                                                  \
    ((_x_) /                                                                                      \
     (((_x_) < 0) ? (1.0 - (_x_)) : (1.0 + (_x_))))  // fast-sigmoid: f(x) = x / (1 + abs(x))
#define mxutil_max(_x_, _y_) (((_x_) > (_y_)) ? (_x_) : (_y_))
#define mxutil_min(_x_, _y_) (((_x_) < (_y_)) ? (_x_) : (_y_))

using namespace MX::Pipe;

class YoloUltralyticsDetect : public MX::Pipe::Pipeline {
  public:
    /** @brief Constructor for using official 80 classes COCO dataset. */
    YoloUltralyticsDetect(const YoloConfig& config);

    cv::Mat preprocess(const cv::Mat& image) override;
    void postprocess(const std::vector<float*>& outputs, Result& result) override;
    void draw(cv::Mat& image, const Result& result) override;

  private:
    /** @brief Structure representing per-layer information of YoloUltralyticsDetect output. */
    struct LayerParams {
        uint8_t coord_ofmap_flow_id;
        uint8_t conf_ofmap_flow_id;
        size_t width;
        size_t height;
        size_t ratio;
        size_t coord_fmap_size;
    };

    bool _is_horizontal_input(int ori_width, int ori_height);
    void _nms(std::list<BBox>& boxes, const BBox& candidate, float iou);
    void _get_detection(std::list<BBox>& boxes,
                        int layer_id,
                        float* conf_buffer,
                        float* coord_buffer,
                        int row,
                        int col);

    void _draw_bbox(cv::Mat& image, const BBox& bbox);
    float _calc_iou(const BBox& bbox_0, const BBox& bbox_1);

    float _conf_to_fastSigmoid_inputVal(float conf);

    static constexpr size_t kNumPostProcessLayers = 3;
    struct LayerParams yolo_post_layers_[kNumPostProcessLayers];

    // Model-specific parameters.
    const char** class_labels_;
    size_t class_count_;
    size_t model_w_;   // Model input width to accelerator
    size_t model_h_;   // Model input height to accelerator
    size_t model_ch_;  // Model input channel to accelerator

    // Colors for labels and bounding boxes.
    std::vector<cv::Scalar> class_label_colors_;
    std::vector<cv::Scalar> bounding_box_colors_;

    // Conf and IOU thresholds.
    float conf_thres_;
    float conf_thres_fastSigmoid_;  // Converted conf threshold for fast-sigmoid
    float iou_thres_;

    // Letterbox ratio and pad.
    float letterbox_ratio_;
    int letterbox_w_;
    int letterbox_h_;
    int pad_h_;
    int pad_w_;

    int ori_w_;
    int ori_h_;
};
