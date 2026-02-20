#pragma once

#include "config.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace MX::Prepost::Util {

    using namespace MX::Runtime;

    /**
     * @brief ScoreManager handles confidence score thresholding and conversion.
     * Fast Sigmoid approximation: f(x) = x / (1 + |x|)
     */
    struct ScoreManager {
        float conf_thres;
        float inv_conf_thres;
        bool fast_sigmoid;

        ScoreManager(float conf_thres_, bool fast_sigmoid_ = false) :
            conf_thres(conf_thres_), fast_sigmoid(fast_sigmoid_) {
            // Must initialize inv_conf_thres AFTER fast_sigmoid is set
            inv_conf_thres = invert(conf_thres);
        }

        /**
         * @brief Logit function (Inverse Sigmoid).
         * Maps probability [0, 1] back to the raw model output space.
         */
        float invert(float p) const {
            // Clamp p to avoid log(0) or division by zero at the boundaries
            p = std::clamp(p, 1e-7f, 1.0f - 1e-7f);

            if (fast_sigmoid) {
                float x = 2.0f * p - 1.0f;         // map [0,1] -> [-1,1]
                return x / (1.0f - std::fabs(x));  // map [-1,1] -> (-inf, inf)
            } else {
                // Standard Logit: x = ln(p / (1 - p))
                return std::log(p / (1.0f - p));
            }
        }

        /**
         * @brief Sigmoid function.
         * Maps raw model output to probability [0, 1].
         */
        float convert(float x) const {
            if (fast_sigmoid) {
                // Algebraic approximation: maps (-inf, inf) to (-1, 1)
                float res = x / (1.0f + std::fabs(x));
                // Shift and scale to (0, 1)
                return (res + 1.0f) * 0.5f;
            } else {
                // Standard Logistic Sigmoid
                return 1.0f / (1.0f + std::exp(-x));
            }
        }
    };

    // Non-maximum suppression (NMS)
    // - class_agnostic=false: boxes of different classes do not suppress each other (class-aware)
    // - class_agnostic=true: boxes suppress each other regardless of class (class-agnostic)
    std::vector<int>
    nms(const std::vector<BBox>& boxes, float iou_thres, bool class_agnostic = false);

    void draw_bbox(cv::Mat& image, const BBox& bbox);

    struct LetterboxParams {
        float ratio = 1.f;
        int letterbox_w = 0;
        int letterbox_h = 0;
        int pad_left = 0;
        int pad_right = 0;
        int pad_top = 0;
        int pad_bottom = 0;
    };

    LetterboxParams compute_letterbox(int ori_w, int ori_h, int model_w, int model_h);

    cv::Mat preprocess(const cv::Mat& image,
                       int letterbox_w,
                       int letterbox_h,
                       int pad_left,
                       int pad_top,
                       int pad_right,
                       int pad_bottom);

    void draw_mask(cv::Mat& image, const Mask& mask, float alpha = 0.3f);

    int get_best_label(float& best_score,
                       float* score_buf,
                       const std::vector<int>& valid_classes,
                       float score_thres);

    std::array<float, 4>
    dfl(float* coord_buf, int row, int col, int stride, int model_w, int model_h);

    struct Grid {
        size_t width;
        size_t height;
    };

    /** @brief Unified structure for YOLO layer parameters.
     * All ports use -1 to indicate unused ports for a given model type.
     */
    struct LayerParams {
        int coord_port = -1;
        int conf_port = -1;
        int keypt_port = -1;
        int mask_coef_port = -1;
        int mask_proto_port = -1;  // For segment models (same for all layers)
        int port_out = -1;         // For YOLOv7
        size_t width;
        size_t height;
        size_t stride;
        std::vector<Point2f> anchors;  // Only pose models populate this
    };

    /**
     * @brief Load YOLO layer configuration from embedded YAML config.
     * @param task Task string (e.g., "yolov8_det", "yolov8_seg", "yolov8_pose")
     * @param model_w Model input width
     * @param model_h Model input height
     * @return Vector of 3 LayerParams (one per layer)
     */
    std::vector<LayerParams>
    loadYoloLayerConfig(const std::string& task, int model_w, int model_h);

    std::string normalize(std::string s);

    std::size_t levenshtein(const std::string& a, const std::string& b);
    std::string colored_diff(const std::string& input, const std::string& target);

}

namespace termcolor {
    static constexpr const char* reset = "\033[0m";
    static constexpr const char* red = "\033[31m";
    static constexpr const char* green = "\033[32m";
}