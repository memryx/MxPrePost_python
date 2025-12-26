#pragma once

#include "config.h"

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

namespace MX::Pipe::Util {

    using namespace MX::Pipe;

#include <algorithm>
#include <cmath>

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

    std::vector<int> nms(const std::vector<BBox>& boxes, float iou_thres);

    inline std::vector<cv::Scalar> make_colors(const std::vector<cv::Scalar>& src_colors,
                                               size_t class_count) {
        std::vector<cv::Scalar> colors;
        colors.reserve(class_count);

        const size_t color_size = src_colors.size();
        for (size_t i = 0; i < class_count; ++i) {
            colors.push_back(src_colors[i % color_size]);
        }
        return colors;
    }

    // initialized once
    inline const std::vector<cv::Scalar> label_colors =
            make_colors(COCO_TEXT_COLORS, COCO_CLASS_NUMBER);

    inline const std::vector<cv::Scalar> box_colors =
            make_colors(COCO_BOX_COLORS, COCO_CLASS_NUMBER);

    void draw_bbox(cv::Mat& image, const BBox& bbox);

    bool is_horizontal_input(int ori_w, int ori_h);

    cv::Mat
    preprocess(const cv::Mat& image, int letterbox_w, int letterbox_h, int pad_w, int pad_h);

    void draw_mask(cv::Mat& image, const Mask& mask, float alpha = 0.3f);

    int get_best_label(float& best_score,
                       float* score_buf,
                       const std::vector<int>& valid_classes,
                       float score_thres);

    std::array<float, 4> dfl(float* coord_buf, int row, int col, int stride);
}
