#pragma once

#include "config.h"

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

namespace MX::Pipe::Util {

    using namespace MX::Pipe;

    // TODO: comment more clearly
    struct ScoreManager {
        bool fast_sigmoid;
        float raw_thres;
        float thres_before_sigmoid;

        ScoreManager(float raw_thres, bool fast_sigmoid) :
            raw_thres(raw_thres), fast_sigmoid(fast_sigmoid) {

            if (fast_sigmoid) {
                // Converts a score value in [0,1] to the corresponding input for the fast-sigmoid
                // function. The fast-sigmoid function: f(x) = x / (1 + |x|), which maps [-inf,
                // +inf] -> [-1, 1].
                //
                // Steps:
                //   1. Map conf [0,1] -> x [-1,1]
                //   2. Return x as input for fast-sigmoid

                float x = raw_thres * 2.0f - 1.0f;                // map [0,1] -> [-1,1]
                thres_before_sigmoid = x / (1.0f - std::abs(x));  // map [-1, 1] -> [-inf, inf]
            } else {
                // x = ln(y / (1 - y))
                thres_before_sigmoid = -logf(raw_thres / (1.0f - raw_thres));
            }
        }

        float convert(float x) const {
            if (fast_sigmoid) {
                x = x / (1.0f + std::fabs(x));
                return x = (x + 1.0f) * 0.5f;  // range (-1, 1) -> (0, 1)
            } else {
                return 1.0f / (1.0f + std::exp(-x));
            }
        }
    };

    Eigen::MatrixXf concat_boxes(const Eigen::MatrixXf& lbox,
                                 const Eigen::MatrixXf& mbox,
                                 const Eigen::MatrixXf& sbox);

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
}
