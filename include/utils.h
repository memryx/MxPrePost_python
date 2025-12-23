#pragma once

#include "config.h"

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

namespace MX::Pipe::Util {

    using namespace MX::Pipe;

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
}
