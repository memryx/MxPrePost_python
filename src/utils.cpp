
#include "utils.h"

#define FONT (cv::FONT_ITALIC)

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using namespace MX::Pipe;

namespace {  // anonymous namespace to avoid symbol conflict
    std::vector<cv::Scalar> make_colors(const std::vector<cv::Scalar>& src_colors,
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
    const std::vector<cv::Scalar> label_colors = make_colors(COCO_TEXT_COLORS, COCO_CLASS_NUMBER);
    const std::vector<cv::Scalar> box_colors = make_colors(COCO_BOX_COLORS, COCO_CLASS_NUMBER);
}

namespace MX::Pipe::Util {

    Eigen::MatrixXf concat_boxes(const Eigen::MatrixXf& lbox,
                                 const Eigen::MatrixXf& mbox,
                                 const Eigen::MatrixXf& sbox) {

        // 1. Calculate total rows (N1 + N2 + N3)
        long total_rows = lbox.rows() + mbox.rows() + sbox.rows();
        int cols = 64;  // Fixed based on your snippet

        // 2. Pre-allocate the large matrix
        RowMatrix boxes(total_rows, cols);

        // 3. Fill the matrix using block operations
        // block(start_row, start_col, num_rows, num_cols)
        boxes.block(0, 0, lbox.rows(), cols) = lbox;
        boxes.block(lbox.rows(), 0, mbox.rows(), cols) = mbox;
        boxes.block(lbox.rows() + mbox.rows(), 0, sbox.rows(), cols) = sbox;

        return boxes;
    }

    std::vector<int> nms(const std::vector<BBox>& boxes, float iou_thres) {
        if (boxes.empty())
            return {};

        const int n = boxes.size();

        // 1. Pre-calculate areas to avoid redundant math in the IoU loop
        std::vector<float> areas(n);
        for (int i = 0; i < n; ++i) {
            areas[i] = (boxes[i].x_max - boxes[i].x_min) * (boxes[i].y_max - boxes[i].y_min);
        }

        // 2. Sort indices based on conf scores
        // We sort indices so we never move the actual BBox structs in memory
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return boxes[i].conf > boxes[j].conf;
        });

        // 3. Bitset-style suppression for efficiency
        std::vector<int> suppressed(n, 0);
        std::vector<int> keep;
        keep.reserve(n);  // Pre-allocate memory

        for (int i = 0; i < n; ++i) {
            int idx_i = indices[i];
            if (suppressed[idx_i])
                continue;

            keep.push_back(idx_i);

            for (int j = i + 1; j < n; ++j) {
                int idx_j = indices[j];
                if (suppressed[idx_j])
                    continue;

                // Manual IoU inline for speed
                float inter_x_min = std::max(boxes[idx_i].x_min, boxes[idx_j].x_min);
                float inter_y_min = std::max(boxes[idx_i].y_min, boxes[idx_j].y_min);
                float inter_x_max = std::min(boxes[idx_i].x_max, boxes[idx_j].x_max);
                float inter_y_max = std::min(boxes[idx_i].y_max, boxes[idx_j].y_max);

                float inter_w = std::max(0.0f, inter_x_max - inter_x_min);
                float inter_h = std::max(0.0f, inter_y_max - inter_y_min);
                float inter_area = inter_w * inter_h;

                if (inter_area <= 0)
                    continue;

                float iou = inter_area / (areas[idx_i] + areas[idx_j] - inter_area);

                if (iou > iou_thres) {
                    suppressed[idx_j] = 1;
                }
            }
        }

        return keep;
    }

    void draw_bbox(cv::Mat& image, const BBox& bbox) {

        int x_min = (int)bbox.x_min;
        int y_min = (int)bbox.y_min;
        int x_max = (int)bbox.x_max;
        int y_max = (int)bbox.y_max;
        int cls_id = bbox.cls_id;
        float conf = bbox.conf;
        cv::Scalar box_color = box_colors[cls_id];
        cv::Scalar text_color = label_colors[cls_id];

        double font_scale = ((double)image.rows / 640.0);
        double bbox_thickness = font_scale * 3;
        double font_thickness = font_scale * 2;
        cv::Size text_size;
        int baseline;
        char text[64];

        /* bounding box rectangle line */
        cv::rectangle(image,
                      cv::Point(x_min, y_min) /*top left*/,
                      cv::Point(x_max, y_max) /*bottom right*/,
                      box_color,
                      bbox_thickness,
                      cv::LINE_4);

        sprintf(text, "%s(%.f%%)", bbox.cls_name.c_str(), 100 * conf);

        text_size = cv::getTextSize(text, FONT, 2 * font_scale, bbox_thickness, &baseline);

        /* label background rectangle */
        cv::rectangle(image,
                      cv::Rect(x_min,
                               std::max(0, y_min - text_size.height),
                               text_size.width * 0.5,
                               text_size.height),  // top left, width, height
                      box_color,
                      cv::FILLED);

        /* label text */
        cv::putText(image,
                    text,
                    cv::Point(x_min,
                              std::max(0, y_min - text_size.height) == 0
                                      ? text_size.height - 5
                                      : y_min - 10 * font_scale),  // bottom left
                    FONT,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv::LINE_AA);
    }

    bool is_horizontal_input(int ori_w, int ori_h) {
        if (ori_h > ori_w) {
            printf("Invalid display image: only horizontal images are supported.\n");
            return false;
        }
        return true;
    }

    cv::Mat
    preprocess(const cv::Mat& image, int letterbox_w, int letterbox_h, int pad_w, int pad_h) {

        // Resize keeping aspect ratio
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(letterbox_w, letterbox_h), 0, 0, cv::INTER_LINEAR);

        // Apply letterbox pad (black border)
        cv::Mat padded;
        cv::copyMakeBorder(resized,
                           padded,
                           pad_h,  // top,
                           pad_h,  // bottom
                           pad_w,  // left
                           pad_w,  // right
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));

        // Convert to float and normalize (0–1)
        padded.convertTo(padded, CV_32F, 1.0 / 255.0);
        return padded;  // shape: (640, 640, 3), range [0,1]
    }

    void draw_mask(cv::Mat& image, const Mask& mask, float alpha) {
        // Create an overlay layer
        cv::Mat overlay = image.clone();

        // Convert your custom points back to cv::Point
        std::vector<cv::Point> cv_points;
        for (const auto& p : mask.xy) {
            cv_points.push_back(cv::Point(p.x, p.y));
        }

        // Fill the polygon on the overlay
        std::vector<std::vector<cv::Point>> fill_contour = {cv_points};
        cv::fillPoly(overlay, fill_contour, box_colors[mask.cls_id]);

        // Blend the overlay with the original image
        cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image);
    }

    int get_best_label(float& best_score,
                       float* score_buf,
                       const std::vector<int>& valid_classes,
                       float score_thres) {

        best_score = -1.f;
        int best_label = -1;

        // loop through valid classes only
        for (int label : valid_classes) {

            float score = score_buf[label];
            if (score <= score_thres)
                continue;

            if (score <= best_score)
                continue;

            // update
            best_score = score;
            best_label = label;
        }

        // no best label found if label == -1
        return best_label;
    }

    /* DFL (Distribution Focal Loss) Decoding */
    std::array<float, 4> dfl(float* coord_buf, int row, int col, int stride) {
        // 1. DFL (Distribution Focal Loss) Decoding
        // YOLO outputs 4 distances (left, top, right, bottom) as probability distributions.
        // We compute the expected value (weighted sum) for each side.
        float dists[4];  // {left, top, right, bottom}

        for (int side = 0; side < 4; ++side) {
            float* side_dist_buf = coord_buf + side * 16;

            // Numerically stable Softmax: find max first
            float local_max = side_dist_buf[0];
            for (int i = 1; i < 16; ++i) {
                if (side_dist_buf[i] > local_max)
                    local_max = side_dist_buf[i];
            }

            float softmax_sum = 0.0f;
            float weighted_sum = 0.0f;

// Single pass for exp calculation to optimize performance
#pragma omp simd reduction(+ : softmax_sum, weighted_sum)
            for (int i = 0; i < 16; ++i) {
                float exp_val = expf(side_dist_buf[i] - local_max);
                softmax_sum += exp_val;
                weighted_sum += (float)i * exp_val;
            }
            dists[side] = weighted_sum / softmax_sum;
        }

        // 2. Decode Distances to Anchor-Relative Coordinates
        // Coordinates are relative to the grid cell center (row, col) multiplied by stride.
        // d[0]=left, d[1]=top, d[2]=right, d[3]=bottom
        float x1 = (col + 0.5f - dists[0]) * stride;
        float y1 = (row + 0.5f - dists[1]) * stride;
        float x2 = (col + 0.5f + dists[2]) * stride;
        float y2 = (row + 0.5f + dists[3]) * stride;

        // make sure coords are within letterbox size
        x1 = std::clamp(x1, 0.0f, (float)MX::Pipe::MODEL_W - 1.f);
        y1 = std::clamp(y1, 0.0f, (float)MX::Pipe::MODEL_H - 1.f);
        x2 = std::clamp(x2, 0.0f, (float)MX::Pipe::MODEL_W - 1.f);
        y2 = std::clamp(y2, 0.0f, (float)MX::Pipe::MODEL_H - 1.f);

        // coords for letterbox
        return {x1, y1, x2, y2};
    }
}
