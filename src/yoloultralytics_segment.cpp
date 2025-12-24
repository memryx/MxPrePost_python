#include "yoloultralytics_segment.h"

#include "utils.h"

using namespace MX::Pipe;
using namespace MX::Pipe::Util;

YoloUltralyticsSegment::~YoloUltralyticsSegment() {
    delete smgr_;
}

YoloUltralyticsSegment::YoloUltralyticsSegment(const YoloConfig& config) {

    // init settings from config
    iou_thres_ = config.iou;

    if (config.valid_classes.empty()) {
        // use all classes
        for (int i = 0; i < COCO_CLASS_NUMBER; ++i) {
            valid_classes_.push_back(i);
        }
    } else {
        // use specified classes
        for (int cls : config.valid_classes) {
            valid_classes_.push_back(cls);
        }
    }

    // compute padding
    // TODO: support vertical images as well
    if (!MX::Pipe::Util::is_horizontal_input(config.ori_width, config.ori_height))
        return;

    ori_w_ = config.ori_width;
    ori_h_ = config.ori_height;

    // letterbox params
    letterbox_ratio_ = (float)MODEL_W / ori_w_;
    letterbox_w_ = ori_w_ * letterbox_ratio_;
    letterbox_h_ = ori_h_ * letterbox_ratio_;

    pad_w_ = (MODEL_W - letterbox_w_) / 2;
    pad_h_ = (MODEL_H - letterbox_h_) / 2;

    // init score manager
    smgr_ = new MX::Pipe::Util::ScoreManager(config.conf, config.fast_sigmoid);

    // init post-process layer params
    yolo_post_layers_[0] = {.coord_port = 0,
                            .conf_port = 1,
                            .mask_coef_port = 3,
                            .width = MODEL_W / 8,   // L0_HW, 640 / 8 = 80
                            .height = MODEL_H / 8,  // L0_HW, 640 / 8 = 80
                            .stride = 8};

    yolo_post_layers_[1] = {.coord_port = 4,
                            .conf_port = 5,
                            .mask_coef_port = 6,
                            .width = MODEL_W / 16,   // L1_HW, 640 / 16 = 40
                            .height = MODEL_H / 16,  // L1_HW, 640 / 16 = 40
                            .stride = 16};

    yolo_post_layers_[2] = {.coord_port = 7,
                            .conf_port = 8,
                            .mask_coef_port = 9,
                            .width = MODEL_W / 32,   // L2_HW, 640 / 32 = 20
                            .height = MODEL_H / 32,  // L2_HW, 640 / 32 = 20
                            .stride = 32};
}

cv::Mat YoloUltralyticsSegment::preprocess(const cv::Mat& image) {
    return MX::Pipe::Util::preprocess(image, letterbox_w_, letterbox_h_, pad_w_, pad_h_);
}

void YoloUltralyticsSegment::draw(cv::Mat& image, const Result& result) {
    // TODO: imrpove drawing by adding bbox + label together with mask
    for (const Mask& mask : result.masks) {
        MX::Pipe::Util::draw_mask(image, mask);
    }
}

void YoloUltralyticsSegment::postprocess(const std::vector<float*>& outputs, Result& result) {

    // Candidate Gathering
    std::vector<BBox> all_boxes;
    std::vector<float*> all_mask_coefs;  // mask coefficient base ptrs
    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);
        float* mask_coef_base = outputs.at(layer.mask_coef_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // get best label and score
            float best_score;
            int best_label = MX::Pipe::Util::get_best_label(best_score,
                                                            conf_base + i * COCO_CLASS_NUMBER,
                                                            valid_classes_,
                                                            smgr_->thres_before_sigmoid);

            // no label with sufficient score
            if (best_label == -1)
                continue;

            // convert best score in [0,1] (e.g. apply sigmoid)
            best_score = smgr_->convert(best_score);

            // decode bbox (Distribution Focal Loss)
            std::array<float, 4> coord = MX::Pipe::Util::dfl(coord_base + i * COORD_FMAP_SIZE,
                                                             i / layer.width /* row */,
                                                             i % layer.width /* col */,
                                                             layer.stride);

            // convert to raw bbox coords
            coord[0] = (coord[0] - pad_w_) / letterbox_ratio_;
            coord[1] = (coord[1] - pad_h_) / letterbox_ratio_;
            coord[2] = (coord[2] - pad_w_) / letterbox_ratio_;
            coord[3] = (coord[3] - pad_h_) / letterbox_ratio_;

            // store bbox
            BBox bbox(coord[0],
                      coord[1],
                      coord[2],
                      coord[3],
                      best_score,
                      best_label,
                      COCO_NAMES[best_label]);

            all_boxes.push_back(bbox);

            // store mask coef pointer
            all_mask_coefs.push_back(mask_coef_base + i * MASK_FMAP_SIZE);
        }
    }

    // apply NMS
    std::vector<int> keep_indices = MX::Pipe::Util::nms(all_boxes, iou_thres_);

    // early exit
    int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    // keep only the selected boxes
    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }

    // declare mask_coefs_mat
    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMatrix mask_coefs_mat(MASK_FMAP_SIZE, num_keep);

    // keep only selected mask coefficients
    for (size_t i = 0; i < num_keep; ++i) {
        mask_coefs_mat.col(i) =
                Eigen::Map<Eigen::VectorXf>(all_mask_coefs[keep_indices[i]], MASK_FMAP_SIZE);
    }

    // Mask Generation via Eigen to speed up (MatMul)
    // Proto: (160*160, 32), Coefs: (32, N)
    Eigen::Map<const RowMatrix> mask_proto(
            outputs[2], MASK_PROTO_H * MASK_PROTO_W, MASK_FMAP_SIZE);
    RowMatrix raw_masks = mask_proto * mask_coefs_mat;  // (160 * 160, N)

    // Wrap the raw data into a 3D-aware shape (H, W, Channels)
    cv::Mat mask_stack(MASK_PROTO_H, MASK_PROTO_W, CV_32FC(num_keep), (void*)raw_masks.data());

    // resize
    cv::Mat resized_stack;
    cv::resize(mask_stack, resized_stack, cv::Size(MODEL_W, MODEL_H), 0, 0, cv::INTER_LINEAR);

    // Define the ROI (Region of Interest) to remove letterbox padding
    cv::Rect roi(pad_w_, pad_h_, letterbox_w_, letterbox_h_);

    // Crop and resize
    cv::Mat cropped_stack = resized_stack(roi);
    cv::Mat final_mask;
    cv::resize(cropped_stack, final_mask, cv::Size(ori_w_, ori_h_), 0, 0, cv::INTER_LINEAR);

    // split
    std::vector<cv::Mat> mask_vec;
    cv::split(final_mask, mask_vec);

    // TODO: Optimization:
    // =================================
    // - Scale the Bounding Box coordinates down to the Proto size (160 x 160).
    // - Crop the mask proto first.
    // - Resize only that tiny crop to the bounding box dimensions.
    // - Run findContours on that small patch.
    // - Approximate Polygons (cv::approxPolyDP) cv::findContours often returns hundreds of points
    //
    // For a single object. If you are sending these points over a network or drawing them, this is
    // a major bottleneck. Use the Ramer-Douglas-Peucker algorithm to simplify the shapes.
    // =================================

    // Assign to result.masks
    for (int i = 0; i < mask_vec.size(); ++i) {
        const cv::Mat& mask = mask_vec[i];
        const BBox& box = result.boxes[i];
        int cls_id = box.cls_id;

        // Threshold the float mask to binary (0 or 255)
        cv::Mat binary_mask;
        cv::threshold(mask, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);

        // Create a Rect from your BBox
        // Ensure coordinates are within image bounds to prevent crashes
        int x = std::max(0, (int)box.x_min);
        int y = std::max(0, (int)box.y_min);
        int w = std::min(binary_mask.cols - x, (int)(box.x_max - box.x_min));
        int h = std::min(binary_mask.rows - y, (int)(box.y_max - box.y_min));
        cv::Rect roi_rect(x, y, w, h);

        // Create a global mask and only keep the ROI area
        cv::Mat roi_only_mask = cv::Mat::zeros(binary_mask.size(), binary_mask.type());
        binary_mask(roi_rect).copyTo(roi_only_mask(roi_rect));

        // Find contours on the clipped mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(roi_only_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Convert cv::Point to your custom Point struct
        for (const auto& contour : contours) {

            Mask mask_struct;
            mask_struct.cls_id = cls_id;  // Assign class ID

            // assign points
            for (const auto& p : contour) {
                mask_struct.xy.push_back({p.x, p.y});
            }

            // Add to your results
            if (!mask_struct.xy.empty()) {
                result.masks.push_back(mask_struct);
            }
        }
    }
}
