#include "config_finalizer.h"

using namespace MX::Pipe;

YoloFinalConfig ConfigFinalizer::finalize(const YoloUserConfig& user) {
    YoloFinalConfig final;

    // Required parameters
    if (user.ori_width <= 0) {
        throw std::invalid_argument("ori_width must be a positive integer.");
    }
    final.ori_width = user.ori_width;

    if (user.ori_height <= 0) {
        throw std::invalid_argument("ori_height must be a positive integer.");
    }
    final.ori_height = user.ori_height;

    // Optional parameters with defaults
    final.conf = user.conf;
    final.iou = user.iou;
    final.fast_sigmoid = user.fast_sigmoid;

    // For Class Labels
    if (user.class_labels.empty()) {
        for (const auto& label : COCO_NAMES) {
            final.class_labels.push_back(label);
        }

    } else {
        for (const auto& label : user.class_labels) {
            final.class_labels.push_back(label);
        }
    }

    // For Valid Classes
    if (user.valid_classes.empty()) {

        for (int i = 0; i < final.class_labels.size(); ++i) {
            final.valid_classes.push_back(i);
        }
    } else {
        for (int cls : user.valid_classes) {
            final.valid_classes.push_back(cls);
        }
    }

    // Compute letterbox parameters
    final.ori_width = user.ori_width;
    final.ori_height = user.ori_height;

    // letterbox params
    final.letterbox_ratio =
            std::min((float)MODEL_W / user.ori_width, (float)MODEL_H / user.ori_height);

    final.letterbox_w = final.ori_width * final.letterbox_ratio;
    final.letterbox_h = final.ori_height * final.letterbox_ratio;

    final.pad_w = (MODEL_W - final.letterbox_w) / 2;
    final.pad_h = (MODEL_H - final.letterbox_h) / 2;

    return final;
}