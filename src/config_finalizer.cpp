#include "config_finalizer.h"

#include "memx/accl/MxAccl.h"

#include <fstream>
using namespace MX::Runtime;

namespace {  // anonymous namespace for helper functions
    std::vector<std::string> _load_classes(const std::string& file_path) {
        std::vector<std::string> class_labels;
        std::ifstream file(file_path);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << file_path << std::endl;
            return class_labels;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Optional: Skip empty lines
            if (!line.empty()) {
                class_labels.push_back(line);
            }
        }

        file.close();
        return class_labels;
    }
}

YoloFinalConfig ConfigFinalizer::finalize(MX::Runtime::MxAccl* accl, const YoloUserConfig& user) {
    YoloFinalConfig final;

    MX::Types::MxModelInfo model_info = accl->get_model_info(0);
    if (model_info.use_model_shape_in == true) {
        throw std::runtime_error(
                "use_model_shape of input must be false for Yolo models in MxPrepost");
    }

    if (model_info.use_model_shape_out == true) {
        throw std::runtime_error(
                "use_model_shape of output must be false for Yolo models in MxPrepost");
    }

    // Required parameters
    if (user.ori_width <= 0) {
        throw std::invalid_argument("ori_width must be provided for YoloUserConfig.");
    }
    final.ori_width = user.ori_width;

    if (user.ori_height <= 0) {
        throw std::invalid_argument("ori_height must be provided for YoloUserConfig.");
    }
    final.ori_height = user.ori_height;

    // Optional parameters with defaults
    final.conf = user.conf;
    final.iou = user.iou;
    final.class_agnostic = user.class_agnostic;
    final.fast_sigmoid = user.fast_sigmoid;

    // For Class Labels
    if (user.classmap_path.empty()) {
        for (const auto& label : COCO_NAMES) {
            final.class_labels.push_back(label);
        }
    } else {
        // read from file
        final.class_labels = _load_classes(user.classmap_path);
    }

    // For Valid Classes
    if (user.valid_classes.empty()) {
        for (int i = 0; i < final.class_labels.size(); ++i) {
            final.valid_classes.push_back(i);
        }
    } else {
        for (int cls : user.valid_classes) {
            if (cls < 0 || cls >= final.class_labels.size()) {
                throw std::invalid_argument("valid_classes contains invalid class ID: " +
                                            std::to_string(cls));
            }
            final.valid_classes.push_back(cls);
        }
    }

    // Compute letterbox parameters
    final.ori_width = user.ori_width;
    final.ori_height = user.ori_height;

    // Get model input dimensions
    final.model_h = model_info.in_featuremap_shapes[0][0];
    final.model_w = model_info.in_featuremap_shapes[0][1];

    // letterbox params
    final.letterbox_ratio = std::min((float) final.model_w / final.ori_width,
                                     (float) final.model_h / final.ori_height);

    final.letterbox_w = (int)std::round(final.ori_width * final.letterbox_ratio);
    final.letterbox_h = (int)std::round(final.ori_height * final.letterbox_ratio);

    // Clamp to model dims
    final.letterbox_w = std::min(final.letterbox_w, final.model_w);
    final.letterbox_h = std::min(final.letterbox_h, final.model_h);

    int dw = final.model_w - final.letterbox_w;
    int dh = final.model_h - final.letterbox_h;

    // Asymmetric padding (handles odd dw/dh)
    final.pad_left = dw / 2;
    final.pad_right = dw - final.pad_left;
    final.pad_top = dh / 2;
    final.pad_bottom = dh - final.pad_top;

    // (Optional) Keep old symmetric fields for legacy code paths
    final.pad_w = final.pad_left;
    final.pad_h = final.pad_top;

    return final;
}