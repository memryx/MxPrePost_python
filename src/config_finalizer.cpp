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

    MX::Types::MxModelInfo model_info = accl->get_model_info(user.model_id);
    if (model_info.use_model_shape_in == true) {
        throw std::runtime_error(
                "use_model_shape of input must be false for Yolo models in MxPrepost");
    }

    if (model_info.use_model_shape_out == true) {
        throw std::runtime_error(
                "use_model_shape of output must be false for Yolo models in MxPrepost");
    }

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

    // Get model input dimensions
    final.model_h = model_info.in_featuremap_shapes[0][0];
    final.model_w = model_info.in_featuremap_shapes[0][1];

    return final;
}