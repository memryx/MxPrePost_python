#include "yolo10_detect.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost::Util;

Yolo10Detect::Yolo10Detect(MX::Runtime::MxAccl* accl,
                           const YoloUserConfig& user_cfg,
                           const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    //------------------------------------------------------------------------------------------------
    auto model_info = accl->get_model_info(cfg_.model_id);

    yolo_post_layers_.resize(NUM_LAYERS);

    std::vector<MX::Types::ShapeVector> expected_coord_shapes;
    std::vector<MX::Types::ShapeVector> expected_conf_shapes;

    for(int i=0; i < NUM_LAYERS; ++i) {
        int stride = STRIDES[i];
        expected_coord_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, COORD_FMAP_SIZE});
        expected_conf_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, static_cast<int64_t>(cfg_.class_labels.size())});
    }

    // in case the user is forcing a particular mapping
    if(cfg_.override_layer_mapping.empty()){
        // match expected shapes with actual model input shapes to find the correct ports for each layer
        int num_ofmaps = model_info.out_featuremap_shapes.size();

        if(cfg_.class_labels.size() != COORD_FMAP_SIZE){
            for (int i=0; i < NUM_LAYERS; ++i){
                yolo_post_layers_[i].height = cfg_.model_h / STRIDES[i];
                yolo_post_layers_[i].width = cfg_.model_w / STRIDES[i];
                yolo_post_layers_[i].stride = STRIDES[i];

                bool found_coord = false;
                bool found_conf = false;
                for (int port = 0; port < num_ofmaps; ++port) {
                    const auto& actual_shape = model_info.out_featuremap_shapes[port];
                    if (actual_shape == expected_coord_shapes[i]) {
                        yolo_post_layers_[i].coord_port = port;
                        found_coord = true;
                    } else if (actual_shape == expected_conf_shapes[i]) {
                        yolo_post_layers_[i].conf_port = port;
                        found_conf = true;
                    }
                }

                if(!found_coord) {
                    throw std::runtime_error("Could not find coordinate output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_coord_shapes[i].to_string());
                }
                if(!found_conf) {
                    throw std::runtime_error("Could not find confidence output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_conf_shapes[i].to_string());
                }

            }
        }
        else {
            // warning for the corner case
            std::cerr << "WARNING: Number of classes (" << cfg_.class_labels.size() << ") is equal to COORD_FMAP_SIZE (" << COORD_FMAP_SIZE << "). "
                      << "This may cause ambiguity in automatically identifying coordinate and confidence ports based on output shapes. "
                      << "Please provide an explicit mapping using the override_layer_mapping entry in YoloUserConfig."
                      << std::endl;

            // fallback to guesstimation based on port order (coord then conf)
            for (int i=0; i < NUM_LAYERS; ++i){
                yolo_post_layers_[i].height = cfg_.model_h / STRIDES[i];
                yolo_post_layers_[i].width = cfg_.model_w / STRIDES[i];
                yolo_post_layers_[i].stride = STRIDES[i];

                yolo_post_layers_[i].coord_port = i * 2; // even ports for coord
                yolo_post_layers_[i].conf_port = i * 2 + 1; // odd ports for conf
            }

            // double check that our guessed shapes match the expected shapes, if not, throw an error
            for (int i=0; i < NUM_LAYERS; ++i){
                const auto& coord_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].coord_port];
                const auto& conf_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].conf_port];

                if (coord_shape != expected_coord_shapes[i]) {
                    throw std::runtime_error("Guessed coordinate port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed coord port: " + std::to_string(yolo_post_layers_[i].coord_port) + " with shape " + coord_shape.to_string() + " and expected shape " + expected_coord_shapes[i].to_string());
                }
                if (conf_shape != expected_conf_shapes[i]) {
                    throw std::runtime_error("Guessed confidence port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed conf port: " + std::to_string(yolo_post_layers_[i].conf_port) + " with shape " + conf_shape.to_string() + " and expected shape " + expected_conf_shapes[i].to_string());
                }
            }
        }

    }
    else {
        // user provided mapping ( <stride (int), vect<coord name, conf name>> )
        // parse the model_info to find the ports corresponding to the provided layer_names,
        // then get assign those ports to the correct layer in yolo_post_layers_ based on the stride,
        // then double check that the shapes match the expected shapes for that stride/layer
        int num_found_layers = 0;
        for (const auto& [stride, layer_names] : cfg_.override_layer_mapping) {
            int layer_id = -1;
            for (int i=0; i < NUM_LAYERS; ++i) {
                if (STRIDES[i] == stride) {
                    layer_id = i;
                    break;
                }
            }

            // invalid stride
            if (layer_id == -1) {
                std::string error_message = "Invalid stride " + std::to_string(stride) + " in override_layer_mapping. Accepted STRIDES are: ";
                for (size_t i = 0; i < STRIDES.size(); ++i) {
                    error_message += std::to_string(STRIDES[i]);
                    if (i != STRIDES.size() - 1) {
                        error_message += ", ";
                    }
                }
                throw std::runtime_error(error_message);
            }

            // assigns based on expected shapes
            yolo_post_layers_[layer_id].height = cfg_.model_h / stride;
            yolo_post_layers_[layer_id].width = cfg_.model_w / stride;
            yolo_post_layers_[layer_id].stride = stride;

            bool found_coord = false;
            bool found_conf = false;
            for (unsigned int port = 0; port < model_info.out_featuremap_shapes.size(); ++port) {
                const auto& actual_shape = model_info.out_featuremap_shapes[port];
                // find the output port by name AND matching shape
                if (model_info.output_layer_names[port] == layer_names[0]) {
                    if (actual_shape == expected_coord_shapes[layer_id]) {
                        yolo_post_layers_[layer_id].coord_port = port;
                        found_coord = true;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[0] + " has shape "
                            + actual_shape.to_string() + " which does not match expected coordinate shape for stride " + std::to_string(stride) + " which is "
                            + expected_coord_shapes[layer_id].to_string();
                        throw std::runtime_error(error_message);
                    }
                } else if (model_info.output_layer_names[port] == layer_names[1]) {
                    if (actual_shape == expected_conf_shapes[layer_id]) {
                        yolo_post_layers_[layer_id].conf_port = port;
                        found_conf = true;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[1] + " has shape "
                            + actual_shape.to_string() + " which does not match expected confidence shape for stride " + std::to_string(stride) + " which is "
                            + expected_conf_shapes[layer_id].to_string();
                        throw std::runtime_error(error_message);
                    }
                }
            }

            if(!found_coord) {
                throw std::runtime_error("Could not find coordinate output port for layer with stride " + std::to_string(stride));
            }
            if(!found_conf) {
                throw std::runtime_error("Could not find confidence output port for layer with stride " + std::to_string(stride));
            }

            num_found_layers++;
        }

        if(num_found_layers != NUM_LAYERS) {
            throw std::runtime_error("override_layer_mapping must contain entries for all " + std::to_string(NUM_LAYERS) + " layers. Found entries for " + std::to_string(num_found_layers) + " layers.");
        }

    }
    //------------------------------------------------------------------------------------------------
}

cv::Mat Yolo10Detect::preprocess(const cv::Mat& image) {
    const int ori_w = image.cols;
    const int ori_h = image.rows;

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    return MX::Prepost::Util::preprocess(image,
                                         lb.letterbox_w,
                                         lb.letterbox_h,
                                         lb.pad_left,
                                         lb.pad_top,
                                         lb.pad_right,
                                         lb.pad_bottom);
}

void Yolo10Detect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void Yolo10Detect::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void Yolo10Detect::postprocess(const std::vector<float*>& outputs,
                               Result& result,
                               const cv::Mat& original_image) {
    if (UNLIKELY(original_image.empty())) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.cols, original_image.rows);
}

void Yolo10Detect::postprocess(const std::vector<float*>& outputs,
                               Result& result,
                               int ori_w,
                               int ori_h) {
    if (UNLIKELY(ori_w <= 0 || ori_h <= 0)) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_w, ori_h);
}

void Yolo10Detect::postprocess_impl(const std::vector<float*>& outputs,
                                    Result& result,
                                    int ori_w,
                                    int ori_h) {

    // FIXME: this is all wrong

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);
    // Candidate Gathering
    std::vector<BBox> all_boxes;
    all_boxes.reserve(total_preds_);
    for (size_t layer_id = 0; layer_id < NUM_LAYERS; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // get best label and score
            // NOTE: use inv_conf_thres here because score is still raw (not applied sigmoid yet).
            // sigmoid is expensive and we only apply it if needed.
            float best_score;
            int best_label =
                    MX::Prepost::Util::get_best_label(best_score,
                                                      conf_base + i * cfg_.class_labels.size(),
                                                      cfg_.valid_classes,
                                                      smgr_->inv_conf_thres);

            // no label with sufficient score
            if (best_label == -1)
                continue;

            // convert best score in [0,1] (e.g. apply sigmoid)
            best_score = smgr_->convert(best_score);

            // decode bbox (Distribution Focal Loss)
            std::array<float, 4> coord = MX::Prepost::Util::dfl(coord_base + i * COORD_FMAP_SIZE,
                                                                i / layer.width /* row */,
                                                                i % layer.width /* col */,
                                                                layer.stride,
                                                                cfg_.model_w,
                                                                cfg_.model_h);

            // convert to raw bbox coords
            coord[0] = (coord[0] - lb.pad_left) / lb.ratio;
            coord[1] = (coord[1] - lb.pad_top) / lb.ratio;
            coord[2] = (coord[2] - lb.pad_left) / lb.ratio;
            coord[3] = (coord[3] - lb.pad_top) / lb.ratio;

            // store bbox
            all_boxes.emplace_back(coord[0],
                                   coord[1],
                                   coord[2],
                                   coord[3],
                                   best_score,
                                   best_label,
                                   cfg_.class_labels[best_label]);
        }
    }

    // early exit
    int n_bboxes = all_boxes.size();
    if (n_bboxes == 0)
        return;

    // Ensure boxes are sorted based on confidence.
    std::sort(all_boxes.begin(), all_boxes.end(), [](const BBox& a, const BBox& b) {
        return a.conf > b.conf;
    });

    result.boxes.reserve(n_bboxes);
    for (BBox bbox : all_boxes) {
        result.boxes.push_back(bbox);
    }
}
