#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::string

#include <map>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include <memx/accl/MxAccl.h>

#include "memx/prepost/MxPrepost.h"

#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace MX::Runtime;
using namespace MX::Prepost;

cv::Mat numpy_to_mat(const py::array& array) {
    py::array arr = py::array::ensure(array, py::array::c_style);
    if (!arr)
        throw std::runtime_error("Input array is not contiguous");

    py::buffer_info info = arr.request();

    if (info.ndim != 2 && info.ndim != 3)
        throw std::runtime_error("Invalid numpy array shape");

    int rows = info.shape[0];
    int cols = info.shape[1];
    int channels = (info.ndim == 3) ? info.shape[2] : 1;

    int type;
    if (info.format == py::format_descriptor<uint8_t>::format())
        type = CV_8UC(channels);
    else if (info.format == py::format_descriptor<float>::format())
        type = CV_32FC(channels);
    else
        throw std::runtime_error("Unsupported dtype");

    // zero-copy conversion
    return cv::Mat(rows, cols, type, info.ptr);
}


// python dict to std::map<int, std::vector<std::string>> (for override_layer_mapping)
std::map<int, std::vector<std::string>> dict_to_layer_mapping(const py::dict& dict) {
    std::map<int, std::vector<std::string>> mapping;
    for (auto item : dict) {
        int key = item.first.cast<int>();
        std::vector<std::string> value = item.second.cast<std::vector<std::string>>();
        mapping[key] = value;
    }
    return mapping;
}

py::array mat_to_numpy(const cv::Mat& mat) {
    // Determine dimensions
    int ndims = mat.dims;
    std::vector<size_t> shape(ndims);
    std::vector<size_t> strides(ndims);

    for (int i = 0; i < ndims; ++i) {
        shape[i] = (size_t)mat.size[i];
        strides[i] = (size_t)mat.step[i];
    }

    // If it's a standard 2D multi-channel image (H, W, C),
    // OpenCV's mat.dims is 2, but numpy expects 3 dimensions.
    if (ndims == 2 && mat.channels() > 1) {
        shape.push_back((size_t)mat.channels());
        strides.push_back((size_t)mat.elemSize1());
    }

    std::string format;
    if (mat.depth() == CV_8U)
        format = py::format_descriptor<uint8_t>::format();
    else if (mat.depth() == CV_32F)
        format = py::format_descriptor<float>::format();
    else
        throw std::runtime_error("Unsupported Mat depth");

    // zero-copy conversion
    return py::array(
            py::buffer_info(mat.data, mat.elemSize1(), format, shape.size(), shape, strides));
}

class BindMxPrepost {
  public:
    BindMxPrepost(py::object pyaccl,
                  const std::string& task,
                  float conf,
                  float iou,
                  std::string classmap_path,
                  std::vector<std::string> custom_class_labels,
                  std::vector<int> valid_classes,
                  bool fast_sigmoid,
                  bool class_agnostic,
                  int model_id,
                  const py::dict& override_layer_mapping) {

        YoloUserConfig config;
        config.conf = conf;
        config.iou = iou;
        config.class_agnostic = class_agnostic;
        config.fast_sigmoid = fast_sigmoid;
        config.model_id = model_id;
        config.override_layer_mapping = dict_to_layer_mapping(override_layer_mapping);

        // convert valid_classes vector to unordered_set
        config.valid_classes =
                std::unordered_set<int>(std::make_move_iterator(valid_classes.begin()),
                                        std::make_move_iterator(valid_classes.end()));

        config.classmap_path = std::move(classmap_path);
        config.custom_class_labels = std::move(custom_class_labels);

        // TODO: check pyaccl type first
        // get PyMxAccl ptr from pyaccl
        py::object ptr = pyaccl.attr("get_raw_ptr")();
        uintptr_t addr = ptr.cast<uintptr_t>();
        MxAccl* accl = reinterpret_cast<MxAccl*>(addr);

        // create MxPrepost using factory method
        prepost_ = MxPrepost::create(accl, task, config);

        // get input shape from model info
        MX::Types::MxModelInfo model_info = accl->get_model_info(config.model_id);
        MX::Types::ShapeVector shape_vec = model_info.in_featuremap_shapes[0];
        for (int i = 0; i < shape_vec.size(); ++i) {
            in_shape_.push_back(shape_vec[i]);
        }
    }

    ~BindMxPrepost() {
        delete prepost_;
    }

    py::array preprocess(const py::array& arr) {

        // convert numpy to cv::Mat
        cv::Mat img = numpy_to_mat(arr);

        // call preprocess
        cv::Mat padded = prepost_->preprocess(img);

        // Reshape for model input
        int sizes[] = {in_shape_[0], in_shape_[1], in_shape_[2], in_shape_[3]};
        cv::Mat reshaped = padded.reshape(1, 4, sizes);

        // convert back to numpy
        return mat_to_numpy(reshaped);
    }

    MX::Prepost::Result postprocess(const std::vector<py::array>& ofmaps) {

        // init ofmap ptrs
        if (ofmap_ptrs_.empty()) {
            ofmap_ptrs_.resize(ofmaps.size());
        }

        // assign ofmap ptrs
        for (int i = 0; i < static_cast<int>(ofmaps.size()); ++i) {
            py::buffer_info info = ofmaps[i].request();
            float* ptr = (float*)info.ptr;
            ofmap_ptrs_[i] = ptr;
        }

        // Force user to provide original image or original shape
        throw std::runtime_error(
                "MxPrepost.postprocess(ofmaps) now requires original image or (ori_w, ori_h).\n"
                "Use:\n"
                "  postprocess(ofmaps, original_image)\n"
                "or\n"
                "  postprocess(ofmaps, ori_w, ori_h)");
    }

    // NEW overload: postprocess(ofmaps, original_image)
    MX::Prepost::Result postprocess(const std::vector<py::array>& ofmaps,
                                    const py::array& original_image) {

        // init ofmap ptrs
        if (ofmap_ptrs_.empty()) {
            ofmap_ptrs_.resize(ofmaps.size());
        }

        // assign ofmap ptrs
        for (int i = 0; i < static_cast<int>(ofmaps.size()); ++i) {
            py::buffer_info info = ofmaps[i].request();
            float* ptr = (float*)info.ptr;
            ofmap_ptrs_[i] = ptr;
        }

        // convert numpy to cv::Mat (original image)
        cv::Mat img = numpy_to_mat(original_image);

        // call postprocess
        MX::Prepost::Result result;
        prepost_->postprocess(ofmap_ptrs_, result, img);  // <-- requires C++ overload
        return result;
    }

    // NEW overload: postprocess(ofmaps, ori_w, ori_h)
    MX::Prepost::Result postprocess(const std::vector<py::array>& ofmaps, int ori_w, int ori_h) {

        // init ofmap ptrs
        if (ofmap_ptrs_.empty()) {
            ofmap_ptrs_.resize(ofmaps.size());
        }

        // assign ofmap ptrs
        for (int i = 0; i < static_cast<int>(ofmaps.size()); ++i) {
            py::buffer_info info = ofmaps[i].request();
            float* ptr = (float*)info.ptr;
            ofmap_ptrs_[i] = ptr;
        }

        // call postprocess
        MX::Prepost::Result result;
        prepost_->postprocess(ofmap_ptrs_, result, ori_w, ori_h);  // <-- requires C++ overload
        return result;
    }
    py::array draw(py::array& arr, const MX::Prepost::Result& result) {
        // convert numpy to cv::Mat
        cv::Mat img = numpy_to_mat(arr);

        // call draw
        prepost_->draw(img, result);

        return mat_to_numpy(img);
    }

  private:
    MxPrepost* prepost_;
    std::vector<float*> ofmap_ptrs_;
    std::vector<int> in_shape_;
};

// helper to safely call import_array()
static int numpy_import_array_wrapper() {
    import_array();  // init numpy array is required in the very beginning
    return 0;
}

PYBIND11_MODULE(mxprepost, m) {
    // helper to safely call import_array(), otherwise got segfault when parsing numpy arrays
    numpy_import_array_wrapper();

    // Register C++ exception → Python exception
    py::register_exception<MX::Prepost::UnsupportedTaskError>(m, "UnsupportedTaskError");

    py::register_exception<MX::Prepost::MxError>(m, "MxError");

    // Result class
    py::class_<MX::Prepost::Result>(m, "Result")
            .def(py::init<>())
            .def_readwrite("boxes", &Result::boxes)
            .def_readwrite("masks", &Result::masks)
            .def_readwrite("keypoints", &Result::keypoints);

    // Box class
    py::class_<MX::Prepost::BBox>(m, "Box")
            .def(py::init<>())
            .def_readwrite("xywh", &BBox::xywh)
            .def_readwrite("xyxy", &BBox::xyxy)
            .def_readwrite("conf", &BBox::conf)
            .def_readwrite("cls_id", &BBox::cls_id)
            .def_readwrite("cls_name", &BBox::cls_name);

    // -----------------------
    // Basic types
    // -----------------------
    py::class_<MX::Prepost::Point2f>(m, "Point2f")
            // .def(py::init<>())
            .def_readwrite("x", &Point2f::x)
            .def_readwrite("y", &Point2f::y);

    // -----------------------
    // Mask
    // -----------------------
    py::class_<MX::Prepost::Mask>(m, "Mask")
            // .def(py::init<>())
            .def_readwrite("xys", &Mask::xys)  // list[Point2f]
            .def_readwrite("cls_id", &Mask::cls_id)
            .def_readwrite("cls_name", &Mask::cls_name);

    // -----------------------
    // Keypoint
    // -----------------------
    py::class_<MX::Prepost::Keypoint>(m, "Keypoint")
            // .def(py::init<>())
            // .def(py::init<Point2f, float>(), py::arg("xy"), py::arg("conf"))
            // .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("conf"))
            .def_readwrite("xy", &Keypoint::xy)  // Point2f
            .def_readwrite("conf", &Keypoint::conf);

    // Prepost class
    py::class_<BindMxPrepost>(m, "MxPrepost")
            .def(py::init<py::object,
                          std::string,
                          float,
                          float,
                          std::string,
                          std::vector<std::string>,
                          std::vector<int>,
                          bool,
                          bool,
                          int,
                          const py::dict&>(),
                 py::arg("accl"),
                 py::arg("task"),
                 py::arg("conf") = 0.3,
                 py::arg("iou") = 0.4,
                 py::arg("classmap_path") = "",
                 py::arg("custom_class_labels") = py::list(),
                 py::arg("valid_classes") = py::list(),
                 py::arg("class_agnostic") = false,
                 py::arg("fast_sigmoid") = false,
                 py::arg("model_id") = 0,
                 py::arg("override_layer_mapping") = py::dict(),
                 R"pbdoc(
Create Prepost.

Args:
  accl (MemryX accl): MemryX accelerator object. 
  task (str): Task for post process. Pass yolov[8|9|10|11]-[det|seg|pose]
  conf (float): Confidence score. [Default is 0.3]
  iou (float): Intersection over Union (IoU) threshold. [Default is 0.4]
  classmap_path (str): The path for a file containing classes separated in each line. [Default using COCO Classes]
  custom_class_labels (list of str): List of custom class labels, provided as list instead of file path.
  valid_classes (list of ints): The classes to consider. [Default is all classes]
  fast_sigmoid (bool): Use fast sigmoid if True. [Default is False]
  class_agnostic (bool): Use class-agnostic NMS if True. [Default is False]
  model_id (int): The ID of the model to be used from MemryX accl. [Default is 0]
  override_layer_mapping (dict): A dictionary mapping stride to list of layer names for coord, conf, mask_coef, keypt ports (if applicable), in case the automatic layer mapping based on output shapes fails. The keys are stride values (e.g., 8, 16, 32) and the values are lists of layer names corresponding to that stride. For example: {16: ["coord_layer_name", "conf_layer_name"], 32: ["coord_layer_name", "conf_layer_name"]}. Use stride=0 for global fmaps such as mask_proto for segmentation. [Default is empty dict, which means auto-mapping will be used]
)pbdoc")
            .def("draw", &BindMxPrepost::draw)
            .def("preprocess", &BindMxPrepost::preprocess)
            // postprocess overloads
            .def("postprocess",
                 py::overload_cast<const std::vector<py::array>&>(&BindMxPrepost::postprocess),
                 py::arg("ofmaps"))
            .def("postprocess",
                 py::overload_cast<const std::vector<py::array>&, const py::array&>(
                         &BindMxPrepost::postprocess),
                 py::arg("ofmaps"),
                 py::arg("original_image"))
            .def("postprocess",
                 py::overload_cast<const std::vector<py::array>&, int, int>(
                         &BindMxPrepost::postprocess),
                 py::arg("ofmaps"),
                 py::arg("ori_w"),
                 py::arg("ori_h"));
}
