#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::string

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include "memx/accl/MxAccl.h"
#include "pipeline.h"

#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace MX::Pipe;

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

py::array mat_to_numpy(const cv::Mat& mat) {
    std::vector<size_t> shape = {(size_t)mat.rows, (size_t)mat.cols, (size_t)mat.channels()};

    std::vector<size_t> strides = {
            (size_t)mat.step, (size_t)mat.elemSize(), (size_t)mat.elemSize1()};

    std::string format;
    if (mat.depth() == CV_8U)
        format = py::format_descriptor<uint8_t>::format();
    else if (mat.depth() == CV_32F)
        format = py::format_descriptor<float>::format();
    else
        throw std::runtime_error("Unsupported Mat depth");

    // zero-copy conversion
    return py::array(py::buffer_info(mat.data, mat.elemSize1(), format, 3, shape, strides));
}

class BindPipeline {
  public:
    BindPipeline(py::object pyaccl,
                 const std::string& task,
                 int ori_width,
                 int ori_height,
                 float conf,
                 float iou,
                 std::string classmap_path,
                 std::vector<int> valid_classes,
                 bool fast_sigmoid) {

        YoloUserConfig config;
        config.ori_width = ori_width;
        config.ori_height = ori_height;
        config.conf = conf;
        config.iou = iou;
        config.fast_sigmoid = fast_sigmoid;

        // convert valid_classes vector to unordered_set
        config.valid_classes =
                std::unordered_set<int>(std::make_move_iterator(valid_classes.begin()),
                                        std::make_move_iterator(valid_classes.end()));

        config.classmap_path = std::move(classmap_path);

        // TODO: check pyaccl type first
        // get PyMxAccl ptr from pyaccl
        py::object ptr = pyaccl.attr("get_raw_ptr")();
        uintptr_t addr = ptr.cast<uintptr_t>();
        MX::Runtime::MxAccl* accl = reinterpret_cast<MX::Runtime::MxAccl*>(addr);

        // create pipeline using factory method
        pipeline_ = Pipeline::create(accl, task, config);
    }

    ~BindPipeline() {
        delete pipeline_;
    }

    py::array preprocess(const py::array& arr) {

        // convert numpy to cv::Mat
        cv::Mat img = numpy_to_mat(arr);

        // call preprocess
        cv::Mat padded = pipeline_->preprocess(img);

        // convert back to numpy
        return mat_to_numpy(padded);
    }

    MX::Pipe::Result postprocess(const std::vector<py::array>& ofmaps) {

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

        // call postrocess
        MX::Pipe::Result result;
        pipeline_->postprocess(ofmap_ptrs_, result);
        return result;
    }

    py::array draw(py::array& arr, const MX::Pipe::Result& result) {
        // convert numpy to cv::Mat
        cv::Mat img = numpy_to_mat(arr);

        // call draw
        pipeline_->draw(img, result);

        return mat_to_numpy(img);
    }

  private:
    Pipeline* pipeline_;
    std::vector<float*> ofmap_ptrs_;
};

// helper to safely call import_array()
static int numpy_import_array_wrapper() {
    import_array();  // init numpy array is required in the very beginning
    return 0;
}

PYBIND11_MODULE(mxpipe, m) {
    // helper to safely call import_array(), otherwise got segfault when parsing numpy arrays
    numpy_import_array_wrapper();

    // Result class
    py::class_<MX::Pipe::Result>(m, "Result")
            .def(py::init<>())
            .def_readwrite("boxes", &Result::boxes)
            .def_readwrite("masks", &Result::masks)
            .def_readwrite("keypoints", &Result::keypoints);

    // Box class
    py::class_<MX::Pipe::BBox>(m, "Box")
            .def(py::init<>())
            .def_readwrite("xywh", &BBox::xywh)
            .def_readwrite("conf", &BBox::conf)
            .def_readwrite("cls_id", &BBox::cls_id)
            .def_readwrite("cls_name", &BBox::cls_name);

    // Pipeline class
    py::class_<BindPipeline>(m, "Pipeline")
            .def(py::init<py::object,
                          std::string,
                          int,
                          int,
                          float,
                          float,
                          std::string,
                          std::vector<int>,
                          bool>(),
                 py::arg("accl"),
                 py::arg("task"),
                 py::arg("ori_width"),
                 py::arg("ori_height"),
                 py::arg("conf") = 0.3,
                 py::arg("iou") = 0.4,
                 py::arg("classmap_path") = "",
                 py::arg("valid_classes") = py::list(),
                 py::arg("fast_sigmoid") = false,
                 R"doc(
Create Pipeline.

Args:
  accl (MemryX accl): MemryX accelerator object. 
  task (str): Task for post process. Pass yolov[8|9|10|11]_[det|seg|pose]
  ori_width (int): Original width of the image.
  ori_height (int): Original height of the image.
  conf (float): Confidence score. [Default is 0.3]
  iou (float): Intersection over Union (IoU) threshold. [Default is 0.4]
  classmap_path (str): The path for a file containing classes separated in each line. [Default using COCO Classes]
  valid_classes (list of ints): The classes to consider. [Default is all classes]
  fast_sigmoid (bool): Use fast sigmoid if True. [Default is False]
)doc")
            .def("draw", &BindPipeline::draw)
            .def("preprocess", &BindPipeline::preprocess)
            .def("postprocess", &BindPipeline::postprocess);
}
