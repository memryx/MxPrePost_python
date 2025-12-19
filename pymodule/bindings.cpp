#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::string

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include "pipeline.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

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
    BindPipeline(const std::string& task,
                 int ori_width,
                 int ori_height,
                 float conf_thres = 0.3,
                 float iou_thres = 0.4) {

        // TODO: factory method for config
        YoloDetectConfig config{ori_width, ori_height, conf_thres, iou_thres};
        pipeline_ = Pipeline::create(task, config);
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
            .def(py::init<std::string, int, int, float, float>(),
                 py::arg("task"),
                 py::arg("ori_width"),
                 py::arg("ori_height"),
                 py::arg("conf_thres") = 0.3f,
                 py::arg("iou_thres") = 0.4f)
            .def("draw", &BindPipeline::draw)
            .def("preprocess", &BindPipeline::preprocess)
            .def("postprocess", &BindPipeline::postprocess);
}
