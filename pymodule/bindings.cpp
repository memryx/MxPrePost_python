#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::string

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include "config.h"
#include "processor.h"
#include "yolov8.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace py = pybind11;

struct Box {
    std::array<float, 4> ltwh;  // (left, top, width, height)
    std::array<float, 4> xywh;  // (x_center, y_center, w, h)
    float conf;                 // confidence conf
    int cls_id;                 // class index
    std::string cls_name;
};

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

    return cv::Mat(rows, cols, type, info.ptr);
}

/* zero copy */
py::array mat_to_numpy(const cv::Mat& mat) {
    std::vector<size_t> shape = {(size_t)mat.rows, (size_t)mat.cols, (size_t)mat.channels()};

    std::vector<size_t> strides = {(size_t)mat.step, (size_t)mat.elemSize(), (size_t)mat.elemSize1()};

    std::string format;
    if (mat.depth() == CV_8U)
        format = py::format_descriptor<uint8_t>::format();
    else if (mat.depth() == CV_32F)
        format = py::format_descriptor<float>::format();
    else
        throw std::runtime_error("Unsupported Mat depth");

    return py::array(py::buffer_info(mat.data, mat.elemSize1(), format, 3, shape, strides));
}

class Pipeline {
  public:
    Pipeline(const std::string& task, int ori_width, int ori_height, float conf_thres = 0.3, float iou_thres = 0.4) {
        YoloDetectConfig config{ori_width, ori_height, conf_thres, iou_thres};
        processor_ = Processor::create(task, config);
    }

    ~Pipeline() {
        delete processor_;
    }

    py::array preprocess(const py::array& arr) {
        cv::Mat img = numpy_to_mat(arr);
        cv::Mat padded = processor_->preprocess(img);
        return mat_to_numpy(padded);
    }

    std::vector<Box> postprocess(const std::vector<py::array>& ofmaps) {
        // TODO: preallocate ofmap_ptrs
        std::vector<float*> ofmap_ptrs;
        for (const auto& ofmap : ofmaps) {
            py::buffer_info info = ofmap.request();
            float* ptr = (float*)info.ptr;
            ofmap_ptrs.push_back(ptr);
        }

        // call real postrocess
        // TODO: remove YOLOv8Result intermediate structure
        YOLOv8Result mid_res;
        processor_->postprocess(ofmap_ptrs, mid_res);

        // construct final results
        std::vector<Box> results;
        while (mid_res.bboxes.empty() == false) {
            BBox bbox = mid_res.bboxes.front();
            mid_res.bboxes.pop();

            Box box;
            // box.xywh = {
            //     (bbox.x_min + bbox.x_max) / 2.0f,
            //     (bbox.y_min + bbox.y_max) / 2.0f,
            //     (bbox.x_max - bbox.x_min),
            //     (bbox.y_max - bbox.y_min)};
            box.xywh = {bbox.x_min, bbox.y_min, (bbox.x_max - bbox.x_min), (bbox.y_max - bbox.y_min)};
            box.conf = bbox.class_score;
            box.cls_id = bbox.class_index;

            // TODO: map class_index to class_name
            box.cls_name = "people";

            results.push_back(box);
        }

        return results;
    }

  private:
    Processor* processor_;
};

// helper to safely call import_array()
static int numpy_import_array_wrapper() {
    import_array();  // init numpy array is required in the very beginning
    return 0;
}

PYBIND11_MODULE(mxproc, m) {
    // helper to safely call import_array(), otherwise got segfault when parsing numpy arrays
    numpy_import_array_wrapper();

    // Box class
    py::class_<Box>(m, "Box")
            .def(py::init<>())
            .def_readwrite("xywh", &Box::xywh)
            .def_readwrite("conf", &Box::conf)
            .def_readwrite("cls_id", &Box::cls_id)
            .def_readwrite("cls_name", &Box::cls_name);

    // Pipeline class
    py::class_<Pipeline>(m, "Pipeline")
            .def(py::init<std::string, int, int, float, float>(),
                 py::arg("task"),
                 py::arg("ori_width"),
                 py::arg("ori_height"),
                 py::arg("conf_thres") = 0.3f,
                 py::arg("iou_thres") = 0.4f)
            .def("preprocess", &Pipeline::preprocess)
            .def("postprocess", &Pipeline::postprocess);
}
