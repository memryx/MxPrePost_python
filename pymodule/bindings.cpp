#include <pybind11/pybind11.h>
#include <pybind11/functional.h> // for std::function
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>        // for py::bytes
#include <pybind11/stl.h>            // for std::string
#include <pybind11/stl/filesystem.h> // for std::filesystem::path

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include <vector>
#include <iostream>

namespace py = pybind11;

struct Box
{
    // Bounding box in (x, y, width, height)
    std::array<float, 4> xywh;

    float conf; // confidence conf
    int cls_id; // class index
    std::string cls_name;
};

class Pipeline
{
public:
    Pipeline(int start = 0) : value_(start) {}

    std::vector<Box> postprocess(const std::vector<py::object> &mxa_output)
    {
        std::vector<Box> results;

        std::cout << "getting mxa_output of size: "
                  << mxa_output.size() << std::endl;

        // Example dummy output
        Box box;
        box.xywh = {10.f, 20.f, 100.f, 200.f};
        box.conf = 0.95f;
        box.cls_id = 1;
        box.cls_name = "person";

        results.push_back(box);
        return results;
    }

private:
    int value_;
};

// helper to safely call import_array()
static int numpy_import_array_wrapper()
{
    import_array(); // init numpy array is required in the very beginning
    return 0;
}

PYBIND11_MODULE(mxpipe, m)
{
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
        .def(py::init<int>(), py::arg("start") = 0)
        .def("postprocess", &Pipeline::postprocess);
}
