#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

class Counter {
public:
    Counter(int start = 0) : value_(start) {}

    void increment() {
        value_++;
        std::cout << "Counter incremented to: " << value_ << std::endl;
    }

    int value() const {
        return value_;
    }

private:
    int value_;
};

PYBIND11_MODULE(mxpipe, m) {
    m.doc() = "Minimal pybind11 example";

    py::class_<Counter>(m, "Counter")
        .def(py::init<int>(), py::arg("start") = 0)
        .def("increment", &Counter::increment)
        .def("value", &Counter::value);
}
