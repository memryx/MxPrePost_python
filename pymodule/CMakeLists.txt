## original cmake
cmake_minimum_required(VERSION 3.14)
project(mxpipe)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O0 -g -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -fopenmp")
endif()

# You can also use find_package(pybind11 REQUIRED) if installed system-wide
add_subdirectory(pybind11)  # Clone from https://github.com/pybind/pybind11

# This will automatically name the output module "mxpipe.<python_abi_tag>.so"
pybind11_add_module(mxpipe bindings.cpp)

# find the path to the site-packages/numpy/core/include/ directory for the currently
# active Python environment, and add that to the target_include_directories
execute_process(
    COMMAND python3 -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(OpenCV REQUIRED)

# Include directories
target_include_directories(mxpipe PRIVATE
                           ${CMAKE_SOURCE_DIR}
                           ${PYTHON_NUMPY_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS}
                           /usr/include/
                           ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(mxpipe PRIVATE ${OpenCV_LIBS} ${CMAKE_CURRENT_SOURCE_DIR}/../build/libmxpipe.so)