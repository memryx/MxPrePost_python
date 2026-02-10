## original cmake
cmake_minimum_required(VERSION 3.14)
project(mxprepost)

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

# This will automatically name the output module "mxprepost.<python_abi_tag>.so"

file(GLOB mxprepost_src ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp)
pybind11_add_module(mxprepost bindings.cpp ${mxprepost_src})

# find the path to the site-packages/numpy/core/include/ directory for the currently
# active Python environment, and add that to the target_include_directories
execute_process(
    COMMAND python3 -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(OpenCV REQUIRED)

# Use FetchContent to get yaml-cpp instead of system package
include(FetchContent)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0  # Use a stable release tag
)
FetchContent_MakeAvailable(yaml-cpp)

# Include directories
target_include_directories(mxprepost PRIVATE
                           ${PYTHON_NUMPY_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS}
                           ${CMAKE_CURRENT_SOURCE_DIR}/../include
                           ${CMAKE_CURRENT_SOURCE_DIR}/../extern/MX_API/mx_accl/include)

target_link_libraries(mxprepost PRIVATE ${OpenCV_LIBS} mx_accl yaml-cpp)

# Set RPATH for the build tree so the executable can find 
# the 'mx_accl' shared library without manual LD_LIBRARY_PATH exports.
set_target_properties(mxprepost PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../extern/MX_API/build/mx_accl"
)