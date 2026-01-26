cmake_minimum_required(VERSION 3.14)
project(mxpipe)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

option(MXPIPE_TEST "Enable unit tests (gtest)" OFF)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O0 -g -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -fopenmp -march=native -DNDEBUG")
endif()


file(GLOB local_src src/*.cpp)

find_package(OpenCV REQUIRED)

add_library(mxpipe SHARED ${local_src})

if(MXPIPE_TEST)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

target_include_directories(mxpipe PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    include/
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/MX_API/mx_accl/include
)

target_link_libraries(mxpipe ${OpenCV_LIBS} mx_accl)

# Set RPATH for the build tree so the executable can find 
# the 'mx_accl' shared library without manual LD_LIBRARY_PATH exports.
set_target_properties(mxpipe PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/extern/MX_API/build/mx_accl"
)
