cmake_minimum_required(VERSION 3.14)
project(mxproc)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

if(BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O0 -g -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -fopenmp")
endif()


file(GLOB local_src src/*.cpp)

find_package(OpenCV REQUIRED)

add_library(mxproc SHARED ${local_src})

target_link_libraries(mxproc ${OpenCV_LIBS})

target_include_directories(mxproc
    PRIVATE ${OpenCV_INCLUDE_DIRS} include/
)
