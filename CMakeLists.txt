cmake_minimum_required(VERSION 3.14)
project(mxpipe)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O0 -g -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -fopenmp -march=native -DNDEBUG")
endif()


file(GLOB local_src src/*.cpp)

find_package(OpenCV REQUIRED)

add_library(mxpipe SHARED ${local_src})

target_link_libraries(mxpipe ${OpenCV_LIBS} mx_accl)

target_include_directories(mxpipe PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    include/
)
