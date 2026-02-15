cmake_minimum_required(VERSION 3.10)
project(yolo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

file(GLOB local_src "*.cpp" "*.c")

# Get first cpp file and extract filename without extension
list(GET local_src 0 MAIN_SRC)
get_filename_component(APP_NAME ${MAIN_SRC} NAME_WE)

add_executable(${APP_NAME} ${local_src})

target_include_directories(${APP_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ROOT_DIR}/include
)

target_link_directories(${APP_NAME} PRIVATE
    ${ROOT_DIR}/build
)

target_link_libraries(${APP_NAME} PRIVATE
    mx_accl
    mxprepost
    ${OpenCV_LIBS}
    Threads::Threads
    dl
)

set_target_properties(${APP_NAME} PROPERTIES
    BUILD_RPATH "${ROOT_DIR}/build"
)
