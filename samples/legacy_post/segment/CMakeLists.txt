cmake_minimum_required(VERSION 3.10)

get_filename_component(app_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O0 -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

include_directories(${OpenCV_INCLUDE_DIRS}
                    )

file(GLOB local_src
    "*.c"
    "*.cpp"
	)

add_executable(${app_name} ${local_src})

target_link_libraries(${app_name}  mx_accl  ${OpenCV_LIBS} mxutils_gui)
target_link_libraries(${app_name} Qt5::Widgets Qt5::Core Qt5::Gui)