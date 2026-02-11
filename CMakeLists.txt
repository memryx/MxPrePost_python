cmake_minimum_required(VERSION 3.14)
project(mxprepost)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

# Compiler flags
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O0 -g -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -fopenmp -march=native -DNDEBUG")
endif()


file(GLOB local_src src/*.cpp)

find_package(OpenCV REQUIRED)

# Use FetchContent to get yaml-cpp instead of system package
include(FetchContent)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0  # Use a stable release tag
)
FetchContent_MakeAvailable(yaml-cpp)

# Embed model-config.yaml into the binary
set(MODEL_CONFIG_YAML "${CMAKE_CURRENT_SOURCE_DIR}/config/model-config.yaml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${MODEL_CONFIG_YAML}")

file(READ "${MODEL_CONFIG_YAML}" MODEL_CONFIG_YAML_CONTENT)

# Generate the .cpp file with embedded YAML (in the build dir; no subdirectory)
set(MODEL_CONFIG_EMBEDDED_CPP "${CMAKE_CURRENT_BINARY_DIR}/model_config_embedded.cpp")

file(WRITE "${MODEL_CONFIG_EMBEDDED_CPP}"
"// Auto-generated file - do not edit
#include <string>

namespace MX::Prepost::Config {
    const std::string& getModelConfigYaml() {
        static const std::string yaml_content = R\"mxprepost_yaml(
${MODEL_CONFIG_YAML_CONTENT}
)mxprepost_yaml\";
        return yaml_content;
    }
}
")

add_library(mxprepost SHARED ${local_src} "${MODEL_CONFIG_EMBEDDED_CPP}")

# --------------------------------------------------
# Testing logic:
# - Default: OFF
# - Auto-ON in Debug
# - User override always wins
# --------------------------------------------------
if(NOT DEFINED BUILD_TESTING)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(BUILD_TESTING ON CACHE BOOL "Enable unit tests" FORCE)
    else()
        set(BUILD_TESTING OFF CACHE BOOL "Enable unit tests" FORCE)
    endif()
endif()

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

target_include_directories(mxprepost PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    include/
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/MX_API/mx_accl/include
)

target_link_libraries(mxprepost ${OpenCV_LIBS} mx_accl yaml-cpp)

# Set RPATH for the build tree so the executable can find 
# the 'mx_accl' shared library without manual LD_LIBRARY_PATH exports.
set_target_properties(mxprepost PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/extern/MX_API/build/mx_accl"
)
