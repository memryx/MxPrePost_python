cmake_minimum_required(VERSION 3.14)
project(mxprepost)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_X86_FLAGS "-mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mfxsr -mcx16 -msahf -mpclmul -mfsgsbase -mmovbe -mprfchw -mxsave -mxsaveopt -mavx -mavx2 -mfma -mbmi -mbmi2 -maes -mf16c -mlzcnt -madx -mxsavec -mxsaves -mclflushopt -mtune=generic")
set(CMAKE_ARM_FLAGS "-march=armv8-a+simd -mtune=cortex-a76")

# Compiler flags
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O0 -g -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -fopenmp -DNDEBUG -ftree-vectorize -fsimd-cost-model=unlimited -fno-math-errno -funsafe-math-optimizations -ftree-loop-ivcanon -ftree-loop-im -ffinite-math-only -fno-signed-zeros -fno-trapping-math -fno-signaling-nans -fcx-limited-range")

    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_X86_FLAGS}")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ARM_FLAGS}")
    else()
        message(WARNING "Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}. No specific optimization flags will be set.")
    endif()
    
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
)

target_link_libraries(mxprepost ${OpenCV_LIBS} mx_accl yaml-cpp)


add_subdirectory(samples/cpp)
