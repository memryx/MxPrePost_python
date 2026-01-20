cmake_minimum_required(VERSION 3.14)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(mxpipe_unit_tests
  unit/test_math_utils.cpp
)

target_include_directories(mxpipe_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/extern/MX_API/mx_accl/include
)

target_link_libraries(mxpipe_unit_tests PRIVATE
  mxpipe
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mxpipe_unit_tests)
