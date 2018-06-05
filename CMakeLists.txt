cmake_minimum_required(VERSION 3.9)
project(FloatX)

option(BUILD_TESTS "Generate build files for unit tests" ON)
option(BUILD_CUDA_TESTS "Generate build files for CUDA unit tests" OFF)
option(BUILD_EXHAUSTIVE_TESTS "Generate build files for exhaustive tests" OFF)
option(DEVEL_TOOLS "Include development tools in build system" ON)
option(BUILD_EXAMPLES "Build examples in the example/ directory" ON)

if(BUILD_CUDA_TESTS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 11)
    # silence false-positive CUDA warnings
    set(CMAKE_CUDA_FLAGS "--disable-warnings ${CMAKE_CUDA_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(third_party)  # third party tools and libraries

add_library(floatx INTERFACE)
target_include_directories(floatx INTERFACE src/)

if(BUILD_TESTS OR BUILD_CUDA_TESTS OR BUILD_EXHAUSTIVE_TESTS)
    enable_testing()
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

if(BUILD_CUDA_TESTS)
    add_subdirectory(cuda_test)
endif()

if(BUILD_EXHAUSTIVE_TESTS)
    add_subdirectory(testx)
endif()


if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
