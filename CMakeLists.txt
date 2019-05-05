cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GP_GPU LANGUAGES CXX CUDA)

if ($ENV{CLION_IDE})
    include_directories(/usr/local/cuda-9.0/include)
endif ()

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")

add_library(GPU
        cudaMain.cu
        cudaMain.h
        cudaHeaders.h
        )

add_executable(GP_GPU
        main.cpp
        )

set_target_properties(
        GP_GPU
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

target_link_libraries(GP_GPU GPU)

set(CMAKE_CXX_FLAGS "-std=c++14")