cmake_minimum_required(VERSION 3.16)
project(mandelbrot CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

find_package(PNGwriter 0.7.0)

add_executable(mandelbrot_cpu mandelbrot.cpp)
add_executable(mandelbrot_gpu mandelbrot.cu)

if(PNGwriter_FOUND)
    target_link_libraries(mandelbrot_cpu PRIVATE PNGwriter::PNGwriter)
    target_link_libraries(mandelbrot_gpu PRIVATE PNGwriter::PNGwriter)
endif(PNGwriter_FOUND)

set_target_properties(mandelbrot_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mandelbrot_gpu PROPERTIES CUDA_ARCHITECTURES 52 61 75)
