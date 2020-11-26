cmake_minimum_required(VERSION 3.14)

project(cuda)
set(CMAKE_CXX_STANDARD 11)
enable_language(CUDA)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60>")

include(GNUInstallDirs)

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)
enable_testing()
