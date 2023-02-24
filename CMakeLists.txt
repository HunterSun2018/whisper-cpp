cmake_minimum_required(VERSION 3.10)

project(whisper)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

set(CMAKE_PREFIX_PATH ~/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#set(CMAKE_C_COMPILER clang-12)
set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED true)

#find_package(Torch REQUIRED)
#include_directories(/usr/include/torch/csrc/api/include/)

aux_source_directory(src SRC)
add_executable(whisper ${SRC})

target_link_libraries(whisper "${TORCH_LIBRARIES}")
set_property(TARGET whisper PROPERTY CXX_STANDARD 14)