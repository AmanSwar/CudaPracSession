cmake_minimum_required(VERSION 3.18)
project(softmax LANGUAGES CXX CUDA)

# enforce C++17 and CUDA separable compilation
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -G -g")
set(Torch_DIR "/home/aman/.local/lib/python3.12/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)


add_executable(softmax
    4.cu
)

target_include_directories(softmax PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(softmax PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Ensure at runtime it can locate libtorch shared objects
set_target_properties(softmax PROPERTIES
    BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
)
