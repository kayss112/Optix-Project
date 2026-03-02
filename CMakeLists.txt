cmake_minimum_required(VERSION 3.18)
project(my_optix_hello LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(GLFW_DIR "E:/OpenGL/OpenGLLibs")
find_package(CUDAToolkit REQUIRED)

set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0" CACHE PATH "PATH to OptiX SDK")
include(${CMAKE_SOURCE_DIR}/common/gdt/cmake/configure_optix.cmake)
include_directories(${OptiX_INSTALL_DIR}/include ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/common/gdt ${GLFW_DIR}/includes)
embed_ptx(
    OUTPUT_TARGET   devicePrograms_embedded
    SOURCES         "${CMAKE_SOURCE_DIR}/Source Files/devicePrograms.cu"
    EMBEDDED_SYMBOL_NAMES  devicePrograms_ptx
)
add_executable(hello "${CMAKE_SOURCE_DIR}/Source Files/main.cpp" "Header Files/LuanchParams.h" "Header Files/SampleRenderer.h" "Source Files/SampleRenderer.cpp" "Header Files/optix7.h" "Header Files/CUDABuffer.h"
"${CMAKE_SOURCE_DIR}/common/glfWindow/GLFWindow.cpp")

target_link_libraries(hello PRIVATE
    CUDA::cuda_driver   # libcuda / cuda.lib
    CUDA::cudart        # cudart.lib
    ${GLFW_DIR}/Libs/glfw3.lib
    devicePrograms_embedded
    opengl32
)