cmake_minimum_required(VERSION 3.18)
project(my_optix_hello LANGUAGES C CXX CUDA)

# 标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 路径设置
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0" 
    CACHE PATH "PATH to OptiX SDK")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/common/gdt/cmake)

# 查找依赖
find_package(CUDAToolkit REQUIRED)

# GLFW 设置（必须在 add_subdirectory gdt 之前）
add_library(glfw STATIC IMPORTED)
set_target_properties(glfw PROPERTIES
    IMPORTED_LOCATION "E:/OpenGL/OpenGLLibs/Libs/glfw3.lib"
    INTERFACE_INCLUDE_DIRECTORIES "E:/OpenGL/OpenGLLibs/Includes"
)

# 添加子目录（必须在 embed_ptx 之前）
add_subdirectory(${CMAKE_SOURCE_DIR}/common/gdt gdt)
add_subdirectory(${CMAKE_SOURCE_DIR}/common/glfWindow glfWindow)

# include 路径
include_directories(
    ${OptiX_INSTALL_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/common/gdt
    ${CMAKE_SOURCE_DIR}/common/glfWindow
    "E:/OpenGL/OpenGLLibs/Include"
)

# PTX 编译
include(${CMAKE_SOURCE_DIR}/common/gdt/cmake/configure_optix.cmake)
embed_ptx(
    OUTPUT_TARGET         devicePrograms_embedded
    SOURCES               "${CMAKE_SOURCE_DIR}/Source Files/devicePrograms.cu"
    EMBEDDED_SYMBOL_NAMES devicePrograms_ptx
)

# 主程序
add_executable(hello
    "${CMAKE_SOURCE_DIR}/Source Files/main.cpp"
    "${CMAKE_SOURCE_DIR}/Source Files/SampleRenderer.cpp"
    "${CMAKE_SOURCE_DIR}/Header Files/LuanchParams.h"
    "${CMAKE_SOURCE_DIR}/Header Files/SampleRenderer.h"
    "${CMAKE_SOURCE_DIR}/Header Files/optix7.h"
    "${CMAKE_SOURCE_DIR}/Header Files/CUDABuffer.h"
)

target_link_libraries(hello PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    glfw
    gdt
    glfWindow
    devicePrograms_embedded
    opengl32
)