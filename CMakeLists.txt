############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.21)
project(cs344)

find_package(OpenCV REQUIRED)

link_libraries(${OpenCV_LIBS} )

# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#
enable_language(CUDA)
if (MSVC)
    # list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4819\"")
    # list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4244\"")
    # add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"/wd 4819\">")
    # add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"/wd 4244\">")
    if (USE_OMP)
        list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/openmp\"")
        list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"/openmp\">")
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>")
    endif()
endif()
find_package(CUDAToolkit REQUIRED)
message(STATUS "-- CUDA --")
message(STATUS "Toolkit Version: " ${CUDAToolkit_VERSION})
message(STATUS "Runtime Directory: " ${CUDAToolkit_LIBRARY_DIR})

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDAToolkit_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/clang; 
  -gencode;arch=compute_50,code=sm_50;")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory("Problem Sets/Problem Set 1")
  # add_subdirectory (HW2)
  # add_subdirectory (HW3)
  # add_subdirectory (HW4)
  # add_subdirectory (HW5)
  # add_subdirectory (HW6)
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
