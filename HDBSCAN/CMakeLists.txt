cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hdbscan_ LANGUAGES CXX CUDA)
  
add_library(hdbcuda STATIC
    graphs/graph.cu
    graphs/graph.cuh
    structs/hdbscan_elements.cu
    structs/hdbscan_elements.cuh
    structs/ECLgraph.h
    initializer/initialize.cu
    initializer/initialize.cuh
    getters/getters.cu
    getters/getters.cuh
    calculates/calculates.cu
    calculates/calculates.cuh
    counts/count.cu
    counts/count.cuh
    mst/mst.cu
    mst/mst.cuh
    trees/tree.cu
    trees/tree.cuh
    merge_sort/merge_sort.cu
    merge_sort/merge_sort.cuh
    core-sg/core-sg.cu
    core-sg/core-sg.cuh
    ../gpuknn/gen_large_knngraph.cu
    ../gpuknn/gen_large_knngraph.cuh
    ../gpuknn/knncuda.cu
    ../gpuknn/knncuda.cuh
    ../gpuknn/nndescent.cu
    ../gpuknn/nndescent.cuh
    ../gpuknn/knncuda_tools.cuh
    ../gpuknn/knncuda_tools.cu
    ../gpuknn/knnmerge.cuh
    ../gpuknn/knnmerge.cu
     ../tools/distfunc.hpp
     ../tools/filetool.hpp
    ../tools/nndescent_element.cuh
    ../tools/knndata_manager.hpp
    ../xmuknn.h
    ../xmuknn.cpp
    ../build_kNNG.cu
    ../build_kNNG.cuh
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDA QUIET)
# set(CMAKE_VERBOSE_MAKEFILE ON)
target_compile_features(hdbcuda PUBLIC cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FILE_OFFSET_BITS=64 -O3 -std=c++14 -arch=sm_75 -rdc=true -Xcompiler -fopenmp -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FILE_OFFSET_BITS=64 -std=c++14 -O3 -pthread -fopenmp")
add_executable(hdbscan_ new.cu)

set_property(TARGET hdbscan_ PROPERTY CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET hdbscan_ PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET hdbscan_ PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(hdbscan_ PRIVATE hdbcuda ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
 
