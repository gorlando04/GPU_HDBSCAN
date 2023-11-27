#include "initialize.cuh"
#include "cuda_runtime.h"


__global__ void initializeVectorCounts(int *vector,int value,int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = value;
    }

}

__global__ void initializeVertex(Vertex *vertexes, int *counts,long int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vertexes[tid].grau = counts[tid];
        vertexes[tid].index = tid;
    }

}


__global__ void initializeVectorCounts_(long int *vector,long int value,int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = value;
    }

}

__global__ void initializeVectorCounts(float *vector,float value,int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = value;
    }

}

__global__ void initializeUntieHubs(Untie_hub *untie, int value, int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        untie[tid].score = value;
    }

}