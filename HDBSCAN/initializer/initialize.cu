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

__global__ void initializeMSTedges(MSTedge *mst_edges,int *nodes, int *edges, float *weight, bool *isInMST,long int size){


    long int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid < size) {
        
        if (isInMST[tid]){

            mst_edges[tid].from_node = nodes[tid];
            mst_edges[tid].to_node = edges[tid];
            mst_edges[tid].weight = weight[tid];
        }
    }


}

__global__ void initializeVectorCountsOFFset(int *vector,int value,int size,long int offset){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset + tid < size) {
        vector[tid + offset] = value;
    }

}
