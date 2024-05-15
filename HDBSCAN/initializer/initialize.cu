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

__global__ void initializeVectorArange(int *vector,int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = tid;
    }
}


HashLabels initializeHash(CondensedTreeNode *condensed_tree,int condensed_size){


    HashLabels hash;
    hash.lambda_array = new float[condensed_size+1];
    int min_parent = getMinParent(condensed_tree,condensed_size);



    for(int i=0;i<condensed_size;i++){

        int position = condensed_tree[i].child;

        hash.lambda_array[position] = condensed_tree[i].lambda_val;

        if (condensed_tree[i].parent == min_parent)
            hash.lambda_array[min_parent] = condensed_tree[i].lambda_val;
    }

    return hash;
}


void createNodeList(int *vector,ECLgraph *g){

    for(int i=0;i<g->nodes;i++){
        
        long int begin = g->nindex[i];
        long int end = g->nindex[i+1];

        for (long int j=begin;j<end;j++)
            vector[j] = i;
    }
}

void createNodeList_gpu(int *vector,GPUECLgraph *g){

    for(int i=0;i<g->nodes;i++){

        long int begin = g->nindex[i];
        long int end = g->nindex[i+1];

        for (long int j=begin;j<end;j++)
            vector[j] = i;
    }
}



void createEdgeList(int *vector,ECLgraph *g){

    for(long int i=0;i<g->edges;i++){
        
        vector[i] = g->nlist[i];
    }
}

void createWeightList(float *vector,ECLgraph *g){

    for(long int i=0;i<g->edges;i++){
        
        vector[i] = g->eweight[i];
    }
}