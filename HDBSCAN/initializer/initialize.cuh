#ifndef INITIALIZE_CUH
#define INITIALIZE_CUH

#include <cuda.h>
#include <iostream>


#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"


__global__ void initializeVectorCounts(int *vector,int value,int size);

__global__ void initializeVertex(Vertex *vertexes, int *counts,long int size);

__global__ void initializeVectorCounts_(long int *vector,long int value,int size);

__global__ void initializeVectorCounts(float *vector,float value,int size);

__global__ void initializeUntieHubs(Untie_hub *untie, int value, int size);


__global__ void initializeMSTedges(MSTedge *mst_edges,int *nodes, int *edges, float *weight, bool *isInMST, long int size);


__global__ void initializeVectorCountsOFFset(int *vector,int value,int size,long int offset);

__global__ void initializeVectorArange(int *vector,int size);

HashLabels initializeHash(CondensedTreeNode *condensed_tree,int condensed_size);



#endif
