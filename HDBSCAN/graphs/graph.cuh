#ifndef GRAPH_CUH
#define GRAPH_CUH

#include <cuda.h>

#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"
#include "../structs/ECLgraph.h"

void joinAntiHubs(int *antihubs,Vertex *vertexes,int not_ties, Untie_hub *unties,int missing_ties);




void createNodeList(int *vector,ECLgraph *g);

void createNodeList_gpu(int *vector,GPUECLgraph *g);



ECLgraph buildECLgraph(int nodes, long int edges,int *kNN , float *distances ,int k,long int mpts, int *antihubs, long int num_antihubs, int mst_gpu=0);


ECLgraph buildEnhancedKNNG(int *h_data,float *distances ,int shards_num,long int numValues,long int k,long int mpts,int mst_gpu=0);


#endif
