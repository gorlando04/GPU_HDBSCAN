#ifndef GRAPH_CUH
#define GRAPH_CUH

#include <cuda.h>

#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"
#include "../structs/ECLgraph.h"

void joinAntiHubs(int *antihubs,Vertex *vertexes,int not_ties, Untie_hub *unties,int missing_ties);




void createNodeList(int *vector,ECLgraph *g);

void createEdgeList(int *vector, ECLgraph *g);

void createWeightList(float *vector,ECLgraph *g);

ECLgraph buildECLgraph(int nodes, long int edges,int *kNN , float *distances ,int k, int *antihubs, long int num_antihubs,float *vectors_data,int dim, long int numValues);



ECLgraph buildEnhancedKNNG(int *h_data,float *distances ,int shards_num, float *vectors_data,int dim,long int numValues);


#endif
