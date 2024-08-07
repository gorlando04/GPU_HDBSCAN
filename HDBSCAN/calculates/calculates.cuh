#ifndef CALCULATES_CUH
#define CALCULATES_CUH

#include <cuda.h>
#include <iostream>


#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"
#include "../structs/ECLgraph.h"
#include "../getters/getters.cuh"

__global__ void calculateScore(int *vectors,int *treshold_idx, Untie_hub *vertex , int *degrees ,long int size,int k,int offset); //

__global__ void calculateCoreDistance_(float *coreDistances,float *kNN_distances,long int offset,long int size,long int k,long int mpts); //

__global__ void ShardVector(int *vec_dev,int *vectors,long int size,long int off_set_size,int offset=0); //

__global__ void calculateMRD(float *d_distances,int *d_nodes,int *d_edges,float *coreDistances,long int size); //


void calculateElements(long int *elementsPerGPU,int shards_num,long int vectorSize); //

void calculateUntieScore(Untie_hub *unties ,long int *indexesPerGPU,int *h_data,int *treshold_idx,int *finalCounts, long int k); //

void calculateCoreDistance(float *kNN_distances, float *coreDistances ,long int numValues,long int k,long int mpts);//

void calculateMutualReachabilityDistance(float *graphDistances,float *coreDistances,int *aux_nodes,int *aux_edges,long int size); //

float calculate_euclidean_distance(float *vector,long int idxa,long int idxb,int dim); //

void calculate_nindex(int nodes, int *kNN, int *flag_knn,ECLgraph *g,int *antihubs,int num_antihubs);

void calculate_nlist(int nodes, int *kNN,int k, int *flag_knn,ECLgraph *g,int *antihubs,int num_antihubs,long int *auxiliar_edges);

void calculate_coreDistance_antihubs(ECLgraph *g,long int *auxiliar_edges,int *antihubs,long int num_antihubs);

int* calculate_degrees(int *kNN,long int vectorSize,long int numValues);


int* calculate_finalAntihubs(Vertex *vertexes,int *kNN,int* finalCounts,int* antihubs,long int numValues,int countsTreshold,
                            int pos_threshold, int value_threshold,long int k);

#endif
