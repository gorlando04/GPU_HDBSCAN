#ifndef GETTERS_CUH
#define GETTERS_CUH

#include <cuda.h>

#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"
#include <vector>
#include <tuple>



int get_NumThreshold(long int numValues_);

int get_TiedVertexes(Vertex *vertexes,int pos_threshold, int value_threshold);


void get_IndexThreshold(int *finalCounts,int *treshold_idx,int value_threshold);

int findKNNlist(int *kNN,long int neig,long int i,int k);


int buscaBinaria(Stability *stabilities, int alvo, int size);


int buscaBinaria(std::vector<std::tuple<int,int>> cluster_size, int alvo);


int buscaBinaria(std::vector<std::tuple<int,bool>> cluster_size, int alvo);

int buscaBinaria(std::vector<int> vector, int alvo);

int buscaBinaria(int* vector, int size, int alvo);



#endif