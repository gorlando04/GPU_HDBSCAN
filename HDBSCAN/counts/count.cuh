#ifndef COUNT_CUH
#define COUNT_CUH

#include <cuda.h>

#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"


// Função do kernel CUDA para calcular a contagem de valores em um vetor
__global__ void countValues(int *data, int *counts, long int size, long int off_set_size,int offset=0); //


// Função do kernel CUDA para calcular a contagem de valores em um vetor
__global__ void countTreshold(Vertex *data, int *counts, long int size, long int off_set_size,int offset,int comparision); //



void countDegrees(int *finalCounts,int *h_data,int shards_num,long int *elementsPerGPU,long int numValues); //



int countThreshold_(long int *elementsPerGPU_,Vertex *vertexes,int value_threshold); //


#endif
