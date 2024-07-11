#include <iostream>
#include <cstdio>
#include <algorithm>
#include <getopt.h>
#include <string.h>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <assert.h>
#include <thread>
#include <vector>

#include "../initializer/initialize.cuh"
#include "../structs/ECLgraph.h"





//void mergeSort(int N,MSTedge *input,MSTedge *output,int device=0);

//void mergeSortMultiGPU(int N,MSTedge *input,MSTedge *output,int offset,int device=0);


//void calculateElements(int *elementsPerGPU,int shards_num,long int vectorSize);

//void merging_partial_results(int *elementsPerGPU, MSTedge **partial_results,int numValues,MSTedge *results);

MSTedge* sort_edges(MSTedge *arr,int numValues);
