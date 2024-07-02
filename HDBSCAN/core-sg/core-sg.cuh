


#ifndef CORESG
#define CORESG


#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../../build_kNNG.cuh"
#include "../structs/hdbscan_elements.cuh"
#include "../structs/ECLgraph.h"
#include "../graphs/graph.cuh"
#include "../mst/mst.cuh"
#include "../initializer/initialize.cuh"
#include "../calculates/calculates.cuh"

#include <omp.h>
#include <pthread.h>

#include <fstream>
#include <string>
#include <cstdio>
#include<unistd.h>

void build_CoreSG(NNDElement *result_graph,long int numValues,long int mpts,int mst_gpu);




#endif
