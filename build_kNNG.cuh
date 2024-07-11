#ifndef kNNGBUILDER_CUH
#define kNNGBUILDER_CUH

#include <assert.h>
#include <unistd.h>
#include <omp.h>
#include <pthread.h>
#include <string.h>


#include <algorithm>
#include <chrono>
#include <iostream>
#include <istream>
#include <vector>

#include "gpuknn/gen_large_knngraph.cuh"
#include "gpuknn/knncuda_tools.cuh"
#include "gpuknn/knnmerge.cuh"
#include "gpuknn/nndescent.cuh"
#include "tools/distfunc.hpp"
#include "tools/filetool.hpp"
#include "tools/knndata_manager.hpp"
#include "tools/timer.hpp"
#include "xmuknn.h"

using namespace std;
using namespace xmuknn;


void ConstructLargeKNNGraph(int shards, string ref, string result);

void PrepareVector(string base,string out);

float* jesus(int *antihubs,int num_antihubs,int *paz);

void write_bool_dict(int* dict,long int numValues,long int k);

int* read_bool_dict();

NNDElement* ReadkNNGgraph();


float* ReadObjects(long int* pf);


void fix_distances(NNDElement *result_graph,long int numValues,long int k);

void write_euclidean_distance_vecs(float* dict,long int numValues);


float* read_euclidean_distance_vecs();



#endif





