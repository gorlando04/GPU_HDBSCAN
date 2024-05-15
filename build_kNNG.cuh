#ifndef kNNGBUILDER_CUH
#define kNNGBUILDER_CUH

#include <assert.h>
#include <unistd.h>

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


void ConstructLargeKNNGraph(int shards, string ref, string result,int K=32);

void PrepareVector(string base,string out);

#endif

