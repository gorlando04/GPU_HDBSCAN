#ifndef MST
#define MST
/*
ECL-MST: This code computes a minimum spanning tree (MST) or a minimum spanning forest (MSF) of an undirected graph.

Copyright (c) 2023, Martin Burtscher and Alex Fallin

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://cs.txstate.edu/~burtscher/research/ECL-MST/ and at https://github.com/burtscher/ECL-MST.

Publication: This work is described in detail in the following paper.
Alex Fallin, Andres Gonzalez, Jarim Seo, and Martin Burtscher. A High-Performance MST Implementation for GPUs. Proceedings of the 2023 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2023.
*/


#include <climits>
#include <algorithm>
#include <tuple>
#include <vector>
#include <sys/time.h>
#include "../structs/ECLgraph.h"
#include "../structs/hdbscan_elements.cuh"
#include <cuda.h>
#include <math.h>
#include "cuda_runtime.h"


static const int Device = 0;
static const int ThreadsPerBlock = 512;

typedef unsigned long long ull;


static inline int serial_find(const int idx, int* const parent);


static inline void serial_join(const int a, const int b, int* const parent);


bool* cpuMST(const ECLgraph& g);


static inline __device__ int find(int curr, const int* const __restrict__ parent);

static inline __device__ void join(int arep, int brep, int* const __restrict__ parent);


static __global__ void initPM(const int nodes, int* const __restrict__ parent, ull* const __restrict__ minv);


template <bool first>
static __global__ void initWL(int4* const __restrict__ wl2, int* const __restrict__ wl2size, const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int* const __restrict__ eweight, ull* const __restrict__ minv, const int* const __restrict__ parent, const int threshold);


static __global__ void kernel1(const int4* const __restrict__ wl1, const int wl1size, int4* const __restrict__ wl2, int* const __restrict__ wl2size, const int* const __restrict__ parent, volatile ull* const __restrict__ minv);

static __global__ void kernel2(const int4* const __restrict__ wl, const int wlsize, int* const __restrict__ parent,ull* const __restrict__ minv, bool* const __restrict__ inMST);

static __global__ void kernel3(const int4* const __restrict__ wl, const int wlsize, volatile ull* const __restrict__ minv);


static void CheckCuda(const int line);

bool* gpuMST(const GPUECLgraph& g, int threshold);


MSTedge* buildMST(ECLgraph g,bool *edges, int shards_num);


MSTedge* buildMST_gpu(GPUECLgraph g,bool *edges, int shards_num,int mult);

#endif
