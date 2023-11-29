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

#include "cuda_runtime.h"


static const int Device = 0;
static const int ThreadsPerBlock = 512;

typedef unsigned long long ull;


static inline int serial_find(const int idx, int* const parent);


static inline void serial_join(const int a, const int b, int* const parent);


bool* cpuMST(const ECLgraph& g);

MSTedge* buildMST(ECLgraph g,bool *edges, int shards_num);

#endif
