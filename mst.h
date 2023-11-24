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
#include <cuda.h>
#include "ECLgraph.h"


static const int Device = 0;
static const int ThreadsPerBlock = 512;

typedef unsigned long long ull;


static inline int serial_find(const int idx, int* const parent)
{
  int curr = parent[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr != (next = parent[curr])) {
      parent[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}


static inline void serial_join(const int a, const int b, int* const parent)
{
  const int arep = serial_find(a, parent);
  const int brep = serial_find(b, parent);
  if (arep > brep) {  // improves locality
    parent[brep] = arep;
  } else {
    parent[arep] = brep;
  }
}


static bool* cpuMST(const ECLgraph& g)
{
  bool* const inMST = new bool [g.edges];
  int* const parent = new int [g.nodes];


  std::fill(inMST, inMST + g.edges, false);
  for (int i = 0; i < g.nodes; i++) parent[i] = i;

  std::vector<std::tuple<float, int, int, int>> list;  // <weight, edge index, from node, to node>
  for (int i = 0; i < g.nodes; i++) {
    for (long int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int n = g.nlist[j];
      if (n > i) {  // only one direction
        list.push_back(std::make_tuple(g.eweight[j], j, i, n));
      }
    }
  }
  std::sort(list.begin(), list.end());

  int count = g.nodes - 1;
  for (int pos = 0; pos < list.size(); pos++) {
    const int a = std::get<2>(list[pos]);
    const int b = std::get<3>(list[pos]);
    const int arep = serial_find(a, parent);
    const int brep = serial_find(b, parent);
    if (arep != brep) {
      const int j = std::get<1>(list[pos]);
      inMST[j] = true;
      serial_join(arep, brep, parent);
      count--;
      if (count == 0) break;
    }
  }



  delete [] parent;
  return inMST;
}

static const int ThreadsPerBlock = 256;

typedef unsigned long long ull;

static inline __device__ int find(int curr, const int* const __restrict__ parent)
{
  int next;
  while (curr != (next = parent[curr])) {
    curr = next;
  }
  return curr;
}


static inline __device__ void join(int arep, int brep, int* const __restrict__ parent)
{
  int mrep;
  do {
    mrep = max(arep, brep);
    arep = min(arep, brep);
  } while ((brep = atomicCAS(&parent[mrep], mrep, arep)) != mrep);
}


static __global__ void initPM(const int nodes, int* const __restrict__ parent, ull* const __restrict__ minv)
{
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < nodes) {
    parent[v] = v;
    minv[v] = ULONG_MAX;
  }
}


template <bool first>
static __global__ void initWL(int4* const __restrict__ wl2, int* const __restrict__ wl2size, const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int* const __restrict__ eweight, ull* const __restrict__ minv, const int* const __restrict__ parent, const int threshold)
{
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  int beg, end, arep, deg = -1;
  if (v < nodes) {
    beg = nindex[v];
    end = nindex[v + 1];
    deg = end - beg;
    arep = first ? v : find(v, parent);
    if (deg < 4) {
      for (int j = beg; j < end; j++) {
        const int n = nlist[j];
        if (n > v) {  // only in one direction
          const int wei = eweight[j];
          if (first ? (wei <= threshold) : (wei > threshold)) {
            const int brep = first ? n : find(n, parent);
            if (first || (arep != brep)) {
              const int k = atomicAdd(wl2size, 1);
              wl2[k] = int4{arep, brep, wei, j};  // <from node, to node, weight, edge index>
            }
          }
        }
      }
    }
  }
  const int WS = 32;  // warp size
  const int lane = threadIdx.x % WS;
  int bal = __ballot_sync(~0, deg >= 4);
  while (bal != 0) {
    const int who = __ffs(bal) - 1;
    bal &= bal - 1;
    const int wi = __shfl_sync(~0, v, who);
    const int wbeg = __shfl_sync(~0, beg, who);
    const int wend = __shfl_sync(~0, end, who);
    const int warep = first ? wi : __shfl_sync(~0, arep, who);
    for (int j = wbeg + lane; j < wend; j += WS) {
      const int n = nlist[j];
      if (n > wi) {  // only in one direction
        const int wei = eweight[j];
        if (first ? (wei <= threshold) : (wei > threshold)) {
          const int brep = first ? n : find(n, parent);
          if (first || (warep != brep)) {
            const int k = atomicAdd(wl2size, 1);
            wl2[k] = int4{warep, brep, wei, j};  // <from node, to node, weight, edge index>
          }
        }
      }
    }
  }
}


static __global__ void kernel1(const int4* const __restrict__ wl1, const int wl1size, int4* const __restrict__ wl2, int* const __restrict__ wl2size, const int* const __restrict__ parent, volatile ull* const __restrict__ minv)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < wl1size) {
    int4 el = wl1[idx];
    const int arep = find(el.x, parent);
    const int brep = find(el.y, parent);
    if (arep != brep) {
      el.x = arep;
      el.y = brep;
      wl2[atomicAdd(wl2size, 1)] = el;
      const ull val = (((ull)el.z) << 32) | el.w;
      if (minv[arep] > val) atomicMin((ull*)&minv[arep], val);
      if (minv[brep] > val) atomicMin((ull*)&minv[brep], val);
    }
  }
}


static __global__ void kernel2(const int4* const __restrict__ wl, const int wlsize, int* const __restrict__ parent, ull* const __restrict__ minv, bool* const __restrict__ inMST)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < wlsize) {
    const int4 el = wl[idx];
    const ull val = (((ull)el.z) << 32) | el.w;
    if ((val == minv[el.x]) || (val == minv[el.y])) {
      join(el.x, el.y, parent);
      inMST[el.w] = true;
    }
  }
}


static __global__ void kernel3(const int4* const __restrict__ wl, const int wlsize, volatile ull* const __restrict__ minv)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < wlsize) {
    const int4 el = wl[idx];
    minv[el.x] = ULONG_MAX;
    minv[el.y] = ULONG_MAX;
  }
}


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}



template <bool filter>
static bool* gpuMST(const ECLgraph& g,const int threshold){

bool* d_inMST = NULL;
  cudaMalloc((void**)&d_inMST, g.edges * sizeof(bool));
  bool* const inMST = new bool [g.edges];

  int* d_parent = NULL;
  cudaMalloc((void**)&d_parent, g.nodes * sizeof(int));

  ull* d_minv = NULL;
  cudaMalloc((void**)&d_minv, g.nodes * sizeof(ull));

  int4* d_wl1 = NULL;
  cudaMalloc((void**)&d_wl1, g.edges / 2 * sizeof(int4));

  int* d_wlsize = NULL;
  cudaMalloc((void**)&d_wlsize, sizeof(int));

  int4* d_wl2 = NULL;
  cudaMalloc((void**)&d_wl2, g.edges / 2 * sizeof(int4));

  int* d_nindex = NULL;
  cudaMalloc((void**)&d_nindex, (g.nodes + 1) * sizeof(int));
  cudaMemcpy(d_nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

  int* d_nlist = NULL;
  cudaMalloc((void**)&d_nlist, g.edges * sizeof(int));
  cudaMemcpy(d_nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice);

  int* d_eweight = NULL;
  cudaMalloc((void**)&d_eweight, g.edges * sizeof(int));
  cudaMemcpy(d_eweight, g.eweight, g.edges * sizeof(int), cudaMemcpyHostToDevice);

  CheckCuda(__LINE__);



  const int blocks = (g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;
  initPM<<<blocks, ThreadsPerBlock>>>(g.nodes, d_parent, d_minv);
  cudaMemset(d_inMST, 0, g.edges * sizeof(bool));

  cudaMemset(d_wlsize, 0, sizeof(int));
  initWL<true><<<blocks, ThreadsPerBlock>>>(d_wl1, d_wlsize, g.nodes, d_nindex, d_nlist, d_eweight, d_minv, d_parent, threshold);

  int wlsize;
  cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
  while (wlsize > 0) {
    cudaMemset(d_wlsize, 0, sizeof(int));
    const int blocks = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;
    kernel1<<<blocks, ThreadsPerBlock>>>(d_wl1, wlsize, d_wl2, d_wlsize, d_parent, d_minv);
    std::swap(d_wl1, d_wl2);
    cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
    if (wlsize > 0) {
      kernel2<<<blocks, ThreadsPerBlock>>>(d_wl1, wlsize, d_parent, d_minv, d_inMST);
      kernel3<<<blocks, ThreadsPerBlock>>>(d_wl1, wlsize, d_minv);
    }
  }

  if (filter) {
    initWL<false><<<blocks, ThreadsPerBlock>>>(d_wl1, d_wlsize, g.nodes, d_nindex, d_nlist, d_eweight, d_minv, d_parent, threshold);

    cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
    while (wlsize > 0) {
      cudaMemset(d_wlsize, 0, sizeof(int));
      const int blocks = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;
      kernel1<<<blocks, ThreadsPerBlock>>>(d_wl1, wlsize, d_wl2, d_wlsize, d_parent, d_minv);
      std::swap(d_wl1, d_wl2);
      cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
      if (wlsize > 0) {
        kernel2<<<blocks, ThreadsPerBlock>>>(d_wl1, wlsize, d_parent, d_minv, d_inMST);
        kernel3<<<blocks, ThreadsPerBlock>>>(d_wl1, wlsize, d_minv);
      }
    }
  }

  cudaDeviceSynchronize();


  cudaMemcpy(inMST, d_inMST, g.edges * sizeof(bool), cudaMemcpyDeviceToHost);

  cudaFree(d_inMST);
  cudaFree(d_parent);
  cudaFree(d_minv);
  cudaFree(d_wl1);
  cudaFree(d_wl2);
  cudaFree(d_wlsize);
  cudaFree(d_nindex);
  cudaFree(d_nlist);
  cudaFree(d_eweight);

  CheckCuda(__LINE__);

  return inMST;
}
#endif