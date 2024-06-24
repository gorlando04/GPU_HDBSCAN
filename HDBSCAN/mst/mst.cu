#include "mst.cuh"
#include "../graphs/graph.cuh"
#include "../calculates/calculates.cuh"
#include "../initializer/initialize.cuh"
#include "../merge_sort/merge_sort.cuh"

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


bool* cpuMST(const ECLgraph& g)
{
  bool* const inMST = new bool [g.edges];
  int* const parent = new int [g.nodes];


  std::fill(inMST, inMST + g.edges, false);
  for (int i = 0; i < g.nodes; i++) parent[i] = i;

  std::vector<std::tuple<float, long int, int, int>> list;  // <weight, edge index, from node, to node>
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
  for (long int pos = 0; pos < list.size(); pos++) {
    const long int a = std::get<2>(list[pos]);
    const long int b = std::get<3>(list[pos]);
    const long int arep = serial_find(a, parent);
    const long int brep = serial_find(b, parent);
    if (arep != brep) {
      const long int j = std::get<1>(list[pos]);
      inMST[j] = true;
      serial_join(arep, brep, parent);
      count--;
      if (count == 0) break;
    }
  }



  delete [] parent;
  return inMST;
}




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


bool* gpuMST(const GPUECLgraph& g, int threshold)
{
  bool* d_inMST = NULL; //
  cudaMalloc((void**)&d_inMST, g.edges * sizeof(bool));
  bool* const inMST = new bool [g.edges];

  int* d_parent = NULL; //
  cudaMalloc((void**)&d_parent, g.nodes * sizeof(int));

  ull* d_minv = NULL; //
  cudaMalloc((void**)&d_minv, g.nodes * sizeof(ull));

  int4* d_wl1 = NULL;
  cudaMalloc((void**)&d_wl1, g.edges / 2 * sizeof(int4));

  int* d_wlsize = NULL;
  cudaMalloc((void**)&d_wlsize, sizeof(int));

  int4* d_wl2 = NULL;
  cudaMalloc((void**)&d_wl2, g.edges / 2 * sizeof(int4));

  int* d_nindex = NULL; // 
  cudaMalloc((void**)&d_nindex, (g.nodes + 1) * sizeof(int));
  cudaMemcpy(d_nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

  int* d_nlist = NULL; //
  cudaMalloc((void**)&d_nlist, g.edges * sizeof(int));
  cudaMemcpy(d_nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice);

  int* d_eweight = NULL; //
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


MSTedge* buildMST(ECLgraph g,bool *edges, int shards_num){



  int *aux_nodes;
  cudaMallocManaged(&aux_nodes,(size_t)g.edges * sizeof(g.nlist[0]));
  createNodeList(aux_nodes,&g);


  MSTedge *finalEdges;
  
  finalEdges = new MSTedge[g.nodes-1];

  int soma =0;


  for (int i = 0; i < g.edges; i++) {

    if(edges[i]){

      finalEdges[soma].from_node = aux_nodes[i];
      finalEdges[soma].to_node = g.nlist[i];
      finalEdges[soma].weight = g.eweight[i];
      soma += 1;
    }
  }   
  if (soma == g.nodes-1)
    printf("MST montada corretamente\n");

  else{
    printf("MST montada incorretamente\n");
    exit(1);
  }


  // Ordena em GPU
  finalEdges = sort_edges(finalEdges,g.nodes-1);


  return finalEdges;

}



MSTedge* buildMST_gpu(GPUECLgraph g,bool *edges, int shards_num,int mult){



  int *aux_nodes;
  cudaMallocManaged(&aux_nodes,(size_t)g.edges * sizeof(g.nlist[0]));
  createNodeList_gpu(aux_nodes,&g);


  MSTedge *finalEdges;
  
  finalEdges = new MSTedge[g.nodes-1];

  int soma =0;


  for (int i = 0; i < g.edges; i++) {

    if(edges[i]){

      finalEdges[soma].from_node = aux_nodes[i];
      finalEdges[soma].to_node = g.nlist[i];


      finalEdges[soma].weight = (1.0 * g.eweight[i]) / pow(10.0,mult) ; 

      soma += 1;
    }
  }   
  

  if (soma == g.nodes-1)
    printf("MST montada corretamente\n");

  else{
    printf("MST montada incorretamente\n");
    exit(1);
  }

  // Ordena em GPU
  finalEdges = sort_edges(finalEdges,g.nodes-1);

  return finalEdges;

}

