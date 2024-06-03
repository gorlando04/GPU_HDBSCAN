#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../tools/filetool.hpp"
#include "../build_kNNG.cuh"
#include "structs/hdbscan_elements.cuh"
#include "structs/ECLgraph.h"
#include "graphs/graph.cuh"
#include "mst/mst.cuh"
#include "trees/tree.cuh"

#include <fstream>
#include <string>
#include <cstdio>
#include<unistd.h>



void ReadTxtVecs(const string &data_path, float **vectors_ptr,
                          long int *num_ptr, long int *dim_ptr,
                          const bool show_process = true) {
    float *&vecs = *vectors_ptr;
    long int &num = *num_ptr;
    long int &dim = *dim_ptr;
    std::ifstream in(data_path);
    if (!in.is_open()) {
      throw(std::string("Failed to open ") + data_path);
    }
    in >> num >> dim;
    vecs = new float[num * dim];

   for (int i = 0; i < num; i++) {

      for (int j = 0; j < dim; j++) {
        in >> vecs[i * dim + j];
      }
    }

    in.close();
    return;
  }

void WriteTxtVecs(const string &data_path, const int *vectors,
                           const int write_num) {
    ofstream out(data_path);

    for (int i = 0; i < write_num; i++) {
        out << vectors[i] << '\t';
      out << endl;
    }
    out.close();
    return;
}


void kNNG_toVecs(int *result_index_graph,float *distances,NNDElement *result_graph,long int knng_num, long int knng_dim){

  // Le os indexes
  cudaMallocManaged(&result_index_graph,(size_t)knng_num*knng_dim * sizeof(int));
  for (long int i = 0; i < knng_num; i++) {
    for (long int j = 0; j < knng_dim; j++) {

      result_index_graph[i * knng_dim + j] = result_graph[i * knng_dim + j].label();
    }

  } 

  CheckCUDA_();

  //Le as distâncias
  cudaMallocManaged(&distances,(size_t)knng_num*knng_dim * sizeof(float));
  for (long int i = 0; i < knng_num; i++) {
    for (long int j = 0; j < knng_dim; j++) {
      distances[i * knng_dim + j] = result_graph[i * knng_dim + j].distance();
    }
  }

  CheckCUDA_();

  delete result_graph;
  result_graph = NULL;

  return;
}

GPUECLgraph buildGPUgraph(string path_to_kNNG,ECLgraph *g,int *multiplicacoes){

  NNDElement *result_graph2;
  long int knng_num2, knng_dim2;
  double min_dist = FileTool::GetMinDistanceFromKNNG(path_to_kNNG,&result_graph2, &knng_num2, &knng_dim2);

  int multiplica = ceil(-log10(fabs(min_dist))) + 2;


  double *eweights = new double[g->edges];
  for(long int i=0;i<g->edges;i++)
    eweights[i] = g->eweight[i];

  GPUECLgraph g_gpu;
  g_gpu.edges = g->edges;
  g_gpu.nodes = g->nodes;

  g_gpu.nlist = new int[g->edges];
  g_gpu.eweight = new int[g->edges];
  g_gpu.nindex = new int[g_gpu.nodes+1];


  for(int i=0;i<g->nodes+1;i++){
    g_gpu.nindex[i] = g->nindex[i];
  }

  for(int i=0;i<g->edges;i++){
    g_gpu.nlist[i] = g->nlist[i];
    g_gpu.eweight[i] =  (int)(pow(10.0,multiplica) *  eweights[i] / 1);
  }

  *multiplicacoes = multiplica;
  delete eweights; eweights = NULL;
  delete result_graph2; result_graph2 = NULL;

  return g_gpu;

}

void freeECLgraph(ECLgraph &g)
{
  if (g.nindex != NULL) cudaFree(g.nindex);
  if (g.nlist != NULL) cudaFree(g.nlist);
  if (g.eweight != NULL) cudaFree(g.eweight);
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
}

void freeECLgraphGPU(GPUECLgraph &g)
{
  if (g.nindex != NULL) free(g.nindex);
  if (g.nlist != NULL) free(g.nlist);
  if (g.eweight != NULL) free(g.eweight);
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
}




int main( int argc, char *argv[]) {

  //Standard parameters
  int shards = 30;
  long int numValues = 1000000;
  long int k = NEIGHB_NUM_PER_LIST;
  long int mpts = 10;
  int mst_gpu = 0;
  int num_buckets = 32;
  int num_threads = 32;

  // ./hdbscan_ NUM_VALUES mpts shards
  if (argc == 5){
    numValues = atoi(argv[1]);
    printf("NUM VALUES SET TO %ld.\n",numValues);

    mpts = atoi(argv[2]);
    printf("mpts SET TO %ld.\n",mpts);

    shards = atoi(argv[3]);
    printf("SHARDS SET TO %d.\n",shards);

    mst_gpu  = atoi(argv[4]);
    printf("MST WILL BE CONSTRUCTED IN GPU: %d\n",mst_gpu);
  }
  
  else{
        printf("NUM_VALUES NOT SETTED\n");
        exit(1);
  }

  assert(k >= mpts);

  //kNNG
  const std::string path_to_data = "/nndescent/GPU_HDBSCAN/data/artificial/SK_data.txt";
  PrepareVector(path_to_data,"/nndescent/GPU_HDBSCAN/data/vectors.fvecs");


  clock_t t; 
  t = clock(); 
  printf("Iniciando o HDBSCAN\n");

  std::string path_to_kNNG = "/nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.kgraph";
  ConstructLargeKNNGraph(shards, "/nndescent/GPU_HDBSCAN/data/vectors", path_to_kNNG);

  // Le o kNNG da memória
  NNDElement *result_graph;
  long int knng_num, knng_dim;
  FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &knng_num, &knng_dim);
  knng_num = numValues;
  printf("kNNG size = %ld e %d\n",knng_num,knng_dim);
    
  // Cast NNDElement to int and float
  int *result_index_graph;
  float *distances;
  kNNG_toVecs(result_index_graph,distances,result_graph,knng_num, knng_dim);

  //HDBSCAN
  int shards_num = numGPUs;
  if (numValues*k > 2000000000 )
    shards_num += numGPUs;
  
  // Constroí o kNNG enhanced
  ECLgraph g;
  g = buildEnhancedKNNG(result_index_graph,distances,shards_num,numValues,k,mpts,mst_gpu);


  bool* edges;
  MSTedge *mst_edges;

  int qntd_nohs = g.nodes;
  if(mst_gpu){
    
    int multiplicacoes=0;
    GPUECLgraph g_gpu = buildGPUgraph(path_to_kNNG,&g,&multiplicacoes);     
    edges = gpuMST(g_gpu, INT_MAX);
    mst_edges = buildMST_gpu(g_gpu,edges,12,multiplicacoes);
     
    cudaFree(distances);
    distances = NULL;
    freeECLgraphGPU(g_gpu);

  }

  else{
    edges = cpuMST(g);
    mst_edges = buildMST(g,edges,12);
  }

  // Liberar G
  freeECLgraph(g);

  SingleLinkageNode *result_arr;
  result_arr = build_Linkage_tree(mst_edges ,knng_num ,qntd_nohs);

  CondensedTreeNode* condensed_tree;
  int condensed_size;
  condensed_tree =  build_Condensed_tree(result_arr, knng_num ,qntd_nohs-1, mpts,&condensed_size);


  Stability *stabilities;
  int stability_size;
  stabilities = compute_stability(condensed_tree,condensed_size,&stability_size);

  int* labels;
  labels = get_clusters(condensed_tree, condensed_size, stabilities,  stability_size, numValues);

  t = clock() - t; 
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds */


//    const std::string out_PATH = "/nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/approximate_result.txt";
    //WriteTxtVecs(out_PATH,labels,numValues);

  return 0;


}

/*
MAPEANDO FUNÇÕES QUE PRECISAM DE SHARDING:





*/
