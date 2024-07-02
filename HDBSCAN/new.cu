#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../tools/filetool.hpp"
#include "../build_kNNG.cuh"
#include "core-sg/core-sg.cuh"


#include <fstream>
#include <string>
#include <cstdio>
#include<unistd.h>




int main( int argc, char *argv[]) {

  //Standard parameters
  int shards = 30;
  long int numValues = 1000000;
  long int k = NEIGHB_NUM_PER_LIST;
  long int mpts = 10;
  int mst_gpu = 0;

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
  const std::string path_to_kNNG = "/nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.kgraph";

clock_t    t ;



    // Preparando o vetor para o formato bin  rio
    PrepareVector(path_to_data,"/nndescent/GPU_HDBSCAN/data/vectors.fvecs");

  t = clock();

    // Constro   o kNNG
    ConstructLargeKNNGraph(shards, "/nndescent/GPU_HDBSCAN/data/vectors", path_to_kNNG);

    // Le o kNNG da memoria
    NNDElement *result_graph;
    long int knng_num, knng_dim;
    FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &knng_num, &knng_dim);
   knng_num = numValues;



 
  //build_CoreSG(path_to_data,path_to_kNNG,shards, numValues,mpts,mst_gpu);
  build_CoreSG(result_graph,numValues,mpts,mst_gpu);

  t = clock() - t;

double   time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds */
  printf("TEMPO :: %lf\n", time_taken);


  return 0;
}
