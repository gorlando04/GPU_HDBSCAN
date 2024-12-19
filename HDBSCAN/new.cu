#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../build_kNNG.cuh"
#include "core-sg/core-sg.cuh"
#include <omp.h>
#include <pthread.h>

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
double   time_taken;     



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

    fix_distances(result_graph,knng_num,k);
    free(result_graph);
    result_graph = NULL;

    FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &knng_num, &knng_dim);

  // Construir o core-SG
  ECLgraph core_sg;
  core_sg = build_CoreSG(result_graph,numValues,mpts,mst_gpu);
  int qntd_nohs = core_sg.nodes;


   clock_t t0 = clock() - t;

   time_taken = ((double)t0)/CLOCKS_PER_SEC; // in seconds 
   printf("TEMPO para a construcao do SCORE-SG ::  %lf\n", time_taken);



   printf("CORE-SG foi construido, vamos iniciar a extracao dos cluters para os diferentes valores de mpts\n");


  for(long int i=k;i>k-2;i--){
     printf("Extraindo clusters para mpts= %ld\n",i);
    if (i != k)  update_core_sg_weights(&core_sg,numValues,i); //Update nos pesos

    //Extrair a hierarquia
    int* labels = extract_clusters(&core_sg,qntd_nohs,mpts,mst_gpu);
  
    t = clock() - t;

   time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
   printf("TEMPO Iter %ld:: %lf\n",(k-i), time_taken);
   t = clock();


  }


  return 0;
}
