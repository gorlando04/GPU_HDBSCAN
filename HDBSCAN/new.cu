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






int main( int argc, char *argv[]) {

  //Standard parameters
  int shards = 30;
  long int numValues = 1000000;
  long int k = NEIGHB_NUM_PER_LIST;
  long int mpts = 10;


  // ./hdbscan_ NUM_VALUES mpts shards
  if (argc == 4){
    numValues = atoi(argv[1]);
    printf("NUM VALUES SET TO %ld.\n",numValues);

    mpts = atoi(argv[2]);
    printf("mpts SET TO %ld.\n",mpts);

    shards = atoi(argv[3]);
    printf("SHARDS SET TO %d.\n",shards);

  }
  
  else{
        printf("NUM_VALUES NOT SETTED\n");
        exit(1);
  }


  assert(k >= mpts);

  //kNNG

    //Preparing vector
    const std::string path_to_data = "/nndescent/GPU_HDBSCAN/data/artificial/SK_data.txt";

    PrepareVector(path_to_data,"/nndescent/GPU_HDBSCAN/data/vectors.fvecs");

    std::string path_to_kNNG = "/nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.kgraph";

    clock_t t; 
    t = clock(); 
    printf("Iniciando o HDBSCAN\n");


    ConstructLargeKNNGraph(shards, "/nndescent/GPU_HDBSCAN/data/vectors", path_to_kNNG);




    // Le o kNNG que esta escrito no arquivo abaixo
    NNDElement *result_graph;
    int knng_num, knng_dim;
    FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &knng_num, &knng_dim);
    knng_num = numValues;
    printf("kNNG size = %ld e %d\n",knng_num,knng_dim);
    
    // Le o vetor de amostras
    float *vectors_data;
    long int data_size, data_dim;

    ReadTxtVecs(path_to_data,&vectors_data,&data_size,&data_dim);
    printf("Data size= %d e %d\n",numValues,data_dim);




    int *result_index_graph;
    cudaMallocManaged(&result_index_graph,(size_t)knng_num*knng_dim * sizeof(int));
    for (long int i = 0; i < knng_num; i++) {
      for (long int j = 0; j < knng_dim; j++) {

        result_index_graph[i * knng_dim + j] = result_graph[i * knng_dim + j].label();
      }

    } 

    CheckCUDA_();

   
    float *distances;
    cudaMallocManaged(&distances,(size_t)numValues*knng_dim * sizeof(float));


    for (long int i = 0; i < knng_num; i++) {
        for (long int j = 0; j < knng_dim; j++) {
          
          distances[i * knng_dim + j] = result_graph[i * knng_dim + j].distance();
        }
    }

    CheckCUDA_();


    //HDBSCAN
    int shards_num = 3;
    ECLgraph g;
    g = buildEnhancedKNNG(result_index_graph,distances,shards_num,vectors_data,data_dim,numValues,k,mpts);

    bool* edges = cpuMST(g);

     


     MSTedge *mst_edges;
     mst_edges = new MSTedge[g.nodes-1];

     mst_edges = buildMST(g,edges,12);

    SingleLinkageNode *result_arr;

    result_arr = build_Linkage_tree(mst_edges ,knng_num ,g.nodes);

    CondensedTreeNode* condensed_tree;
    int condensed_size;
    condensed_tree =  build_Condensed_tree(result_arr, knng_num ,g.nodes-1, mpts,&condensed_size);


    Stability *stabilities;
    int stability_size;
    
    stabilities = compute_stability(condensed_tree,condensed_size,&stability_size);

    int* labels;
    labels = get_clusters(condensed_tree, condensed_size, stabilities,  stability_size, numValues);

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("Demorou %lf segundos\n",time_taken);

    const std::string out_PATH = "/nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/approximate_result.txt";
    WriteTxtVecs(out_PATH,labels,numValues);



  return 0;


}
