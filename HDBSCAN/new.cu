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
    std::cerr << num << " " << dim << std::endl;
    vecs = new float[num * dim];
   printf("%ld deu bom?\n",num*dim);

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


  // ./hdbscan_ NUM_VALUES shards
  if (argc == 3){
    numValues = atoi(argv[1]);
printf("NUM VALUES SET TO %ld.\n",numValues);

    shards = atoi(argv[2]);
    printf("SHARDS SET TO %d.\n",shards);

  }
  
  else{
        printf("NUM_VALUES NOT SETTED\n");
        exit(1);
  }



  //kNNG

    //Preparing vector
    const std::string path_to_data = "/nndescent/GPU_HDBSCAN/data/artificial/SK_data.txt";

    PrepareVector(path_to_data,"/nndescent/GPU_HDBSCAN/data/vectors.fvecs");

    std::string path_to_kNNG = "/nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.kgraph";
    clock_t t; 
    t = clock(); 


    ConstructLargeKNNGraph(shards, "/nndescent/GPU_HDBSCAN/data/vectors", path_to_kNNG);

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("Demorou %lf segundos para construir o kNNG\n",time_taken);



    // Le o kNNG que esta escrito no arquivo abaixo
    NNDElement *result_graph;
    int num, dim;
    FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &num, &dim);
    num = numValues;
    printf("kNNG size = %ld e %d\n",num,dim);
    
    // Le o vetor de amostras
    float *vectors_data;
    long int vecs_size, dim_;

    ReadTxtVecs(path_to_data,&vectors_data,&vecs_size,&dim_);
    printf("Data size= %d e %d\n",numValues,dim_);




    int *result_index_graph;
    cudaMallocManaged(&result_index_graph,(size_t)num*dim * sizeof(int));
    for (long int i = 0; i < num; i++) {
      for (long int j = 0; j < dim; j++) {

        result_index_graph[i * dim + j] = result_graph[i * dim + j].label();
      }

    } 

    CheckCUDA_();

   
    float *distances;
    cudaMallocManaged(&distances,(size_t)numValues*dim * sizeof(float));


    for (long int i = 0; i < num; i++) {
        for (long int j = 0; j < dim; j++) {
          
          distances[i * dim + j] = result_graph[i * dim + j].distance();
        }
    }

    CheckCUDA_();

    printf("Iniciando o HDBSCAN\n");
    t = clock();

    //HDBSCAN
    int shards_num = 3;
    ECLgraph g;
    g = buildEnhancedKNNG(result_index_graph,distances,shards_num,vectors_data,dim,numValues);

    bool* edges = cpuMST(g);

     


     MSTedge *mst_edges;
     mst_edges = new MSTedge[g.nodes-1];

     mst_edges = buildMST(g,edges,12);

    SingleLinkageNode *result_arr;

    result_arr = build_Linkage_tree(mst_edges ,num ,g.nodes);

    CondensedTreeNode* condensed_tree;
    int condensed_size;
    condensed_tree =  build_Condensed_tree(result_arr, num ,g.nodes-1, k,&condensed_size);


    Stability *stabilities;
    int stability_size;
    
    stabilities = compute_stability(condensed_tree,condensed_size,&stability_size);

    int* labels;
    labels = get_clusters(condensed_tree, condensed_size, stabilities,  stability_size, numValues);

    t = clock() - t; 
    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("Demorou %lf segundos para o HDBSCAN\n",time_taken);

    const std::string out_PATH = "/nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/approximate_result.txt";
    WriteTxtVecs(out_PATH,labels,numValues);



  return 0;


}
