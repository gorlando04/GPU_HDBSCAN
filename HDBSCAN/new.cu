#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "/nndescent/GPU_HDBSCAN/experiments/tools/filetool.hpp"
#include "structs/hdbscan_elements.cuh"
#include "structs/ECLgraph.h"
#include "graphs/graph.cuh"
#include "mst/mst.cuh"
#include "trees/tree.cuh"


void generate_random(int *h_data){

   // Gera o vetor de teste
  for (long int i = 0; i < vectorSize; i++) {
        h_data[i] = /*i / k;*/rand() % numValues;
    }   

    return;
}


void generate_random(float *h_data){

   // Gera o vetor de teste
  for (long int i = 0; i < vectorSize; i++) {
        h_data[i] = /*i / k;*/rand() % 50 + 0.15 * (rand() % 50);
    }   

    return;
}








int main() {


    int shards_num = 3;

     clock_t t; 
    t = clock(); 
    // Le o kNNG que esta escrito no arquivo abaixo
    std::string path_to_kNNG = "/nndescent/GPU_HDBSCAN/experiments/results/NNDescent-KNNG.kgraph";
    NNDElement *result_graph;
    int num, dim;
    FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &num, &dim);
    num = numValues;
    printf("%d e %d\n",num,dim);




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



    ECLgraph g;
    g = buildEnhancedKNNG(result_index_graph,distances,shards_num);

    printf("O grafo tem %d NOHS e %ld arestas\n",g.nodes,g.edges);
     bool* edges = cpuMST(g);

     
    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("Demorou %lf segundos para tudo\n",time_taken);

     MSTedge *mst_edges;
     mst_edges = new MSTedge[g.nodes-1];

     mst_edges = buildMST(g,edges,12);

    UnionFind U = UnionFind(num);


    SingleLinkageNode *result_arr;
    result_arr = new SingleLinkageNode[g.nodes-1];

    for (long int i=0;i<(g.nodes-1);i++){

      int a = mst_edges[i].from_node;
      int b = mst_edges[i].to_node;
      float delta = mst_edges[i].weight;

      int aa = U.FastFind(a);
      int bb = U.FastFind(b);
      
      result_arr[i].left_node = aa;
      result_arr[i].right_node = bb;
      result_arr[i].weight = delta;
      result_arr[i].node_size = (U.getSize(aa)) + (U.getSize(bb));

      U.Union(aa,bb);

    }

    printf("Single Linkage Tree montada\n");


  return 0;


}
