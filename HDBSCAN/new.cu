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

    SingleLinkageNode *result_arr;

    result_arr = build_Linkage_tree(mst_edges ,num ,g.nodes);

    CondensedTreeNode* condensed_tree;
    int condensed_size;
    condensed_tree =  build_Condensed_tree(result_arr, num ,g.nodes-1, k,&condensed_size);


    Stability *stabilities;
    int stability_size;
    
    stabilities = compute_stability(condensed_tree,condensed_size,&stability_size);


    get_clusters(condensed_tree, condensed_size, stabilities,  stability_size, numValues);




  return 0;


}
