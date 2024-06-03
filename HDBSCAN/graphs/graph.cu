#include "graph.cuh"
#include "../initializer/initialize.cuh"
#include "../getters/getters.cuh"
#include "../calculates/calculates.cuh"
#include "../counts/count.cuh"
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <vector>
#include <omp.h>
#include <pthread.h>


ECLgraph buildECLgraph(int nodes, long int edges,int *kNN, float *distances,int k,long int mpts, int *antihubs, long int num_antihubs,int mst_gpu)
{


    long int numValues = nodes;
    ECLgraph g;


    g.nodes = nodes;


   /*Isso significa, que o nó 0 está conectado com Y-x NÓS,
   O nó 1 está conetado com z-y nós, e etc...*/
    cudaMallocManaged(&g.nindex,(size_t)(g.nodes + 1) * sizeof(g.nindex[0])); // nindex[0] = X, nindex[1] = y, nindex[2] = z
    int gridSize = (g.nodes + 1 + blockSize - 1) / blockSize;
    initializeVectorCounts_<<<gridSize,blockSize>>>(g.nindex,0,g.nodes+1); // Aqui usar GPU
    Check();
    cudaMemPrefetchAsync(g.nindex,(size_t)(g.nodes + 1) * sizeof(g.nindex[0]),cudaCpuDeviceId);

    bool *flag_knn = (bool*)malloc(numValues*k * sizeof(bool));

    calculate_nindex(nodes, kNN, flag_knn,&g,antihubs,num_antihubs);

    // Nesse pontos os nós já estão calculados, agora precisamos inserir as arestas. Essa parte será bem demorada.
    long int *auxiliar_edges;

    cudaMallocManaged(&auxiliar_edges,(size_t)(g.nodes) * sizeof(long int)); // nindex[0] = X, nindex[1] = y, nindex[2] = z
    gridSize = (g.nodes + 1  + blockSize - 1) / blockSize;
    initializeVectorCounts_<<<gridSize,blockSize>>>(auxiliar_edges,0,g.nodes); //Aqui usar GPU
    avoid_pageFault(auxiliar_edges,g.nodes,true);
    Check();


    cudaMallocManaged(&g.nlist,(size_t)(g.nindex[nodes]) * sizeof(int));
    g.edges = g.nindex[nodes];
    calculate_nlist(nodes, kNN,k, flag_knn,&g,antihubs,num_antihubs,auxiliar_edges);


    cudaMallocManaged(&g.eweight,(size_t)g.edges * sizeof(g.eweight[0]));

	int *aux_nodes;
    cudaMallocManaged(&aux_nodes,(size_t)g.edges * sizeof(g.nlist[0]));
    createNodeList(aux_nodes,&g) ; //Aqui usar GPU
    Check();

         
    long int elementsPerGPU[numGPUs];
    calculateElements(elementsPerGPU,numGPUs,numValues); 

    float *coreDistances;
    cudaMallocManaged(&coreDistances,(size_t)(numValues) * sizeof(float)); 
    calculateCoreDistance(distances,coreDistances,elementsPerGPU,k,mpts); //Aqui usa GPU
    Check();

     // Testar esse substituição
    /*
    float *coreDistances;
    cudaMallocManaged(&coreDistances,(size_t)(numValues) * sizeof(float)); 
    calculateCoreDistance(distances,coreDistances,numValues,k,mpts); //Aqui usa GPU
    */


    cudaFree(kNN);
    kNN = NULL;

    if(mst_gpu != 1){cudaFree(distances); distances = NULL; }


    calculateMutualReachabilityDistance(g.eweight,coreDistances,aux_nodes,g.nlist,g.edges);  //Aqui usa GPU
    Check();

    // Read vector txtx
    calculate_coreDistance_antihubs(&g,auxiliar_edges,antihubs,num_antihubs);

  return g;   
}




ECLgraph buildEnhancedKNNG(int *kNN, float *distances, int shards_num, long int numValues,long int k,long int mpts ,int mst_gpu){

    long int vectorSize = numValues*k;

    int *finalCounts = calculate_degrees(kNN,vectorSize,shards_num,numValues);

    Vertex *vertexes;
	cudaMallocManaged(&vertexes,(size_t)numValues * sizeof(Vertex));
    int gridSize = (numValues + blockSize - 1) / blockSize;    
    initializeVertex<<<gridSize,blockSize>>>(vertexes,finalCounts,numValues);
    Check();

    // Pegar os threshold  + Ordenação Parcial + Pegar o valor do threshold 
    int pos_threshold = get_NumThreshold(numValues);
    std::partial_sort(vertexes, vertexes + pos_threshold, vertexes + numValues, compareVertexByDegree); 
    int value_threshold = vertexes[pos_threshold-1].grau;
    printf("A posicao do threshold eh: %d e o valor eh: %d\n",pos_threshold-1,value_threshold);


    avoid_pageFault(vertexes,numValues);

    // Encontrar quantos valores são iguais ao threshold
    long int elementsPerGPU_[numGPUs];
    calculateElements(elementsPerGPU_,numGPUs,numValues);

    // Encontra quantos valores são iguais ao threshold
    int countsTreshold = countThreshold_(elementsPerGPU_,vertexes,value_threshold);

    int *antihubs;
    antihubs = new int[pos_threshold];

     if (countsTreshold > 1){
   
        calculate_finalAntihubs(vertexes,kNN,finalCounts,antihubs,numValues,countsTreshold,
                            pos_threshold, value_threshold,k);
     }

    else{
        // Bota os não empatados
        for(int i=0;i< pos_threshold;i++)
            antihubs[i] = vertexes[i].index;
    }

    // Ordena pelo índice para inserir na MST
    std::sort(antihubs,antihubs+pos_threshold);


    // Libera a galera
    cudaFree(finalCounts);
    cudaFree(vertexes);
    finalCounts = NULL;
    vertexes = NULL;


    ECLgraph g;
    
    g = buildECLgraph(numValues, vectorSize,kNN, distances,k,mpts, antihubs, pos_threshold,mst_gpu);

    return g;
}
