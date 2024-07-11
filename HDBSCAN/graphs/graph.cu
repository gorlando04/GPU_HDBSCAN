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

    int *flag_knn = (int*)malloc(numValues*k * sizeof(int));
    calculate_nindex(nodes, kNN, flag_knn,&g,antihubs,num_antihubs);

    // Nesse pontos os nós já estão calculados, agora precisamos inserir as arestas. Essa parte será bem demorada.
    long int *auxiliar_edges;

    cudaMallocManaged(&auxiliar_edges,(size_t)(g.nodes) * sizeof(long int)); // nindex[0] = X, nindex[1] = y, nindex[2] = z
    gridSize = (g.nodes + 1  + blockSize - 1) / blockSize;
    initializeVectorCounts_<<<gridSize,blockSize>>>(auxiliar_edges,0,g.nodes); //Aqui usar GPU
    avoid_pageFault(g.nodes,auxiliar_edges,true);
    Check();

printf("PROTAGONISTA 2\n");

    cudaMallocManaged(&g.nlist,(size_t)(g.nindex[nodes]) * sizeof(int));
    g.edges = g.nindex[nodes];
    calculate_nlist(nodes, kNN,k, flag_knn,&g,antihubs,num_antihubs,auxiliar_edges);

     free(flag_knn);
     flag_knn = NULL;

    cudaFree(kNN);
    kNN = NULL;
printf("PROTAGONISTA 3\n");

    cudaMallocManaged(&g.eweight,(size_t)g.edges * sizeof(g.eweight[0]));

    int *aux_nodes;
    cudaMallocManaged(&aux_nodes,(size_t)g.edges * sizeof(g.nlist[0]));
    createNodeList(aux_nodes,&g) ; //Aqui usar GPU
    Check();

         
    long int elementsPerGPU[numGPUs];
    calculateElements(elementsPerGPU,numGPUs,numValues); 



printf("PROTAGONISTA 4\n");


    float *coreDistances;
    cudaMallocManaged(&coreDistances,(size_t)(numValues) * sizeof(float)); 
    calculateCoreDistance(distances,coreDistances,numValues,k,k-1);   
    Check();


    cudaFree(distances); distances = NULL; 

printf("PROTAGONISTA 5\n");

    calculateMutualReachabilityDistance(g.eweight,coreDistances,aux_nodes,g.nlist,g.edges);  //Aqui usa GPU

    Check();

    cudaFree(aux_nodes);
    aux_nodes = NULL;

    cudaFree(coreDistances);
    coreDistances = NULL;

printf("PROTAGONISTA 6\n");

    // Read vector txtx
    calculate_coreDistance_antihubs(&g,auxiliar_edges,antihubs,num_antihubs);

printf("PROTAGONISTA 7\n");

  return g;   
}




ECLgraph buildEnhancedKNNG(int *kNN, float *distances, int shards_num, long int numValues,long int k,long int mpts ,int mst_gpu){




    long int vectorSize = numValues*k;
    int *finalCounts = calculate_degrees(kNN,vectorSize,numValues);
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


    avoid_pageFault(numValues,vertexes,false);

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

printf("PROTAGONISTA 1\n");

    
    g = buildECLgraph(numValues, vectorSize,kNN, distances,k,mpts, antihubs, pos_threshold,mst_gpu);






printf("PROTAGONISTA 8\n");

    return g;
}
