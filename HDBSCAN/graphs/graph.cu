#include "graph.cuh"
#include "../initializer/initialize.cuh"
#include "../getters/getters.cuh"
#include "../calculates/calculates.cuh"
#include "../counts/count.cuh"
#include <time.h>
#include <unistd.h>
#include "../../tools/filetool.hpp"

#include <algorithm>
#include <vector>



void joinAntiHubs(int *antihubs,Vertex *vertexes,int not_ties, Untie_hub *unties,int missing_ties){

    // Bota os não empatados
    for(int i=0;i< not_ties;i++){
        antihubs[i] = vertexes[i].index;

    }

    for(int i=0;i<missing_ties;i++){
        antihubs[i+not_ties] = unties[i].index;
    }

    return ;
}

void Check(){

    cudaDeviceSynchronize();

    CheckCUDA_();
    return ;
}





ECLgraph buildECLgraph(int nodes, long int edges,int *kNN, float *distances,int k,long int mpts, int *antihubs, long int num_antihubs,int mst_gpu)
{

    long int numValues = nodes;

    ECLgraph g; g.nodes = nodes;

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

    Check();

    cudaMemPrefetchAsync(auxiliar_edges,(size_t)(g.nodes ) * sizeof(long int),cudaCpuDeviceId); //Inserir no código


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


    cudaFree(kNN);
    
    kNN = NULL;

    if(mst_gpu != 1){
	cudaFree(distances); distances = NULL; }


    calculateMutualReachabilityDistance(g.eweight,coreDistances,aux_nodes,g.nlist,g.edges);  //Aqui usa GPU

    Check();


    // Read vector 
    float *vectors_data;
    const std::string path_to_data_binary = "/nndescent/GPU_HDBSCAN/data/vectors.fvecs";
    long int data_size2, data_dim2;

    FileTool::ReadBinaryAntihubs(path_to_data_binary,&vectors_data, &data_size2, &data_dim2,antihubs,num_antihubs);    	

    for (long int i=0;i<num_antihubs;i++){

        int idx_a = antihubs[i];
        long int pos_begin = g.nindex[idx_a] + auxiliar_edges[idx_a];
        for (long int j=i;j<num_antihubs-1;j++){
            int idx_b = antihubs[j+1];
            
            //Calcula distancia euclidiana
            float euclidean_distance = calculate_euclidean_distance(vectors_data,i,j+1,data_dim2);

            if (g.eweight[pos_begin + j] < euclidean_distance){
                g.eweight[pos_begin+j] = euclidean_distance;

                long int pos_begin2 = g.nindex[idx_b] + auxiliar_edges[idx_b];
                g.eweight[pos_begin2 + i] = euclidean_distance;
            }
        }
    }

  return g;   
}





ECLgraph buildEnhancedKNNG(int *h_data, float *distances, int shards_num, long int numValues,long int k,long int mpts,int mst_gpu){

    long int vectorSize = numValues*k;

    cudaMemPrefetchAsync(h_data,(size_t)vectorSize * sizeof(int), cudaCpuDeviceId);
 
    long int elementsPerGPU[shards_num];

    // Calcula a quantidade de elementos por GPU
    calculateElements(elementsPerGPU,shards_num,vectorSize);


    // Realiza a contagem de graus para todos os vértices
    int *finalCounts; // Contagens finais após a combinação das GPUs

    cudaMallocManaged(&finalCounts,(size_t)numValues * sizeof(int));

    
    int gridSize = (numValues + blockSize - 1) / blockSize;

    // Inicializa o vetor
    initializeVectorCounts<<<gridSize,blockSize>>>(finalCounts,0,numValues);

    Check();

    // Conta os graus de cada vértice
    countDegrees(finalCounts,h_data,numGPUs,elementsPerGPU,numValues);

    Vertex *vertexes;

    //vertexes = new Vertex[numValues];
    // Aloca memória para o vetor na CPU
	cudaMallocManaged(&vertexes,(size_t)numValues * sizeof(Vertex));


    //Configura a grade de threads
    gridSize = (numValues + blockSize - 1) / blockSize;    
    
    // Inicializa o vetor de vértices, com  os graus específicos.
    initializeVertex<<<gridSize,blockSize>>>(vertexes,finalCounts,numValues);

    Check();

    // Pegar os threshold
    int pos_threshold = get_NumThreshold(numValues);

    // Ordenação Parcial
    std::partial_sort(vertexes, vertexes + pos_threshold, vertexes + numValues, compareVertexByDegree);


    // Pegar o valor do threshold 
    int value_threshold = vertexes[pos_threshold-1].grau;

    printf("A posicao do threshold eh: %d e o valor eh: %d\n",pos_threshold-1,value_threshold);

    // Evita page fault
    for (int i=0;i<numGPUs;i++){
        cudaSetDevice(i);
        cudaMemPrefetchAsync(vertexes,(size_t)numValues * sizeof(Vertex),i);
    }

    // Encontrar quantos valores são iguais ao threshold
    long int elementsPerGPU_[numGPUs];
    calculateElements(elementsPerGPU_,numGPUs,numValues);


    // Encontra quantos valores são iguais ao threshold
    int countsTreshold = countThreshold_(elementsPerGPU_,vertexes,value_threshold);

    int *treshold_idx;

    int *antihubs;

    antihubs = new int[pos_threshold];

     Untie_hub *unties;

     if (countsTreshold > 1){
    // Aloca memória para o vetor na CPU
	cudaMallocManaged(&treshold_idx,(size_t)countsTreshold * sizeof(int));
  

    CheckCUDA_();

    // Pega os índices dos pontos que são iguais ao threshold
    get_IndexThreshold(finalCounts,treshold_idx,value_threshold,numValues);



    for (int i=0;i<numGPUs;i++){
        cudaSetDevice(i);
        cudaMemPrefetchAsync(treshold_idx,(size_t)countsTreshold * sizeof(int),i);
    }

    long int indexesPerGPU[numGPUs];

    // Calcula quantos elementos serão processados por cada GPU
    calculateElements(indexesPerGPU,numGPUs,countsTreshold);


    unties = new Untie_hub[countsTreshold];

    // Calculata os scores dos empates
    calculateUntieScore(unties,indexesPerGPU,h_data,treshold_idx,finalCounts,k);

    // Pega quantos empates temos na lista final
    int missing_ties = get_TiedVertexes(vertexes,pos_threshold,value_threshold);
    int not_ties = pos_threshold - missing_ties;

    std::partial_sort(unties, unties + missing_ties, unties + countsTreshold, compareVertexByScore);

    // Junta todos os antihubs em um vetor
    joinAntiHubs(antihubs,vertexes,not_ties,unties,missing_ties);

    delete unties;
    unties = NULL;


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
    cudaFree(treshold_idx);

    finalCounts = NULL;
    vertexes = NULL;
    treshold_idx = NULL;


    ECLgraph g;
    
    g = buildECLgraph(numValues, vectorSize,h_data, distances,k,mpts, antihubs, pos_threshold,mst_gpu);

    return g;
}