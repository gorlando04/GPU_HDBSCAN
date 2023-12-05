#include "graph.cuh"
#include "../initializer/initialize.cuh"
#include "../getters/getters.cuh"
#include "../calculates/calculates.cuh"
#include "../counts/count.cuh"


#include <algorithm>



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





void createNodeList(int *vector,ECLgraph *g){

    for(int i=0;i<g->nodes;i++){
        
        long int begin = g->nindex[i];
        long int end = g->nindex[i+1];

        for (long int j=begin;j<end;j++)
            vector[j] = i;
    }
}

void createEdgeList(int *vector,ECLgraph *g){

    for(long int i=0;i<g->edges;i++){
        
        vector[i] = g->nlist[i];
    }
}

void createWeightList(float *vector,ECLgraph *g){

    for(long int i=0;i<g->edges;i++){
        
        vector[i] = g->eweight[i];
    }
}

ECLgraph buildECLgraph(int nodes, long int edges,int *kNN, float *distances,int k, int *antihubs, long int num_antihubs,float *vectors_data,int dim)
{



  ECLgraph g;


  g.nodes = nodes;




   /*Isso significa, que o nó 0 está conectado com Y-x NÓS,
   O nó 1 está conetado com z-y nós, e etc...*/
    cudaMallocManaged(&g.nindex,(size_t)(g.nodes + 1) * sizeof(g.nindex[0])); // nindex[0] = X, nindex[1] = y, nindex[2] = z

    int gridSize = (g.nodes + 1 + blockSize - 1) / blockSize;


    initializeVectorCounts_<<<gridSize,blockSize>>>(g.nindex,0,g.nodes+1);

    cudaDeviceSynchronize();

    CheckCUDA_();

    cudaMemPrefetchAsync(g.nindex,(size_t)(g.nodes + 1) * sizeof(g.nindex[0]),cudaCpuDeviceId);



    // Calcula quantas arestas cada noh terá, levando em conta que eh um grafo não direcional.
    for (long int i=0;i<nodes;i++){
        long int soma = 0;
        for (long int j=0;j<k;j++){

            long int neig = kNN[i*k + j];

            //Verifica se i esta na lista de neig
            int FLAG = findKNNlist(kNN,neig,i,k);

            g.nindex[neig+1] += 1;

            if (!FLAG)
                soma += 1;
        }
        g.nindex[i+1] += soma;
    }


    // Adicionar os antihubs
    int contador = 0;

    for (long int i=0;i<nodes;i++)

        if (i == antihubs[contador]){
            contador ++;
            g.nindex[i+1] += (num_antihubs-1);
    }
    
    //Calcular offsets
    for (long int i=1;i<nodes+1;i++){

        g.nindex[i] = g.nindex[i-1] + g.nindex[i];

    }

    // Nesse pontos os nós já estão calculados, agora precisamos inserir as arestas. Essa parte será bem demorada.

    long int *auxiliar_edges;

    cudaMallocManaged(&auxiliar_edges,(size_t)(g.nodes) * sizeof(long int)); // nindex[0] = X, nindex[1] = y, nindex[2] = z

    gridSize = (g.nodes + 1 + blockSize - 1) / blockSize;


    initializeVectorCounts_<<<gridSize,blockSize>>>(auxiliar_edges,0,g.nodes);

    cudaDeviceSynchronize();

    CheckCUDA_();



    g.nlist = (int*)malloc(g.nindex[nodes] * sizeof(g.nlist[0]));
    g.edges = g.nindex[nodes];

    // Adicionar as arestas sem antihubs
    long int k2 = k;

    for (long int i=0;i<nodes;i++)

        for (long int j=0;j<k2;j++){

            // Posição de insertion da edge no vetor nlist.     
            long int edge_offset = g.nindex[i];
            int insertion_offset = auxiliar_edges[i];
            long int pos = edge_offset + insertion_offset;

            
            // Pega o índice do vizinho
            long int neig = kNN[i*k2+j];

            // Verifica se o ponto esta na lista dos vizinhos do outro ponto
            int FLAG = findKNNlist(kNN,neig,i,k2);

            // Calcula o offset do vizinho
            long int neig_edge_offset = g.nindex[neig];
            int neig_insertion_offset = auxiliar_edges[neig];
            long int neig_pos = neig_edge_offset + neig_insertion_offset;

            auxiliar_edges[neig] += 1;

            g.nlist[neig_pos] = i;

            // Se não tiver na lista de kNN do vizinho
            if (!FLAG){
                g.nlist[pos] = neig;
                // Adiciona mais uma inserção no vetor auxiliar
                auxiliar_edges[i] += 1;
	   }
        }

    // Adiciona os antiHubs

    for (long int i=0;i<num_antihubs;i++){

        int current = antihubs[i];
        long int pos_begin = g.nindex[current];
        long int offset = auxiliar_edges[current];

        for (long int j=0;j<num_antihubs;j++){
            int neig = antihubs[j];

            if (neig != current){
                g.nlist[pos_begin+offset] = neig;
                offset++;
            }
        }
    }
        g.eweight = (float*)malloc(g.edges * sizeof(g.eweight[0]));

	 int *aux_nodes;
        cudaMallocManaged(&aux_nodes,(size_t)g.edges * sizeof(g.nlist[0]));
        createNodeList(aux_nodes,&g);

         
        long int elementsPerGPU[numGPUs];
        calculateElements(elementsPerGPU,numGPUs,numValues);


        float *coreDistances;
        cudaMallocManaged(&coreDistances,(size_t)(numValues) * sizeof(float)); 
        calculateCoreDistance(distances,coreDistances,elementsPerGPU,k);


        float *graphDistances;
        cudaMallocManaged(&graphDistances,(size_t)g.edges * sizeof(g.eweight[0]));

        int *aux_edges;
        cudaMallocManaged(&aux_edges,(size_t)g.edges * sizeof(g.nlist[0]));
        for (long int i=0;i<g.edges;i++){
            aux_edges[i] = g.nlist[i];
        }

        calculateMutualReachabilityDistance(graphDistances,coreDistances,aux_nodes,aux_edges,g.edges); 


    for (long int i=0;i<num_antihubs;i++){

        int idx_a = antihubs[i];
        long int pos_begin = g.nindex[idx_a] + auxiliar_edges[idx_a];
        for (long int j=i;j<num_antihubs-1;j++){
            int idx_b = antihubs[j+1];
            
            //Calcula distancia euclidiana
            float euclidean_distance = calculate_euclidean_distance(vectors_data,idx_a,idx_b,dim);

            if (graphDistances[pos_begin + j] < euclidean_distance){
                graphDistances[pos_begin+j] = euclidean_distance;

                long int pos_begin2 = g.nindex[idx_b] + auxiliar_edges[idx_b];
                graphDistances[pos_begin2 + i] = euclidean_distance;
            }


        }
    }

    for (long int i=0;i<g.edges;i++){
        g.eweight[i] = graphDistances[i];
    } 


 
  return g;   
}


ECLgraph buildEnhancedKNNG(int *h_data, float *distances, int shards_num,float *vectors_data,int dim){


     clock_t t; 
    t = clock(); 
    // Aqui da pra colocar um perfectch
    for (int i=0;i<numGPUs;i++){
        cudaSetDevice(i);
        cudaMemPrefetchAsync(h_data,(size_t)vectorSize * sizeof(int),i);
    }

    printf("Comecand a busca\n");

 
    long int elementsPerGPU[shards_num];

    // Calcula a quantidade de elementos por GPU
    calculateElements(elementsPerGPU,shards_num);


    // Realiza a contagem de graus para todos os vértices
    int *finalCounts; // Contagens finais após a combinação das GPUs

    cudaMallocManaged(&finalCounts,(size_t)numValues * sizeof(int));

    
    int gridSize = (numValues + blockSize - 1) / blockSize;

    // Inicializa o vetor
    initializeVectorCounts<<<gridSize,blockSize>>>(finalCounts,0,numValues);

    cudaDeviceSynchronize();

    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s hehehehehe",cudaGetErrorString(cuda_status));
        exit(-1);
    }

    // Conta os graus de cada vértice
    countDegrees(finalCounts,h_data,numGPUs,elementsPerGPU);


    Vertex *vertexes;

    //vertexes = new Vertex[numValues];
    // Aloca memória para o vetor na CPU
	cudaMallocManaged(&vertexes,(size_t)numValues * sizeof(Vertex));


  
    //Configura a grade de threads
    gridSize = (numValues + blockSize - 1) / blockSize;    
    
    // Inicializa o vetor de vértices, com  os graus específicos.
    initializeVertex<<<gridSize,blockSize>>>(vertexes,finalCounts,numValues);

    cudaDeviceSynchronize();

CheckCUDA_();

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

    // Aloca memória para o vetor na CPU
	cudaMallocManaged(&treshold_idx,(size_t)countsTreshold * sizeof(int));
  


    CheckCUDA_();

    // Pega os índices dos pontos que são iguais ao threshold
    get_IndexThreshold(finalCounts,treshold_idx,value_threshold);



    for (int i=0;i<numGPUs;i++){
        cudaSetDevice(i);
        cudaMemPrefetchAsync(treshold_idx,(size_t)countsTreshold * sizeof(int),i);
    }

    long int indexesPerGPU[numGPUs];

    // Calcula quantos elementos serão processados por cada GPU
    calculateElements(indexesPerGPU,numGPUs,countsTreshold);


    Untie_hub *unties;
    unties = new Untie_hub[countsTreshold];

    // Calculata os scores dos empates
    calculateUntieScore(unties,indexesPerGPU,h_data,treshold_idx,finalCounts);

    // Pega quantos empates temos na lista final
    int missing_ties = get_TiedVertexes(vertexes,pos_threshold,value_threshold);
    int not_ties = pos_threshold - missing_ties;

    std::partial_sort(unties, unties + missing_ties, unties + countsTreshold, compareVertexByScore);

    int *antihubs;

    antihubs = new int[pos_threshold];


    // Junta todos os antihubs em um vetor
    joinAntiHubs(antihubs,vertexes,not_ties,unties,missing_ties);

    // Ordena pelo índice para inserir na MST
    std::sort(antihubs,antihubs+pos_threshold);


    printf("Iniciando a construcao do grafo\n");

    ECLgraph g;
    
    g = buildECLgraph(numValues, vectorSize,h_data, distances,k, antihubs, pos_threshold,vectors_data,dim);


    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("Demorou %lf segundos para montar o enhanced kNNG\n",time_taken);

    return g;
}
