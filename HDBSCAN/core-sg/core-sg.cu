#include "core-sg.cuh"






GPUECLgraph buildGPUgraph(string path_to_kNNG,ECLgraph *g,int *multiplicacoes){

  NNDElement *result_graph2;
  long int knng_num2, knng_dim2;
  double min_dist = FileTool::GetMinDistanceFromKNNG(path_to_kNNG,&result_graph2, &knng_num2, &knng_dim2);
  int multiplica = ceil(-log10(fabs(min_dist))) + 2;


  double *eweights = new double[g->edges];
  for(long int i=0;i<g->edges;i++)
    eweights[i] = g->eweight[i];

  GPUECLgraph g_gpu;
  g_gpu.edges = g->edges;
  g_gpu.nodes = g->nodes;

  g_gpu.nlist = new int[g->edges];
  g_gpu.eweight = new int[g->edges];
  g_gpu.nindex = new int[g_gpu.nodes+1];


  for(int i=0;i<g->nodes+1;i++){
    g_gpu.nindex[i] = g->nindex[i];
  }

  for(int i=0;i<g->edges;i++){
    g_gpu.nlist[i] = g->nlist[i];
    g_gpu.eweight[i] =  (int)(pow(10.0,multiplica) *  eweights[i] / 1);
  }

  *multiplicacoes = multiplica;
  delete eweights; eweights = NULL;
  delete result_graph2; result_graph2 = NULL;

  return g_gpu;

}

void freeECLgraph(ECLgraph &g)
{
  if (g.nindex != NULL) cudaFree(g.nindex);
  if (g.nlist != NULL) cudaFree(g.nlist);
  if (g.eweight != NULL) cudaFree(g.eweight);
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
}

void freeECLgraphGPU(GPUECLgraph &g)
{
  if (g.nindex != NULL) free(g.nindex);
  if (g.nlist != NULL) free(g.nlist);
  if (g.eweight != NULL) free(g.eweight);
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
}

int* extract_knng_index(NNDElement *kNN_,long int knng_num,long int knng_dim){

    int *kNN = new int[(size_t)knng_num*knng_dim * sizeof(int)];

    #pragma omp parallel for
    for (long int i = 0; i < knng_num; i++) {
      for (long int j = 0; j < knng_dim; j++) {
        kNN[i * knng_dim + j] = kNN_[i * knng_dim + j].label();
}

    } 

    CheckCUDA_();

    return kNN;

}


float* extract_knng_distance(NNDElement *kNN_,long int knng_num,long int knng_dim){

    float *kNN = new float[(size_t)knng_num*knng_dim * sizeof(float)];

    #pragma omp parallel for
    for (long int i = 0; i < knng_num; i++) {
      for (long int j = 0; j < knng_dim; j++) {

        kNN[i * knng_dim + j] = kNN_[i * knng_dim + j].distance();
}

    } 

    CheckCUDA_();

    return kNN;

}


void core_sg_calculate_nindex(long int nodes,int* flag_knn,int *kNN,ECLgraph *g,MSTedge* mst_edges){

    //1.2 Fazer o processo de construção do grafo para o kNNG
     // Calcula quantas arestas cada noh terá, levando em conta que eh um grafo não direcional.
    for (long int i=0;i<nodes;i++){

        long int soma = 0;


        for (long int j=0;j<k;j++){

            long int neig = kNN[i*k + j];

            //Verifica se i esta na lista de neig
            int FLAG = flag_knn[i*k + j];

            if (FLAG > 1){ g->nindex[neig+1] += FLAG-1; g->nindex[i+1] -= (FLAG-1);}

            g->nindex[neig+1] +=1;


            if (!FLAG)
                soma += 1;
        }
        g->nindex[i+1] += soma;
    }

    // Adicionar a MST, semelhantemente aos antihubs.
    for (long int i=0;i<nodes-1;i++){

      int to_node = mst_edges[i].to_node;
      int from_node = mst_edges[i].from_node;

      g->nindex[to_node+1] += 1;
      g->nindex[from_node+1] += 1;

    }
     

    //Calcular offsets
    for (long int i=1;i<nodes+1;i++){

        g->nindex[i] = g->nindex[i-1] + g->nindex[i];

    }

  return ;
}


void core_sg_calculate_nlist(long int nodes,int* flag_knn,int *kNN,ECLgraph *g,MSTedge* mst_edges,long int* auxiliar_edges,long int* offsets,
  float *euclidean_distance,float* distance){

     long int k2 = k;

    // Adiciona os vizinhos paralelamente
    omp_set_num_threads(32);
    #pragma omp parallel for 
    for (long int i = 0; i < nodes; i++) {
        
        // Calcula o offset do ponto
        long int edge_offset = g->nindex[i];

        long int pos = edge_offset + auxiliar_edges[i];    

        for (long int j = 0; j < k2; j++) {
        
            // Pega o   ndice do vizinho  
            long int neig = kNN[i * k2 + j];

            euclidean_distance[pos] = distance[i*k2+j];
            g->nlist[pos] = neig;
            auxiliar_edges[i] += 1;

            pos += 1; 

        }
    }

    // Adiciona vizinhos que não são mútuos
    for (long int i = 0; i < nodes; i++) {
        

        for (long int j = 0; j < k2; j++) {
        
            // Pega o   indice do vizinho  
            long int neig = kNN[i * k2 + j];

            int FLAG = flag_knn[i*k2+j]; 

            // Deu problema
            if (!FLAG){

                //Calcula Propriedades de NEIG em NList
                long int neig_edge_offset = g->nindex[neig];


                long int neig_pos = neig_edge_offset + auxiliar_edges[neig];
                // Adicionando o idx i na lista do neig
                auxiliar_edges[neig] += 1;
                g->nlist[neig_pos] = i;
                euclidean_distance[neig_pos] = distance[i*k2+j];

            }
        }
    }


    cudaMemcpy(offsets, auxiliar_edges, g->nodes * sizeof(long int), cudaMemcpyDeviceToDevice);
    avoid_pageFault(g->nodes,offsets,true);




   for (long int i=0;i<nodes-1;i++){

        int to_node = mst_edges[i].to_node;
        int from_node = mst_edges[i].from_node;

        long int pos_begin_to = g->nindex[to_node];
        long int offset_to = auxiliar_edges[to_node];

        long int pos_begin_from = g->nindex[from_node];
        long int offset_from = auxiliar_edges[from_node];


        g->nlist[pos_begin_to+offset_to] = from_node;
        g->nlist[pos_begin_from+offset_from] = to_node;

        auxiliar_edges[to_node] += 1;
        auxiliar_edges[from_node] += 1;
    }



    return ;
}




void core_sg_calculate_MRD_mst(ECLgraph *g,long int *auxiliar_edges,MSTedge *mst_edges,long int nodes,float *euclidean_distance_){


    // Pegar os pontos (todos, Utilizando a função que já está pronta e está na primeira versão)
    long int data_dim = -1;
    float* vectors_data = ReadObjects(&data_dim);


     assert(data_dim != -1);


    // Faz o loop. Calcula a euclideana entre os dois pontos e comparar com a core_distance. Lembra de incrementar auxiliar edges pra não dar merda.
    for(long int i=0;i<nodes-1;i++){

      int idx_a = mst_edges[i].from_node;
      int idx_b = mst_edges[i].to_node;

      float distance = mst_edges[i].weight;

      float euclidean_distance = calculate_euclidean_distance(vectors_data,idx_a,idx_b,data_dim);

      if (distance < euclidean_distance)
        distance = euclidean_distance;
      

      long int pos_begin_a = g->nindex[idx_a] + auxiliar_edges[idx_a];
      long int pos_begin_b = g->nindex[idx_b] + auxiliar_edges[idx_b];

      auxiliar_edges[idx_a] += 1;
      auxiliar_edges[idx_b] += 1;

      g->eweight[pos_begin_a] = distance;
      g->eweight[pos_begin_b] = distance;

      euclidean_distance_[pos_begin_a] = euclidean_distance;
      euclidean_distance_[pos_begin_b] = euclidean_distance;

    }

    free(vectors_data); vectors_data = NULL;

    return;

}


ECLgraph Union_kNNG_MST(long int nodes,MSTedge *mst_edges){

    // Calcular nindex, com auxílio do dicionário que será lido por aqui
      //1.1 Ler dicionário de booleanos
    int* flag_knn = read_bool_dict();
    long int knng_num = nodes;
    long int knng_dim = k;
    NNDElement *kNN_ = ReadkNNGgraph();

    int *kNN = extract_knng_index(kNN_,knng_num,knng_dim);
    float *distances = extract_knng_distance(kNN_,knng_num,knng_dim);

printf("PROTAGONISTA 11\n");


    CheckCUDA_();
    delete kNN_;
    kNN_ = NULL;


    ECLgraph g;
    g.nodes = nodes;


    cudaMallocManaged(&g.nindex,(size_t)(g.nodes + 1) * sizeof(long int));
    int gridSize = (g.nodes + 1 + blockSize - 1) / blockSize;
    initializeVectorCounts_<<<gridSize,blockSize>>>(g.nindex,0,g.nodes+1); // Aqui usar GPU
    Check();
    cudaMemPrefetchAsync(g.nindex,(size_t)(g.nodes + 1) * sizeof(g.nindex[0]),cudaCpuDeviceId);       


    core_sg_calculate_nindex(nodes,flag_knn,kNN,&g,mst_edges);

printf("PROTAGONISTA 12\n");

    long int *offsets;
    cudaMallocManaged(&offsets,(size_t)(g.nodes) * sizeof(long int)); // nindex[0] = X, nindex[1] = y, nindex[2] = z

    // Adicionar as arestas do jeito maneiro. 
    long int *auxiliar_edges;
    cudaMallocManaged(&auxiliar_edges,(size_t)(g.nodes) * sizeof(long int)); // nindex[0] = X, nindex[1] = y, nindex[2] = z
    gridSize = (g.nodes + 1  + blockSize - 1) / blockSize;
    initializeVectorCounts_<<<gridSize,blockSize>>>(auxiliar_edges,0,g.nodes); //Aqui usar GPU
    avoid_pageFault(g.nodes,auxiliar_edges,true);
    Check();

printf("PROTAGONISTA 13\n");

    cudaMallocManaged(&g.nlist,(size_t)(g.nindex[nodes]) * sizeof(int));
    g.edges = g.nindex[nodes];
    float *euclidean_distance = new float[g.edges];

printf("PROTAGONISTA 14\n");

    core_sg_calculate_nlist(nodes,flag_knn,kNN,&g,mst_edges,auxiliar_edges,offsets,euclidean_distance,distances);

printf("PROTAGONISTA 15\n");

    free(kNN);
    kNN = NULL;

    cudaFree(auxiliar_edges);
    auxiliar_edges = NULL;

    // Adicionar o peso maneiro. Já ta tudo calculado. Precisa calcular somente a core_distance. Da pra adicionar de modo paralelo. Provavelmente
	  int *aux_nodes;
    cudaMallocManaged(&aux_nodes,(size_t)g.edges * sizeof(g.nlist[0]));
    createNodeList(aux_nodes,&g) ; //Aqui usar GPU
    Check();
         
    long int elementsPerGPU[numGPUs];
    calculateElements(elementsPerGPU,numGPUs,nodes); 


printf("PROTAGONISTA 16\n");

    float *coreDistances;
    cudaMallocManaged(&coreDistances,(size_t)(nodes) * sizeof(float)); 
    calculateCoreDistance(distances,coreDistances,nodes,k,mpts-1);   
    Check();

    free(distances);
    distances = NULL;

printf("PROTAGONISTA 17\n");

  cudaMallocManaged(&g.eweight,(size_t)g.edges * sizeof(g.eweight[0]));


    calculateMutualReachabilityDistance(g.eweight,coreDistances,aux_nodes,g.nlist,g.edges);  //Aqui usa GPU

     cudaFree(aux_nodes);
    aux_nodes = NULL;

printf("PROTAGONISTA 18\n");

    // Calcular a MRD para os pontos da MST  
    core_sg_calculate_MRD_mst(&g,offsets,mst_edges,nodes,euclidean_distance);
    // Retornar o grafo

printf("PROTAGONISTA 19\n");

    write_euclidean_distance_vecs(euclidean_distance,g.edges);  


    free(euclidean_distance); euclidean_distance = NULL;
    cudaFree(offsets); offsets = NULL;
    free(mst_edges);mst_edges = NULL;
    cudaFree(coreDistances);coreDistances = NULL;
printf("PROTAGONISTA 20 EDGES = %ld\n",g.edges);

    return g;

}






// Perguntar para o hermes se existe essa possibilidade?
void calculate_euclidean_distance_core_sg(ECLgraph* g,float* vector_data,long int dim,float *euclidean_distances,int *aux_nodes){



  long int totl_edges = g->edges;
  #pragma parallel omp parallel 
  for(long int i =0;i<totl_edges;i++){

    long int idx_a = g->nlist[i];
    long int idx_b = aux_nodes[i];

    euclidean_distances[i] = calculate_euclidean_distance(vector_data,idx_a,idx_b,dim);

  }

  return ;
}


void update_core_sg_weights(ECLgraph* g ,int nodes,long int mpts){


  // Ler o kNNG
  NNDElement *kNN_ = ReadkNNGgraph();
  long int knng_num = nodes;
  long int knng_dim = k;

  float *distances = extract_knng_distance(kNN_,knng_num,knng_dim);

  CheckCUDA_();
  delete kNN_;
  kNN_ = NULL;

  //Calcula o novo vetor de core-distances
   float *coreDistances;
  cudaMallocManaged(&coreDistances,(size_t)(nodes) * sizeof(float)); 
  calculateCoreDistance(distances,coreDistances,nodes,k,mpts-1);   
  Check();


  // Temos o grafo G e as arestas, porém sem PESO
  // MRD -> MAX(core_distance_i, core_distance_j, distancia_i_j)
  // Core distance já temos
  // Não temos -> Distância euclidiana
   int *aux_nodes;
   cudaMallocManaged(&aux_nodes,(size_t)g->edges * sizeof(g->nlist[0]));
   createNodeList(aux_nodes,g) ; 
   Check();


   cudaFree(g->eweight); g->eweight = NULL;


   g->eweight = read_euclidean_distance_vecs();    


// Calcular a MRD, temos o vetor g->eweight com a distancia euclidiana dos pontos. MUito simples agora. Shard em aux_node, g eweight e g nlist
   
}





ECLgraph  build_CoreSG(NNDElement *result_graph,long int numValues,long int mpts,int mst_gpu){

    long int knng_num = numValues;
    long int knng_dim = k;

    // Cast NNDElement to int and float
    int *result_index_graph;
    float *distances;


    cudaMallocManaged(&result_index_graph,(size_t)knng_num*knng_dim * sizeof(int));
    for (long int i = 0; i < knng_num; i++) {
      for (long int j = 0; j < knng_dim; j++) {
        
        result_index_graph[i * knng_dim + j] = result_graph[i * knng_dim + j].label();
        }

    } 

    CheckCUDA_();

   
    cudaMallocManaged(&distances,(size_t)numValues*knng_dim * sizeof(float));


    for (long int i = 0; i < knng_num; i++) {
        for (long int j = 0; j < knng_dim; j++) {
          
          distances[i * knng_dim + j] = result_graph[i * knng_dim + j].distance();
        }
    }





    CheckCUDA_();
    delete result_graph;
    result_graph = NULL;

    //HDBSCAN
    int shards_num = numGPUs;
    if (numValues*k > 2000000000 )
        shards_num += numGPUs;

    // Constroí o kNNG enhanced
    ECLgraph g;
    g = buildEnhancedKNNG(result_index_graph,distances,shards_num,numValues,k,mpts,mst_gpu);

printf("PROTAGONISTA 9\n");

    // Variáveis para a MST
    bool* edges;
    MSTedge *mst_edges;

    int qntd_nohs = g.nodes;
    if(mst_gpu){

        int multiplicacoes=0;
        GPUECLgraph g_gpu = buildGPUgraph("/nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.kgraph",&g,&multiplicacoes);     
        edges = gpuMST(g_gpu, INT_MAX);
        mst_edges = buildMST_gpu(g_gpu,edges,multiplicacoes);
            
        freeECLgraphGPU(g_gpu);

    }

    else{

        edges = cpuMST(g);
        mst_edges = buildMST(g,edges);
    }

    // Liberar G
    freeECLgraph(g);
    //g = NULL;


printf("PROTAGONISTA 10\n");


    ECLgraph core_sg;
    // União kNNG + MST
    core_sg = Union_kNNG_MST(numValues,mst_edges);

    printf("PROTAGONISTA ultimo\n");

    // EScrever o grafo na memória
    update_core_sg_weights(&core_sg,numValues,k-2);
    return core_sg;
}




