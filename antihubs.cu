#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <time.h>
#include "ECLgraph.h"
#include <string>

#include <climits>
#include <tuple>
#include <vector>
#include "tools/filetool.hpp"
#include "mst.h"



const int numGPUs = 3;
const long int numValues = /*400000000;*/1000000; // Número de valores possíveis no vetor
const long int vectorSize = /*12000000000*/32000000;// Tamanho do vetor
const int blockSize = 256;
const int k = 32;

struct Vertex {

    int index;
    int grau;
};

struct Untie_hub
{
    int index;
    int score=0;
};


void CheckCUDA_(){

    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }
}
__global__ void initializeVectorCounts(int *vector,int value,int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = value;
    }

}

// Função do kernel CUDA para calcular a contagem de valores em um vetor
__global__ void countValues(int *data, int *counts, long int size, long int off_set_size,int offset=0) {


    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;
    if (tid < size) {
        long int value = data[tid+ off_set_size*offset];
        atomicAdd(&counts[value], 1);
    }
}


// Custom comparator to sort MyStruct based on the 'value' field
bool compareVertexByDegree(const Vertex &a, const Vertex &b) {
    return a.grau < b.grau;
}



__global__ void initializeVertex(Vertex *vertexes, int *counts,long int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vertexes[tid].grau = counts[tid];
        vertexes[tid].index = tid;
    }

}



// Função do kernel CUDA para calcular a contagem de valores em um vetor
__global__ void countTreshold(Vertex *data, int *counts, long int size, long int off_set_size,int offset,int comparision) {

    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;
    if (tid < size) {
        long int value = data[tid + off_set_size*offset].grau;

        if (value == comparision)
            atomicAdd(&counts[0], 1);
    }
}



__global__ void calculateScore(int *vectors,int *treshold_idx, Untie_hub *vertex , int *degrees ,long int size,int k,int offset){

    //Idx que iremos calcular o score
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;

    if (tid < size){
        int idx = treshold_idx[tid+ (offset) ];
        vertex[tid].index = idx;

        for (int j=0;j<k;j++){
            int neig = vectors[tid+j];
            vertex[tid].score += degrees[neig];
        }
    }

}

// Custom comparator to sort MyStruct based on the 'value' field
bool compareVertexByScore(const Untie_hub &a, const Untie_hub &b) {
    return a.score < b.score;
}


void generate_random(int *h_data){

   // Gera o vetor de teste
  for (long int i = 0; i < vectorSize; i++) {
        h_data[i] = /*i / k;*/rand() % numValues;
    }   

    return;
}

void calculateElements(long int *elementsPerGPU,int shards_num,long int vectorSize=vectorSize){

        //Define o tamanho de cada shard
    for (int i=0;i<shards_num;i++){

        if (i != (shards_num)-1){ elementsPerGPU[i] = vectorSize / (shards_num);}

        else{
            elementsPerGPU[i] = vectorSize;

            for (int j=0;j<i;j++)
                elementsPerGPU[i] -= elementsPerGPU[j];
            

        }
    }

    return;
}


void countDegrees(int *finalCounts,int *h_data,int shards_num,long int *elementsPerGPU){

    int *d_counts[shards_num];  // Vetores na GPU
    int *h_counts[shards_num]; // Contagens na CPU para cada GPU

        // Inicializa contagens na CPU para cada GPU
    for (int i = 0; i < shards_num; i++) {
        h_counts[i] = new int[numValues];
    }

    for (int i = 0; i < numGPUs; i++) {

        int idx = i ;

        cudaSetDevice(i);

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&d_counts[idx], numValues * sizeof(int)); // Vetor de frequências


        // Configura a grade de threads
        long int numBlocks = ( elementsPerGPU[idx]/ blockSize) +1;
        long int gridSize = (numValues + blockSize - 1) / blockSize;
        

        initializeVectorCounts<<<gridSize,blockSize>>>(d_counts[idx],0,numValues);
        
        countValues<<<numBlocks, blockSize>>>(h_data, d_counts[idx], elementsPerGPU[idx],elementsPerGPU[0],idx);


    
    }

    //Libera a memória

    for (int i = 0; i < numGPUs; i++) {

        int idx = i ;
        cudaSetDevice(i);

        cudaDeviceSynchronize();

        cudaMemcpy(h_counts[idx], d_counts[idx], numValues * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_counts[idx]);  

    }
  

    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s hehehehehe",cudaGetErrorString(cuda_status));
        exit(-1);
    }


     // Combina as contagens de todas as GPUs na CPU

    for (int i = 0; i < shards_num; i++) {
        for (int j = 0; j < numValues; j++) {
            finalCounts[j] += h_counts[i][j];
        }
    }   


}

int get_NumThreshold(long int numValues_){

    return sqrt(numValues_);
}

int get_TiedVertexes(Vertex *vertexes,int pos_threshold, int value_threshold){

        // Pega quantos empates temos na lista final
    int missing = 0;
    for (int i=pos_threshold-1;i >= 0; i--){
        if (vertexes[i].grau  != value_threshold){
            missing = pos_threshold - i ;
            break;
        }
    }

    return missing;
}

int countThreshold_(long int *elementsPerGPU_,Vertex *vertexes,int value_threshold){


    int *gpu_count[numGPUs];

    int *cpu_counts[numGPUs]; // Contagens na CPU para cada GPU


    int countsTreshold = 0; // Contagens finais após a combinação das GPUs

    // Inicializa contagens na CPU para cada GPU
    for (int i = 0; i < numGPUs; i++) {
        cpu_counts[i] = new int[numGPUs];
        memset(cpu_counts[i], 0, numGPUs * sizeof(int));
    }


    printf("Buscando os vertices\n");

    for (int i = 0; i < numGPUs; i++) {

        cudaSetDevice(i);

        // Aloca memória para o vetor e contagens na GPU


        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&gpu_count[i],numGPUs*sizeof(int)); // Vetor de valores


        // Configura a grade de threads
        long int numBlocks = ( elementsPerGPU_[i]/ blockSize) +1;
        
        
        countTreshold<<<numBlocks, blockSize>>>(vertexes, gpu_count[i], elementsPerGPU_[i],elementsPerGPU_[0],i,value_threshold);

           

        auto cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("%s hehehehehe",cudaGetErrorString(cuda_status));
            exit(-1);
           }
    }



    for (int i = 0; i < numGPUs; i++) {
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            cudaMemcpy(cpu_counts[i], gpu_count[i], numGPUs * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(gpu_count[i]);  

    }    


    for (int i=0;i<numGPUs;i++){
        // Quantidade de pontos que são iguais ao threshold
        countsTreshold += cpu_counts[i][0];
    }

    return countsTreshold;
}


void get_IndexThreshold(int *finalCounts,int *treshold_idx,int value_threshold){

    int control = 0;

    // Pega os índices iguais ao threshold.
    for (int i=0;i<numValues;i++){

        if (finalCounts[i] == value_threshold){
            treshold_idx[control] = i;
            control++;
        }
    }
}

void calculateUntieScore(Untie_hub *unties ,long int *indexesPerGPU,int *h_data,int *treshold_idx,int *finalCounts){

     Untie_hub *unties_gpus[numGPUs];

    Untie_hub *unties_cpus[numGPUs];

    

    for (int i = 0; i < numGPUs; i++) {

        cudaSetDevice(i);

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&unties_gpus[i],indexesPerGPU[i] * sizeof(Untie_hub)); // Vetor de valores


        // Configura a grade de threads
        long int numBlocks = ( indexesPerGPU[i]/ blockSize) +1;
        
        calculateScore<<<numBlocks,blockSize>>>(h_data,treshold_idx,unties_gpus[i],finalCounts,indexesPerGPU[i],k,i*indexesPerGPU[0]);

           

        auto cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("%s hehehehehe",cudaGetErrorString(cuda_status));
            exit(-1);
           }
    }

    // Juntando tudo em CPU

    for (int i = 0; i < numGPUs; i++) {
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            unties_cpus[i] = new Untie_hub[indexesPerGPU[i]];
            cudaMemcpy(unties_cpus[i], unties_gpus[i], indexesPerGPU[i] * sizeof(Untie_hub), cudaMemcpyDeviceToHost);

            cudaFree(unties_gpus[i]);

        auto cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("%s hehehehehe",cudaGetErrorString(cuda_status));
            exit(-1);
           }

    }   


    // Junta em CPU
    for (int i=0;i<numGPUs;i++){

        for (int j=0;j<indexesPerGPU[i];j++){
            
            unties[ j + (i*indexesPerGPU[0])].index = unties_cpus[i][j].index;
            unties[ j + (i*indexesPerGPU[0])].score = unties_cpus[i][j].score;
        }
    }

    return;
}


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





__global__ void initializeVectorCounts_(long int *vector,long int value,int size){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = value;
    }

}

int findKNNlist(int *kNN,long int neig,long int i,int k){

    for (long int j=0;j<k;j++)
        if (kNN[neig*k+j] == i)
            return 1;
    return 0;
}




ECLgraph buildECLgraph(int nodes, long int edges,int *kNN,int k, int *antihubs, long int num_antihubs)
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

            // Adiciona mais uma inserção no vetor auxiliar
            auxiliar_edges[i] += 1;
            
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
            if (!FLAG)
                g.nlist[pos] = neig;
            
        }

        g.eweight = (float*)malloc(g.edges * sizeof(g.eweight[0]));





        // Gera o vetor de teste
        for (long int i = 0; i < g.edges; i++) {
            g.eweight[i] = (rand() % 50) * 1.35;
    }  



 
  return g;   
}






ECLgraph buildEnhancedKNNG(int *h_data,int shards_num){


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

    printf("SCORE %d: %d\n",unties[0].index,unties[0].score);

    // Junção dos antihubs
    int *antihubs;

    antihubs = new int[pos_threshold];


    // Junta todos os antihubs em um vetor
    joinAntiHubs(antihubs,vertexes,not_ties,unties,missing_ties);

    // Ordena pelo índice para inserir na MST
    std::sort(antihubs,antihubs+pos_threshold);


    printf("Iniciando a construcao do grafo\n");

    ECLgraph g;
    
    g = buildECLgraph(numValues, vectorSize,h_data,k, antihubs, pos_threshold);


    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("Demorou %lf segundos para montar o enhanced kNNG\n",time_taken);

    return g;
}
int main() {


    int shards_num = 3;

    // Le o kNNG que esta escrito no arquivo abaixo
    std::string path_to_kNNG = "/nndescent/GPU_HDBSCAN/experiments/results/NNDescent-KNNG.kgraph";

    NNDElement *result_graph;
    int num, dim;
    FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &num, &dim);

    num = numValues;
    printf("%d e %d\n",num,dim);

    int *result_index_graph;

    cudaMallocManaged(&result_index_graph,(size_t)num*dim * sizeof(int));

  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      result_index_graph[i * dim + j] = result_graph[i * dim + j].label();
    }

  }

    CheckCUDA_();

   
    ECLgraph g;

    g = buildEnhancedKNNG(result_index_graph,shards_num);


    printf("Iniciando a construcao da MST\n");


    // run CPU code and compare result
    bool* cpuMSTedges = cpuMST(g);


    int soma = 0;
    for (int i=0;i<g.edges;i++){

        if (cpuMSTedges[i]){
            soma += 1;
        }
    }

    printf("A quantidade de arestas eh: %d\n",soma);



  return 0;


}
