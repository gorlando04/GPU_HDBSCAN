#include "calculates.cuh"
#include "../initializer/initialize.cuh"
#include "cuda_runtime.h"
#include "math.h"
#include <unistd.h>
#include "../../tools/filetool.hpp"
#include "../counts/count.cuh"

#include <algorithm>
#include <vector>
#include <omp.h>
#include <pthread.h>




__global__ void calculateScore(int *vectors,int *treshold_idx, Untie_hub *vertex , int *degrees ,long int size,int k,int offset){

    //Idx que iremos calcular o score
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;

    if (tid < size){
        long int idx = treshold_idx[tid+ (offset) ];
        vertex[tid].index = idx;

        for (long int j=0;j<k;j++){
            int neig = vectors[idx*k +j];
            vertex[tid].score += degrees[neig];
        }
    }

}

__global__ void calculateCoreDistance_(float *coreDistances,float *kNN_distances,long int offset,long int size,long int k,long int mpts){

    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        
        float distance = kNN_distances[(tid+offset)*k+mpts];
        coreDistances[tid] = distance;
    }

}


__global__ void ShardVector(int *vec_dev,int *vectors,long int size,long int off_set_size,int offset){


    long int tid = (blockIdx.x * blockSize) + threadIdx.x;

    if (tid < size){

        vec_dev[tid] = vectors[tid + off_set_size*offset];
    }
}

__global__ void calculateMRD(float *d_distances,int *d_nodes,int *d_edges,float *coreDistances,long int size){

    long int tid = (blockIdx.x * blockSize) + threadIdx.x;

    if (tid < size){

        long int idx_a = d_nodes[tid];
        long int idx_b = d_edges[tid];

        float core_dA = coreDistances[idx_a];
        float core_dB = coreDistances[idx_b];

        if (core_dA > core_dB){
            d_distances[tid] = core_dA;
	    return;
	}

        d_distances[tid] = core_dB;
        return;
    }

}

void calculateElements(long int *elementsPerGPU,int shards_num,long int vectorSize){

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

void calculateUntieScore(Untie_hub *unties ,long int *indexesPerGPU,int *h_data,int *treshold_idx,int *finalCounts,long int k){

     Untie_hub *unties_gpus[numGPUs];

    Untie_hub *unties_cpus[numGPUs];

    

    for (int i = 0; i < numGPUs; i++) {

        cudaSetDevice(i);

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&unties_gpus[i],indexesPerGPU[i] * sizeof(Untie_hub)); // Vetor de valores


        // Configura a grade de threads
        long int numBlocks = ( indexesPerGPU[i]/ blockSize) +1;

        initializeUntieHubs<<<numBlocks,blockSize>>>(unties_gpus[i],0,indexesPerGPU[i]);
        
        calculateScore<<<numBlocks,blockSize>>>(h_data,treshold_idx,unties_gpus[i],finalCounts,indexesPerGPU[i],k,i*indexesPerGPU[0]);

           

        auto cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("%s unties",cudaGetErrorString(cuda_status));
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

int calculate_shards_num(long int size,int long_=1){

    int shards_num = numGPUs;

    double total_size_mb = 3 * sizeof(int) *long_ * size  / pow(10,9);

    while (total_size_mb / shards_num > GPU_SIZE){
        shards_num += numGPUs;
    }
}



void calculateCoreDistance(float *kNN_distances, float *coreDistances ,long int numValues,long int k,long int mpts){



    omp_set_num_threads(32);
    #pragma omp parallel for 
    for (long int i = 0; i < numValues; i++) 
        coreDistances[i] = kNN_distances[i*k+mpts];


    return;
}


void calculateMutualReachabilityDistance(float *graphDistances,float *coreDistances,int *aux_nodes,int *aux_edges,long int size){

    int shards_num = calculate_shards_num(size);

    int *d_nodes[shards_num], *d_edges[shards_num]; // Vetores na GPU
    float *d_distances[shards_num];  // Vetores na GPU


    long int elementsPerGPU[shards_num];
    //Define o tamanho de cada shard
    calculateElements(elementsPerGPU,shards_num,size);




    // Calcula o número de iterações necessárias
    int iters = shards_num / numGPUs;


    for (int s=0;s < iters;s++){

	    for (int i = 0; i < numGPUs; i++) {

            long int idx = i + (s*numGPUs);
            cudaSetDevice(i);

            // Aloca memória para o vetor de nohs na GPU
            cudaMalloc(&d_nodes[idx],elementsPerGPU[idx] * sizeof(int)); 
            // Aloca memória para o vetor e contagens na GPU
            cudaMalloc(&d_edges[idx],elementsPerGPU[idx] * sizeof(int)); 
            // Aloca memória para o vetor e contagens na GPU
            cudaMalloc(&d_distances[idx], elementsPerGPU[idx] * sizeof(float)); 


            // Configura a grade de threads
            long int numBlocks = ( elementsPerGPU[idx]/ blockSize) +1;
            long int offset = elementsPerGPU[0] * idx;

            cudaMemcpy(d_nodes[idx], &aux_nodes[offset], elementsPerGPU[idx]*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_edges[idx], &aux_edges[offset], elementsPerGPU[idx]*sizeof(int), cudaMemcpyHostToDevice);
        
        
            calculateMRD<<<numBlocks,blockSize>>>(d_distances[idx],d_nodes[idx],d_edges[idx],coreDistances,elementsPerGPU[idx]);
        }

        //Libera a memória
        for (int i = 0; i < numGPUs; i++) {
            int idx = i + (s*numGPUs);
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            //cudaMemcpy(h_distances[idx], d_distances[idx], elementsPerGPU[idx] * sizeof(int), cudaMemcpyDeviceToHost);

            long int offset = elementsPerGPU[0] * idx;

            cudaMemcpy(&graphDistances[offset],d_distances[idx], elementsPerGPU[idx] * sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaFree(d_nodes[idx]);
            cudaFree(d_edges[idx]);
            cudaFree(d_distances[idx]);  
            d_nodes[idx] = NULL;
            d_edges[idx] = NULL;
            d_distances[idx] = NULL;
        }    
        CheckCUDA_();
    }
    
   return;

}


float calculate_euclidean_distance(float *vector,long int idxa,long int idxb,int dim){

    float soma = 0.0;
    for(long int i=0;i<dim;i++){
        soma += ( pow(vector[idxa*dim+i] - vector[idxb*dim+i],2) );
    }

    return sqrt(soma);
}


void calculate_nindex(int nodes, int *kNN, bool *flag_knn,ECLgraph *g,int *antihubs,int num_antihubs){

 // Calcula quantas arestas cada noh terá, levando em conta que eh um grafo não direcional.
    for (long int i=0;i<nodes;i++){

        long int soma = 0;


        for (long int j=0;j<k;j++){

            long int neig = kNN[i*k + j];

            //Verifica se i esta na lista de neig
            int FLAG = findKNNlist(kNN,neig,i,k);
	        flag_knn[i*k + j] = FLAG;

            if (FLAG > 1){ g->nindex[neig+1] += FLAG-1; g->nindex[i+1] -= (FLAG-1);}

            g->nindex[neig+1] += 1;
           
            if (!FLAG)
                soma += 1;
        }
        g->nindex[i+1] += soma;
    }

    // Adicionar os antihubs
    int contador = 0;

    #pragma omp parallel for
    for (long int i=0;i<nodes;i++)

        if (i == antihubs[contador]){
            contador ++;
            g->nindex[i+1] += (num_antihubs-1);
    }
     
    
    //Calcular offsets
    for (long int i=1;i<nodes+1;i++){

        g->nindex[i] = g->nindex[i-1] + g->nindex[i];

    }

}

void calculate_nlist(int nodes, int *kNN,int k, bool *flag_knn,ECLgraph *g,int *antihubs,int num_antihubs,long int *auxiliar_edges){

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
            }
        }
    }

    //Adiciona os antihubs
    for (long int i=0;i<num_antihubs;i++){

        int current = antihubs[i];
        long int pos_begin = g->nindex[current];
        long int offset = auxiliar_edges[current];

        for (long int j=0;j<num_antihubs;j++){
            int neig = antihubs[j];

            if (neig != current){
                g->nlist[pos_begin+offset] = neig;
                offset++;
            }
        }
    }

}


void calculate_coreDistance_antihubs(ECLgraph *g,long int *auxiliar_edges,int *antihubs,long int num_antihubs){


    // Read vector txtx
    const std::string path_to_data_binary = "/nndescent/GPU_HDBSCAN/data/vectors.fvecs";
    long int data_size2, data_dim2;
    float *vectors_data;

    FileTool::ReadBinaryAntihubs(path_to_data_binary,&vectors_data, &data_size2, &data_dim2,antihubs,num_antihubs);    	

    for (long int i=0;i<num_antihubs;i++){

        int idx_a = antihubs[i];
        long int pos_begin = g->nindex[idx_a] + auxiliar_edges[idx_a];
        for (long int j=i;j<num_antihubs-1;j++){
            int idx_b = antihubs[j+1];
            
            //Calcula distancia euclidiana
            float euclidean_distance = calculate_euclidean_distance(vectors_data,i,j+1,data_dim2);

            if (g->eweight[pos_begin + j] < euclidean_distance){
                g->eweight[pos_begin+j] = euclidean_distance;

                long int pos_begin2 = g->nindex[idx_b] + auxiliar_edges[idx_b];
                g->eweight[pos_begin2 + i] = euclidean_distance;
            }


        }
    }
}


int* calculate_degrees(int *kNN,long int vectorSize,long int numValues){

    cudaMemPrefetchAsync(kNN,(size_t)vectorSize * sizeof(int), cudaCpuDeviceId);

    int shards_num = calculate_shards_num(vectorSize + numValues,2);

    int *d_nodes[shards_num], *d_edges[shards_num]; // Vetores na GPU
    float *d_distances[shards_num];  // Vetores na GPU


    // Calcula a quantidade de elementos por GPU
    long int elementsPerGPU[shards_num];
    calculateElements(elementsPerGPU,shards_num,vectorSize);

    // Realiza a contagem de graus para todos os vértices
    int *finalCounts; // Contagens finais após a combinação das GPUs
    cudaMallocManaged(&finalCounts,(size_t)numValues * sizeof(int));

    
    int gridSize = (numValues + blockSize - 1) / blockSize;
    // Inicializa o vetor
    initializeVectorCounts<<<gridSize,blockSize>>>(finalCounts,0,numValues);
    Check();        

    // Conta os graus de cada vértice
    countDegrees(finalCounts,kNN,shards_num,elementsPerGPU,numValues);


    return finalCounts;
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

int* calculate_finalAntihubs(Vertex *vertexes,int *kNN,int* finalCounts,int* antihubs,long int numValues,int countsTreshold,
                            int pos_threshold, int value_threshold,long int k){

    // Pega os índices dos pontos que são iguais ao threshold
    int *treshold_idx;
	cudaMallocManaged(&treshold_idx,(size_t)countsTreshold * sizeof(int));
    get_IndexThreshold(finalCounts,treshold_idx,value_threshold,numValues);

    avoid_pageFault(treshold_idx,countsTreshold);

    // Calcula quantos elementos serão processados por cada GPU
    long int indexesPerGPU[numGPUs];
    calculateElements(indexesPerGPU,numGPUs,countsTreshold);

    // Calculata os scores dos empates
    Untie_hub *unties = new Untie_hub[countsTreshold];
    calculateUntieScore(unties,indexesPerGPU,kNN,treshold_idx,finalCounts,k);

    // Pega quantos empates temos na lista final
    int missing_ties = get_TiedVertexes(vertexes,pos_threshold,value_threshold);
    int not_ties = pos_threshold - missing_ties;

    std::partial_sort(unties, unties + missing_ties, unties + countsTreshold, compareVertexByScore);

    // Junta todos os antihubs em um vetor
    joinAntiHubs(antihubs,vertexes,not_ties,unties,missing_ties);

    delete unties;
    unties = NULL;

    return antihubs;

}
