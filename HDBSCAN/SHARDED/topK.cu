#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <time.h>


const int numGPUs = 3;
const long int numValues = 1000000; // Número de valores possíveis no vetor
const long int vectorSize = 32000000;// Tamanho do vetor
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



__global__ void ShardVector(int *vec_dev,int *vectors,long int off_set_size,long int end,int offset=0){


    long int list_id = blockIdx.x;
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;

    if (tid < end){
    vec_dev[tid] = vectors[tid + off_set_size*offset];
    }
}


// Função do kernel CUDA para calcular a contagem de valores em um vetor
__global__ void countValues(int *data, int *counts, long int size) {


    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;
    if (tid < size) {
        long int value = data[tid];
        atomicAdd(&counts[value], 1);
    }
}


__global__ void ShardVertex(Vertex *vec_dev,Vertex *vectors,long int off_set_size,long int end,int offset=0){


    long int list_id = blockIdx.x;
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;

    if (tid < end){
    vec_dev[tid].grau = vectors[tid + off_set_size*offset].grau;

    }
}


// Função do kernel CUDA para calcular a contagem de valores em um vetor
__global__ void countTreshold(Vertex *data, int *counts, long int size,int comparision,int gpu_id) {


    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;
    if (tid < size) {
        long int value = data[tid].grau;

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
bool compareVertexByDegree(const Vertex &a, const Vertex &b) {
    return a.grau < b.grau;
}



int main() {
    int shards_num = 9;
    int *h_data;  // Vetor na CPU
    int *d_data[shards_num], *d_counts[shards_num];  // Vetores na GPU
    int *h_counts[shards_num]; // Contagens na CPU para cada GPU
    int *finalCounts; // Contagens finais após a combinação das GPUs

    // Aloca memória para o vetor na CPU
	cudaMallocManaged(&h_data,(size_t)vectorSize * sizeof(int));
  

    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }

// Gera o vetor de teste
  for (long int i = 0; i < vectorSize; i++) {
        h_data[i] = /*i / k;// */rand() % numValues;
    }

    printf("Comecand a busca\n");


    // Inicializa contagens na CPU para cada GPU
    for (int i = 0; i < shards_num; i++) {
        h_counts[i] = new int[numValues];
        memset(h_counts[i], 0, numValues * sizeof(int));
    }

    long int elementsPerGPU[shards_num];

    //Define o tamanho de cada shard
    for (int i=0;i<shards_num;i++){

        if (i != (shards_num)-1){ elementsPerGPU[i] = vectorSize / (shards_num);}

        else{
            elementsPerGPU[i] = vectorSize;

            for (int j=0;j<i;j++)
                elementsPerGPU[i] -= elementsPerGPU[j];
            

        }
    }

// Realiza a contagem de graus para todos os vértices

    long int end = 0;
 int iters = shards_num / numGPUs;

    for (int s=0;s < iters;s++){

	for (int i = 0; i < numGPUs; i++) {

        int idx = i + (s*numGPUs);
        cudaSetDevice(i);

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&d_data[idx],elementsPerGPU[idx] * sizeof(int)); // Vetor de valores
        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&d_counts[idx], numValues * sizeof(int)); // Vetor de frequências


        // Configura a grade de threads
        long int numBlocks = ( elementsPerGPU[idx]/ blockSize) +1;
        

        ShardVector<<< numBlocks,blockSize>>>(d_data[idx],h_data,elementsPerGPU[0],elementsPerGPU[idx], idx);

        
        //cudaMemcpy(h_data2[idx], d_data[idx], elementsPerGPU[idx] * sizeof(int), cudaMemcpyDeviceToHost);
        countValues<<<numBlocks, blockSize>>>(d_data[idx], d_counts[idx], elementsPerGPU[idx]);

        
        }

        //Libera a memória

        for (int i = 0; i < numGPUs; i++) {
            int idx = i + (s*numGPUs);
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            cudaMemcpy(h_counts[idx], d_counts[idx], numValues * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_data[idx]);
            cudaFree(d_counts[idx]);  

        }    

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("%s hehehehehe",cudaGetErrorString(cuda_status));
            exit(-1);
        }
    
	printf("Iter %d finalizada.\n",s);
    }






    printf("\naaa");
    
 // Combina as contagens de todas as GPUs na CPU
    finalCounts = new int[numValues];

    memset(finalCounts, 0, numValues * sizeof(int));

    for (int i = 0; i < shards_num; i++) {
        for (int j = 0; j < numValues; j++) {
            finalCounts[j] += h_counts[i][j];
        }
    }   


    /*for (long int i=0;i<numValues;i++){
	if (finalCounts[i] != 30){ printf("Resultado errado"); return 0;}
        printf("%ld = %d\n",i,finalCounts[i]);
    }*/

    // Atribui o vetor de contagens a uma struct que contém os índices

    Vertex *vertexes;

    //vertexes = new Vertex[numValues];
    // Aloca memória para o vetor na CPU
	cudaMallocManaged(&vertexes,(size_t)numValues * sizeof(Vertex));
  

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }

    for (int i=0;i<numValues;i++){
        vertexes[i].grau = finalCounts[i];
        vertexes[i].index = i;
    }

    //Libera a memória


    // Ordenar o vetor de structs pelo valor do grau.

    /* Isso vai ser feito aqui.*/


    // Pegar o threshold
    int pos_threshold = sqrt(numValues)+1;


    clock_t t; 
    t = clock(); 

    std::partial_sort(vertexes, vertexes + pos_threshold, vertexes + numValues, compareVertexByDegree);

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

    printf("Tempo para a ordencao: %.3lf\n",time_taken);



    FILE *pFile;



    pFile=fopen("Time_1000000_32.txt", "a");

    if(pFile==NULL) {
        perror("Error opening file.");
    }
else {

        fprintf(pFile, "%lf", time_taken);
    }

fclose(pFile);


    return 0;
}
