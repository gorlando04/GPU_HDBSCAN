#include "cuda_runtime.h"
#include "count.cuh"
#include "../initializer/initialize.cuh"
#include <omp.h>
#include <pthread.h>



// Função do kernel CUDA para calcular a contagem de valores em um vetor
__global__ void countValues(int *data, int *counts, long int size/*, long int off_set_size,int offset*/) {


    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long int tid = (blockIdx.x * blockSize) + threadIdx.x;
    if (tid < size) {
        long int value = data[tid];
        atomicAdd(&counts[value], 1);
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



void countDegrees(int *finalCounts,int *h_data,int shards_num,long int *elementsPerGPU, long int numValues){

    int *d_counts[shards_num];  // Vetores na GPU
    int *data_device[shards_num]; // Contagens na CPU para cada GPU
    int *h_counts[shards_num]; // Contagens na CPU para cada GPU
    
    // Inicializa contagens na CPU para cada GPU
    for (int i = 0; i < shards_num; i++) {
        h_counts[i] = new int[numValues];
    }

  // Calcula o número de iterações necessárias
    int iters = shards_num / numGPUs;

    for (int s=0;s < iters;s++){

	    for (int i = 0; i < numGPUs; i++) {

            long int idx = i + (s*numGPUs);
            cudaSetDevice(i);

            // Aloca memória para o vetor e contagens na GPU
            cudaMalloc(&d_counts[idx], numValues * sizeof(int)); // Vetor de frequências
	    cudaMalloc(&data_device[idx], elementsPerGPU[idx] * sizeof(int)); // Vetor de frequencias

            long int offset = elementsPerGPU[0] * idx;
            cudaMemcpy(data_device[idx], &h_data[offset], elementsPerGPU[idx]*sizeof(int), cudaMemcpyHostToDevice);

            // Configura a grade de threads
            long int numBlocks = ( elementsPerGPU[idx]/ blockSize) +1;
            long int gridSize = (numValues + blockSize - 1) / blockSize;

            initializeVectorCounts<<<gridSize,blockSize>>>(d_counts[idx],0,numValues);
            countValues<<<numBlocks, blockSize>>>(data_device[idx], d_counts[idx], elementsPerGPU[idx]);

        }

        //Libera a memória
        for (int i = 0; i < numGPUs; i++) {
            int idx = i + (s*numGPUs);
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            cudaMemcpy(h_counts[idx], d_counts[idx], numValues * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_counts[idx]); 
            d_counts[idx] = NULL;
        }    
        CheckCUDA_();
    }

    // Combina as contagens de todas as GPUs na CPU
    omp_set_num_threads(32);
    #pragma omp parallel for
    for (int j = 0; j < numValues; j++) {
        for (int i = 0; i < shards_num; i++) {
            finalCounts[j] += h_counts[i][j];
        }
    }


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
