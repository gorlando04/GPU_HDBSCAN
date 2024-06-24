#include <iostream>
#include <cstdio>
#include <algorithm>
#include <getopt.h>
#include <string.h>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <assert.h>
#include <thread>
#include <vector>

#include "merge_sort.cuh"

#include <time.h>


__device__ bool compareEdgeByWeightGPU(const MSTedge &a, const MSTedge &b){
    return a.weight < b.weight;
}

__device__ bool isWeightEqual(const MSTedge &a, const MSTedge &b){
    return a.weight == b.weight;
}

__device__ void merge(int l,int m,int r,MSTedge data[],MSTedge tmp[])
{
    int i=l,j=m,k=l;
    while (i<m&&j<r)
    {
        if (isWeightEqual(tmp[i],tmp[j]) || compareEdgeByWeightGPU(tmp[i],tmp[j]))
        {
            data[k].weight = tmp[i].weight; 
            data[k].from_node = tmp[i].from_node;
            data[k].to_node = tmp[i].to_node; 
            k++;i++;
        }
        else
        {
            data[k].weight = tmp[j].weight; 
            data[k].from_node = tmp[j].from_node;
            data[k].to_node = tmp[j].to_node; 
            k++;j++;        
        }
    }
    while (i<m) {
        data[k].weight = tmp[i].weight; 
        data[k].from_node = tmp[i].from_node;
        data[k].to_node = tmp[i].to_node; 
        k++;i++; 
    }
    while (j<r) {
    
        data[k].weight = tmp[j].weight; 
        data[k].from_node = tmp[j].from_node;
        data[k].to_node = tmp[j].to_node; 
        k++;j++;
    }
}

__global__ void merge_kernel(int N, int chunk,MSTedge data[],MSTedge tmp[]) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index>N) return ;
    int start=index*chunk;
    if (start>=N || start<0) return ;

    int left=start;
    int mid=min(start+(int)(chunk/2),N);
    int right=min(start+chunk,N);


    merge(left, mid,right,data,tmp);
}

void mergeSort(int N,MSTedge *input,MSTedge *output,int device)
{
    cudaSetDevice(device);

    MSTedge *device_i;
    MSTedge *tmp;
    cudaMalloc((void **)&device_i, N*sizeof(MSTedge));
    cudaMalloc((void **)&tmp, N*sizeof(MSTedge));


    cudaMemcpy(device_i, input, N*sizeof(MSTedge), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp, input, N*sizeof(MSTedge), cudaMemcpyHostToDevice);

    
    for (int chunk=2;chunk<2*N;chunk*=2)
    {
        // const int threadsPerBlock = 512;
        const int threadsPerBlock=1;
        const int blocks = ((N + threadsPerBlock*chunk - 1) / (threadsPerBlock*chunk));
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,device_i,tmp);
        cudaDeviceSynchronize();
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,tmp,device_i);
        cudaDeviceSynchronize();
    }
    

    cudaMemcpy(output, device_i, N*sizeof(MSTedge), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    cudaFree(device_i);
    cudaFree(tmp);

    return;
}


void mergeSortMultiGPU(int N,MSTedge *input,MSTedge *output,int offset,int device)
{
    cudaSetDevice(device);

    MSTedge *device_i;
    MSTedge *tmp;
    cudaMalloc((void **)&device_i, N*sizeof(MSTedge));
    cudaMalloc((void **)&tmp, N*sizeof(MSTedge));


    cudaMemcpy(device_i, &input[offset], N*sizeof(MSTedge), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp, &input[offset], N*sizeof(MSTedge), cudaMemcpyHostToDevice);


    
    for (int chunk=2;chunk<2*N;chunk*=2)
    {
        const int threadsPerBlock=1;
        const int blocks = ((N + threadsPerBlock*chunk - 1) / (threadsPerBlock*chunk));
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,device_i,tmp);
        cudaDeviceSynchronize();
        merge_kernel<<<blocks,threadsPerBlock>>>(N,chunk,tmp,device_i);
        cudaDeviceSynchronize();
    }
    

    cudaMemcpy(output, device_i, N*sizeof(MSTedge), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }



    cudaFree(device_i);
    cudaFree(tmp);

    return;
}


void calculateElements(int *elementsPerGPU,int shards_num,long int vectorSize){

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

void merging_partial_results(int *elementsPerGPU, MSTedge **partial_results,int numValues,MSTedge *results){

    int control[numGPUs];

    for(int i=0;i<numGPUs;i++)
        control[i] = 0;


    for(int i=0;i<numValues;i++){

        float menor = 1e8;
        int idx_shard_menor = -1;
        int idx_menor = -1;

        for(int j=0;j<numGPUs;j++){

            if (control[j] == elementsPerGPU[j]){continue;}
            
            else if (j == 0){ 
                menor = partial_results[j][control[j]].weight;
                idx_menor = control[j];
                idx_shard_menor = j;
            }

            else{ 
                if (menor > partial_results[j][control[j]].weight ) {
                    menor = partial_results[j][control[j]].weight;
                    idx_menor = control[j];
                    idx_shard_menor = j;
                }
            }           
        }
        control[idx_shard_menor] += 1;
        results[i].from_node = partial_results[idx_shard_menor][idx_menor].to_node;
        results[i].to_node =  partial_results[idx_shard_menor][idx_menor].from_node; 
        results[i].weight = partial_results[idx_shard_menor][idx_menor].weight;
    }

    return;
}


MSTedge* sort_edges(MSTedge *arr,int numValues){


   clock_t t;
   t = clock();
    //Create CPU based Arrays
    MSTedge*  result = new MSTedge[numValues];


    bool flag_gpu = false;
    if (numValues < 10000000) flag_gpu = true;
 

    // ORDENAÇÃO APENAS EM 1 GPU
    if (numGPUs == 1 || flag_gpu){

        mergeSort(numValues,arr,result,0);
        return result;
    }

    // Ordenação em mais de 1 GPU
    int *elements_per_GPU;
    elements_per_GPU = (int*)malloc(numGPUs*sizeof(int));
    calculateElements(elements_per_GPU,numGPUs,numValues);

    MSTedge *partial_results[numGPUs];
    for(int i=0;i<numGPUs;i++)
        partial_results[i] = new MSTedge[elements_per_GPU[i]];

    std::vector<std::thread> threads;
    for (int i=0;i<numGPUs;i++)
        threads.push_back (std::thread ([elements_per_GPU, i,arr,partial_results] () {
            mergeSortMultiGPU(elements_per_GPU[i],arr,partial_results[i],i*elements_per_GPU[0],i);
        })); 
    for (auto &t: threads)
        t.join ();  





    merging_partial_results(elements_per_GPU, partial_results,numValues,result);

     t = clock() - t; 
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds */

  printf("Ordenacao das asrestas demorou: %lf \n",time_taken);
    return result;
}
