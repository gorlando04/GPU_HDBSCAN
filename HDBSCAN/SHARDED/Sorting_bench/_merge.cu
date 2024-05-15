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

struct MSTedge{

    float from_node;
    float to_node;
    float weight;
};

#define NUM_GPUS 1

__device__ bool compareEdgeByWeight(const MSTedge &a, const MSTedge &b){
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
        if (isWeightEqual(tmp[i],tmp[j]) || compareEdgeByWeight(tmp[i],tmp[j]))
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

void mergeSort(int N,MSTedge *input,MSTedge *output,int device=0)
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
}


void mergeSortMultiGPU(int N,MSTedge *input,MSTedge *output,int offset,int device=0)
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

  int control[NUM_GPUS];

  for(int i=0;i<NUM_GPUS;i++)
    control[i] = 0;


  for(int i=0;i<numValues;i++){

    int menor = INT_MAX;
    int idx_shard_menor = -1;
    int idx_menor = -1;

    for(int j=0;j<NUM_GPUS;j++){

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
        control[idx_menor] += 1;
        results[i].from_node = partial_results[idx_shard_menor][idx_menor].to_node;
        results[i].to_node =  partial_results[idx_shard_menor][idx_menor].from_node; 
        results[i].weight = partial_results[idx_shard_menor][idx_menor].weight;

  }


  

}


int main(int argc, char *argv[]){

  //Standard parameters
  long int numValues = 1000;

  std::string vec = "1000";
  std::string iter = "0";

  if (argc == 3){
    numValues = atoi(argv[1]);
    vec = argv[1];
    iter = argv[2];
    printf("NUM VALUES SET TO %ld.\n",numValues);
  }

  else{
        printf("NUM_VALUES NOT SETTED\n");
        exit(1);
  }

  //Create CPU based Arrays
  MSTedge* arr = new MSTedge[numValues];
  MSTedge*  result = new MSTedge[numValues];

 
  int j=0;
  for (int i=numValues-1;i>=0;i--){
    arr[j].weight = (float)rand() /(RAND_MAX); arr[j].from_node = arr[j].weight; arr[j].to_node = arr[j].weight;
    j += 1;
  }

  bool flag_gpu = false;
  if (numValues < 1000000) flag_gpu = true;
  clock_t t; 
  t = clock(); 


  if (NUM_GPUS == 1 || flag_gpu){
    mergeSort(numValues,arr,result,0);
  }

  else {
    int *elements_per_GPU;
    elements_per_GPU = (int*)malloc(NUM_GPUS*sizeof(int));
    MSTedge *partial_results[NUM_GPUS];


    calculateElements(elements_per_GPU,NUM_GPUS,numValues);

    for(int i=0;i<NUM_GPUS;i++)
      partial_results[i] = new MSTedge[elements_per_GPU[i]];

    std::vector<std::thread> threads;
 
    for (int i=0;i<NUM_GPUS;i++){
      threads.push_back (std::thread ([elements_per_GPU, i,arr,partial_results] () {



        mergeSortMultiGPU(elements_per_GPU[i],arr,partial_results[i],i*elements_per_GPU[0],i);
      })); 
    }

    for (auto &t: threads)
      t.join ();  


	


    merging_partial_results(elements_per_GPU, partial_results,numValues,result);


  }

  
  t = clock() - t; 
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

int flag = 1;
    if (result[0].weight != result[0].from_node || result[0].weight != result[0].to_node) flag = 0;

    
    for (int i = 1; i < numValues; i++) {
        // Unsorted pair found
        if (result[i - 1].weight > result[i].weight || result[i].weight != result[i].from_node||result[i].weight != result[i].to_node)
            flag = 0;
    }

printf("EH ORDENADO MEU AMIGO %d\n",flag);
  std::string folder = "results-";
  std::string standart_1 = "Time_";
  std::string standart_2 = ".txt";
  std::string path = folder + vec  + "/"  + standart_1 + vec + "_" + iter + standart_2;
  FILE *pFile;

    pFile=fopen(path.c_str(), "a");

  if(pFile==NULL) {
        perror("Error opening file.");
    }
else {

        fprintf(pFile, "%lf\n", time_taken); fprintf(pFile, "EH ORDENADO: %d", flag);
    }

fclose(pFile);


  return 0;

}
