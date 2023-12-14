#include "calculates.cuh"
#include "../initializer/initialize.cuh"
#include "cuda_runtime.h"
#include "math.h"




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

void calculateCoreDistance(float *kNN_distances, float *coreDistances ,long int *indexesPerGPU,long int k,long int mpts){

    float  *coreDistances_cpu[numGPUs];

    float  *coreDistances_gpu[numGPUs];


    for (int i = 0; i < numGPUs; i++) {

        cudaSetDevice(i);

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&coreDistances_gpu[i],indexesPerGPU[i] * sizeof(float)); // Vetor de valores

        // Inicializa o vetor em GPU de core_distances
        long int numBlocks_ = (indexesPerGPU[i]/blockSize) + 1;
        initializeVectorCounts<<<numBlocks_,blockSize>>>(coreDistances_gpu[i],0,indexesPerGPU[i]);



        // Configura a grade de threads
        long int numBlocks = ( indexesPerGPU[i]/ blockSize) +1;

        calculateCoreDistance_<<<numBlocks,blockSize>>>(coreDistances_gpu[i],kNN_distances,indexesPerGPU[0],indexesPerGPU[i],k,mpts-1);
        
   
        CheckCUDA_();
    }

    // Juntando tudo em CPU

    for (int i = 0; i < numGPUs; i++) {
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            coreDistances_cpu[i] = new float[indexesPerGPU[i]];
            cudaMemcpy(coreDistances_cpu[i], coreDistances_gpu[i], indexesPerGPU[i] * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(coreDistances_gpu[i]);

            CheckCUDA_();

    }   

    // Junta em CPU
    for (int i=0;i<numGPUs;i++){
        for (int j=0;j<indexesPerGPU[i];j++)
            coreDistances[(i*indexesPerGPU[0]) + j] = coreDistances_cpu[i][j];
    }

    return;
}




void calculateMutualReachabilityDistance(float *graphDistances,float *coreDistances,int *aux_nodes,int *aux_edges,long int size){


    int shards_num = 9;
    int *d_nodes[shards_num], *d_edges[shards_num]; // Vetores na GPU
    float *d_distances[shards_num];  // Vetores na GPU
    float *h_distances[shards_num]; // Contagens na CPU para cada GPU



    CheckCUDA_();


    long int elementsPerGPU[shards_num];

    //Define o tamanho de cada shard
    for (int i=0;i<shards_num;i++){

        if (i != (shards_num)-1){ elementsPerGPU[i] = size/ (shards_num);}

        else{
            elementsPerGPU[i] = size;

            for (int j=0;j<i;j++)
                elementsPerGPU[i] -= elementsPerGPU[j];
            

        }
    }

    // Inicializa contagens na CPU para cada GPU
    for (int i = 0; i < shards_num; i++) {
        h_distances[i] = new float[elementsPerGPU[i]];
        memset(h_distances[i], 0,elementsPerGPU[i] * sizeof(float));
    }

    // Realiza a contagem de graus para todos os vértices

    int iters = shards_num / numGPUs;

    for (int s=0;s < iters;s++){

	for (int i = 0; i < numGPUs; i++) {

        int idx = i + (s*numGPUs);
        cudaSetDevice(i);

        // Aloca memória para o vetor de nohs na GPU
        cudaMalloc(&d_nodes[idx],elementsPerGPU[idx] * sizeof(int)); 

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&d_edges[idx],elementsPerGPU[idx] * sizeof(int)); 

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&d_distances[idx], elementsPerGPU[idx] * sizeof(float)); 


        // Configura a grade de threads
        long int numBlocks = ( elementsPerGPU[idx]/ blockSize) +1;
        
        ShardVector<<<numBlocks,blockSize>>>(d_nodes[idx],aux_nodes,elementsPerGPU[idx],elementsPerGPU[0],idx);
        ShardVector<<<numBlocks,blockSize>>>(d_edges[idx],aux_edges,elementsPerGPU[idx],elementsPerGPU[0],idx);

        
        calculateMRD<<<numBlocks,blockSize>>>(d_distances[idx],d_nodes[idx],d_edges[idx],coreDistances,elementsPerGPU[idx]);

        
        }

        //Libera a memória

        for (int i = 0; i < numGPUs; i++) {
            int idx = i + (s*numGPUs);
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            cudaMemcpy(h_distances[idx], d_distances[idx], elementsPerGPU[idx] * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_nodes[idx]);
            cudaFree(d_edges[idx]);
            cudaFree(d_distances[idx]);  

        }    

        CheckCUDA_();
    
    }


    for (int i = 0; i < shards_num; i++) {
        for (int j = 0; j < elementsPerGPU[i]; j++) {
		graphDistances[j+ (i * elementsPerGPU[0])] = h_distances[i][j];
        }
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
