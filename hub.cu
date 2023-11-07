#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

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

void merge(struct Vertex arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Crie arrays temporários para a metade esquerda e direita
    struct Vertex L[n1];
    struct Vertex R[n2];

    // Copie os dados para os arrays temporários L[] e R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Mescla os arrays temporários de volta em arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i].grau <= R[j].grau) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copie os elementos restantes de L[], se houver
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copie os elementos restantes de R[], se houver
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(struct Vertex arr[], int l, int r) {
    if (l < r) {
        // Encontra o ponto médio do array
        int m = l + (r - l) / 2;

        // Classifica a primeira metade
        mergeSort(arr, l, m);

        // Classifica a segunda metade
        mergeSort(arr, m + 1, r);

        // Mescla as duas metades ordenadas
        merge(arr, l, m, r);
    }
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
        h_data[i] = i / k;//rand() % numValues;
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


    // Pegar os threshold
    int pos_threshold = sqrt(numValues);

    // Pegar o valor do threshold 
    int value_threshold = vertexes[pos_threshold].grau;

    printf("A posicao do threshold eh: %d e o valor eh: %d\n",pos_threshold,value_threshold);
    // Encontrar quantos valores são iguais ao threshold
    Vertex *degree_count[numGPUs];  // Vetores na GPU

    long int elementsPerGPU_[numGPUs];

    //Define o tamanho de cada shard
    for (int i=0;i<numGPUs;i++){

        if (i != (numGPUs)-1){ elementsPerGPU_[i] = numValues / (numGPUs);}

        else{
            elementsPerGPU_[i] = numValues;

            for (int j=0;j<i;j++)
                elementsPerGPU_[i] -= elementsPerGPU_[j];
            

        }
    }

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
        cudaMalloc(&degree_count[i],elementsPerGPU_[i] * sizeof(Vertex)); // Vetor de valores
        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&gpu_count[i],numGPUs*sizeof(int)); // Vetor de valores


        // Configura a grade de threads
        long int numBlocks = ( elementsPerGPU_[i]/ blockSize) +1;
        

        ShardVertex<<< numBlocks,blockSize>>>(degree_count[i],vertexes,elementsPerGPU_[0],elementsPerGPU_[i], i);

        
        countTreshold<<<numBlocks, blockSize>>>(degree_count[i], gpu_count[i], elementsPerGPU_[i],value_threshold,i);

           

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("%s hehehehehe",cudaGetErrorString(cuda_status));
            exit(-1);
           }
    }



    for (int i = 0; i < numGPUs; i++) {
            cudaSetDevice(i);

            cudaDeviceSynchronize();

            cudaMemcpy(cpu_counts[i], gpu_count[i], numGPUs * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(degree_count[i]);
            cudaFree(gpu_count[i]);  

    }    


    for (int i=0;i<numGPUs;i++){
        // Quantidade de pontos que são iguais ao threshold
        countsTreshold += cpu_counts[i][0];
    }



    int *treshold_idx;

    // Aloca memória para o vetor na CPU
	cudaMallocManaged(&treshold_idx,(size_t)countsTreshold * sizeof(int));
  

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }

    // Prepara o vetor com os índices dos pontos que são iguais ao threshold
    

    int control = 0;


    // Pega os índices iguais ao threshold.
    for (int i=0;i<numValues;i++){

        if (finalCounts[i] == value_threshold){
            treshold_idx[control] = i;
            control++;
        }
    }

    long int indexesPerGPU[numGPUs];

    //Define o tamanho de cada shard
    for (int i=0;i<numGPUs;i++){

        if (i != (numGPUs)-1){ indexesPerGPU[i] = countsTreshold/ (numGPUs);}

        else{
            indexesPerGPU[i] = countsTreshold;

            for (int j=0;j<i;j++)
                indexesPerGPU[i] -= indexesPerGPU[j];
            

        }
    }



    Untie_hub *unties_gpus[numGPUs];

    Untie_hub *unties_cpus[numGPUs];

    

    int *degree_counts_gpu[numGPUs];


    for (int i = 0; i < numGPUs; i++) {

        cudaSetDevice(i);

        // Aloca memória para o vetor e contagens na GPU
        cudaMalloc(&unties_gpus[i],indexesPerGPU[i] * sizeof(Untie_hub)); // Vetor de valores

        cudaMalloc(&degree_counts_gpu[i], numValues * sizeof(int)); // Vetor de valores

        cudaMemcpy(degree_counts_gpu[i], finalCounts, numValues * sizeof(int), cudaMemcpyHostToDevice);


        // Configura a grade de threads
        long int numBlocks = ( indexesPerGPU[i]/ blockSize) +1;
        
        calculateScore<<<numBlocks,blockSize>>>(h_data,treshold_idx,unties_gpus[i],degree_counts_gpu[i],indexesPerGPU[i],k,i*indexesPerGPU[0]);

           

        cuda_status = cudaGetLastError();
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
            cudaFree(degree_counts_gpu[i]);  

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("%s hehehehehe",cudaGetErrorString(cuda_status));
            exit(-1);
           }

    }   

    Untie_hub *unties;
    unties = new Untie_hub[countsTreshold];


    // Junta em CPU
    for (int i=0;i<numGPUs;i++){

        for (int j=0;j<indexesPerGPU[i];j++){
            
            unties[ j + (i*indexesPerGPU[0])].index = unties_cpus[i][j].index;
            unties[ j + (i*indexesPerGPU[0])].score = unties_cpus[i][j].score;
        }
    }


    mergeSort(vertexes, 0, numValues-1) ;

    return 0;
}
