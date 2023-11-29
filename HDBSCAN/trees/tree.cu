#include "tree.cuh"
#include "../initializer/initialize.cuh"


UnionFind::UnionFind(int N) {


    this->next_label = N;

    // Parent array
    int *aux_parent;
    cudaMallocManaged(&aux_parent,(size_t)(2 * N - 1) * sizeof(int));

    
    int gridSize = ((2 * N - 1) + blockSize - 1) / blockSize;
    initializeVectorCounts<<<gridSize,blockSize>>>(aux_parent,-1,(2 * N - 1));

    cudaDeviceSynchronize();
    CheckCUDA_();

    this->parent_arr = new int[(2*N -1)];
    cudaMemcpy(this->parent_arr, aux_parent,(size_t)(2 * N - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(aux_parent);

    cudaDeviceSynchronize();
    CheckCUDA_();

    // Size Array
    int *aux_size;
    cudaMallocManaged(&aux_size,(size_t)(2 * N - 1) * sizeof(int));

    
    gridSize = (N  + blockSize - 1) / blockSize;
    initializeVectorCounts<<<gridSize,blockSize>>>(aux_size,1,N);

    cudaDeviceSynchronize();
    CheckCUDA_();

    gridSize = ((N-1)  + blockSize - 1) / blockSize;
    initializeVectorCountsOFFset<<<gridSize,blockSize>>>(aux_size,0,(2*N-1),N);

    cudaDeviceSynchronize();
    CheckCUDA_();

    this->size_arr = new int[(2*N -1)];
    cudaMemcpy(this->size_arr , aux_size ,(size_t)(2 * N - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(aux_size);

    cudaDeviceSynchronize();
    CheckCUDA_();

    /*this->parent = new int[(2*N -1)];
    cudaMemcpy(this->parent, this->parent_arr,(size_t)(2 * N - 1) * sizeof(int), cudaMemcpyHostToHost);
    cudaDeviceSynchronize();
    CheckCUDA_();*/

    
}

void UnionFind::Union(int m, int n) {


    this->size_arr[this->next_label] = this->size_arr[m] + this->size_arr[n];
    this->parent_arr[m] = this->next_label;
    this->parent_arr[n] = this->next_label;
    this->next_label += 1;

    return;
}

int UnionFind::FastFind(int n) {

    int p = n;
    while (this->parent_arr[n] != -1) {
	n = this->parent_arr[n];
}

    while (this->parent_arr[p] != n){ 

        p = this->parent_arr[p];
        this->parent_arr[p] = n;
    }
    return n;
}

int UnionFind::getSize(int n){

    return this->size_arr[n];
}


int UnionFind::getNextLabel(){
    return this->next_label;
}
