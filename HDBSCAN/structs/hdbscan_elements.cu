#include "hdbscan_elements.cuh"
#include <iostream>
#include "cuda_runtime.h"


// Custom comparator to sort MyStruct based on the 'value' field
bool compareVertexByDegree(const Vertex &a, const Vertex &b) {
    return a.grau < b.grau;
}

// Custom comparator to sort MyStruct based on the 'value' field
bool compareVertexByScore(const Untie_hub &a, const Untie_hub &b) {
    return a.score < b.score;
}


bool compareEdgeByWeight(const MSTedge &a, const MSTedge &b){
    return a.weight < b.weight;
}


void CheckCUDA_(){

    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("%s ",cudaGetErrorString(cuda_status));
        exit(-1);
    }
}
