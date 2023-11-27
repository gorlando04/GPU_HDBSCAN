#include "cuda_runtime.h"
#include "getters.cuh"
#include "math.h"


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

int findKNNlist(int *kNN,long int neig,long int i,int k){

    for (long int j=0;j<k;j++)
        if (kNN[neig*k+j] == i)
            return 1;
    return 0;
}