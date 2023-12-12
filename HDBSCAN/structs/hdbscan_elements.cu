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

int getMaxChild(CondensedTreeNode *condensed_tree, int size){

    int max = condensed_tree[0].child;

    for (long int i=0;i<size;i++){
        if (condensed_tree[i].child > max)
            max = condensed_tree[i].child;
    }


    return max;
    
}

int getMaxParent(CondensedTreeNode *condensed_tree, int size){

    int max = condensed_tree[0].parent;

    for (long int i=0;i<size;i++){
        if (condensed_tree[i].parent > max)
            max = condensed_tree[i].parent;
    }


    return max;
    
}


int getMinParent(CondensedTreeNode *condensed_tree, int size){

    int min = condensed_tree[0].parent;

    for (long int i=0;i<size;i++){
        if (condensed_tree[i].parent < min)
            min = condensed_tree[i].parent;
    }


    return min;
    
}


float getMaxLambda(CondensedTreeNode *condensed_tree, int size){
    float max = condensed_tree[0].lambda_val;

    for (long int i=0;i<size;i++){
        if (condensed_tree[i].lambda_val > max)
            max = condensed_tree[i].lambda_val;
    }


    return max;

}

std::vector<int> getIndexes(CondensedTreeNode *condesed_tree,int size, int node){

    std::vector<int> vec;
    for(int i=0;i<size;i++){

        if (condesed_tree[i].parent == node)
            vec.push_back(i);
    }

    return vec;

}
