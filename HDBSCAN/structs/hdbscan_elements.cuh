#ifndef STRUCT_HDBSCAN_ELEMENT_CUH
#define STRUCT_HDBSCAN_ELEMENT_CUH

#include <cuda.h>

#include "cuda_runtime.h"
#include "vector"

const int numGPUs = 3;
const long int numValues = 1000000;//400000000; // Número de valores possíveis no vetor
const long int vectorSize = 32000000;//12000000000;// Tamanho do vetor
const int blockSize = 256;
const int k = 32;
const int mpts=k;

struct Vertex {

    int index;
    int grau;


};

// Custom comparator to sort MyStruct based on the 'value' field
bool compareVertexByDegree(const Vertex &a, const Vertex &b);

struct Untie_hub
{
    int index;
    int score;
};



// Custom comparator to sort MyStruct based on the 'value' field
bool compareVertexByScore(const Untie_hub &a, const Untie_hub &b);

void CheckCUDA_();


struct MSTedge{

    int from_node;
    int to_node;
    float weight;
};

bool compareEdgeByWeight(const MSTedge &a, const MSTedge &b);


struct SingleLinkageNode{

    int left_node;
    int right_node;
    float weight;
    int node_size;
};



struct CondensedTreeNode{

    int parent;
    int child;
    float lambda_val;
    int child_size;
};


int getMaxChild(CondensedTreeNode *condensed_tree, int size);

int getMaxParent(CondensedTreeNode *condensed_tree, int size);


int getMinParent(CondensedTreeNode *condensed_tree, int size);

float getMaxLambda(CondensedTreeNode *condensed_tree, int size);

std::vector<int> getIndexes(CondensedTreeNode *condesed_tree, int size, int node);

struct Stability{

    int cluster_id;
    float lambda;
};


#endif
