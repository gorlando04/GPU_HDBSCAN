#ifndef TREE_CUH
#define TREE_CUH

#include <cuda.h>

#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"
#include <vector>


class UnionFind {
public:
     UnionFind(int N);

    void Union(int m, int n);

    int FastFind(int n);

    int getSize(int n);

    int getNextLabel();

private:
    int  *parent_arr;
    int *size_arr;
    int next_label;
};



#endif
