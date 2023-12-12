#ifndef TREE_CUH
#define TREE_CUH

#include <cuda.h>

#include "cuda_runtime.h"
#include "../structs/hdbscan_elements.cuh"
#include <vector>

#define INFINITE INT_MAX;


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



SingleLinkageNode* build_Linkage_tree( MSTedge *mst_edges ,int num,int num_nodes);



std::vector<int> BFS_from_hierarchy(SingleLinkageNode *hierarchy,int bfs_root, int num_nodes);

CondensedTreeNode* build_Condensed_tree(SingleLinkageNode *hierarchy,int num ,int num_nodes, int mpts, int *size);

Stability* compute_stability(CondensedTreeNode *condensed_tee, int size,int *pointer);

std::vector<int> BFS_from_cluster_tree(CondensedTreeNode *condensed_tree, int bfs_root, int condensed_size);

int* get_clusters(CondensedTreeNode *condensed_tree, int condensed_size, Stability *stabilities, int stability_size, int numValues);

int* do_labelling(CondensedTreeNode *condensed_tree, int condensed_size, std::vector<int> cluster, std::vector<std::tuple<int,int>> cluster_map);


class TreeUnionFind{


public:
    TreeUnionFind(int size);

    void Union(int x, int y);

    int Find(int x);

private:
  int *_data0;
  int *_data1;
  int size;

};




#endif
