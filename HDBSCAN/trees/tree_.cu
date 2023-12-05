#include "tree.cuh"
#include "../initializer/initialize.cuh"
#include <algorithm>    
#include <tuple>
#include "../getters/getters.cuh"


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



TreeUnionFind::TreeUnionFind(int size){

    this->size = size;
    // Parent array
    int *aux;
    cudaMallocManaged(&aux,(size_t)size * sizeof(int));

    
    int gridSize = ((size) + blockSize - 1) / blockSize;
    initializeVectorCounts<<<gridSize,blockSize>>>(aux,0,size);

    cudaDeviceSynchronize();
    CheckCUDA_();

    this->_data1 = new int[size];
    cudaMemcpy(this->_data1, aux,(size_t)size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(aux);

    cudaDeviceSynchronize();
    CheckCUDA_();

    //Initiate data0
    this->_data0 = new int[size];

    cudaMallocManaged(&aux,(size_t)size * sizeof(int));

    
    gridSize = ((size) + blockSize - 1) / blockSize;
    initializeVectorArange<<<gridSize,blockSize>>>(aux,size);

    cudaDeviceSynchronize();
    CheckCUDA_();

    cudaMemcpy(this->_data0, aux,(size_t)size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(aux);

    cudaDeviceSynchronize();
    CheckCUDA_();

}

void TreeUnionFind::Union(int x, int y){


    int x_root = Find(x);
    int y_root = Find(y);


    if (this->_data1[x_root] < this->_data1[y_root])
        this->_data0[x_root] = y_root;
    else if (this->_data1[x_root] > this->_data1[y_root])
        this->_data0[y_root] = x_root;
    else{
        this->_data0[y_root] = x_root;
        this->_data1[x_root] += 1;
    }

    return ;
}

int TreeUnionFind::Find(int x){

    if (this->_data0[x] != x)
        this->_data0[x] = Find(this->_data0[x]);

    return this->_data0[x];
}

SingleLinkageNode* build_Linkage_tree( MSTedge *mst_edges ,int num,int num_nodes){
    
    
    UnionFind U = UnionFind(num);


    SingleLinkageNode *result_arr;
    result_arr = new SingleLinkageNode[num_nodes-1];

    for (long int i=0;i<(num_nodes-1);i++){

      int a = mst_edges[i].from_node;
      int b = mst_edges[i].to_node;
      float delta = mst_edges[i].weight;

      int aa = U.FastFind(a);
      int bb = U.FastFind(b);
      
      result_arr[i].left_node = aa;
      result_arr[i].right_node = bb;
      result_arr[i].weight = delta;
      result_arr[i].node_size = (U.getSize(aa)) + (U.getSize(bb));

      U.Union(aa,bb);

    }

//    printf("Single Linkage Tree montada\n");

    return result_arr;

}



std::vector<int> BFS_from_hierarchy(SingleLinkageNode *hierarchy,int bfs_root, int num_nodes){

    int dim = num_nodes;
    int max_node = 2 * num_nodes;
    int num_points = max_node - dim + 1;

    std::vector<int> to_process;
    to_process.push_back(bfs_root);

    std::vector<int>  result;

    while (to_process.size() > 0){



        //merge result com to_process
        result.insert( result.end(), to_process.begin(), to_process.end() );

        //Aux vector
       std::vector<int> aux;

        for(int i=0;i<to_process.size();i++){
            if (to_process[i] >= num_points)
		aux.push_back(to_process[i] - num_points);

        }

         
        std::vector<int> aux_;
        if (aux.size() > 0){
            
            for (int i=0;i<aux.size();i++){
                aux_.push_back(hierarchy[aux[i]].left_node);

                aux_.push_back(hierarchy[aux[i]].right_node);
            }
        }

        to_process.swap(aux_);

    }


    return result;


}


CondensedTreeNode* build_Condensed_tree(SingleLinkageNode *hierarchy,int num ,int num_nodes, int mpts,int *size){



    int root = 2 * num_nodes;

    int num_points = root / 2;
    num_points += 1;
    int nextlabel = num_points+1;

    std::vector<int> node_list;  
    node_list = BFS_from_hierarchy(hierarchy, root,num_nodes);
    int node_list_size = node_list.size();




    int *relabel;
    relabel = new int[root+1];

    relabel[root] = num_points;

    CondensedTreeNode *result_list;
    result_list = new CondensedTreeNode[root];


    int *ignore;
    cudaMallocManaged(&ignore,(size_t)node_list_size * sizeof(int));


    int gridSize = (node_list_size + blockSize - 1) / blockSize;
    initializeVectorCounts<<<gridSize,blockSize>>>(ignore,0,node_list_size);


    int node, left, right, left_count,right_count;
    float lambda_value;

    int append_variable = 0;


    std::vector<int> aux_node_list_left;
    std::vector<int> aux_node_list_right;



    for (long int i=0;i<node_list_size;i++){
        node = node_list[i];
        if (ignore[node] || node < num_points)
            continue;
        

        left = hierarchy[node-num_points].left_node;
        right = hierarchy[node-num_points].right_node;
        lambda_value = INFINITE;
        


        if (hierarchy[node-num_points].weight > 0.0)
            lambda_value = 1.0 / hierarchy[node-num_points].weight;
        
        left_count = 1;
        if ( left >= num_points) {
            left_count = hierarchy[left - num_points].node_size;
        }

        right_count = 1;
        if (right >= num_points){
            right_count = hierarchy[right - num_points].node_size;
        }

        if (left_count >= mpts && right_count >= mpts){
            relabel[left] = nextlabel;
            nextlabel += 1;

            result_list[append_variable].parent = relabel[node];
            result_list[append_variable].child = relabel[left];
            result_list[append_variable].lambda_val = lambda_value;
            result_list[append_variable].child_size = left_count;

            append_variable += 1;


            relabel[right] = nextlabel;
            nextlabel += 1;

            result_list[append_variable].parent = relabel[node];
            result_list[append_variable].child = relabel[right];
            result_list[append_variable].lambda_val = lambda_value;
            result_list[append_variable].child_size = right_count;

            append_variable += 1;
        }

        else if (left_count < mpts && right_count < mpts){

            aux_node_list_left = BFS_from_hierarchy(hierarchy, left, num_nodes);
            int sub_node;
            for (int j=0;j<aux_node_list_left.size();j++){
                sub_node = aux_node_list_left[j];

                if (sub_node < num_points){
                    
                    result_list[append_variable].parent = relabel[node];
                    result_list[append_variable].child = sub_node;
                    result_list[append_variable].lambda_val = lambda_value;
                    result_list[append_variable].child_size = 1;
                    append_variable += 1;
                }
                ignore[sub_node] = 1;
            }

            aux_node_list_right = BFS_from_hierarchy(hierarchy, right, num_nodes);
            for (int j=0;j<aux_node_list_right.size();j++){
                sub_node = aux_node_list_right[j];

                if (sub_node < num_points){
                    
                    result_list[append_variable].parent = relabel[node];
                    result_list[append_variable].child = sub_node;
                    result_list[append_variable].lambda_val = lambda_value;
                    result_list[append_variable].child_size = 1;
                    append_variable += 1;
                }
                ignore[sub_node] = 1;
            }
            
        }


        else if (left_count < mpts){

            relabel[right] = relabel[node];
            aux_node_list_left = BFS_from_hierarchy(hierarchy, left, num_nodes);
            int sub_node;
            for (int j=0;j<aux_node_list_left.size();j++){
                sub_node = aux_node_list_left[j];

                if (sub_node < num_points){
                    
                    result_list[append_variable].parent = relabel[node];
                    result_list[append_variable].child = sub_node;
                    result_list[append_variable].lambda_val = lambda_value;
                    result_list[append_variable].child_size = 1;
                    append_variable += 1;
                }
                ignore[sub_node] = 1;
            }
        }
        else {
            relabel[left] = relabel[node];
            aux_node_list_right = BFS_from_hierarchy(hierarchy, right, num_nodes);
	    int sub_node;
            for (int j=0;j<aux_node_list_right.size();j++){
                sub_node = aux_node_list_right[j];

                if (sub_node < num_points){
                    
                    result_list[append_variable].parent = relabel[node];
                    result_list[append_variable].child = sub_node;
                    result_list[append_variable].lambda_val = lambda_value;
                    result_list[append_variable].child_size = 1;
                    append_variable += 1;
                }
                ignore[sub_node] = 1;
            }
        }



    }

    CondensedTreeNode *result_list_;
    result_list_ = new CondensedTreeNode[append_variable];

    for(long int i=0;i<append_variable;i++){
        result_list_[i].parent =   result_list[i].parent;
        result_list_[i].child = result_list[i].child;
        result_list_[i].lambda_val = result_list[i].lambda_val;
        result_list_[i].child_size = result_list[i].child_size;

    }

    *size = append_variable;

    return result_list_;
}

Stability* compute_stability(CondensedTreeNode *condensed_tee, int size,int *pointer){


    int max_child = getMaxChild(condensed_tee,size);
    int max_parent =getMaxParent(condensed_tee,size);
    int min_parent = getMinParent(condensed_tee,size);

    int largest_child = std::max(max_child, min_parent);
    int num_clusters = max_parent - min_parent + 1;

    int smalles_cluster = min_parent;

 std::vector<std::tuple<int,float>> sorted;


    for(int i=0;i<size;i++)
        sorted.push_back(std::make_tuple(condensed_tee[i].child,condensed_tee[i].lambda_val));

    std::sort(sorted.begin(),sorted.end());
    float* birth_arr;
    birth_arr = new float[largest_child+1];

    int current_child = -1;
    float min_lambda = 0.0;

    for (long int i=0;i<size;i++){
        int child = std::get<0>(sorted[i]);
        float lambda_ = std::get<1>(sorted[i]);

        if (child == current_child)
            min_lambda = std::min(min_lambda,lambda_);
        else if (current_child != -1){
            birth_arr[current_child] = min_lambda;
            current_child = child;
            min_lambda = lambda_;
        }
        else {
            current_child = child;
            min_lambda = lambda_;
        }

    }

    if (current_child != -1)
        birth_arr[current_child] = min_lambda;

    birth_arr[smalles_cluster] = 0.0;

    // Parent array
    float *result_array;
    cudaMallocManaged(&result_array,(size_t)num_clusters * sizeof(float));

    
    int gridSize = (num_clusters + blockSize - 1) / blockSize;
    initializeVectorCounts<<<gridSize,blockSize>>>(result_array,0.0,num_clusters);

    cudaDeviceSynchronize();
    CheckCUDA_();

    for (long int i=0;i<size;i++){
        int parent = condensed_tee[i].parent;
        float lambda_ = condensed_tee[i].lambda_val;
        int child_size = condensed_tee[i].child_size;

        int result_index = parent - smalles_cluster;

        result_array[result_index] += ( (lambda_ - birth_arr[parent] ) * child_size );
    }


    int size_ = max_parent + 1 - smalles_cluster;    

    Stability *result;
    result = new Stability[size_];
    *pointer = size_;

    for (long int i=0;i<size_;i++){
        result[i].cluster_id = smalles_cluster + i;
        result[i].lambda = result_array[i];
    }

    return result;

}






std::vector<int> BFS_from_cluster_tree(CondensedTreeNode *condensed_tree, int bfs_root, int condensed_size){


    std::vector<int> to_process;
    to_process.push_back(bfs_root);

    std::vector<int>  result;
     while (to_process.size() > 0){
	sort(to_process.begin(),to_process.end());
        //merge result com to_process
        result.insert( result.end(), to_process.begin(), to_process.end() );

        //Aux vector
        std::vector<int> aux;
        for(int i=0;i<condensed_size;i++){
            int pos = buscaBinaria(to_process, condensed_tree[i].parent);
            if(pos != -1){
                aux.push_back(i);
            }
        }

        std::vector<int> aux_;
        for(int i=0;i<aux.size();i++)
            aux_.push_back(condensed_tree[aux[i]].child);

        to_process.swap(aux_);
    }
    sort(result.begin(),result.end());
    return result;

}

int* do_labelling(CondensedTreeNode *condensed_tree, int condensed_size, std::vector<int> cluster, std::vector<std::tuple<int,int>> cluster_map){

    int root_cluster = getMinParent(condensed_tree,condensed_size);
    int max_parent = getMaxParent(condensed_tree,condensed_size);
    
    int *result_arr;
//    printf("ROOT = %d\n",root_cluster);
    result_arr = new int[root_cluster];

    HashLabels arrays;

    arrays = initializeHash(condensed_tree,condensed_size);

    //classe maluca, passando max_parent_array+1
    TreeUnionFind union_find = TreeUnionFind(max_parent+1);

    for (long int n=0;n<condensed_size;n++){
        int child = condensed_tree[n].child;
        int parent = condensed_tree[n].parent;

        //Mais um overload
        if ( buscaBinaria(cluster,child) == -1){
            union_find.Union(parent, child);
        }
    }

   int soma=0;
    for (long int n=0;n<root_cluster;n++){

        int cluster = union_find.Find(n);

        if (cluster < root_cluster)
            result_arr[n] = -1;
        else if (cluster == root_cluster)
            result_arr[n] = -1;
	else {

            float point_lambda = arrays.lambda_array[n];

            float cluster_lambda = arrays.lambda_array[cluster];
            if (point_lambda > cluster_lambda){
                soma += 1;
		  int cluster_pos = buscaBinaria(cluster_map,cluster);
                result_arr[n] = std::get<1>(cluster_map[cluster_pos]);
            }
            else
                result_arr[n] = -1;
        }
    
    }
	printf("soma %d\n",soma);

    return result_arr;
}

int* get_clusters(CondensedTreeNode *condensed_tree, int condensed_size, Stability *stabilities, int stability_size, int numValues){

    int *node_list;
    node_list = new int[stability_size-1];
    int size_node_list = 0;
    for(int i=stability_size-1;i>0;i--){
        node_list[size_node_list] = stabilities[i].cluster_id;
        size_node_list += 1;
    }

    int count = 0;
    for(long int i=0;i<condensed_size;i++)
        if (condensed_tree[i].child_size > 1)
            count += 1;
    
    CondensedTreeNode *cluster_tree;
    cluster_tree = new CondensedTreeNode[count];

    count = 0;
    for(long int i=0;i<condensed_size;i++){
        if (condensed_tree[i].child_size > 1){
            cluster_tree[count].child = condensed_tree[i].child;
            cluster_tree[count].lambda_val = condensed_tree[i].lambda_val;
            cluster_tree[count].child_size = condensed_tree[i].child_size;
            cluster_tree[count].parent = condensed_tree[i].parent;
            count += 1;
        }
    }

    std::vector<bool> is_cluster;

    // Hashed is_cluster
    int max_parent = getMaxParent(condensed_tree,condensed_size);
    int min_parent = getMinParent(condensed_tree,condensed_size);

    for (int i=0;i < max_parent+1;i++){
        if (i > min_parent)
        is_cluster.push_back(true);
        else
        is_cluster.push_back(false);

    }


    int num_points = numValues;

    float max_lambda = getMaxLambda(condensed_tree,condensed_size);

    int max_cluster_size = num_points+1;

    std::vector<int> cluster_sizes;


    for (int i=0;i< (max_parent-min_parent+1); i++)
        cluster_sizes.push_back(0);

    for (int i=0;i< count; i++){
        int pos = (cluster_tree[i].child - min_parent);
        cluster_sizes[pos] = cluster_tree[i].child_size;
    }



    for (long int i=0;i<size_node_list;i++){

        //Idxs
        std::vector<int> child_selection = getIndexes(cluster_tree,count,node_list[i]);

        //Subtree_stability
        float subtree_stability = 0;
        for(int j=0;j<child_selection.size();j++){

            int child = cluster_tree[child_selection[j]].child;
            int pos = buscaBinaria(stabilities, child, stability_size);
            subtree_stability += stabilities[pos].lambda;
        }

        int pos = buscaBinaria(stabilities, node_list[i], stability_size);


        if (subtree_stability > stabilities[pos].lambda || cluster_sizes[node_list[i]-min_parent] > max_cluster_size){
            is_cluster[node_list[i]] = false;
            stabilities[pos].lambda = subtree_stability;
        }

        else{
            std::vector<int> sub_nodes;
            sub_nodes = BFS_from_cluster_tree(condensed_tree, node_list[i], condensed_size);
            for (int j=0;j<sub_nodes.size();j++){
                if (sub_nodes[j] != node_list[i]){
                    is_cluster[sub_nodes[j]] = false;
                }

            }
        }

    }

    std::vector<int> cluster;
    for (int i=0;i < is_cluster.size();i++)
        if (is_cluster[i]){
            cluster.push_back(i);
	}
  printf("CLUSTERS FIND = %ld \n",cluster.size());
  sort(cluster.begin(),cluster.end());


    std::vector<std::tuple<int,int>> cluster_map;
    int current_clust = 0;
    for (int i=0;i<cluster.size();i++){
        cluster_map.push_back( std::make_tuple(cluster[i] , current_clust)  );
	current_clust += 1;
    }
    int *labels;
    labels = do_labelling(condensed_tree,condensed_size,cluster,cluster_map);
/*    for (int i=numValues;i>-1;i--){

        printf("{ID: %d, Clus_ID: %d}, ",i,labels[i]);
    }*/
    return labels;
}
