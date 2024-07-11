#include "build_kNNG.cuh"


float calculate_euclidean_distance_fix(float *vector,long int idxa,long int idxb,int dim){

    float  soma = 0.0;
    for(long int i=0;i<dim;i++){
        soma += ( pow(vector[idxa*dim+i] - vector[idxb*dim+i],2) );
    }

    return sqrt(soma);
}


void ConstructLargeKNNGraph(int shards, string ref, string result) {


  string ref_path = ref;

  string result_path = result;



  int K = 32;
  
  GenLargeKNNGraph(ref_path, result_path, K,shards);




    return;
}

void PrepareVector(string base,string out){

    string base_path = base;
    float *vectors;
    long int vecs_size, vecs_dim;

    FileTool::ReadTxtVecs(base_path,&vectors,&vecs_size,&vecs_dim);


    // Arquivo em que será criado o .fvecs que será utilizado
    string out_path = out;

    // Escrita em binário
    FileTool::WriteBinaryVecs(out_path, vectors,
                              vecs_size,
                              vecs_dim);

    long int data_size2, data_dim2;
    float *vectors_data;
    FileTool::ReadBinaryVecs(out_path,&vectors_data,&data_size2,&data_dim2);
    return;
}


float* jesus(int *antihubs,int num_antihubs,int* paz){

    long int data_size2, data_dim2;
    float *vectors_data;
    const string out_path = "/nndescent/GPU_HDBSCAN/data/vectors.fvecs";

    FileTool::ReadBinaryVecs(out_path,&vectors_data,&data_size2,&data_dim2);


    float* antihubs_objects = new float[data_dim2 * num_antihubs];

    for (long int i=0;i<num_antihubs;i++){
	int idx = antihubs[i];
        for(long int j=0;j<data_dim2;j++)
   		antihubs_objects[i*data_dim2 + j] = vectors_data[idx*data_dim2 + j];
    }
    *paz = data_dim2;
    return antihubs_objects;
}


void write_bool_dict(int* dict,long int numValues,long int k){

    const string out_path = "/nndescent/GPU_HDBSCAN/results/dict.binary";

    FileTool::WriteBinaryVecs(out_path, dict, numValues,k); 

    return;

}


int* read_bool_dict(){

    const string out_path = "/nndescent/GPU_HDBSCAN/results/dict.binary";
    long int data_size2, data_dim2;
    int *vectors_data;

    FileTool::ReadBinaryVecs(out_path,&vectors_data,&data_size2,&data_dim2);


     return vectors_data;

}

NNDElement* ReadkNNGgraph(){

    const std::string path_to_kNNG = "/nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.kgraph";
    
    NNDElement *result_graph;
    long int knng_num, knng_dim;
    FileTool::ReadBinaryVecs(path_to_kNNG , &result_graph, &knng_num, &knng_dim);

    return result_graph;


}


float* ReadObjects(long int* pf){

    float *vectors_data;
    long int data_size, data_dim;

    const std::string path_to_data ="/nndescent/GPU_HDBSCAN/data/vectors.fvecs";
    FileTool::ReadBinaryVecs(path_to_data , &vectors_data, &data_size, &data_dim);

    *pf = data_dim;
    return vectors_data;

}


void fix_distances(NNDElement *result_graph,long int numValues,long int k){

    long int knng_num = numValues;
    long int knng_dim = k;

    // Cast NNDElement to int and float
    int *result_index_graph;


    long int data_dim = -1;
    float* vectors_data = ReadObjects(&data_dim);


//    cudaMallocManaged(&result_index_graph,(size_t)knng_num*knng_dim * sizeof(int));
    result_index_graph = new int[knng_num*knng_dim];

    #pragma parallel for
    for (long int i = 0; i < knng_num; i++) {
      for (long int j = 0; j < knng_dim; j++) {
        
        result_index_graph[i * knng_dim + j] = result_graph[i * knng_dim + j].label();
        }

    } 


   
  //  cudaMallocManaged(&distances,(size_t)numValues*knng_dim * sizeof(float));
    #pragma parallel for
    for (long int i = 0; i < knng_num; i++) {
        for (long int j = 0; j < knng_dim; j++) {

          result_graph[i * knng_dim + j].SetDistance(calculate_euclidean_distance_fix(vectors_data,i,result_index_graph[i * knng_dim + j],data_dim));

          }
    }

     FileTool::WriteBinaryVecs("/nndescent/GPU_HDBSCAN/results/NNDescent-KNNG.kgraph", result_graph,
                              numValues,
                              k);

    return ;
}



void write_euclidean_distance_vecs(float* dict,long int numValues){

    const std::string path_to_write = "/nndescent/GPU_HDBSCAN/results/euclidean.binary";


    FileTool::WriteBinaryDistances_(path_to_write, dict,
                              numValues);

    return;

}


float* read_euclidean_distance_vecs(){

    const std::string path_to_read = "/nndescent/GPU_HDBSCAN/results/euclidean.binary";
  float *vector;
  long int knng_num;
  FileTool::ReadBinaryDistances(path_to_read, &vector,
                              &knng_num);


     return vector;

}
