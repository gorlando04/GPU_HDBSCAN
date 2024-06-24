#include "build_kNNG.cuh"


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
