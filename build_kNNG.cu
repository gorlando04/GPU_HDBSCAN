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

    return;
}