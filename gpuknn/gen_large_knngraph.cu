#include <algorithm>
#include <thread>
#include <vector>

#include "../tools/knndata_manager.hpp"
#include "../tools/nndescent_element.cuh"
#include "../tools/timer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gen_large_knngraph.cuh"
#include "knncuda_tools.cuh"
#include "knnmerge.cuh"
#include "nndescent.cuh"
#include <unistd.h>


#define NUM_GPU 3

using namespace std;

void ReadGraph(const string &graph_path, NNDElement **knn_graph_ptr,
               const int read_pos, const int read_num) {
  NNDElement *&knn_graph = *knn_graph_ptr;
  int dim;
  FileTool::ReadBinaryVecs(graph_path, &knn_graph, &dim, read_pos, read_num);
}

void WriteGraph(const string &graph_path, const NNDElement *knn_graph,
                const int graph_size, const int k, const int write_pos) {
  FileTool::WriteBinaryVecs(graph_path, knn_graph, write_pos, graph_size, k);
}

void WriteTXTGraph(const string &graph_path, const NNDElement *knn_graph,
                   const int graph_size, const int k, const int write_pos) {
  ofstream out(graph_path);
  for (int i = 0; i < graph_size; i++) {
    out << k << "\t";
    for (int j = 0; j < k; j++) {
      auto elem = knn_graph[i * k + j];
      out << elem.distance() << "\t" << elem.label() << "\t";
    }
    out << endl;
  }
  out.close();
}

void BuildEachShard(KNNDataManager &data_manager, const string &out_data_path,int id) {
  Timer knn_timer;

  int i = id;

  cudaSetDevice(id % 3);

  mutex mtx;
  
  NNDElement *knn_graph;
  data_manager.ActivateShard(i); // 0.12s
  float *vectors_dev;
  cudaMalloc(&vectors_dev, (size_t)data_manager.GetVecsNum(i) *
                                data_manager.GetDim() * sizeof(float));
  cudaMemcpy(vectors_dev, data_manager.GetVectors(i),
              (size_t)data_manager.GetVecsNum(i) * data_manager.GetDim() *
                  sizeof(float),
              cudaMemcpyHostToDevice);
  knn_timer.start();
  gpuknn::NNDescent(&knn_graph, vectors_dev, data_manager.GetVecsNum(i),
                    data_manager.GetDim(), 6, false);

  thread th1([&data_manager, knn_graph, out_data_path, i, &mtx]() {
      mtx.lock();
      Timer writer_timer;
      writer_timer.start();
      WriteGraph(data_manager.GetGraphDataPath(), knn_graph,
                data_manager.GetVecsNum(i), data_manager.GetK(),
                data_manager.GetBeginPosition(i));
      for (int j = 0; j < data_manager.GetVecsNum(i) * data_manager.GetK(); j++) {
        knn_graph[j].SetLabel(knn_graph[j].label() +
                              data_manager.GetBeginPosition(i));
      }
      WriteGraph(out_data_path, knn_graph, data_manager.GetVecsNum(i),
                data_manager.GetK(), data_manager.GetBeginPosition(i));
      data_manager.DiscardShard(i);
      delete[] knn_graph;
      mtx.unlock();
    });

  th1.detach();
  cudaFree(vectors_dev);

  mtx.lock();
  mtx.unlock();
}

int MergeList(const NNDElement *A, const int m, const NNDElement *B,
              const int n, NNDElement *C, const int max_size) {
  int i = 0, j = 0, cnt = 0;
  while ((i < m) && (j < n)) {
    if (A[i] <= B[j]) {
      C[cnt++] = A[i++];
      if (cnt >= max_size) goto EXIT;
    } else {
      C[cnt++] = B[j++];
      if (cnt >= max_size) goto EXIT;
    }
  }

  if (i == m) {
    for (; j < n; j++) {
      C[cnt++] = B[j];
      if (cnt >= max_size) goto EXIT;
    }
  } else {
    for (; i < m; i++) {
      C[cnt++] = A[i];
      if (cnt >= max_size) goto EXIT;
    }
  }
EXIT:
  return cnt;
}

void UpdateKNNGraph(NNDElement **old_graph_ptr, const NNDElement *new_graph,
                    const int graph_size, const int k) {
  NNDElement *&old_graph = *old_graph_ptr;
  NNDElement *tmp_list = new NNDElement[k * 2];
  for (int i = 0; i < graph_size; i++) {
    MergeList(&old_graph[i * k], k, &new_graph[i * k], k, tmp_list, k * 2);
    unique(tmp_list, tmp_list + k * 2);
    for (int j = 0; j < k; j++) {
      old_graph[i * k + j] = tmp_list[j];
    }
  }
  delete[] tmp_list;
}

__device__ void UniqueMergeSequential(const NNDElement* A, const int m,
                                      const NNDElement* B, const int n,
                                      NNDElement* C, const int k) {
    int i = 0, j = 0, cnt = 0;
    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[cnt++] = A[i++];
            if (cnt >= k) goto EXIT;
            while (i < m && A[i] <= C[cnt-1]) i++;
            while (j < n && B[j] <= C[cnt-1]) j++;
        } else {
            C[cnt++] = B[j++];
            if (cnt >= k) goto EXIT;
            while (i < m && A[i] <= C[cnt-1]) i++;
            while (j < n && B[j] <= C[cnt-1]) j++;
        }
    }

    if (i == m) {
        for (; j < n; j++) {
            if (B[j] > C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) goto EXIT;
        }
        for (; i < m; i++) {
            if (A[i] > C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) goto EXIT;
        }
    } else {
        for (; i < m; i++) {
            if (A[i] > C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) goto EXIT;
        }
        for (; j < n; j++) {
            if (B[j] > C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) goto EXIT;
        }
    }

EXIT:
    return;
}

__global__ void UpdateKNNGraphKernel(NNDElement *result_graph,
                                     const NNDElement *new_graph,
                                     const int start_pos) {
  int list_id = blockIdx.x + start_pos;
  int result_list_id = blockIdx.x;
  int tx = threadIdx.x;
  __shared__ NNDElement a_cache[NEIGHB_NUM_PER_LIST];
  __shared__ NNDElement b_cache[NEIGHB_NUM_PER_LIST];
  __shared__ NNDElement c_cache[NEIGHB_NUM_PER_LIST];
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int pos = i * WARP_SIZE + tx;
    if (pos < NEIGHB_NUM_PER_LIST) {
      a_cache[pos] = result_graph[result_list_id * NEIGHB_NUM_PER_LIST + pos];
      b_cache[pos] = new_graph[list_id * NEIGHB_NUM_PER_LIST + pos];
    }
  }
  if (tx == 0) {
    UniqueMergeSequential(a_cache, NEIGHB_NUM_PER_LIST, b_cache,
                          NEIGHB_NUM_PER_LIST, c_cache, NEIGHB_NUM_PER_LIST);
  }
  for (int i = 0; i < it_num; i++) {
    int pos = i * WARP_SIZE + tx;
    if (pos < NEIGHB_NUM_PER_LIST) {
      result_graph[result_list_id * NEIGHB_NUM_PER_LIST + pos] = c_cache[pos];
    }
  }
}

__global__ void PreProcIDKernel(NNDElement *result_knn_graph,
                                const int first_graph_size,
                                const int graph_size, const int k,
                                const int offset_a, const int offset_b) {
  int list_id = blockIdx.x;
  int neighb_id = threadIdx.x;
  auto &elem = result_knn_graph[list_id * k + neighb_id];
  if (elem.label() >= first_graph_size) {
    elem.SetLabel(elem.label() - first_graph_size + offset_b);
  } else {
    elem.SetLabel(elem.label() + offset_a);
  }
  return;
}

void PreProcID(NNDElement *result_knn_graph_host, const int first_graph_size,
               const int graph_size, const int k, const int offset_a,
               const int offset_b) {
  for (int i = 0; i < graph_size; i++) {
    for (int j = 0; j < k; j++) {
      auto &elem = result_knn_graph_host[i * k + j];
      if (elem.label() >= first_graph_size) {
        elem.SetLabel(elem.label() - first_graph_size + offset_b);
      } else {
        elem.SetLabel(elem.label() + offset_a);
      }
    }
  }
}


recursive_mutex multi_merge;


void MultiMerge(KNNDataManager &data_manager, const string &out_data_path,int id_gpu,int begin, int allow_next, bool isLast=false){

 //Define a GPU que irá realizar a tarefa

  int shards_num = data_manager.GetShardsNum();

  if (begin >= shards_num -2 && !isLast)
    return;

  cudaSetDevice(id_gpu);


  int i = begin;

  // Sync
  multi_merge.lock();

  mutex mtx;

  Timer merge_timer;
  merge_timer.start();

  // 0.45s
  NNDElement *result_first = 0, *result_second = 0;
  data_manager.ActivateShard(i);
  ReadGraph(out_data_path, &result_first, data_manager.GetBeginPosition(i),
            data_manager.GetVecsNum(i));
  


  for (int j = i + 1; j < shards_num; j++) { 




    NNDElement *result_knn_graph_dev;
    Timer timer;
    timer.start();
    data_manager.ActivateShard(j);





    gpuknn::KNNMergeFromHost(
        &result_knn_graph_dev, data_manager.GetVectors(i),
        data_manager.GetVecsNum(i), data_manager.GetKNNGraph(i),
        data_manager.GetVectors(j), data_manager.GetVecsNum(j),
        data_manager.GetKNNGraph(j));

    mtx.lock();
    mtx.unlock();

    Timer update_graph_timer;
    update_graph_timer.start();


    ReadGraph(out_data_path, &result_second, data_manager.GetBeginPosition(j),
              data_manager.GetVecsNum(j));
    NNDElement *result_first_dev;
    cudaMalloc(&result_first_dev, (size_t)data_manager.GetVecsNum(i) *
                                      data_manager.GetK() *
                                      sizeof(NNDElement));
    cudaMemcpy(result_first_dev, result_first,
                (size_t)data_manager.GetVecsNum(i) * data_manager.GetK() *
                    sizeof(NNDElement),
                cudaMemcpyHostToDevice);
    NNDElement *result_second_dev;
    cudaMalloc(&result_second_dev, (size_t)data_manager.GetVecsNum(j) *
                                        data_manager.GetK() *
                                        sizeof(NNDElement));
    cudaMemcpy(result_second_dev, result_second,
                (size_t)data_manager.GetVecsNum(j) * data_manager.GetK() *
                    sizeof(NNDElement),
                cudaMemcpyHostToDevice);
    PreProcIDKernel<<<data_manager.GetVecsNum(i) + data_manager.GetVecsNum(j),
                      data_manager.GetK()>>>(
        result_knn_graph_dev, data_manager.GetVecsNum(i),
        data_manager.GetVecsNum(i) + data_manager.GetVecsNum(j),
        data_manager.GetK(), data_manager.GetBeginPosition(i),
        data_manager.GetBeginPosition(j));
    cudaDeviceSynchronize();
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
      cerr << "PreProcID failed" << endl;
      cerr << cudaGetErrorString(status) << endl;
      exit(-1);
    }
    UpdateKNNGraphKernel<<<data_manager.GetVecsNum(i), WARP_SIZE>>>(
        result_first_dev, result_knn_graph_dev, 0);
    UpdateKNNGraphKernel<<<data_manager.GetVecsNum(j), WARP_SIZE>>>(
        result_second_dev, result_knn_graph_dev, data_manager.GetVecsNum(i));
    cudaDeviceSynchronize();
    cudaMemcpy(result_first, result_first_dev,
                (size_t)data_manager.GetVecsNum(i) * data_manager.GetK() *
                    sizeof(NNDElement),
                cudaMemcpyDeviceToHost);
    cudaMemcpy(result_second, result_second_dev,
                (size_t)data_manager.GetVecsNum(j) * data_manager.GetK() *
                    sizeof(NNDElement),
                cudaMemcpyDeviceToHost);
     status = cudaGetLastError();
     if (status != cudaSuccess) {
       cerr << cudaGetErrorString(status) << endl;
       exit(-1);
    }
    cudaFree(result_first_dev);
    cudaFree(result_second_dev);
    cudaFree(result_knn_graph_dev);
    int vecs_num = data_manager.GetVecsNum(j);
    int k = data_manager.GetK();
    int begin_pos = data_manager.GetBeginPosition(j);
    data_manager.DiscardShard(j);
    
        //Espera o atual ser discartado
    if (j == allow_next){
      multi_merge.unlock();
    }


    thread write_th([&result_second_dev, &result_second, vecs_num, k,
                      begin_pos, j, &out_data_path, &mtx]() {
      // Timer timer;
      // timer.start();
      mtx.lock();
      Timer write_graph_timer;
      write_graph_timer.start();
      WriteGraph(out_data_path, result_second, vecs_num,
                  k, begin_pos);
      delete[] result_second;
      mtx.unlock();
    });
    //write_th.detach();
    write_th.join();
  }

  //Isso aqui não tem pq ser thread.
  int vecs_num = data_manager.GetVecsNum(i);
  int k = data_manager.GetK();
  int begin_pos = data_manager.GetBeginPosition(i);
  WriteGraph(out_data_path, result_first, vecs_num, k, begin_pos);
  data_manager.DiscardShard(i);

    
    delete[] result_first;



  float merge_time = merge_timer.end();
  

       auto status = cudaGetLastError();
     if (status != cudaSuccess) {
       cerr << cudaGetErrorString(status) << endl;
       exit(-1);
    }


}

void GenLargeKNNGraph(const string &vecs_data_path, const string &out_data_path,
                      const int k,int num_shards) {
  assert(k == NEIGHB_NUM_PER_LIST);
  KNNDataManager data_manager(vecs_data_path, k, num_shards, 10000000);
  assert(data_manager.GetDim() == VEC_DIM);
  data_manager.CheckStatus();

    //Blank knngraph costs: 0.07
  FileTool::CreateBlankKNNGraph(data_manager.GetGraphDataPath(),
                                data_manager.GetVecsNum(), data_manager.GetK());
  FileTool::CreateBlankKNNGraph(out_data_path, data_manager.GetVecsNum(),
                                data_manager.GetK());
  Timer knn_timer;
  knn_timer.start();

  int shards_num = data_manager.GetShardsNum();

  int iters = num_shards/ NUM_GPU;

  for (int s=0;s<iters;s++){

    vector<thread> threads;

    for (int i=0;i<NUM_GPU;i++)
      threads.push_back (thread ([&data_manager, out_data_path,s,i] () {

        BuildEachShard(data_manager, out_data_path,(3*s+i));
      })); 

    for (auto &t: threads)
      t.join ();   
  }



  sleep(2);





  for (int s=0;s<iters;s++){

    vector<thread> threads;

    for (int i=0;i<NUM_GPU;i++){
      sleep(1);
      int allow_begin = NUM_GPU*s+i + NUM_GPU;

      if (allow_begin >= num_shards)
        allow_begin = num_shards -1;



      threads.push_back (thread ([&data_manager, out_data_path,s,i,allow_begin] () {

        

        MultiMerge(data_manager,out_data_path, i, NUM_GPU*s+i ,allow_begin);
      })); 
      sleep(1);
    }
    for (auto &t: threads)
      t.join (); 

    threads.clear();
  }

  MultiMerge(data_manager,out_data_path, NUM_GPU-1, shards_num - 2 ,-1,true);
  
 


  return;

}
