# HDBSCAN

This repository presents a multi-GPU version of HDBSCAN, and uses the kNNG implementation presented in [Multi-GPU kNNG](https://github.com/gorlando04/Scalable-distributed-algorithms-for-approximating-the-kNNG) and benefits from paralelism provided by the multiple GPUs.


## Software Especifications

The comparsion was performed in a GPU server with the following specifications

<aside>
💡 RAM: 128 GB
	
💡 Processor: 2 Intel(R) Xeon(R) Silver 4208 CPU

💡 GPU: 3 NVIDIA GeForce RTX 2080 Super

💡 Ubuntu 22.04

💡 CUDA 12.0

💡 Docker Version: 20.10.21

</aside>


## Docker images 

In this experiment, we used one docker-image for supporting our work.

To build the container for HDBSCAN , the following image was used:

```
docker run --gpus '"device=0,1,2"' --name GPU_HDBSCAN --rm -it -v /home/gabriel:/nndescent -w /nndescent --shm-size=1g --ulimit memlock=-1 nvidia/cuda:12.0.1-devel-ubuntu22.04
```

When acessing the container, the following steps were done:

```bash
#Installing Nano
apt update
apt install nano
apt install -y tmux
clear 

apt install -y cmake

apt install -y wget

apt install -y python

apt install -y python3-pip

pip install scikit-learn

pip install numpy

pip install pandas

pip install cython
pip install hdbscan



```

After the beggining of every execution, it was tested if the container was with GPUs working with the **************nvidia-smi************** command. 


## Observations

Firstly, it is important to say that in order to compile correctly the source code it is important to follow this instructions:

1. Check GPU compute capability in [NVIDIA](https://developer.nvidia.com/cuda-gpus). After that, it is important to change the value for the correct compute capability in [CMakeLists.txt](https://github.com/gorlando04/GPU_HDBSCAN/blob/main/HDBSCAN/CMakeLists.txt). In the CMake file, the following value must be changed to the correct compute capability: 

```
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FILE_OFFSET_BITS=64 -O3 -std=c++14 -arch=sm_(COMPUTE_CAPABILITY) -rdc=true -Xcompiler -fopenmp -pthread")
```

2. After that the following commands must be done:

```
cd HDBSCAN/cmake
cmake ..
make
```

3. Finally, the executable file will be avaiable, and its called **hdbscan_**.


## Parameters

It is important to say that in [nndescent.cuh](https://github.com/gorlando04/GPU_HDBSCAN/main/gpuknn/nndescent.cuh) the primary parameters of the algorithm are set. This is done in the first lines.

```cpp
const int VEC_DIM = 12; // Vectors dimension
const int NEIGHB_NUM_PER_LIST = 32; //Value of K in kNN
const int SAMPLE_NUM = 16;  // assert(SAMPLE_NUM * 2 <= NEIGHB_NUM_PER_LIST);
const int NND_ITERATION = 6; // Iterations of the algorithm
const int MERGE_SAMPLE_NUM = 12;
const int MERGE_ITERATION = 11;

```

Also in [hdbscan_elements.cuh](https://github.com/gorlando04/GPU_HDBSCAN/blob/main/HDBSCAN/structs/hdbscan_elements.cuh) it is necessary to define the parameters of HDBSCAN. This can be done by modyfing the first lines:

```cpp
const int numGPUs = 3;
const int blockSize = 256;
const int k = 32;
const int mpts=k;

```

## Experiments




## Reference






