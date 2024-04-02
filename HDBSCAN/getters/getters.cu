#include "cuda_runtime.h"
#include "getters.cuh"
#include "math.h"


int get_NumThreshold(long int numValues_){

    return sqrt(numValues_);
}

int get_TiedVertexes(Vertex *vertexes,int pos_threshold, int value_threshold){

     // Pega quantos empates temos na lista final
    int missing = 0;
    
    for (int i=0;i < pos_threshold; i++){
        if (vertexes[i].grau  == value_threshold){
            missing = pos_threshold - i ;
            break;
        }
    }

    return missing;
}


void get_IndexThreshold(int *finalCounts,int *treshold_idx,int value_threshold, long int numValues){

    int control = 0;

    // Pega os índices iguais ao threshold.
    for (int i=0;i<numValues;i++){

        if (finalCounts[i] == value_threshold){
            treshold_idx[control] = i;
            control++;
        }
    }
}

int findKNNlist(int *kNN,long int neig,long int i,int k){

    for (long int j=0;j<k;j++)
        if (kNN[neig*k+j] == i)
            return 1;
    return 0;
}


int buscaBinaria(Stability *stabilities, int alvo, int size) {
    int inicio = 0;
    int fim = size - 1;

    while (inicio <= fim) {
        int meio = inicio + (fim - inicio) / 2;

        // Verifica se o elemento do meio é o alvo
        if (stabilities[meio].cluster_id == alvo) {
            return meio;
        }

        // Se o alvo estiver à esquerda, atualiza o índice final
        else if (stabilities[meio].cluster_id > alvo) {
            fim = meio - 1;
        }

        // Se o alvo estiver à direita, atualiza o índice inicial
        else {
            inicio = meio + 1;
        }
    }

    // Retorna -1 se o elemento não for encontrado
    return -1;
}


int buscaBinaria(std::vector<std::tuple<int,int>> cluster_size, int alvo){
    int inicio = 0;
    int fim = cluster_size.size() - 1;

    while (inicio <= fim) {
        int meio = inicio + (fim - inicio) / 2;

        // Verifica se o elemento do meio é o alvo
        if (std::get<0>(cluster_size[meio]) == alvo) {
            return meio;
        }

        // Se o alvo estiver à esquerda, atualiza o índice final
        else if (std::get<0>(cluster_size[meio]) > alvo) {
            fim = meio - 1;
        }

        // Se o alvo estiver à direita, atualiza o índice inicial
        else {
            inicio = meio + 1;
        }
    }

    // Retorna -1 se o elemento não for encontrado
    return -1;
}


int buscaBinaria(std::vector<std::tuple<int,bool>> cluster_size, int alvo){
    int inicio = 0;
    int fim = cluster_size.size() - 1;

    while (inicio <= fim) {
        int meio = inicio + (fim - inicio) / 2;

        // Verifica se o elemento do meio é o alvo
        if (std::get<0>(cluster_size[meio]) == alvo) {
            return meio;
        }

        // Se o alvo estiver à esquerda, atualiza o índice final
        else if (std::get<0>(cluster_size[meio]) > alvo) {
            fim = meio - 1;
        }

        // Se o alvo estiver à direita, atualiza o índice inicial
        else {
            inicio = meio + 1;
        }
    }

    // Retorna -1 se o elemento não for encontrado
    return -1;
}


int buscaBinaria(std::vector<int> vector, int alvo){

    int inicio = 0;
    int fim = vector.size() - 1;

    while (inicio <= fim) {
        int meio = inicio + (fim - inicio) / 2;

        // Verifica se o elemento do meio é o alvo
        if (vector[meio] == alvo) {
            return meio;
        }

        // Se o alvo estiver à esquerda, atualiza o índice final
        else if (vector[meio] > alvo) {
            fim = meio - 1;
        }

        // Se o alvo estiver à direita, atualiza o índice inicial
        else {
            inicio = meio + 1;
        }
    }

    // Retorna -1 se o elemento não for encontrado
    return -1;


}

int buscaBinaria(int* vector, int size, int alvo){

    int inicio = 0;
    int fim = size - 1;

    while (inicio <= fim) {
        int meio = inicio + (fim - inicio) / 2;

        // Verifica se o elemento do meio é o alvo
        if (vector[meio] == alvo) {
            return meio;
        }

        // Se o alvo estiver à esquerda, atualiza o índice final
        else if (vector[meio] > alvo) {
            fim = meio - 1;
        }

        // Se o alvo estiver à direita, atualiza o índice inicial
        else {
            inicio = meio + 1;
        }
    }

    // Retorna -1 se o elemento não for encontrado
    return -1;
}
