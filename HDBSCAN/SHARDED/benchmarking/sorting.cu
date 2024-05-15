#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <string>



long int vectorSize = 100000;// Tamanho do vetor


struct MSTedge{

    int from_node;
    int to_node;
    float weight;
};


bool compareEdgeByWeight(const MSTedge &a, const MSTedge &b){
    return a.weight < b.weight;
}

float randomFloat()
{
    return (float)(rand()) / (float)(rand());
}


int main( int argc, char *argv[]) {
   
    std::string vec = "100000";
    std::string iter = "0";
    if (argc == 3){
    vectorSize = atoi(argv[1]);
    vec = argv[1];
    iter = argv[2];
    printf("NUM VALUES SET TO %ld.\n",vectorSize);
  }

    MSTedge *finalEdges;
  
    finalEdges = new MSTedge[vectorSize];



  for (int i = 0; i < vectorSize; i++) {
      finalEdges[i].from_node = 0;
      finalEdges[i].to_node = 0;
      finalEdges[i].weight = randomFloat();
    
  } 
    clock_t t; 
    t = clock(); 

    std::sort(finalEdges,finalEdges+(vectorSize),compareEdgeByWeight);

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

    printf("Tempo para a ordencao: %.3lf\n",time_taken);


    std::string folder = "results-";
    std::string standart_1 = "Time_";
    std::string standart_2 = ".txt";
    std::string path = folder + vec  + "/"  + standart_1 + vec + "_" + iter + standart_2;
    FILE *pFile;

    pFile=fopen(path.c_str(), "a");

    if(pFile==NULL) {
        perror("Error opening file.");
    }
else {

        fprintf(pFile, "%lf", time_taken);
    }

fclose(pFile);


    return 0;
}
