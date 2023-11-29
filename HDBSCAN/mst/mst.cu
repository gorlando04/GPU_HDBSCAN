#include "mst.cuh"
#include "../graphs/graph.cuh"
#include "../calculates/calculates.cuh"
#include "../initializer/initialize.cuh"


static inline int serial_find(const int idx, int* const parent)
{
  int curr = parent[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr != (next = parent[curr])) {
      parent[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}


static inline void serial_join(const int a, const int b, int* const parent)
{
  const int arep = serial_find(a, parent);
  const int brep = serial_find(b, parent);
  if (arep > brep) {  // improves locality
    parent[brep] = arep;
  } else {
    parent[arep] = brep;
  }
}


bool* cpuMST(const ECLgraph& g)
{
  bool* const inMST = new bool [g.edges];
  int* const parent = new int [g.nodes];


  std::fill(inMST, inMST + g.edges, false);
  for (int i = 0; i < g.nodes; i++) parent[i] = i;

  std::vector<std::tuple<float, long int, int, int>> list;  // <weight, edge index, from node, to node>
  for (int i = 0; i < g.nodes; i++) {
    for (long int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int n = g.nlist[j];
      if (n > i) {  // only one direction
        list.push_back(std::make_tuple(g.eweight[j], j, i, n));
      }
    }
  }
  std::sort(list.begin(), list.end());

  int count = g.nodes - 1;
  for (long int pos = 0; pos < list.size(); pos++) {
    const long int a = std::get<2>(list[pos]);
    const long int b = std::get<3>(list[pos]);
    const long int arep = serial_find(a, parent);
    const long int brep = serial_find(b, parent);
    if (arep != brep) {
      const long int j = std::get<1>(list[pos]);
      inMST[j] = true;
      serial_join(arep, brep, parent);
      count--;
      if (count == 0) break;
    }
  }



  delete [] parent;
  return inMST;
}


MSTedge* buildMST(ECLgraph g,bool *edges, int shards_num){



  int *aux_nodes;
  cudaMallocManaged(&aux_nodes,(size_t)g.edges * sizeof(g.nlist[0]));
  createNodeList(aux_nodes,&g);


  MSTedge *finalEdges;
  
  finalEdges = new MSTedge[g.nodes-1];

  int soma =0;


  for (int i = 0; i < g.edges; i++) {

    if(edges[i]){

      finalEdges[soma].from_node = aux_nodes[i];
      finalEdges[soma].to_node = g.nlist[i];
      finalEdges[soma].weight = g.eweight[i];
      soma += 1;
    }
  }   
  
  if (soma == g.nodes-1)
    printf("MST montada corretamente\n");

  else{
    printf("MST montada incorretamente\n");
    exit(1);
  }

  std::sort(finalEdges,finalEdges+(g.nodes-1),compareEdgeByWeight);

  return finalEdges;

}
