#include "mst.h"







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


static bool* cpuMST(const ECLgraph& g)
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

