import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import seed, random

# for generating gaussian frequency distribution

for i in range(500):
  seed(i)
  values = randn(500) # function to generate 500 values having a Gaussian distribution with mean 0 and variance 1
  values = np.array(values)
  np.savetxt("_file_for_saving_the_omega_values_", values)
  
# for generating the graphs

for i in range(250):
  seed = i
  # for ER random graphs with size N and average degree k
  G = nx.erdos_renyi_graph(N, (k/N), seed=seed, directed=False)
  F = nx.to_numpy_matrix(G)
  with open("_file_for_saving_the_ith_ER_graph_adjacency_matrix_","wb") as f:
    for line in F:
        np.savetxt(f, line, fmt='%.2f')
        
  # for ER random graphs with size N and average degree k
  C = nx.barabasi_albert_graph(N, (k/2), seed=seed)
  H = nx.to_numpy_matrix(C)
  with open("_file_for_saving_the_ith_ER_graph_adjacency_matrix_","wb") as f:
    for line in H:
        np.savetxt(f, line, fmt='%.2f')
