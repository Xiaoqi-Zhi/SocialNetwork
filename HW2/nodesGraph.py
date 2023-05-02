import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

data_nodes = pd.read_csv("anodes.csv")
data_edges  = pd.read_csv("aedges.csv")

nodes = data_nodes["Id"]
source = data_edges["Source"]
target = data_edges["Target"]

G = nx.Graph()
G.add_nodes_from(nodes)
edges = []
for i in range(len(source)):
    edges.append((source[i],target[i]))
G.add_edges_from(edges)
pos = nx.circular_layout(G)
print(nx.adjacency_matrix(G).todense())
print(len(nodes),len(edges))
nx.draw(G,pos)
plt.show()
