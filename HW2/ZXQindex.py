import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# 导入数据
data_nodes = pd.read_csv("anodes.csv")
data_edges = pd.read_csv("aedges.csv")

# 提取节点和链接
nodes = data_nodes["Id"]
source = data_edges["Source"]
target = data_edges["Target"]

# 生成图并插入节点和链接
G = nx.Graph()
G.add_nodes_from(nodes)
edges = []
for i in range(len(source)):
    edges.append((source[i], target[i]))
G.add_edges_from(edges)

print(edges)