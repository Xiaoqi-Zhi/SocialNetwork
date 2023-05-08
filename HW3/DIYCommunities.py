import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
from matplotlib import cm
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.colors import ListedColormap
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
pos = nx.circular_layout(G)

start_time = time.time()
# 构建图并计算CN指标矩阵
def build_cn_matrix(G):
    nodes = list(G.nodes())
    n = len(nodes)
    cn_matrix = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j > i:
                common_neighbors = set(G[u]) & set(G[v])
                cn_matrix[i, j] = cn_matrix[j, i] = len(common_neighbors)
    return cn_matrix

# 对节点进行聚类
# k是簇的数量
def cluster_nodes(cn_matrix, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(cn_matrix)
    labels = kmeans.labels_
    return labels

# 进行社团识别
cn_matrix = build_cn_matrix(G)
labels = cluster_nodes(cn_matrix, 3)
colors = ['r', 'g', 'b']
node_colors = [colors[label] for label in labels]
cmap = ListedColormap(colors)
nx.draw(G, pos, node_color=node_colors)
end_time = time.time()
print(end_time-start_time)
plt.show()
for i, node in enumerate(G.nodes()):
    print(f"Node '{node}' belongs to cluster {labels[i]}")