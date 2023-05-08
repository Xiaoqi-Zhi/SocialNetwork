import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
from matplotlib import cm
from sklearn.cluster import KMeans
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import girvan_newman

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
# nx.draw(G,pos,with_labels=True)

flag = int(input("1:planted I-partation 2:Kmeans 3:Greedy Modularity Communities 4:Girvan Newman Choose algorithm: "))
if flag == 1:
    # planted I-partation graph (Number of groups,Number of vertices in each group,probability of connecting vertices within a group,probability of connected vertices between groups)
    start_time = time.time()
    G_LP = nx.planted_partition_graph(4, 32, 0.5, 0.01, seed=114514)
    nx.draw(G_LP, with_labels=True)
    end_time = time.time()
    print(end_time - start_time)
    plt.show()

elif flag == 2:
    # Kmeans
    start_time = time.time()
    A = pd.DataFrame(nx.adjacency_matrix(G).todense())
    kmeans = KMeans(n_clusters=4, random_state=114514)
    kmeans.fit(A)
    y = kmeans.predict(A)
    colors = []
    for i in range(len(G.nodes)):
        colors.append(cm.Set1(y[i] + 1))
    nx.draw(G, pos=nx.circular_layout(G), node_color=colors)
    end_time = time.time()
    print(end_time - start_time)
    plt.show()

elif flag == 3:
    # Greedy Modularity Communities
    start_time = time.time()
    c = greedy_modularity_communities(G)
    a = dict(enumerate(c))
    b = [0] * len(G.nodes)
    j = 0
    for k, v in a.items():
        for v1 in range(len(v)):
            b[j + v1] = k
        j += len(v)
    colors = []
    for i in range(len(G.nodes)):
        colors.append(cm.Set1(b[i] + 1))
    nx.draw(G, pos, node_color=colors)
    end_time = time.time()
    print(end_time - start_time)
    plt.show()

elif flag == 4:
    # Girvan Newman
    start_time = time.time()
    comp = girvan_newman(G)
    # a = dict(enumerate(next(comp)))
    # a = dict(enumerate(next(comp)))
    a = dict(enumerate(next(comp)))
    a = dict(enumerate(next(comp)))
    b = [0] * len(G.nodes)
    j = 0
    for k, v in a.items():
        for v1 in range(len(v)):
            b[j + v1] = k
        j += len(v)
    colors = []
    for i in range(len(G.nodes)):
        colors.append(cm.Set1(b[i] + 1))
    nx.draw(G, pos, node_color=colors)
    end_time = time.time()
    print(end_time - start_time)
    plt.show()
