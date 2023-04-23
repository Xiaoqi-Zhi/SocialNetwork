import matplotlib.pyplot as plt
import networkx as nx
# networkx 和 matplotplib 可能存在版本冲突问题，需要把matplotlib降级到2.2.3或者升级networkx到2.2以上版本（目前为2.2.6版本）
# 且需要在终端安装以下包才可运行
# conda install scipy
# conda install msgpack
# conda install qt=5.6 pyqt=5.6 sip=4.18

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
G.add_edges_from([(1, 3), (3, 5), (5, 7), (7, 9), (9, 2), (2, 4), (4, 6), (6, 8),(8,10)])
nx.draw(G, with_labels=True)
print(nx.adjacency_matrix(G).todense())
plt.show()