from pyecharts import options as opts
from pyecharts.charts import Graph
import pandas as pd
import webbrowser
data_nodes = pd.read_csv("anodes.csv")
data_edges = pd.read_csv("aedges.csv")

source = data_edges["Source"]
target = data_edges["Target"]
nodes = set()
links = []
for i in range(len(source)):
    s = source[i]
    t = target[i]
    nodes.add(s)
    nodes.add(t)
    links.append(opts.GraphLink(source=s,target=t))
nodes = [opts.GraphNode(name = node) for node in nodes]
graph = Graph()

graph.set_global_opts(
    title_opts=opts.TitleOpts(title = "作业一关系图"),
    legend_opts = opts.LegendOpts(is_show= False)
)

graph.add(
    "",
    nodes,
    links,
    layout="circular"
)
graph.render("HW1pyecharts.html")
webbrowser.open("HW1pyecharts.html")