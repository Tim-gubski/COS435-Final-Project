import json, networkx as nx
net = json.load(open("examples/customroadnet.json"))
G = nx.DiGraph()
for r in net["roads"]:
    G.add_edge(r["startIntersection"], r["endIntersection"], id=r["id"])

print(nx.has_path(G, "5", "7"))