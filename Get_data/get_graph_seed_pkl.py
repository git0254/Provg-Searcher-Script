import pickle, hashlib
from collections import defaultdict

# 统一的wl_hash实现
def wl_hash(graph):
    """
    用于计算子图的WL-hash，基于节点属性和边属性进行哈希
    """
    nodes_str = '|'.join(sorted([f"{n}:{d}" for n, d in graph.nodes(data=True)]))
    edges_str = '|'.join(sorted([f"{u}->{v}:{d}" for u, v, d in graph.edges(data=True)]))
    txt = nodes_str + "#" + edges_str
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


with open("k_3test.pt", "rb") as f:
    k3_data = pickle.load(f)

hash2graph = defaultdict(dict)
hash2seed = defaultdict(set)

for center_uuid, ego_graph in k3_data.items():
    h = wl_hash(ego_graph)
    hash2graph[center_uuid][h] = ego_graph
    hash2seed[h].add(center_uuid)

with open("3hash2graph.pkl", "wb") as f:
    pickle.dump(hash2graph, f, protocol=4)
with open("3hash2seed.pkl", "wb") as f:
    pickle.dump(hash2seed, f, protocol=4)

print("3hash2graph.pkl、3hash2seed.pkl已生成")
