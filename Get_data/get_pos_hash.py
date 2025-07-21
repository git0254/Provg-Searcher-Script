import pickle
import hashlib
import torch
import pandas as pd
from collections import defaultdict

# 1. 读取k_3test.pt数据
with open('k_3test.pt', 'rb') as f:
    ego_dict = pickle.load(f)

# 2. 读取all_nodes_data.csv，建立node_name -> uuid映射
df = pd.read_csv('all_nodes_data.csv')
name2uuid = dict(zip(df['path'], df['uuid']))

# 3. 正样本生成：计算子图hash并填充结果字典
# 统一的wl_hash实现
def wl_hash(graph):
    """
    用于计算子图的WL-hash，基于节点属性和边属性进行哈希
    """
    nodes_str = '|'.join(sorted([f"{n}:{d}" for n, d in graph.nodes(data=True)]))
    edges_str = '|'.join(sorted([f"{u}->{v}:{d}" for u, v, d in graph.edges(data=True)]))
    txt = nodes_str + "#" + edges_str
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


# 4. 构建3posQueryHashes.pkl 和 3posQueryHashStats.pkl
query_hashes = {}
hash_stats = {}

def process_graphs():
    for center_uuid, graph in ego_dict.items():
        # 获取每个子图的哈希
        graph_hash = wl_hash(graph)
        # 获取该节点的名称
        node_name = next((k for k, v in name2uuid.items() if v == center_uuid), None)
        if node_name is None:
            continue

        # 将哈希值与查询条件、中心节点uuid关联
        if node_name not in query_hashes:
            query_hashes[node_name] = {}
        if center_uuid not in query_hashes[node_name]:
            query_hashes[node_name][center_uuid] = set()
        query_hashes[node_name][center_uuid].add(graph_hash)

        # 统计每个哈希在训练/测试集中的频次
        dataset = 'train' if center_uuid.startswith('train') else 'test'  # 根据实际情况调整
        if dataset not in hash_stats:
            hash_stats[dataset] = {}
        if node_name not in hash_stats[dataset]:
            hash_stats[dataset][node_name] = {}
        if graph_hash not in hash_stats[dataset][node_name]:
            hash_stats[dataset][node_name][graph_hash] = 0
        hash_stats[dataset][node_name][graph_hash] += 1

process_graphs()

# 5. 保存为pkl文件
with open('3posQueryHashes.pkl', 'wb') as f:
    pickle.dump(query_hashes, f, protocol=4)

with open('3posQueryHashStats.pkl', 'wb') as f:
    pickle.dump(hash_stats, f, protocol=4)

print("3posQueryHashes.pkl 和 3posQueryHashStats.pkl 已生成")
