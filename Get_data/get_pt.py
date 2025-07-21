import networkx as nx
import torch
import pandas as pd
import numpy as np
from embedders import get_embedder_by_name
from partition import detect_communities
from pathlib import Path
from datahandlers import get_handler
import pickle
import random
import torch



# 计算两图的GED
def simple_ged(g1, g2):
    ged_vertex_diff = abs(g1.vcount() - g2.vcount())
    ged_edge_diff = len(set(g1.get_edgelist()).symmetric_difference(g2.get_edgelist()))
    ged_total = ged_vertex_diff + ged_edge_diff
    return ged_total

def load_node_data():
    # 加载 all_nodes_data.csv 获取路径到UUID的映射
    nodes_data = pd.read_csv('all_nodes_data.csv')
    path_to_uuid = dict(zip(nodes_data['path'], nodes_data['uuid']))  # path -> uuid 映射
    path_to_type = dict(zip(nodes_data['path'], nodes_data['type_index']))  # path -> type 映射
    return path_to_uuid, path_to_type
# 得到图G的两个communities_list
def _get_pair(positive, communities_list, idx, G):
    """Generate one pair of graphs from a given community structure.
    (取第 idx 个社区并在原图 G 上切割得到子图)
    Args:
        positive (bool): 是否是正样本-->              positive=True=>小扰动   positive=False => 大扰动
        communities (dict): {社区ID: [节点列表]}
        G (igraph.Graph): 原始的完整图

    Returns:
        permuted_g (igraph.Graph): 经过节点重排的社区子图
        changed_g (igraph.Graph): 经过边修改的社区子图
    """
    # 从 `communities` 里选一个社区
    if idx > len(communities_list) - 1:
        idx = idx % len(communities_list)  # 可选：循环利用
    community_nodes = communities_list[idx]
    g = G.subgraph(community_nodes)

    if idx + 1 > len(communities_list) - 1:
        next = idx % len(communities_list)
    else:
        next = idx + 1
    changed_community_nodes = communities_list[next]
    g_add = G.subgraph(changed_community_nodes)
    # 根据 `positive` 选择边的修改数量
    n_changes = 0.1 if positive else 0.8
    # 对子图 `g` 进行边修改
    changed_g = substitute_random_edges_ig(g, g_add, positive, n_changes)
    return g, changed_g


def substitute_random_edges_ig(G, G_add, positive, ratio=0.1):
    """在 igraph.Graph (有向图) 中随机替换 `n` 条边"""
    G = G.copy()  # 复制图，避免修改原始图
    n_nodes = G.vcount()  # 获取节点数
    edges = G.get_edgelist()  # 获取所有边的列表

    ############################操作边#################################################
    # 1、随机选择 `n` 条边进行删除
    total_edges = len(edges)
    n_changes_edges = int(total_edges * ratio)
    # 1、随机选择 `n_changes_edges` 条边进行删除
    e_remove_idx = np.random.choice(total_edges, n_changes_edges, replace=False)  # 选 `n_changes_edges` 条边索引
    e_remove = [edges[i] for i in e_remove_idx]  # 获取要删除的边
    edge_set = set(map(tuple, edges))  # 转换为集合，方便查重

    # 2、随机生成 `n_changes_edges` 条新边，确保新边不重复且不和删除的边相同
    max_attempts = 1000
    attempts = 0
    e_add = set()
    while len(e_add) < n_changes_edges and attempts < max_attempts:
        e = tuple(np.random.choice(n_nodes, 2, replace=False))
        if e not in edge_set and e not in e_remove and e not in e_add:
            e_add.add(e)
        attempts += 1

    # 3、执行删除和添加
    G.delete_edges(e_remove)  # 删除选定的 `n` 条边
    G.add_edges(list(e_add))  # 添加 `n` 条新边

    #############################操作点##################################
    # 删点 关联的边删掉
    # 删除节点的比例
    nodes_to_remove_count = int(n_nodes * ratio)  # 按比例确定删除的节点数
    nodes_to_remove = np.random.choice(G.vs.indices, size=nodes_to_remove_count, replace=False)  # 随机选取节点
    nodes_to_remove_set = set(nodes_to_remove)
    G.delete_vertices(nodes_to_remove_set)
    # 加点
    if not positive:
        # 获取 G_add 中的节点数与 G 中的原节点数
        n_nodes_add = G_add.vcount()
        n_nodes_origin = G.vcount()
        nodes_to_add_count = min(n_nodes_add, nodes_to_remove_count)  # 要添加的节点数量与删除的节点数量相同
        # 遍历添加节点到 G 中
        new_nodes = []
        for node_add in range(nodes_to_add_count):
            G.add_vertex()  # 添加一个新节点
            new_node_index = len(G.vs) - 1  # 获取新节点的索引
            # 复制 G_add 中对应节点的属性到新节点
            G.vs[new_node_index]['name'] = G_add.vs[node_add]['name']
            G.vs[new_node_index]['type'] = G_add.vs[node_add]['type']
            G.vs[new_node_index]['properties'] = G_add.vs[node_add]['properties']
            new_nodes.append(new_node_index)  # 保存新节点的索引
        # 将新添加的节点与现有节点连接
        new_edges = []
        for node in new_nodes:
            # 随机选择一个已存在的节点来连接
            existing_node = np.random.choice(n_nodes_origin)  # 从原有节点中随机选择一个节点
            new_edges.append((existing_node, node))
        G.add_edges(new_edges)

    return G  # 返回修改后的图


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据集
    data_handler = get_handler("atlas", True)

    # 加载数据
    data_handler.load()
    # 成整个大图+捕捉特征语料+简化策略这里添加
    features, edges, mapp, relations, G = data_handler.build_graph()
    # 大图分割
    communities = detect_communities(G)

    # 嵌入构造特征向量
    embedder_class = get_embedder_by_name("word2vec")
    embedder = embedder_class(G, features, mapp)
    embedder.train()
    node_embeddings = embedder.embed_nodes()
    edge_embeddings = embedder.embed_edges()

    # 读取 all_nodes_data.csv 获取路径到UUID的映射
    path_to_uuid, path_to_type = load_node_data()

    # 创建 MultiDiGraph
    Graph = nx.MultiDiGraph()

    # 图的构建
    for i in range(len(communities)):
        positive = (i % 2 == 0)
        g, changed_g = _get_pair(positive, communities, i, G)

        # 将社区子图中的节点和边添加到 MultiDiGraph 中
        for node in g.vs:
            path = node['name']
            if path in path_to_uuid:
                uuid = path_to_uuid[path]  # 使用path获取对应的UUID
                node_feature = node_embeddings.get(path, np.zeros(54))  # 获取节点特征，缺失时使用默认值
                Graph.add_node(uuid, node_feature=node_feature)

        for edge_idx, edge in enumerate(g.es, start=1):
            node1 = g.vs[edge.source]['name']
            node2 = g.vs[edge.target]['name']
            edge_type = edge_idx  # 默认值为 "unknown"

            if node1 in path_to_uuid and node2 in path_to_uuid:
                node1_uuid = path_to_uuid[node1]
                node2_uuid = path_to_uuid[node2]
                if node1 and node2:
                    Graph.add_edge(node1_uuid, node2_uuid, type_edge=edge_type)

        # 调试打印：查看每次构建的 MultiDiGraph 中的节点和边
    #     print(f"Current MultiDiGraph after iteration {i}:")
    #     print(f"Nodes: {list(Graph.nodes)}")  # 打印当前图中的所有节点
    #     print(f"Edges: {list(Graph.edges)}")  # 打印当前图中的所有边
    #     print(f"Edge attributes: {list(Graph.edges(data=True))}")  # 打印边及其属性（包括type_edge）
    #
    # print("Graph construction complete")

G = Graph
EGO_K = 3  # 设置为3跳，生成k_3test.pt

# 生成每个节点的 3 跳 ego-graph
pt_dict = {}
for center in G.nodes:
    # 获取 3 跳邻域的节点
    ego_nodes = nx.single_source_shortest_path_length(G, center, cutoff=EGO_K).keys()
    # 从 G 中提取该邻域的子图
    subg = G.subgraph(ego_nodes).copy()

    # 保留节点的所有属性
    for n in subg.nodes:
        subg.nodes[n].update(G.nodes[n])  # 保留原有的节点属性（如果有）

    # 保存该节点的子图
    pt_dict[center] = subg

# # 输出到 k_3test.pt 文件
# with open('k_3test.pt', 'wb') as f:
#     pickle.dump(pt_dict, f, protocol=4)
#
# print(f"✅ 已输出 k_3test.pt，包含 {len(pt_dict)} 个 3跳 ego-graphs")
# 生成训练集和测试集的函数
def split_dataset(dataset, train_ratio=0.8):
    """
    将整个数据集按指定比例划分为训练集和测试集。
    """
    # 随机打乱数据集
    all_nodes = list(dataset.keys())
    random.shuffle(all_nodes)

    # 根据给定的比例划分
    split_idx = int(len(all_nodes) * train_ratio)
    train_nodes = all_nodes[:split_idx]
    test_nodes = all_nodes[split_idx:]

    # 创建训练集和测试集
    train_set = {node: dataset[node] for node in train_nodes}
    test_set = {node: dataset[node] for node in test_nodes}

    return train_set, test_set

# 加载原始数据
with open('k_3test.pt', 'rb') as f:
    dataset = pickle.load(f)

# 按比例划分数据集
train_set, test_set = split_dataset(dataset)

# 保存训练集和测试集
with open('k_3train.pt', 'wb') as f:
    pickle.dump(train_set, f, protocol=4)

with open('k_3test.pt', 'wb') as f:
    pickle.dump(test_set, f, protocol=4)

print(f"训练集大小: {len(train_set)}，测试集大小: {len(test_set)}")