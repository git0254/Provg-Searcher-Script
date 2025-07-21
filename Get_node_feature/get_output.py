import torch
from datahandlers import get_handler
from embedders import get_embedder_by_name
from partition import detect_communities
import numpy as np
from typing import List
import json
from pathlib import Path

#得到图G的两个communities_list
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
#
def substitute_random_edges_ig(G, G_add, positive, ratio=0.1):
    """在 igraph.Graph (有向图) 中随机替换 `n` 条边"""
    G = G.copy()  # 复制图，避免修改原始图
    n_nodes = G.vcount()  # 获取节点数
    edges = G.get_edgelist()  # 获取所有边的列表

    ############################操作边#################################################
    # 1、随机选择 `n` 条边进行删除
    total_edges = len(edges)
    n_changes_edges= int(total_edges * ratio)
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
data_handler = get_handler("atlas",True)
# data_handler = get_handler("theia", True)
# data_handler = get_handler("cadets", True)
# data_handler = get_handler("clearScope", True)
# data_handler = get_handler("trace", True)

# 加载数据
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G  = data_handler.build_graph()
# 将图 G 中所有节点的 name 和 type 属性分别保存到文件中

# 保存 name 属性到 names_output.txt
with open('names_output.txt', 'w') as f_names:
    for node in G.vs:  # 遍历图 G 的所有节点
        node_name = node['name']  # 获取 name 属性
        f_names.write(f"{node_name}\n")  # 每行写入一个 name

# 保存 type 属性到 types_output.txt
with open('types_output.txt', 'w') as f_types:
    for node in G.vs:  # 遍历图 G 的所有节点
        node_type = node['type']  # 获取 type 属性
        f_types.write(f"{node_type}\n")  # 每行写入一个 type

print("✅ 节点的 name 属性已成功保存到 names_output.txt 文件中。")
print("✅ 节点的 type 属性已成功保存到 types_output.txt 文件中。")
