# Provg-Searcher-Script

Get_node_feature用来生成node_feature下的四个文件

Get_data用来生成data文件夹下的文件



Get_node_feature

​	1.get_output.py：生成get_node_feature.py所需要的两个文件

​	2.get_node_feature.py：实现csv文件和pc文件的生成



Get_data：

​	get_pt.py：（需要在当前文件夹先放入已经生成的all_nodes_data.csv），生成k_3test.pt和k_3train.pt

​	get_graph_seed_pkl.py：运行生成3hash2graph.pkl和3hash2graph.pkl

​	get_pos_hash.py：运行生成3posQueryHashes.pkl和3posQueryHashStats.pkl
