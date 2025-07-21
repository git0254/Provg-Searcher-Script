import os
import pickle
import uuid
import pandas as pd
import numpy as np

# 导入type_enum.py中的类型映射
from type_enum import ObjectType

# DATA_ROOT = "./data"  # 假设数据根目录
OUT_DIR = "./output"  # 输出目录
FEATURE_DIM = 54  # 假设特征维度为37
NAMESPACE = uuid.UUID('12345678-1234-5678-1234-567812345678')


# 定义从文件中获取路径和类型的方法
def get_paths_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def get_types_from_file(file_path):
    with open(file_path, 'r') as file:
        return [int(line.strip()) for line in file.readlines()]


# 从type_enum.py查找对应的类型字符串
def get_type_from_enum(type_index):
    try:
        return ObjectType(type_index).name
    except ValueError:
        return "UNKNOWN"


# 从路径生成uuid
def key2uuid(key):
    return str(uuid.uuid5(NAMESPACE, key))


# 确保输出目录存在
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_node_data(names_file, types_file):
    paths = get_paths_from_file(names_file)
    type_indices = get_types_from_file(types_file)

    all_rows = []
    for path, type_index in zip(paths, type_indices):
        node_uuid = key2uuid(path)
        node_type = get_type_from_enum(type_index)
        node_data = {
            "uuid": node_uuid,
            "path": path,
            "type": node_type,
            "type_index": type_index
        }
        all_rows.append(node_data)

    return pd.DataFrame(all_rows)


def save_to_csv_and_pickle(df):
    ensure_dir(OUT_DIR)

    # 保存csv文件
    df.to_csv(os.path.join(OUT_DIR, "all_nodes_data.csv"), index=False)

    # 创建type_index到type的映射
    abstract_indexer = {t: idx for idx, t in enumerate(df["type"].unique())}
    with open(os.path.join(OUT_DIR, "abstarct_indexer.pc"), "wb") as f:
        pickle.dump(abstract_indexer, f, protocol=4)

    # 创建type字符串到特征的映射
    type2array = {t: np.zeros(FEATURE_DIM) for t in abstract_indexer}
    for idx, type_str in enumerate(abstract_indexer):
        type2array[type_str][idx % FEATURE_DIM] = 1  # 示例特征编码方式
    with open(os.path.join(OUT_DIR, "type2array.pc"), "wb") as f:
        pickle.dump(type2array, f, protocol=4)

    print("✅ CSV and PC files saved to", os.path.abspath(OUT_DIR))


def main():
    names_file = 'names_output.txt'  # 路径文件
    types_file = 'type_output.txt'  # 类型文件
    df = generate_node_data(names_file, types_file)
    save_to_csv_and_pickle(df)


if __name__ == "__main__":
    main()
