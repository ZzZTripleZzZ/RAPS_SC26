import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === 配置 ===
BASE_DATA_DIR = "../data"
BASE_PLOT_DIR = "../plots"
RANKS_PER_NODE = 8 

def process_app_folder(app_name):
    print(f"--- Processing {app_name} ---")
    
    # 路径构造
    input_csv = os.path.join(BASE_DATA_DIR, app_name, "traffic_matrix.csv")
    output_dir = os.path.join(BASE_PLOT_DIR, app_name)
    
    if not os.path.exists(input_csv):
        print(f"Skipping {app_name}: No traffic_matrix.csv found.")
        return

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    max_rank = max(df["Source"].max(), df["Target"].max()) + 1
    
    # 2. 节点映射与聚合
    df["Source_Node"] = df["Source"] // RANKS_PER_NODE
    df["Target_Node"] = df["Target"] // RANKS_PER_NODE
    
    # 过滤节点内流量
    network_traffic = df[df["Source_Node"] != df["Target_Node"]].copy()
    
    # 聚合
    node_df = network_traffic.groupby(["Source_Node", "Target_Node"])["Bytes"].sum().reset_index()
    
    # 构建矩阵
    num_nodes = (max_rank + RANKS_PER_NODE - 1) // RANKS_PER_NODE
    matrix = node_df.pivot(index="Source_Node", columns="Target_Node", values="Bytes").fillna(0)
    matrix = matrix.reindex(index=range(num_nodes), columns=range(num_nodes), fill_value=0)

    # 3. 画图 (热力图)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='g', cmap="Reds", 
                cbar_kws={'label': 'Bytes Transmitted'})
    
    plt.title(f"{app_name.upper()} Node-to-Node Traffic\n(Aggregated from {max_rank} Ranks)")
    plt.xlabel("Destination Node ID")
    plt.ylabel("Source Node ID")
    
    output_path = os.path.join(output_dir, "node_pattern.png")
    plt.savefig(output_path)
    plt.close() # 关闭画布释放内存
    
    print(f"Success! Saved plot to {output_path}")

def main():
    # 扫描 data 目录下的所有子文件夹
    if not os.path.exists(BASE_DATA_DIR):
        print("Data directory not found!")
        return

    subdirs = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    
    if not subdirs:
        print("No app data found in ../data/")
        return

    for app in subdirs:
        process_app_folder(app)

if __name__ == "__main__":
    main()