import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === 配置 ===
BASE_DATA_DIR = "../data"
PLOT_DIR = "../plots"
RANKS_PER_NODE = 8  # Frontier 架构设定

def get_node_matrix(app_name):
    csv_path = os.path.join(BASE_DATA_DIR, app_name, "traffic_matrix.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: No data found for {app_name}")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # 计算最大 Rank ID 以确定矩阵大小
    max_rank = max(df["Source"].max(), df["Target"].max()) + 1
    
    # === 核心逻辑：Rank -> Node 映射 ===
    df["Source_Node"] = df["Source"] // RANKS_PER_NODE
    df["Target_Node"] = df["Target"] // RANKS_PER_NODE
    
    # 过滤掉节点内部流量 (Intra-node)
    # 我们只关心走光纤的流量
    network_traffic = df[df["Source_Node"] != df["Target_Node"]].copy()
    
    # 聚合流量
    node_df = network_traffic.groupby(["Source_Node", "Target_Node"])["Bytes"].sum().reset_index()
    
    # 转换为矩阵
    num_nodes = (max_rank + RANKS_PER_NODE - 1) // RANKS_PER_NODE
    matrix = node_df.pivot(index="Source_Node", columns="Target_Node", values="Bytes").fillna(0)
    
    # 补全矩阵确保是方阵 (8x8)
    matrix = matrix.reindex(index=range(num_nodes), columns=range(num_nodes), fill_value=0)
    
    return matrix

def plot_comparison():
    # 1. 扫描数据目录
    if not os.path.exists(BASE_DATA_DIR):
        print("Error: Data directory not found.")
        return

    apps = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    apps.sort() # 排序保证顺序一致
    
    valid_apps = []
    matrices = []
    
    for app in apps:
        mat = get_node_matrix(app)
        if mat is not None and not mat.empty: # 确保矩阵非空
            valid_apps.append(app)
            matrices.append(mat)
    
    if not valid_apps:
        print("No valid traffic data found to compare.")
        return

    # 2. 创建画布
    num_apps = len(valid_apps)
    fig, axes = plt.subplots(1, num_apps, figsize=(6 * num_apps, 5), constrained_layout=True)
    
    # 如果只有一个应用，axes 不是列表，需要转换
    if num_apps == 1:
        axes = [axes]

    # 3. 循环画图
    for i, app in enumerate(valid_apps):
        ax = axes[i]
        matrix = matrices[i]
        
        # 使用对数颜色映射 (Log Scale) 可能会看得更清楚，
        # 但这里为了直观展示绝对量级，我们用线性映射，但在标题注明总量
        total_bytes = matrix.sum().sum()
        max_val = matrix.max().max()
        
        # 画热力图
        sns.heatmap(matrix, ax=ax, cmap="Reds", annot=True, fmt='.2g',
                    cbar_kws={'label': 'Bytes' if i == num_apps-1 else ''})
        
        ax.set_title(f"{app.upper()}\nTotal Traffic: {total_bytes/1e9:.2f} GB")
        ax.set_xlabel("Dest Node")
        if i == 0:
            ax.set_ylabel("Source Node")
        else:
            ax.set_ylabel("") # 中间的图不显示Y轴标签，省空间

    # 4. 保存
    os.makedirs(PLOT_DIR, exist_ok=True)
    output_path = os.path.join(PLOT_DIR, "comparison_pattern.png")
    plt.savefig(output_path, dpi=150)
    print(f"Success! Comparison plot saved to {output_path}")

if __name__ == "__main__":
    plot_comparison()