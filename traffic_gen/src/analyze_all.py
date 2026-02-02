import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_ROOT = "../data"
PLOTS_ROOT = "../plots"

def analyze_folder(exp_name):
    data_dir = os.path.join(DATA_ROOT, exp_name)
    csv_path = os.path.join(data_dir, "traffic_timeseries.csv")
    
    if not os.path.exists(csv_path):
        return # 跳过没有数据的文件夹

    print(f"Creating plots for: {exp_name} ...")
    
    # 创建输出目录
    plot_dir = os.path.join(PLOTS_ROOT, exp_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # --- 图表 1: 总带宽随时间变化 ---
    time_bin = 0.05
    df['Time_Bin'] = (df['Time'] // time_bin) * time_bin
    timeline = df.groupby('Time_Bin')['Bytes'].sum().reset_index()
    timeline['Bandwidth_GBs'] = (timeline['Bytes'] / 1e9) / time_bin
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=timeline, x='Time_Bin', y='Bandwidth_GBs', linewidth=2)
    plt.fill_between(timeline['Time_Bin'], timeline['Bandwidth_GBs'], alpha=0.3)
    plt.title(f"Bandwidth Over Time: {exp_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("GB/s")
    plt.savefig(os.path.join(plot_dir, "bandwidth.png"))
    plt.close()
    
    # --- 图表 2: 通信矩阵 (Node-level) ---
    RANKS_PER_NODE = 8
    df["Source_Node"] = df["Source"] // RANKS_PER_NODE
    df["Target_Node"] = df["Target"] // RANKS_PER_NODE
    # 过滤节点内通信
    net_traffic = df[df["Source_Node"] != df["Target_Node"]]
    
    if not net_traffic.empty:
        matrix = net_traffic.groupby(["Source_Node", "Target_Node"])["Bytes"].sum().unstack(fill_value=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, cmap="Reds", annot=False) # 批量图就不显示具体数字了，太乱
        plt.title(f"Node-to-Node Traffic: {exp_name}")
        plt.savefig(os.path.join(plot_dir, "heatmap.png"))
        plt.close()
    
    print(f"  -> Saved to {plot_dir}")

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print("No data directory found.")
        exit()

    subdirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    for exp_name in subdirs:
        analyze_folder(exp_name)
    
    print("✅ All plots generated.")