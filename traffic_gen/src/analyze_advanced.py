import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 默认配置 ===
DATA_ROOT = "../data"
PLOTS_ROOT = "../plots_advanced"
RANKS_PER_NODE = 8 # 假设每个物理节点有 8 个 Rank (1 GCD per Rank)

# ==========================================
# 1. 拓扑映射模块 (Topology Mapper)
# ==========================================
class TopologyMapper:
    def __init__(self, mode="linear"):
        self.mode = mode

    def get_coords(self, rank_id):
        """
        输入: MPI Rank ID
        输出: (Group_ID, Local_ID) - Group_ID 用于画宏观拓扑图
        """
        node_id = rank_id // RANKS_PER_NODE
        
        if self.mode == "dragonfly":
            # --- 模拟 Frontier 的 Dragonfly 拓扑 (简化版) ---
            # 假设: 64 Ranks -> 8 Nodes.
            # 逻辑: 每 2 个 Node 组成一个 Chassis/Group (共 4 个 Group)
            # 这是一个全互连的小型 Dragonfly
            group_id = node_id // 2 
            return group_id, node_id % 2

        elif self.mode == "fattree":
            # --- 模拟 Fat-Tree (2-Level) ---
            # 假设: 64 Ranks -> 8 Nodes.
            # 逻辑: 每 4 个 Node 接入一个 Edge Switch (Pod) (共 2 个 Pod)
            pod_id = node_id // 4
            return pod_id, node_id % 4
            
        else: # "linear" or "mesh"
            # 默认: 直接把每个 Node 当作一个 Group
            return node_id, 0

    def get_label_name(self):
        if self.mode == "dragonfly": return "Dragonfly Group"
        if self.mode == "fattree": return "Fat-Tree Pod"
        return "Node ID"

# ==========================================
# 2. Incast 检测模块 (Incast Detector)
# ==========================================
def detect_incast(df, time_bin=0.01, threshold=4):
    """
    Incast 定义: 在极短时间窗口内，有超过 threshold 个不同的 Source 向同一个 Target 发送数据。
    """
    print(f"   🕵️  Detecting Incast (Window={time_bin}s, Threshold={threshold} sources)...")
    
    # 1. 对时间分箱
    df['Time_Bin'] = (df['Time'] // time_bin) * time_bin
    
    # 2. 统计每个时间窗内，每个 Target 有多少个唯一的 Source
    incast_stats = df.groupby(['Time_Bin', 'Target'])['Source'].nunique().reset_index()
    incast_stats.rename(columns={'Source': 'FanIn_Degree'}, inplace=True)
    
    # 3. 筛选出超过阈值的事件
    incast_events = incast_stats[incast_stats['FanIn_Degree'] >= threshold].copy()
    
    return incast_events

# ==========================================
# 3. 主分析逻辑
# ==========================================
def analyze_folder(exp_name, topology_mode):
    data_dir = os.path.join(DATA_ROOT, exp_name)
    csv_path = os.path.join(data_dir, "traffic_timeseries.csv")
    
    if not os.path.exists(csv_path):
        return

    print(f"\n>>> Analyzing {exp_name} with Topology: [{topology_mode.upper()}] ...")
    
    # 创建输出目录
    plot_dir = os.path.join(PLOTS_ROOT, f"{exp_name}_{topology_mode}")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # --- A. 拓扑流量图 (Topology Heatmap) ---
    mapper = TopologyMapper(topology_mode)
    
    # 将 Rank 映射为拓扑坐标
    df['Source_Group'] = df['Source'].apply(lambda x: mapper.get_coords(x)[0])
    df['Target_Group'] = df['Target'].apply(lambda x: mapper.get_coords(x)[0])
    
    # 过滤掉组内流量 (Local Traffic 通常不经过全局光纤)
    global_traffic = df[df['Source_Group'] != df['Target_Group']]
    
    if not global_traffic.empty:
        matrix = global_traffic.groupby(['Source_Group', 'Target_Group'])['Bytes'].sum().unstack(fill_value=0)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt='.2g')
        label = mapper.get_label_name()
        plt.title(f"Global Traffic Pattern ({topology_mode.upper()})\nApp: {exp_name}")
        plt.xlabel(f"Dest {label}")
        plt.ylabel(f"Source {label}")
        plt.savefig(os.path.join(plot_dir, "topology_traffic.png"))
        plt.close()
    else:
        print("   ⚠️  No global traffic detected between groups.")

    # --- B. Incast 风险图 (Incast Timeline) ---
    # 默认开启
    incast_events = detect_incast(df, time_bin=0.05, threshold=4) # 阈值可调
    
    if not incast_events.empty:
        plt.figure(figsize=(10, 5))
        # 画散点图：X轴是时间，Y轴是受害节点(Target)，颜色深浅代表Fan-In程度
        sns.scatterplot(data=incast_events, x='Time_Bin', y='Target', 
                        hue='FanIn_Degree', size='FanIn_Degree', 
                        palette='viridis', sizes=(20, 200))
        
        plt.title(f"Incast Events Detected (Many-to-One > 4)\nApp: {exp_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Victim Rank ID")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(plot_dir, "incast_risk.png"))
        plt.close()
        print(f"   ⚠️  Found {len(incast_events)} incast events! See incast_risk.png")
    else:
        print("   ✅ No significant Incast events detected.")

    print(f"   -> Results saved to {plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="linear", 
                        choices=["linear", "dragonfly", "fattree"],
                        help="Choose network topology for aggregation")
    args = parser.parse_args()

    if not os.path.exists(DATA_ROOT):
        print("No data directory found.")
        exit()

    subdirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    for exp_name in subdirs:
        analyze_folder(exp_name, args.topology)