#!/usr/bin/env python3
"""
SC26 实验结果可视化
生成四个用例的图表：
1. 自适应路由算法比较
2. 节点放置策略效果
3. 作业调度算法性能
4. 功耗分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 加载数据
RESULTS_DIR = Path("/app/data/results")
OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "sc26_experiments_detailed.csv")

# 创建综合图表
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# =============================================
# 图1: 自适应路由算法比较 (Use Case 1)
# =============================================
ax1 = axes[0, 0]
routing_df = df[df['use_case'] == 'adaptive_routing'].copy()

# 按系统和算法聚合
routing_summary = routing_df.groupby(['system', 'algorithm']).agg({
    'latency': 'mean',
    'throughput': 'mean',
    'max_link_util': 'mean'
}).reset_index()

# 准备数据
systems = ['lassen', 'frontier']
colors = {'minimal': '#2E86AB', 'ecmp': '#A23B72', 'ugal': '#F18F01', 'valiant': '#C73E1D'}

x = np.arange(len(systems))
width = 0.2

# 延迟比较
for i, algo in enumerate(['minimal', 'ugal', 'valiant', 'ecmp']):
    algo_data = routing_summary[routing_summary['algorithm'] == algo]
    if not algo_data.empty:
        latencies = []
        for sys in systems:
            sys_data = algo_data[algo_data['system'] == sys]
            latencies.append(sys_data['latency'].values[0] if len(sys_data) > 0 else 0)
        ax1.bar(x + i*width - 1.5*width, latencies, width, label=algo.upper(),
                color=colors.get(algo, '#666666'), alpha=0.85)

ax1.set_xlabel('System', fontsize=11)
ax1.set_ylabel('Average Latency (hops)', fontsize=11)
ax1.set_title('Use Case 1: Adaptive Routing - Latency Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['Lassen\n(Fat-Tree k=32)', 'Frontier\n(Dragonfly)'])
ax1.legend(title='Algorithm', loc='upper right')
ax1.set_ylim(0, max(routing_summary['latency'].max() * 1.2, 6))

# =============================================
# 图2: 节点放置策略 (Use Case 2)
# =============================================
ax2 = axes[0, 1]
placement_df = df[df['use_case'] == 'node_placement'].copy()

# 按策略和系统聚合
placement_summary = placement_df.groupby(['system', 'strategy']).agg({
    'communication_cost': 'mean'
}).reset_index()

strategies = ['contiguous', 'random', 'locality', 'spectral']
strategy_labels = ['Contiguous\n(Baseline)', 'Random', 'Locality-\nAware', 'Spectral\nClustering']
colors_placement = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']

bar_width = 0.35
x = np.arange(len(strategies))

for i, sys in enumerate(systems):
    sys_data = placement_summary[placement_summary['system'] == sys]
    costs = []
    for strategy in strategies:
        strat_data = sys_data[sys_data['strategy'] == strategy]
        cost = strat_data['communication_cost'].values[0] / 1e9 if len(strat_data) > 0 else 0
        costs.append(cost)
    ax2.bar(x + (i-0.5)*bar_width, costs, bar_width,
            label=f'{sys.capitalize()}', alpha=0.85)

ax2.set_xlabel('Placement Strategy', fontsize=11)
ax2.set_ylabel('Communication Cost (GB)', fontsize=11)
ax2.set_title('Use Case 2: Node Placement - Communication Cost', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(strategy_labels, fontsize=9)
ax2.legend(title='System')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# =============================================
# 图3: 作业调度性能 (Use Case 3)
# =============================================
ax3 = axes[1, 0]
sched_df = df[df['use_case'] == 'scheduling'].copy()

# 过滤掉异常值
sched_df = sched_df[sched_df['makespan'] > 10]

sched_summary = sched_df.groupby(['system', 'scheduler']).agg({
    'makespan': 'mean',
    'energy_kwh': 'mean',
    'utilization': 'mean'
}).reset_index()

schedulers = ['fcfs', 'backfill', 'multitenant']
scheduler_labels = ['FCFS', 'Backfill', 'Multi-tenant']
colors_sched = ['#E8D5B7', '#B8D4E3', '#D4B8E8']

bar_width = 0.25
x = np.arange(len(schedulers))

# Makespan
for i, sys in enumerate(systems):
    sys_data = sched_summary[sched_summary['system'] == sys]
    makespans = []
    for scheduler in schedulers:
        sched_data = sys_data[sys_data['scheduler'] == scheduler]
        makespan = sched_data['makespan'].values[0] if len(sched_data) > 0 else 0
        makespans.append(makespan)
    ax3.bar(x + (i-0.5)*bar_width, makespans, bar_width,
            label=f'{sys.capitalize()}', alpha=0.85)

ax3.set_xlabel('Scheduler', fontsize=11)
ax3.set_ylabel('Makespan (seconds)', fontsize=11)
ax3.set_title('Use Case 3: Job Scheduling - Makespan Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(scheduler_labels)
ax3.legend(title='System')

# =============================================
# 图4: 功耗分析 (Use Case 4)
# =============================================
ax4 = axes[1, 1]
power_df = df[df['use_case'] == 'power'].copy()

power_summary = power_df.groupby(['system', 'scenario']).agg({
    'total_power_mw': 'mean',
    'compute_power_mw': 'mean'
}).reset_index()

scenarios = ['baseline', 'power_cap', 'frequency_scaling', 'job_packing']
scenario_labels = ['Baseline', 'Power\nCapping', 'Frequency\nScaling', 'Job\nPacking']
colors_power = ['#FF9F1C', '#2EC4B6', '#E71D36', '#011627']

bar_width = 0.35
x = np.arange(len(scenarios))

for i, sys in enumerate(systems):
    sys_data = power_summary[power_summary['system'] == sys]
    powers = []
    for scenario in scenarios:
        scen_data = sys_data[sys_data['scenario'] == scenario]
        power = scen_data['total_power_mw'].values[0] if len(scen_data) > 0 else 0
        powers.append(power)
    ax4.bar(x + (i-0.5)*bar_width, powers, bar_width,
            label=f'{sys.capitalize()}', alpha=0.85)

ax4.set_xlabel('Power Management Scenario', fontsize=11)
ax4.set_ylabel('Total Power (MW)', fontsize=11)
ax4.set_title('Use Case 4: Power Consumption Analysis', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(scenario_labels, fontsize=9)
ax4.legend(title='System')

# 调整布局
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sc26_experiments_overview.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Overview chart saved to: {OUTPUT_DIR / 'sc26_experiments_overview.png'}")

# =============================================
# 额外图表: 按应用程序的详细分析
# =============================================

# 图5: 路由算法吞吐量对比
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# 吞吐量比较
ax = axes2[0]
throughput_summary = routing_df.groupby(['system', 'algorithm'])['throughput'].mean().reset_index()
x_sys = np.arange(len(systems))

for i, algo in enumerate(['minimal', 'ugal', 'valiant', 'ecmp']):
    algo_data = throughput_summary[throughput_summary['algorithm'] == algo]
    if not algo_data.empty:
        throughputs = []
        for sys in systems:
            sys_data = algo_data[algo_data['system'] == sys]
            throughputs.append(sys_data['throughput'].values[0] if len(sys_data) > 0 else 0)
        ax.bar(x_sys + i*width - 1.5*width, throughputs, width, label=algo.upper(),
               color=colors.get(algo, '#666666'), alpha=0.85)

ax.set_xlabel('System', fontsize=11)
ax.set_ylabel('Throughput (GB/s)', fontsize=11)
ax.set_title('Routing Algorithm - Throughput Comparison', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Lassen (Fat-Tree)', 'Frontier (Dragonfly)'])
ax.legend(title='Algorithm')

# 链路利用率比较
ax = axes2[1]
util_summary = routing_df.groupby(['system', 'algorithm'])['max_link_util'].mean().reset_index()

for i, algo in enumerate(['minimal', 'ugal', 'valiant', 'ecmp']):
    algo_data = util_summary[util_summary['algorithm'] == algo]
    if not algo_data.empty:
        utils = []
        for sys in systems:
            sys_data = algo_data[algo_data['system'] == sys]
            utils.append(sys_data['max_link_util'].values[0] if len(sys_data) > 0 else 0)
        ax.bar(x_sys + i*width - 1.5*width, utils, width, label=algo.upper(),
               color=colors.get(algo, '#666666'), alpha=0.85)

ax.set_xlabel('System', fontsize=11)
ax.set_ylabel('Max Link Utilization', fontsize=11)
ax.set_title('Routing Algorithm - Link Utilization', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Lassen (Fat-Tree)', 'Frontier (Dragonfly)'])
ax.legend(title='Algorithm')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sc26_routing_detailed.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Routing detail chart saved to: {OUTPUT_DIR / 'sc26_routing_detailed.png'}")

# =============================================
# 图6: 按Mini-App分类的通信模式
# =============================================
fig3, ax = plt.subplots(figsize=(12, 6))

# 提取应用名称
routing_df['app'] = routing_df['matrix'].str.extract(r'^([a-z]+)_')
routing_df = routing_df.dropna(subset=['app'])
app_summary = routing_df.groupby(['app', 'inferred_pattern']).size().reset_index(name='count')

apps = [a for a in routing_df['app'].unique() if isinstance(a, str)]
patterns = routing_df['inferred_pattern'].dropna().unique()

pattern_colors = {'stencil-3d': '#3498DB', 'sparse-random': '#E74C3C',
                  'all-to-all': '#2ECC71', 'nearest-neighbor': '#9B59B6',
                  'unknown': '#95A5A6'}

x_pos = np.arange(len(apps))
bottom = np.zeros(len(apps))

for pattern in patterns:
    counts = []
    for app in apps:
        cnt = app_summary[(app_summary['app'] == app) & (app_summary['inferred_pattern'] == pattern)]
        counts.append(cnt['count'].values[0] if len(cnt) > 0 else 0)
    ax.bar(x_pos, counts, bottom=bottom, label=pattern.replace('-', ' ').title(),
           color=pattern_colors.get(pattern, '#666666'), alpha=0.85)
    bottom += counts

ax.set_xlabel('Mini-Application', fontsize=11)
ax.set_ylabel('Number of Experiments', fontsize=11)
ax.set_title('Communication Patterns by Mini-Application', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([a.upper() for a in apps], fontsize=10)
ax.legend(title='Pattern', loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sc26_patterns.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Pattern chart saved to: {OUTPUT_DIR / 'sc26_patterns.png'}")

# =============================================
# 图7: 系统配置比较
# =============================================
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

# 系统规格
system_specs = {
    'Lassen': {'nodes': 4608, 'topology': 'Fat-Tree (k=32)', 'bw': 12.5, 'power': 4.7},
    'Frontier': {'nodes': 9472, 'topology': 'Dragonfly', 'bw': 25.0, 'power': 15.1}
}

# 节点数量
ax = axes4[0]
ax.bar(['Lassen', 'Frontier'], [4608, 9472], color=['#3498DB', '#E74C3C'], alpha=0.85)
ax.set_ylabel('Number of Nodes', fontsize=11)
ax.set_title('System Scale', fontsize=12, fontweight='bold')
for i, v in enumerate([4608, 9472]):
    ax.text(i, v + 200, str(v), ha='center', fontsize=11)

# 网络带宽
ax = axes4[1]
ax.bar(['Lassen\n(InfiniBand EDR)', 'Frontier\n(Slingshot)'], [12.5, 25.0],
       color=['#3498DB', '#E74C3C'], alpha=0.85)
ax.set_ylabel('Network Bandwidth (GB/s)', fontsize=11)
ax.set_title('Network Bandwidth', fontsize=12, fontweight='bold')
for i, v in enumerate([12.5, 25.0]):
    ax.text(i, v + 0.5, f'{v}', ha='center', fontsize=11)

# 功耗
ax = axes4[2]
power_summary_sys = power_df.groupby('system')['total_power_mw'].mean()
ax.bar(['Lassen', 'Frontier'], [power_summary_sys.get('lassen', 4.7),
       power_summary_sys.get('frontier', 15.1)], color=['#3498DB', '#E74C3C'], alpha=0.85)
ax.set_ylabel('Total Power (MW)', fontsize=11)
ax.set_title('Power Consumption', fontsize=12, fontweight='bold')
for i, v in enumerate([power_summary_sys.get('lassen', 4.7), power_summary_sys.get('frontier', 15.1)]):
    ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sc26_system_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"System comparison chart saved to: {OUTPUT_DIR / 'sc26_system_comparison.png'}")

# =============================================
# 打印实验总结
# =============================================
print("\n" + "="*60)
print("EXPERIMENT RESULTS SUMMARY")
print("="*60)

print("\n1. ADAPTIVE ROUTING:")
print("-" * 40)
routing_agg = routing_df.groupby(['system', 'algorithm']).agg({
    'latency': 'mean',
    'throughput': 'mean',
}).round(3)
print(routing_agg)

print("\n2. NODE PLACEMENT:")
print("-" * 40)
placement_agg = placement_df.groupby(['system', 'strategy']).agg({
    'communication_cost': lambda x: f"{x.mean()/1e9:.2f} GB"
})
print(placement_agg)

print("\n3. JOB SCHEDULING:")
print("-" * 40)
sched_valid = sched_df[sched_df['makespan'] > 10]
if len(sched_valid) > 0:
    sched_agg = sched_valid.groupby(['system', 'scheduler']).agg({
        'makespan': 'mean',
        'energy_kwh': 'mean'
    }).round(2)
    print(sched_agg)
else:
    print("Using mock scheduling results")

print("\n4. POWER ANALYSIS:")
print("-" * 40)
power_agg = power_df.groupby(['system', 'scenario']).agg({
    'total_power_mw': 'mean'
}).round(3)
print(power_agg)

print("\n" + "="*60)
print(f"Charts saved to: {OUTPUT_DIR}")
print("="*60)
