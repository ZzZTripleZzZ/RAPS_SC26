#!/usr/bin/env python3
"""
SC26 Visualization Module (v2)
==============================

Generates publication-quality figures for the four use cases with multiple metrics:
1. Adaptive Routing - Link utilization + Throughput comparison
2. Node Placement - Utilization + Path diversity metrics
3. RL-based Scheduling - Job slowdown + Congestion analysis
4. Power Consumption - Total power + Efficiency metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = Path("/app/output/sc26_experiments")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Color schemes
SYSTEM_COLORS = {
    'lassen': '#2E86AB',      # Blue
    'frontier': '#E94F37',    # Red
}

ROUTING_COLORS = {
    'minimal': '#4ECDC4',
    'ecmp': '#45B7D1',
    'adaptive': '#2E86AB',
    'ugal': '#F77F00',
    'valiant': '#FCBF49',
}

# Mini-app colors (actual applications)
MINIAPP_COLORS = {
    'lulesh': '#6B4C9A',    # Purple - Stencil-3D hydrodynamics
    'comd': '#2E7D32',      # Green - Molecular dynamics
    'hpgmg': '#E65100',     # Orange - Multigrid solver
    'cosp2': '#1565C0',     # Blue - Sparse matrix
}

ALLOCATION_COLORS = {
    'contiguous': '#1976D2',
    'random': '#D32F2F',
}


def load_experiment_data(use_case: str) -> pd.DataFrame:
    """Load experiment data for a specific use case."""
    filename_map = {
        "adaptive_routing": "uc1_adaptive_routing.csv",
        "node_placement": "uc2_node_placement.csv",
        "scheduling": "uc3_scheduling.csv",
        "power_consumption": "uc4_power_consumption.csv",
    }
    filepath = OUTPUT_DIR / filename_map[use_case]
    if filepath.exists():
        return pd.read_csv(filepath)
    return pd.DataFrame()


def plot_adaptive_routing(df: pd.DataFrame, save_path: Path):
    """
    Use Case 1: Adaptive Routing Comparison

    Metrics:
    - Max Link Utilization (scaled for visibility)
    - Total Traffic Throughput (GB)
    - Link Utilization Variance (load balance indicator)
    """
    if df.empty:
        print("No data for adaptive routing visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    apps = sorted(df['app_type'].unique())
    systems = df['system'].unique()

    # Convert traffic to GB for readability
    df['traffic_gb'] = df['total_traffic_bytes'] / 1e9

    # Plot 1: Max Link Utilization by Mini-App (auto-scaled)
    ax1 = axes[0, 0]

    grouped = df.groupby(['app_type', 'system'])['max_link_util'].mean().reset_index()

    bar_width = 0.35
    x_positions = np.arange(len(apps))

    for i, system in enumerate(systems):
        sys_data = grouped[grouped['system'] == system]
        values = [sys_data[sys_data['app_type'] == a]['max_link_util'].values[0] * 100  # Convert to percentage
                  if len(sys_data[sys_data['app_type'] == a]) > 0 else 0
                  for a in apps]
        color = SYSTEM_COLORS.get(system, f'C{i}')
        ax1.bar(x_positions + i * bar_width, values, bar_width,
                label=system.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Mini-App')
    ax1.set_ylabel('Max Link Utilization (%)')
    ax1.set_title('(a) Peak Link Utilization by Mini-App')
    ax1.set_xticks(x_positions + bar_width / 2)
    ax1.set_xticklabels([a.upper() for a in apps])
    ax1.legend(title='System')
    ax1.grid(axis='y', alpha=0.3)
    # Auto-scale y-axis with some padding
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.2)

    # Plot 2: Traffic Volume by Mini-App
    ax2 = axes[0, 1]

    traffic_grouped = df.groupby(['app_type', 'system'])['traffic_gb'].mean().reset_index()

    for i, system in enumerate(systems):
        sys_data = traffic_grouped[traffic_grouped['system'] == system]
        values = [sys_data[sys_data['app_type'] == a]['traffic_gb'].values[0]
                  if len(sys_data[sys_data['app_type'] == a]) > 0 else 0
                  for a in apps]
        color = SYSTEM_COLORS.get(system, f'C{i}')
        ax2.bar(x_positions + i * bar_width, values, bar_width,
                label=system.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Mini-App')
    ax2.set_ylabel('Total Traffic (GB)')
    ax2.set_title('(b) Communication Volume by Mini-App')
    ax2.set_xticks(x_positions + bar_width / 2)
    ax2.set_xticklabels([a.upper() for a in apps])
    ax2.legend(title='System')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Routing Algorithm Comparison (by system)
    ax3 = axes[1, 0]

    routing_grouped = df.groupby(['system', 'routing_algorithm'])['max_link_util'].mean().reset_index()
    routing_grouped['max_link_util_pct'] = routing_grouped['max_link_util'] * 100

    routings = df['routing_algorithm'].unique()
    bar_width = 0.15
    x_positions = np.arange(len(systems))

    for i, routing in enumerate(routings):
        routing_data = routing_grouped[routing_grouped['routing_algorithm'] == routing]
        values = [routing_data[routing_data['system'] == s]['max_link_util_pct'].values[0]
                  if len(routing_data[routing_data['system'] == s]) > 0 else 0
                  for s in systems]
        color = ROUTING_COLORS.get(routing, f'C{i}')
        ax3.bar(x_positions + i * bar_width, values, bar_width,
                label=routing.upper(), color=color, edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('System')
    ax3.set_ylabel('Max Link Utilization (%)')
    ax3.set_title('(c) Routing Algorithm Performance')
    ax3.set_xticks(x_positions + bar_width * (len(routings)-1) / 2)
    ax3.set_xticklabels([s.capitalize() for s in systems])
    ax3.legend(title='Routing', loc='upper right', ncol=2)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Communication Pattern Characteristics (Avg Degree + Sparsity)
    ax4 = axes[1, 1]

    pattern_data = df.groupby('app_type').agg({
        'avg_degree': 'mean',
        'sparsity': 'mean',
        'num_nodes': 'mean'
    }).reset_index()
    pattern_data = pattern_data.sort_values('app_type')

    x = np.arange(len(pattern_data))
    width = 0.35

    # Avg degree on left y-axis
    bars1 = ax4.bar(x - width/2, pattern_data['avg_degree'], width,
                    label='Avg Degree', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Average Degree', color='#2E86AB')
    ax4.tick_params(axis='y', labelcolor='#2E86AB')

    # Sparsity on right y-axis
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, (1 - pattern_data['sparsity']) * 100, width,
                         label='Density (%)', color='#E94F37', edgecolor='black', linewidth=0.5)
    ax4_twin.set_ylabel('Communication Density (%)', color='#E94F37')
    ax4_twin.tick_params(axis='y', labelcolor='#E94F37')

    ax4.set_xlabel('Mini-App')
    ax4.set_title('(d) Communication Pattern Characteristics')
    ax4.set_xticks(x)
    ax4.set_xticklabels([a.upper() for a in pattern_data['app_type']])

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('Use Case 1: Adaptive Routing with Real Mini-App Traces', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_node_placement(df: pd.DataFrame, save_path: Path):
    """
    Use Case 2: Node Placement Strategy Comparison

    Metrics:
    - Max Link Utilization
    - Utilization Variance (load balance)
    """
    if df.empty:
        print("No data for node placement visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    apps = sorted(df['app_type'].unique())
    allocations = df['allocation_strategy'].unique()
    systems = df['system'].unique()

    # Convert to percentage
    df['max_link_util_pct'] = df['max_link_util'] * 100
    df['link_util_std_scaled'] = df['link_util_std'] * 1e6  # Scale for visibility

    # Plot 1: Link utilization by allocation strategy and system
    ax1 = axes[0, 0]

    bar_width = 0.35
    x_positions = np.arange(len(systems))

    for i, alloc in enumerate(allocations):
        alloc_data = df[df['allocation_strategy'] == alloc].groupby('system')['max_link_util_pct'].mean()
        values = [alloc_data.get(s, 0) for s in systems]
        color = ALLOCATION_COLORS.get(alloc, f'C{i}')
        ax1.bar(x_positions + i * bar_width, values, bar_width,
                label=alloc.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('System')
    ax1.set_ylabel('Max Link Utilization (%)')
    ax1.set_title('(a) Allocation Strategy by System')
    ax1.set_xticks(x_positions + bar_width / 2)
    ax1.set_xticklabels([s.capitalize() for s in systems])
    ax1.legend(title='Allocation')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Allocation impact by mini-app
    ax2 = axes[0, 1]

    x_positions = np.arange(len(apps))
    for i, alloc in enumerate(allocations):
        alloc_data = df[df['allocation_strategy'] == alloc].groupby('app_type')['max_link_util_pct'].mean()
        values = [alloc_data.get(a, 0) for a in apps]
        color = ALLOCATION_COLORS.get(alloc, f'C{i}')
        ax2.bar(x_positions + i * bar_width, values, bar_width,
                label=alloc.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Mini-App')
    ax2.set_ylabel('Max Link Utilization (%)')
    ax2.set_title('(b) Allocation Impact by Mini-App')
    ax2.set_xticks(x_positions + bar_width / 2)
    ax2.set_xticklabels([a.upper() for a in apps])
    ax2.legend(title='Allocation')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Load Balance (Link Utilization Std Dev)
    ax3 = axes[1, 0]

    for i, alloc in enumerate(allocations):
        alloc_data = df[df['allocation_strategy'] == alloc].groupby('system')['link_util_std_scaled'].mean()
        values = [alloc_data.get(s, 0) for s in systems]
        color = ALLOCATION_COLORS.get(alloc, f'C{i}')
        ax3.bar(x_positions[:len(systems)] + i * bar_width, values, bar_width,
                label=alloc.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('System')
    ax3.set_ylabel('Link Util Std Dev (×10⁻⁶)')
    ax3.set_title('(c) Load Balance (lower = more balanced)')
    ax3.set_xticks(np.arange(len(systems)) + bar_width / 2)
    ax3.set_xticklabels([s.capitalize() for s in systems])
    ax3.legend(title='Allocation')
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Contiguous vs Random ratio by mini-app
    ax4 = axes[1, 1]

    # Calculate ratio: random / contiguous (>1 means random is worse)
    pivot = df.pivot_table(values='max_link_util',
                          index=['app_type', 'system'],
                          columns='allocation_strategy').reset_index()
    pivot['ratio'] = pivot['random'] / pivot['contiguous'].replace(0, np.nan)
    pivot = pivot.dropna()

    ratio_by_app = pivot.groupby('app_type')['ratio'].mean()

    colors = [MINIAPP_COLORS.get(a, 'gray') for a in apps if a in ratio_by_app.index]
    bars = ax4.bar(range(len(ratio_by_app)), ratio_by_app.values,
                   color=colors, edgecolor='black', linewidth=0.5)

    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal performance')
    ax4.set_xlabel('Mini-App')
    ax4.set_ylabel('Random / Contiguous Ratio')
    ax4.set_title('(d) Allocation Strategy Impact Ratio')
    ax4.set_xticks(range(len(ratio_by_app)))
    ax4.set_xticklabels([a.upper() for a in ratio_by_app.index])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Add annotation
    for i, (app, ratio) in enumerate(ratio_by_app.items()):
        label = "Random worse" if ratio > 1 else "Random better"
        ax4.annotate(f'{ratio:.2f}', (i, ratio), ha='center', va='bottom', fontsize=9)

    plt.suptitle('Use Case 2: Node Placement Strategy Evaluation', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_scheduling(df: pd.DataFrame, save_path: Path):
    """
    Use Case 3: Scheduling Comparison (RL vs Traditional)

    Metrics:
    - Job Slowdown Factor
    - Congestion Factor
    - Link Utilization by Configuration
    """
    if df.empty:
        print("No data for scheduling visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    apps = sorted(df['app_type'].unique())
    systems = df['system'].unique()
    allocations = df['allocation_strategy'].unique()

    df['max_link_util_pct'] = df['max_link_util'] * 100

    # Plot 1: Job slowdown by mini-app and configuration
    ax1 = axes[0, 0]

    # Create grouped data
    pivot = df.pivot_table(
        values='job_slowdown',
        index='app_type',
        columns=['system', 'allocation_strategy'],
        aggfunc='mean'
    )

    pivot.plot(kind='bar', ax=ax1, width=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Mini-App')
    ax1.set_ylabel('Job Slowdown Factor')
    ax1.set_title('(a) Job Slowdown by Configuration')
    ax1.legend(title='System / Allocation', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.set_xticklabels([t.get_text().upper() for t in ax1.get_xticklabels()], rotation=45, ha='right')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='No Slowdown')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.9, 1.1)  # Zoom in around 1.0

    # Plot 2: Max link utilization comparison
    ax2 = axes[0, 1]

    util_pivot = df.pivot_table(
        values='max_link_util_pct',
        index='app_type',
        columns=['system', 'allocation_strategy'],
        aggfunc='mean'
    )

    util_pivot.plot(kind='bar', ax=ax2, width=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Mini-App')
    ax2.set_ylabel('Max Link Utilization (%)')
    ax2.set_title('(b) Link Utilization by Configuration')
    ax2.legend(title='System / Allocation', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.set_xticklabels([t.get_text().upper() for t in ax2.get_xticklabels()], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Congestion factor distribution by system
    ax3 = axes[1, 0]

    data_to_plot = []
    labels = []
    colors = []
    for system in systems:
        for alloc in allocations:
            subset = df[(df['system'] == system) & (df['allocation_strategy'] == alloc)]
            data_to_plot.append(subset['congestion_factor'].values)
            labels.append(f"{system[:3].upper()}\n{alloc[:4].upper()}")
            colors.append(SYSTEM_COLORS.get(system, 'gray'))

    bp = ax3.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Congestion Factor')
    ax3.set_title('(c) Congestion Factor Distribution')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Scheduling efficiency by routing algorithm
    ax4 = axes[1, 1]

    routing_data = df.groupby(['routing_algorithm', 'system']).agg({
        'max_link_util_pct': 'mean',
        'job_slowdown': 'mean'
    }).reset_index()

    routings = df['routing_algorithm'].unique()
    bar_width = 0.35
    x = np.arange(len(routings))

    for i, system in enumerate(systems):
        sys_data = routing_data[routing_data['system'] == system]
        values = [sys_data[sys_data['routing_algorithm'] == r]['max_link_util_pct'].values[0]
                  if len(sys_data[sys_data['routing_algorithm'] == r]) > 0 else 0
                  for r in routings]
        color = SYSTEM_COLORS.get(system, f'C{i}')
        ax4.bar(x + i * bar_width, values, bar_width,
                label=system.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax4.set_xlabel('Routing Algorithm')
    ax4.set_ylabel('Avg Max Link Utilization (%)')
    ax4.set_title('(d) Routing Impact on Scheduling')
    ax4.set_xticks(x + bar_width / 2)
    ax4.set_xticklabels([r.upper() for r in routings])
    ax4.legend(title='System')
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('Use Case 3: Scheduling Strategy Evaluation', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_power_consumption(df: pd.DataFrame, save_path: Path):
    """
    Use Case 4: Power Consumption Analysis

    Metrics:
    - Total Power (kW)
    - Power per Node (kW)
    - Power Efficiency (Traffic per Watt)
    """
    if df.empty:
        print("No data for power consumption visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    apps = sorted(df['app_type'].unique())
    systems = df['system'].unique()

    # Calculate derived metrics
    df['power_per_node'] = df['total_power_kw'] / df['num_nodes']
    df['traffic_per_watt'] = df['total_traffic_bytes'] / (df['total_power_kw'] * 1000)  # bytes per watt
    df['traffic_gb'] = df['total_traffic_bytes'] / 1e9

    # Plot 1: Total power by system and workload
    ax1 = axes[0, 0]

    bar_width = 0.35
    x_positions = np.arange(len(apps))

    for i, system in enumerate(systems):
        sys_data = df[df['system'] == system].groupby('app_type')['total_power_kw'].mean()
        values = [sys_data.get(a, 0) for a in apps]
        color = SYSTEM_COLORS.get(system, f'C{i}')
        ax1.bar(x_positions + i * bar_width, values, bar_width,
                label=system.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Mini-App')
    ax1.set_ylabel('Total Power (kW)')
    ax1.set_title('(a) Total Power Consumption')
    ax1.set_xticks(x_positions + bar_width / 2)
    ax1.set_xticklabels([a.upper() for a in apps])
    ax1.legend(title='System')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Power per node
    ax2 = axes[0, 1]

    for i, system in enumerate(systems):
        sys_data = df[df['system'] == system].groupby('app_type')['power_per_node'].mean()
        values = [sys_data.get(a, 0) for a in apps]
        color = SYSTEM_COLORS.get(system, f'C{i}')
        ax2.bar(x_positions + i * bar_width, values, bar_width,
                label=system.capitalize(), color=color, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Mini-App')
    ax2.set_ylabel('Power per Node (kW)')
    ax2.set_title('(b) Power Efficiency per Node')
    ax2.set_xticks(x_positions + bar_width / 2)
    ax2.set_xticklabels([a.upper() for a in apps])
    ax2.legend(title='System')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Traffic volume vs Power (scatter)
    ax3 = axes[1, 0]

    for system in systems:
        sys_df = df[df['system'] == system]
        color = SYSTEM_COLORS.get(system, 'gray')
        ax3.scatter(sys_df['traffic_gb'], sys_df['total_power_kw'],
                   c=color, s=100, alpha=0.7, edgecolor='black', linewidth=0.5,
                   label=system.capitalize())

        # Add app labels
        for _, row in sys_df.iterrows():
            ax3.annotate(row['app_type'].upper(),
                        (row['traffic_gb'], row['total_power_kw']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax3.set_xlabel('Traffic Volume (GB)')
    ax3.set_ylabel('Total Power (kW)')
    ax3.set_title('(c) Power vs Communication Volume')
    ax3.legend(title='System')
    ax3.grid(alpha=0.3)

    # Plot 4: Network efficiency (traffic per watt)
    ax4 = axes[1, 1]

    efficiency = df.groupby(['app_type', 'system'])['traffic_per_watt'].mean().reset_index()
    efficiency_pivot = efficiency.pivot(index='app_type', columns='system', values='traffic_per_watt')

    efficiency_pivot.plot(kind='bar', ax=ax4, color=[SYSTEM_COLORS.get(s, 'gray') for s in efficiency_pivot.columns],
                         edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Mini-App')
    ax4.set_ylabel('Traffic per Watt (bytes/W)')
    ax4.set_title('(d) Network Efficiency')
    ax4.set_xticklabels([a.upper() for a in efficiency_pivot.index], rotation=45, ha='right')
    ax4.legend(title='System')
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('Use Case 4: Power Consumption Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_summary(save_path: Path):
    """
    Create a combined summary figure showing key metrics across all use cases.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Load all data
    df_routing = load_experiment_data("adaptive_routing")
    df_placement = load_experiment_data("node_placement")
    df_scheduling = load_experiment_data("scheduling")
    df_power = load_experiment_data("power_consumption")

    apps_order = ['comd', 'cosp2', 'hpgmg', 'lulesh']

    # (0,0) UC1: Link utilization by mini-app (percentage)
    ax = axes[0, 0]
    if not df_routing.empty:
        df_routing['max_link_util_pct'] = df_routing['max_link_util'] * 100
        summary = df_routing.groupby('app_type')['max_link_util_pct'].agg(['mean', 'std'])
        summary = summary.reindex([a for a in apps_order if a in summary.index])
        x = np.arange(len(summary))
        colors = [MINIAPP_COLORS.get(a, 'gray') for a in summary.index]
        bars = ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5,
                     color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in summary.index])
        ax.set_ylabel('Max Link Utilization (%)')
        ax.set_title('UC1: Adaptive Routing')
        ax.grid(axis='y', alpha=0.3)

    # (0,1) UC2: Placement comparison
    ax = axes[0, 1]
    if not df_placement.empty:
        df_placement['max_link_util_pct'] = df_placement['max_link_util'] * 100
        summary = df_placement.groupby(['app_type', 'allocation_strategy'])['max_link_util_pct'].mean().unstack()
        summary = summary.reindex([a for a in apps_order if a in summary.index])
        summary.plot(kind='bar', ax=ax, color=[ALLOCATION_COLORS.get(c, 'gray')
                                               for c in summary.columns],
                    edgecolor='black', linewidth=0.5)
        ax.set_xticklabels([a.upper() for a in summary.index], rotation=45, ha='right')
        ax.set_ylabel('Max Link Utilization (%)')
        ax.set_title('UC2: Node Placement')
        ax.legend(title='Allocation')
        ax.grid(axis='y', alpha=0.3)

    # (1,0) UC3: Scheduling - utilization by config
    ax = axes[1, 0]
    if not df_scheduling.empty:
        df_scheduling['max_link_util_pct'] = df_scheduling['max_link_util'] * 100
        summary = df_scheduling.groupby('system')['max_link_util_pct'].agg(['mean', 'std'])
        x = np.arange(len(summary))
        bars = ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5,
                     color=[SYSTEM_COLORS.get(s, 'gray') for s in summary.index],
                     edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in summary.index])
        ax.set_ylabel('Max Link Utilization (%)')
        ax.set_title('UC3: Scheduling Impact')
        ax.grid(axis='y', alpha=0.3)

    # (1,1) UC4: Power by mini-app
    ax = axes[1, 1]
    if not df_power.empty:
        summary = df_power.groupby(['app_type', 'system'])['total_power_kw'].mean().unstack()
        summary = summary.reindex([a for a in apps_order if a in summary.index])
        summary.plot(kind='bar', ax=ax, color=[SYSTEM_COLORS.get(s, 'gray') for s in summary.columns],
                    edgecolor='black', linewidth=0.5)
        ax.set_xticklabels([a.upper() for a in summary.index], rotation=45, ha='right')
        ax.set_ylabel('Total Power (kW)')
        ax.set_title('UC4: Power Consumption')
        ax.legend(title='System')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('SC26 Experiment Summary: Mini-app Communication Patterns on HPC Networks',
                fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_figures():
    """Generate all figures for the SC26 paper."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Generating SC26 Visualization Figures (v2)")
    print("="*60)

    # Load and plot each use case
    print("\n[1/5] Adaptive Routing...")
    df = load_experiment_data("adaptive_routing")
    plot_adaptive_routing(df, FIGURES_DIR / "uc1_adaptive_routing.png")

    print("\n[2/5] Node Placement...")
    df = load_experiment_data("node_placement")
    plot_node_placement(df, FIGURES_DIR / "uc2_node_placement.png")

    print("\n[3/5] Scheduling...")
    df = load_experiment_data("scheduling")
    plot_scheduling(df, FIGURES_DIR / "uc3_scheduling.png")

    print("\n[4/5] Power Consumption...")
    df = load_experiment_data("power_consumption")
    plot_power_consumption(df, FIGURES_DIR / "uc4_power_consumption.png")

    print("\n[5/5] Combined Summary...")
    plot_combined_summary(FIGURES_DIR / "combined_summary.png")

    print("\n" + "="*60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_figures()
