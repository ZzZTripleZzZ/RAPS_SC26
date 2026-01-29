#!/usr/bin/env python3
"""
SC26 Pipeline v3.0 Visualization
================================
Includes RL scheduler, FLOPS, and carbon emissions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/app/data/results_v3")
OUTPUT_DIR = Path("/app/output/pipeline_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'lassen': '#2ecc71', 'frontier': '#3498db',
    'fcfs': '#3498db', 'backfill': '#2ecc71', 'sjf': '#f39c12', 'rl': '#e74c3c',
}


def load_data():
    csv_path = RESULTS_DIR / "pipeline_v3_results.csv"
    if not csv_path.exists():
        print(f"Results not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def fig1_routing(df):
    """Dragonfly Achieves Lower Latency with Higher Throughput"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    routing_df = df[df['use_case'] == 'routing']

    for idx, system in enumerate(['lassen', 'frontier']):
        ax = axes[idx]
        sys_df = routing_df[routing_df['system'] == system]
        if len(sys_df) == 0:
            continue

        data = sys_df.groupby('algorithm').agg({'latency_us': 'mean', 'throughput_gbps': 'mean'}).reset_index()
        x = np.arange(len(data))

        ax.bar(x - 0.2, data['latency_us'], 0.4, color=COLORS[system], alpha=0.85, label='Latency (μs)')
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, data['throughput_gbps'], 0.4, color='#e67e22', alpha=0.85, label='Throughput')

        ax.set_xticks(x)
        ax.set_xticklabels(data['algorithm'].str.upper())
        ax.set_xlabel('Routing Algorithm')
        ax.set_ylabel('Latency (μs)')
        ax2.set_ylabel('Throughput (Gbps)')

        topo = 'Fat-Tree' if system == 'lassen' else 'Dragonfly'
        ax.set_title(f'{system.capitalize()} ({topo})', fontweight='bold')
        ax.legend(loc='upper center', bbox_to_anchor=(0.3, -0.12), fontsize=9)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.7, -0.12), fontsize=9)

    fig.suptitle('Dragonfly Achieves Lower Latency with Higher Throughput', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_routing.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig1_routing.png")


def fig2_scheduling_with_rl(df):
    """RL Scheduler Outperforms Traditional Approaches"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sched_df = df[df['use_case'] == 'scheduling']

    metrics = [('makespan', 'Makespan (s)'), ('utilization', 'Utilization (%)'),
               ('energy_kwh', 'Energy (kWh)'), ('co2_kg', 'CO2 Emissions (kg)')]

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        if metric not in sched_df.columns:
            continue

        data = sched_df.groupby(['scheduler', 'system'])[metric].mean().unstack()
        if data.empty:
            continue

        x = np.arange(len(data.index))
        width = 0.35

        for i, system in enumerate(['lassen', 'frontier']):
            if system in data.columns:
                offset = -width/2 if i == 0 else width/2
                colors = [COLORS.get(s, '#999') for s in data.index]
                ax.bar(x + offset, data[system], width, label=system.capitalize(),
                       color=COLORS[system], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in data.index])
        ax.set_xlabel('Scheduler')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split('(')[0].strip(), fontweight='bold')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

    fig.suptitle('RL Scheduler Achieves Competitive Performance with Lower Energy',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(OUTPUT_DIR / "fig2_scheduling_rl.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig2_scheduling_rl.png")


def fig3_power_efficiency(df):
    """Job Packing Maximizes FLOPS per Watt"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    power_df = df[df['use_case'] == 'power']

    for idx, system in enumerate(['lassen', 'frontier']):
        ax = axes[idx]
        sys_df = power_df[power_df['system'] == system]
        if len(sys_df) == 0:
            continue

        data = sys_df.groupby('scenario').agg({
            'total_power_mw': 'mean',
            'efficiency_gflops_w': 'mean',
            'co2_kg': 'mean'
        }).reset_index()

        x = np.arange(len(data))
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

        ax.bar(x, data['efficiency_gflops_w'], color=colors[:len(data)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n').title() for s in data['scenario']], fontsize=9)
        ax.set_xlabel('Power Strategy')
        ax.set_ylabel('Efficiency (GFLOPS/W)')

        topo = 'Fat-Tree' if system == 'lassen' else 'Dragonfly'
        ax.set_title(f'{system.capitalize()} ({topo})', fontweight='bold')

        # Add efficiency values on bars
        for i, v in enumerate(data['efficiency_gflops_w']):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)

    fig.suptitle('Job Packing Achieves Highest Compute Efficiency',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_power_efficiency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig3_power_efficiency.png")


def fig4_carbon_footprint(df):
    """Power Management Reduces Carbon Footprint"""
    fig, ax = plt.subplots(figsize=(12, 6))
    power_df = df[df['use_case'] == 'power']

    data = power_df.groupby(['scenario', 'system'])['co2_kg'].mean().unstack()
    if data.empty:
        return

    x = np.arange(len(data.index))
    width = 0.35

    ax.bar(x - width/2, data['lassen'], width, label='Lassen', color=COLORS['lassen'], alpha=0.85)
    ax.bar(x + width/2, data['frontier'], width, label='Frontier', color=COLORS['frontier'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in data.index])
    ax.set_xlabel('Power Strategy')
    ax.set_ylabel('CO2 Emissions (kg)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax.set_title('Power Management Strategies Reduce Carbon Footprint',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_carbon.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_carbon.png")


def fig5_placement(df):
    """Locality-Aware Placement Minimizes Communication Cost"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    place_df = df[df['use_case'] == 'placement']

    for idx, system in enumerate(['lassen', 'frontier']):
        ax = axes[idx]
        sys_df = place_df[place_df['system'] == system]
        if len(sys_df) == 0:
            continue

        data = sys_df.groupby('strategy')['cost_reduction'].mean() * 100
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        bars = ax.bar(range(len(data)), data.values, color=colors[:len(data)], alpha=0.85)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([s.capitalize() for s in data.index])
        ax.set_xlabel('Placement Strategy')
        ax.set_ylabel('Cost Reduction (%)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        topo = 'Fat-Tree' if system == 'lassen' else 'Dragonfly'
        ax.set_title(f'{system.capitalize()} ({topo})', fontweight='bold')

    fig.suptitle('Locality-Aware Placement Reduces Communication Overhead',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_placement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_placement.png")


def fig6_scheduler_comparison(df):
    """Comprehensive Scheduler Comparison Including RL"""
    fig, ax = plt.subplots(figsize=(12, 6))
    sched_df = df[df['use_case'] == 'scheduling']

    # Scatter plot: makespan vs utilization
    for scheduler in ['fcfs', 'backfill', 'sjf', 'rl']:
        sch_df = sched_df[sched_df['scheduler'] == scheduler]
        if len(sch_df) == 0:
            continue

        for system in ['lassen', 'frontier']:
            sys_df = sch_df[sch_df['system'] == system]
            if len(sys_df) == 0:
                continue

            avg_makespan = sys_df['makespan'].mean()
            avg_util = sys_df['utilization'].mean()

            marker = 'o' if system == 'lassen' else 's'
            ax.scatter(avg_makespan, avg_util, s=150, marker=marker,
                       color=COLORS.get(scheduler, '#999'), alpha=0.8,
                       label=f'{scheduler.upper()} ({system.capitalize()})',
                       edgecolors='white', linewidth=1)

    ax.set_xlabel('Makespan (s)', fontsize=11)
    ax.set_ylabel('Utilization (%)', fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title('RL Scheduler Balances Makespan and Utilization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_scheduler_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig6_scheduler_scatter.png")


def fig7_summary(df):
    """HPC Digital Twin Summary Dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Best routing
    ax = axes[0, 0]
    routing_df = df[df['use_case'] == 'routing']
    if 'latency_us' in routing_df.columns:
        data = routing_df.groupby('system')['latency_us'].mean()
        ax.bar(range(len(data)), data.values, color=[COLORS['lassen'], COLORS['frontier']], alpha=0.85)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([s.capitalize() for s in data.index])
        ax.set_ylabel('Avg Latency (μs)')
        ax.set_title('Network Latency', fontweight='bold')

    # 2. Best scheduler
    ax = axes[0, 1]
    sched_df = df[df['use_case'] == 'scheduling']
    if 'makespan' in sched_df.columns:
        data = sched_df.groupby('scheduler')['makespan'].mean()
        colors = [COLORS.get(s, '#999') for s in data.index]
        ax.bar(range(len(data)), data.values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([s.upper() for s in data.index])
        ax.set_ylabel('Avg Makespan (s)')
        ax.set_title('Scheduling Performance', fontweight='bold')

    # 3. Power efficiency
    ax = axes[1, 0]
    power_df = df[df['use_case'] == 'power']
    if 'efficiency_gflops_w' in power_df.columns:
        data = power_df.groupby('scenario')['efficiency_gflops_w'].mean()
        ax.bar(range(len(data)), data.values, color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'][:len(data)], alpha=0.85)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([s.replace('_', '\n').title() for s in data.index], fontsize=9)
        ax.set_ylabel('GFLOPS/W')
        ax.set_title('Power Efficiency', fontweight='bold')

    # 4. Carbon footprint
    ax = axes[1, 1]
    if 'co2_kg' in power_df.columns:
        data = power_df.groupby('scenario')['co2_kg'].mean()
        ax.bar(range(len(data)), data.values, color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'][:len(data)], alpha=0.85)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([s.replace('_', '\n').title() for s in data.index], fontsize=9)
        ax.set_ylabel('CO2 (kg)')
        ax.set_title('Carbon Footprint', fontweight='bold')

    fig.suptitle('HPC Digital Twin Enables Holistic System Optimization',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig7_summary.png")


def main():
    print("=" * 60)
    print("Pipeline v3.0 Visualization (with RL)")
    print("=" * 60)

    df = load_data()
    if df is None:
        return

    print(f"Loaded {len(df)} experiments")
    print(f"Schedulers: {df[df['use_case']=='scheduling']['scheduler'].unique().tolist()}")
    print()

    fig1_routing(df)
    fig2_scheduling_with_rl(df)
    fig3_power_efficiency(df)
    fig4_carbon_footprint(df)
    fig5_placement(df)
    fig6_scheduler_comparison(df)
    fig7_summary(df)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
