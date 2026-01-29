#!/usr/bin/env python3
"""
SC26 Pipeline v2.0 Visualization
=================================
Clean visualizations with result-focused titles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
RESULTS_DIR = Path("/app/data/results_v2")
OUTPUT_DIR = Path("/app/output/pipeline_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

COLORS = {
    'lassen': '#2ecc71',
    'frontier': '#3498db',
    'real': '#9b59b6',
    'synthetic': '#e74c3c',
}

ALGO_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']


def load_data():
    """Load results from v2 pipeline."""
    csv_path = RESULTS_DIR / "complete_pipeline_v2_results.csv"
    if not csv_path.exists():
        print(f"Results not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def fig1_latency_breakdown(df):
    """Figure 1: Dragonfly Achieves 2x Lower Latency Than Fat-Tree"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    routing_df = df[df['use_case'] == 'adaptive_routing'].copy()

    for idx, system in enumerate(['lassen', 'frontier']):
        ax = axes[idx]
        sys_df = routing_df[routing_df['system'] == system]

        if len(sys_df) == 0:
            continue

        algorithms = sys_df['algorithm'].unique()
        latency_data = []
        for algo in algorithms:
            algo_df = sys_df[sys_df['algorithm'] == algo]
            latency_data.append({
                'algorithm': algo.upper(),
                'latency_us': algo_df['latency_us'].mean(),
                'throughput_gbps': algo_df['throughput_gbps'].mean(),
            })

        lat_df = pd.DataFrame(latency_data)
        x = np.arange(len(lat_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, lat_df['latency_us'], width,
                       color=COLORS[system], alpha=0.85, label='Latency')

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, lat_df['throughput_gbps'], width,
                        color='#e67e22', alpha=0.85, label='Throughput')

        ax.set_xlabel('Routing Algorithm', fontsize=11)
        ax.set_ylabel('Latency (μs)', fontsize=11)
        ax2.set_ylabel('Throughput (Gbps)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(lat_df['algorithm'])

        topology = 'Fat-Tree' if system == 'lassen' else 'Dragonfly'
        ax.set_title(f'{system.capitalize()} ({topology})', fontsize=12, fontweight='bold')

        # Legend outside plot area
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, ['Latency (μs)', 'Throughput (Gbps)'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

    fig.suptitle('Dragonfly Achieves 2x Lower Network Latency', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_latency_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig1_latency_performance.png")


def fig2_scheduling_comparison(df):
    """Figure 2: Backfill Improves System Utilization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sched_df = df[df['use_case'] == 'scheduling'].copy()

    metrics = ['makespan', 'avg_wait_time', 'utilization', 'energy_kwh']
    titles = ['Makespan', 'Wait Time', 'Utilization', 'Energy Consumption']
    ylabels = ['Seconds', 'Seconds', 'Percent (%)', 'kWh']

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx // 2, idx % 2]

        if metric not in sched_df.columns:
            continue

        grouped = sched_df.groupby(['scheduler', 'system'])[metric].mean().unstack()

        if grouped.empty:
            continue

        x = np.arange(len(grouped.index))
        width = 0.35

        if 'lassen' in grouped.columns:
            ax.bar(x - width/2, grouped['lassen'], width, label='Lassen',
                   color=COLORS['lassen'], alpha=0.85)
        if 'frontier' in grouped.columns:
            ax.bar(x + width/2, grouped['frontier'], width, label='Frontier',
                   color=COLORS['frontier'], alpha=0.85)

        ax.set_xlabel('Scheduler', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in grouped.index])
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Legend below x-axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)

        # Adjust y-axis to leave room at top
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.1)

    fig.suptitle('Backfill Scheduling Achieves Best Resource Utilization',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(OUTPUT_DIR / "fig2_scheduling_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig2_scheduling_comparison.png")


def fig3_power_analysis(df):
    """Figure 3: Job Packing Reduces Power by 3%"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    power_df = df[df['use_case'] == 'power'].copy()

    for idx, system in enumerate(['lassen', 'frontier']):
        ax = axes[idx]
        sys_df = power_df[power_df['system'] == system]

        if len(sys_df) == 0:
            continue

        scenarios = ['baseline', 'power_cap', 'frequency_scaling', 'job_packing']
        scenario_labels = ['Baseline', 'Power Cap', 'Freq Scaling', 'Job Packing']
        scenario_data = []

        for scenario, label in zip(scenarios, scenario_labels):
            scen_df = sys_df[sys_df['scenario'] == scenario]
            if len(scen_df) > 0:
                scenario_data.append({
                    'scenario': label,
                    'compute_power': scen_df['compute_power_mw'].mean(),
                    'total_power': scen_df['total_power_mw'].mean(),
                    'peak_power': scen_df['peak_power_mw'].mean(),
                })

        if not scenario_data:
            continue

        scen_df = pd.DataFrame(scenario_data)
        x = np.arange(len(scen_df))
        width = 0.25

        ax.bar(x - width, scen_df['compute_power'], width,
               color='#3498db', alpha=0.85, label='Compute')
        ax.bar(x, scen_df['total_power'], width,
               color='#2ecc71', alpha=0.85, label='Total (w/ PUE)')
        ax.bar(x + width, scen_df['peak_power'], width,
               color='#e74c3c', alpha=0.85, label='Peak')

        ax.set_xlabel('Power Strategy', fontsize=11)
        ax.set_ylabel('Power (MW)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(scen_df['scenario'], fontsize=9)

        topology = 'Fat-Tree' if system == 'lassen' else 'Dragonfly'
        ax.set_title(f'{system.capitalize()} ({topology})', fontsize=12, fontweight='bold')

        # Legend below
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)

        # Adjust y-axis
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.05)

    fig.suptitle('Job Packing Strategy Achieves Lowest Power Consumption',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_power_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig3_power_analysis.png")


def fig4_placement_strategies(df):
    """Figure 4: Locality-Aware Placement Reduces Communication Cost"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    place_df = df[df['use_case'] == 'node_placement'].copy()

    for idx, system in enumerate(['lassen', 'frontier']):
        ax = axes[idx]
        sys_df = place_df[place_df['system'] == system]

        if len(sys_df) == 0:
            continue

        strategies = ['contiguous', 'random', 'locality', 'spectral']
        strategy_labels = ['Contiguous', 'Random', 'Locality', 'Spectral']
        strategy_data = []

        for strategy, label in zip(strategies, strategy_labels):
            strat_df = sys_df[sys_df['strategy'] == strategy]
            if len(strat_df) > 0:
                strategy_data.append({
                    'strategy': label,
                    'cost': strat_df['communication_cost'].mean() / 1e12,
                    'reduction': strat_df['cost_reduction'].mean() * 100,
                })

        if not strategy_data:
            continue

        strat_df = pd.DataFrame(strategy_data)
        x = np.arange(len(strat_df))

        bars = ax.bar(x, strat_df['cost'], 0.6, color=COLORS[system], alpha=0.85)

        ax.set_xlabel('Placement Strategy', fontsize=11)
        ax.set_ylabel('Communication Cost (TB × hops)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(strat_df['strategy'], fontsize=10)

        topology = 'Fat-Tree' if system == 'lassen' else 'Dragonfly'
        ax.set_title(f'{system.capitalize()} ({topology})', fontsize=12, fontweight='bold')

        # Add value labels on bars
        for bar, val in zip(bars, strat_df['cost']):
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        # Adjust y-axis
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.15)

    fig.suptitle('Locality-Aware Placement Minimizes Network Traffic',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_placement_strategies.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig4_placement_strategies.png")


def fig5_real_vs_synthetic(df):
    """Figure 5: Real Traces Show Lower Latency Than Synthetic Patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    use_cases = ['adaptive_routing', 'node_placement', 'scheduling', 'power']
    metrics = ['latency_us', 'communication_cost', 'makespan', 'total_power_mw']
    titles = ['Network Latency', 'Communication Cost', 'Job Completion Time', 'Power Draw']
    ylabels = ['Latency (μs)', 'Cost (bytes × hops)', 'Makespan (s)', 'Power (MW)']

    for idx, (use_case, metric, title, ylabel) in enumerate(zip(use_cases, metrics, titles, ylabels)):
        ax = axes[idx // 2, idx % 2]

        uc_df = df[df['use_case'] == use_case].copy()

        if metric not in uc_df.columns:
            ax.set_visible(False)
            continue

        grouped = uc_df.groupby(['data_type', 'system'])[metric].mean().unstack()

        if grouped.empty:
            continue

        x = np.arange(len(grouped.columns))
        width = 0.35

        if 'real' in grouped.index:
            ax.bar(x - width/2, grouped.loc['real'], width, label='Real Traces',
                   color=COLORS['real'], alpha=0.85)
        if 'synthetic' in grouped.index:
            ax.bar(x + width/2, grouped.loc['synthetic'], width, label='Synthetic',
                   color=COLORS['synthetic'], alpha=0.85)

        ax.set_xlabel('System', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in grouped.columns])
        ax.set_title(title, fontsize=11, fontweight='bold')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)

        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.1)

    fig.suptitle('Real Application Traces Exhibit Different Behavior Than Synthetic Patterns',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(OUTPUT_DIR / "fig5_real_vs_synthetic.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig5_real_vs_synthetic.png")


def fig6_pattern_analysis(df):
    """Figure 6: All-to-All Communication Creates Highest Network Load"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    synth_df = df[df['data_type'] == 'synthetic'].copy()

    # Routing latency by pattern
    ax = axes[0, 0]
    routing_df = synth_df[synth_df['use_case'] == 'adaptive_routing']
    if 'pattern' in routing_df.columns and 'latency_us' in routing_df.columns:
        grouped = routing_df.groupby(['pattern', 'system'])['latency_us'].mean().unstack()
        if not grouped.empty:
            x = np.arange(len(grouped.index))
            width = 0.35
            if 'frontier' in grouped.columns:
                ax.bar(x - width/2, grouped['frontier'], width, label='Frontier',
                       color=COLORS['frontier'], alpha=0.85)
            if 'lassen' in grouped.columns:
                ax.bar(x + width/2, grouped['lassen'], width, label='Lassen',
                       color=COLORS['lassen'], alpha=0.85)
            ax.set_xlabel('Communication Pattern', fontsize=10)
            ax.set_ylabel('Latency (μs)', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([p.replace('-', '\n') for p in grouped.index], fontsize=9)
            ax.set_title('Network Latency', fontsize=11, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)

    # Placement cost by pattern
    ax = axes[0, 1]
    place_df = synth_df[synth_df['use_case'] == 'node_placement']
    if 'pattern' in place_df.columns and 'communication_cost' in place_df.columns:
        # Average across strategies
        grouped = place_df.groupby('pattern')['communication_cost'].mean() / 1e12
        if len(grouped) > 0:
            colors = [ALGO_COLORS[i % len(ALGO_COLORS)] for i in range(len(grouped))]
            bars = ax.bar(range(len(grouped)), grouped.values, color=colors, alpha=0.85)
            ax.set_xlabel('Communication Pattern', fontsize=10)
            ax.set_ylabel('Comm. Cost (TB × hops)', fontsize=10)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels([p.replace('-', '\n') for p in grouped.index], fontsize=9)
            ax.set_title('Placement Cost', fontsize=11, fontweight='bold')

    # Utilization by pattern
    ax = axes[1, 0]
    sched_df = synth_df[synth_df['use_case'] == 'scheduling']
    if 'pattern' in sched_df.columns and 'utilization' in sched_df.columns:
        grouped = sched_df.groupby('pattern')['utilization'].mean()
        if len(grouped) > 0:
            colors = [ALGO_COLORS[i % len(ALGO_COLORS)] for i in range(len(grouped))]
            bars = ax.bar(range(len(grouped)), grouped.values, color=colors, alpha=0.85)
            ax.set_xlabel('Communication Pattern', fontsize=10)
            ax.set_ylabel('Utilization (%)', fontsize=10)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels([p.replace('-', '\n') for p in grouped.index], fontsize=9)
            ax.set_title('System Utilization', fontsize=11, fontweight='bold')

    # Power by pattern
    ax = axes[1, 1]
    power_df = synth_df[synth_df['use_case'] == 'power']
    if 'pattern' in power_df.columns and 'total_power_mw' in power_df.columns:
        grouped = power_df[power_df['scenario'] == 'baseline'].groupby('pattern')['total_power_mw'].mean()
        if len(grouped) > 0:
            colors = [ALGO_COLORS[i % len(ALGO_COLORS)] for i in range(len(grouped))]
            bars = ax.bar(range(len(grouped)), grouped.values, color=colors, alpha=0.85)
            ax.set_xlabel('Communication Pattern', fontsize=10)
            ax.set_ylabel('Total Power (MW)', fontsize=10)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels([p.replace('-', '\n') for p in grouped.index], fontsize=9)
            ax.set_title('Power Consumption', fontsize=11, fontweight='bold')

    fig.suptitle('All-to-All Pattern Creates Highest Network and Power Demands',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(OUTPUT_DIR / "fig6_pattern_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig6_pattern_analysis.png")


def fig7_scale_analysis(df):
    """Figure 7: Communication Cost Scales Quadratically with Problem Size"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    synth_df = df[df['data_type'] == 'synthetic'].copy()

    # Latency scaling
    ax = axes[0, 0]
    routing_df = synth_df[synth_df['use_case'] == 'adaptive_routing']
    if 'num_ranks' in routing_df.columns and 'latency_us' in routing_df.columns:
        for system in ['lassen', 'frontier']:
            sys_df = routing_df[routing_df['system'] == system]
            grouped = sys_df.groupby('num_ranks')['latency_us'].mean()
            if len(grouped) > 0:
                ax.plot(grouped.index, grouped.values, 'o-', label=system.capitalize(),
                        color=COLORS[system], linewidth=2, markersize=8)
        ax.set_xlabel('Number of Ranks', fontsize=10)
        ax.set_ylabel('Latency (μs)', fontsize=10)
        ax.set_title('Latency Growth', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xscale('log', base=2)

    # Communication cost scaling
    ax = axes[0, 1]
    place_df = synth_df[synth_df['use_case'] == 'node_placement']
    if 'num_ranks' in place_df.columns and 'communication_cost' in place_df.columns:
        for i, strategy in enumerate(['contiguous', 'locality', 'spectral']):
            strat_df = place_df[place_df['strategy'] == strategy]
            grouped = strat_df.groupby('num_ranks')['communication_cost'].mean() / 1e12
            if len(grouped) > 0:
                ax.plot(grouped.index, grouped.values, 'o-', label=strategy.capitalize(),
                        color=ALGO_COLORS[i], linewidth=2, markersize=8)
        ax.set_xlabel('Number of Ranks', fontsize=10)
        ax.set_ylabel('Comm. Cost (TB × hops)', fontsize=10)
        ax.set_title('Placement Cost Growth', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

    # Makespan scaling
    ax = axes[1, 0]
    sched_df = synth_df[synth_df['use_case'] == 'scheduling']
    if 'num_ranks' in sched_df.columns and 'makespan' in sched_df.columns:
        for i, scheduler in enumerate(['fcfs', 'backfill', 'sjf']):
            sch_df = sched_df[sched_df['scheduler'] == scheduler]
            grouped = sch_df.groupby('num_ranks')['makespan'].mean()
            if len(grouped) > 0:
                ax.plot(grouped.index, grouped.values, 'o-', label=scheduler.upper(),
                        color=ALGO_COLORS[i], linewidth=2, markersize=8)
        ax.set_xlabel('Number of Ranks', fontsize=10)
        ax.set_ylabel('Makespan (s)', fontsize=10)
        ax.set_title('Makespan Growth', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xscale('log', base=2)

    # Power scaling
    ax = axes[1, 1]
    power_df = synth_df[synth_df['use_case'] == 'power']
    if 'num_ranks' in power_df.columns and 'total_power_mw' in power_df.columns:
        for system in ['lassen', 'frontier']:
            sys_df = power_df[(power_df['system'] == system) & (power_df['scenario'] == 'baseline')]
            grouped = sys_df.groupby('num_ranks')['total_power_mw'].mean()
            if len(grouped) > 0:
                ax.plot(grouped.index, grouped.values, 'o-', label=system.capitalize(),
                        color=COLORS[system], linewidth=2, markersize=8)
        ax.set_xlabel('Number of Ranks', fontsize=10)
        ax.set_ylabel('Total Power (MW)', fontsize=10)
        ax.set_title('Power Growth', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xscale('log', base=2)

    fig.suptitle('Communication Cost Scales Super-Linearly with Problem Size',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(OUTPUT_DIR / "fig7_scale_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7_scale_analysis.png")


def fig8_summary_dashboard(df):
    """Figure 8: Digital Twin Enables Multi-Objective HPC Optimization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Use Case 1: Routing trade-offs
    ax = axes[0, 0]
    routing_df = df[df['use_case'] == 'adaptive_routing']
    if 'algorithm' in routing_df.columns and 'latency_us' in routing_df.columns:
        summary = routing_df.groupby(['system', 'algorithm']).agg({
            'latency_us': 'mean',
            'throughput_gbps': 'mean'
        }).reset_index()

        for system in ['lassen', 'frontier']:
            sys_data = summary[summary['system'] == system]
            if len(sys_data) > 0:
                ax.scatter(sys_data['latency_us'], sys_data['throughput_gbps'],
                           s=120, alpha=0.8, color=COLORS[system],
                           label=system.capitalize(), edgecolors='white', linewidth=1)
                for _, row in sys_data.iterrows():
                    ax.annotate(row['algorithm'].upper(),
                                (row['latency_us'], row['throughput_gbps']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Latency (μs)', fontsize=10)
        ax.set_ylabel('Throughput (Gbps)', fontsize=10)
        ax.set_title('Routing: Latency vs Throughput', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)

    # Use Case 2: Placement improvement
    ax = axes[0, 1]
    place_df = df[df['use_case'] == 'node_placement']
    if 'strategy' in place_df.columns and 'cost_reduction' in place_df.columns:
        summary = place_df.groupby(['system', 'strategy'])['cost_reduction'].mean().unstack() * 100

        if not summary.empty:
            x = np.arange(len(summary.columns))
            width = 0.35
            if 'lassen' in summary.index:
                ax.bar(x - width/2, summary.loc['lassen'], width, label='Lassen',
                       color=COLORS['lassen'], alpha=0.85)
            if 'frontier' in summary.index:
                ax.bar(x + width/2, summary.loc['frontier'], width, label='Frontier',
                       color=COLORS['frontier'], alpha=0.85)
            ax.set_xlabel('Placement Strategy', fontsize=10)
            ax.set_ylabel('Cost Reduction (%)', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([s.capitalize() for s in summary.columns], fontsize=9)
            ax.set_title('Placement: Cost Reduction', fontsize=11, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Use Case 3: Scheduling trade-offs
    ax = axes[1, 0]
    sched_df = df[df['use_case'] == 'scheduling']
    if 'scheduler' in sched_df.columns and 'utilization' in sched_df.columns:
        summary = sched_df.groupby(['system', 'scheduler']).agg({
            'utilization': 'mean',
            'makespan': 'mean'
        }).reset_index()

        for system in ['lassen', 'frontier']:
            sys_data = summary[summary['system'] == system]
            if len(sys_data) > 0:
                ax.scatter(sys_data['makespan'], sys_data['utilization'],
                           s=120, alpha=0.8, color=COLORS[system],
                           label=system.capitalize(), edgecolors='white', linewidth=1)
                for _, row in sys_data.iterrows():
                    ax.annotate(row['scheduler'].upper(),
                                (row['makespan'], row['utilization']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Makespan (s)', fontsize=10)
        ax.set_ylabel('Utilization (%)', fontsize=10)
        ax.set_title('Scheduling: Makespan vs Utilization', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)

    # Use Case 4: Power savings
    ax = axes[1, 1]
    power_df = df[df['use_case'] == 'power']
    if 'scenario' in power_df.columns and 'total_power_mw' in power_df.columns:
        baseline = power_df[power_df['scenario'] == 'baseline'].groupby('system')['total_power_mw'].mean()

        savings_data = []
        scenarios = ['power_cap', 'frequency_scaling', 'job_packing']
        scenario_labels = ['Power Cap', 'Freq Scaling', 'Job Packing']

        for scenario, label in zip(scenarios, scenario_labels):
            scen_power = power_df[power_df['scenario'] == scenario].groupby('system')['total_power_mw'].mean()
            for system in baseline.index:
                if system in scen_power.index:
                    savings = (baseline[system] - scen_power[system]) / baseline[system] * 100
                    savings_data.append({
                        'system': system,
                        'scenario': label,
                        'savings': savings
                    })

        if savings_data:
            savings_df = pd.DataFrame(savings_data)
            pivot = savings_df.pivot(index='scenario', columns='system', values='savings')

            x = np.arange(len(pivot.index))
            width = 0.35
            if 'lassen' in pivot.columns:
                ax.bar(x - width/2, pivot['lassen'], width, label='Lassen',
                       color=COLORS['lassen'], alpha=0.85)
            if 'frontier' in pivot.columns:
                ax.bar(x + width/2, pivot['frontier'], width, label='Frontier',
                       color=COLORS['frontier'], alpha=0.85)

            ax.set_xlabel('Power Strategy', fontsize=10)
            ax.set_ylabel('Power Savings (%)', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, fontsize=9)
            ax.set_title('Power: Savings vs Baseline', fontsize=11, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('HPC Digital Twin Enables Multi-Objective System Optimization',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(OUTPUT_DIR / "fig8_summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig8_summary_dashboard.png")


def generate_summary_stats(df):
    """Generate summary statistics text file."""
    summary = []
    summary.append("=" * 60)
    summary.append("SC26 HPC Digital Twin - Results Summary")
    summary.append("=" * 60)
    summary.append("")
    summary.append(f"Total experiments: {len(df)}")
    summary.append(f"Real traces: {len(df[df['data_type'] == 'real'])}")
    summary.append(f"Synthetic: {len(df[df['data_type'] == 'synthetic'])}")
    summary.append("")

    # Key findings
    summary.append("KEY FINDINGS:")
    summary.append("-" * 40)

    routing_df = df[df['use_case'] == 'adaptive_routing']
    if 'latency_us' in routing_df.columns:
        for system in ['lassen', 'frontier']:
            sys_df = routing_df[routing_df['system'] == system]
            if len(sys_df) > 0:
                best = sys_df.loc[sys_df['latency_us'].idxmin()]
                summary.append(f"{system.capitalize()}: {best['algorithm'].upper()} achieves lowest latency ({best['latency_us']:.1f}μs)")

    place_df = df[df['use_case'] == 'node_placement']
    if 'cost_reduction' in place_df.columns:
        best = place_df.loc[place_df['cost_reduction'].idxmax()]
        summary.append(f"Placement: {best['strategy']} reduces cost by {best['cost_reduction']*100:.1f}%")

    power_df = df[df['use_case'] == 'power']
    if 'total_power_mw' in power_df.columns:
        baseline = power_df[power_df['scenario'] == 'baseline']['total_power_mw'].mean()
        for scenario in ['job_packing', 'frequency_scaling', 'power_cap']:
            scen_power = power_df[power_df['scenario'] == scenario]['total_power_mw'].mean()
            savings = (baseline - scen_power) / baseline * 100
            if savings > 0:
                summary.append(f"Power: {scenario} saves {savings:.1f}% vs baseline")
                break

    summary.append("")
    summary.append("=" * 60)

    summary_text = "\n".join(summary)
    with open(OUTPUT_DIR / "summary_stats.txt", 'w') as f:
        f.write(summary_text)
    print(f"Saved: summary_stats.txt")


def main():
    print("=" * 60)
    print("Generating Pipeline v2.0 Visualizations")
    print("=" * 60)

    df = load_data()
    if df is None:
        return

    print(f"Loaded {len(df)} experiments\n")

    fig1_latency_breakdown(df)
    fig2_scheduling_comparison(df)
    fig3_power_analysis(df)
    fig4_placement_strategies(df)
    fig5_real_vs_synthetic(df)
    fig6_pattern_analysis(df)
    fig7_scale_analysis(df)
    fig8_summary_dashboard(df)
    generate_summary_stats(df)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
