#!/usr/bin/env python3
"""
SC26 HPC Digital Twin - Final Visualization
============================================
4 figures covering 4 use cases, demonstrating:
1. Simulation runs successfully (baseline + algorithms)
2. Insights from simulation results
3. Sim2Real gap analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/app/data/results_v3")
OUTPUT_DIR = Path("/app/output/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# Colors
C_LASSEN = '#2ecc71'
C_FRONTIER = '#3498db'
C_REAL = '#9b59b6'
C_SYNTH = '#e74c3c'
SCHED_COLORS = {'fcfs': '#3498db', 'backfill': '#2ecc71', 'sjf': '#f39c12', 'rl': '#e74c3c'}
ALGO_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']


def load_data():
    df = pd.read_csv(RESULTS_DIR / "pipeline_v3_results.csv")
    print(f"Loaded {len(df)} experiments")
    return df


def fig1_simulation_overview(df):
    """
    Figure 1: Digital Twin Successfully Simulates HPC Workloads

    Shows: All 4 use cases run on both systems with multiple algorithms
    Story: The simulation framework works end-to-end
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid: 2x2 for the 4 use cases
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # ========== Use Case 1: Adaptive Routing ==========
    ax1 = fig.add_subplot(gs[0, 0])
    routing_df = df[df['use_case'] == 'routing']

    # Count successful runs per algorithm per system
    routing_counts = routing_df.groupby(['system', 'algorithm']).size().unstack(fill_value=0)

    x = np.arange(len(routing_counts.columns))
    width = 0.35
    ax1.bar(x - width/2, routing_counts.loc['lassen'] if 'lassen' in routing_counts.index else [0]*len(x),
            width, label='Lassen (Fat-Tree)', color=C_LASSEN, alpha=0.85)
    ax1.bar(x + width/2, routing_counts.loc['frontier'] if 'frontier' in routing_counts.index else [0]*len(x),
            width, label='Frontier (Dragonfly)', color=C_FRONTIER, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in routing_counts.columns])
    ax1.set_ylabel('Experiments Run')
    ax1.set_title('Use Case 1: Adaptive Routing', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)

    # Add latency values as text
    for system in ['lassen', 'frontier']:
        sys_df = routing_df[routing_df['system'] == system]
        for algo in routing_counts.columns:
            algo_df = sys_df[sys_df['algorithm'] == algo]
            if len(algo_df) > 0:
                lat = algo_df['latency_us'].mean()
                idx = list(routing_counts.columns).index(algo)
                offset = -width/2 if system == 'lassen' else width/2
                count = routing_counts.loc[system, algo] if system in routing_counts.index else 0
                ax1.annotate(f'{lat:.0f}μs', xy=(idx + offset, count),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=8, color='gray')

    # ========== Use Case 2: Node Placement ==========
    ax2 = fig.add_subplot(gs[0, 1])
    place_df = df[df['use_case'] == 'placement']

    place_counts = place_df.groupby(['system', 'strategy']).size().unstack(fill_value=0)

    x = np.arange(len(place_counts.columns))
    ax2.bar(x - width/2, place_counts.loc['lassen'] if 'lassen' in place_counts.index else [0]*len(x),
            width, label='Lassen', color=C_LASSEN, alpha=0.85)
    ax2.bar(x + width/2, place_counts.loc['frontier'] if 'frontier' in place_counts.index else [0]*len(x),
            width, label='Frontier', color=C_FRONTIER, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in place_counts.columns])
    ax2.set_ylabel('Experiments Run')
    ax2.set_title('Use Case 2: Node Placement', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)

    # ========== Use Case 3: Job Scheduling ==========
    ax3 = fig.add_subplot(gs[1, 0])
    sched_df = df[df['use_case'] == 'scheduling']

    sched_counts = sched_df.groupby(['system', 'scheduler']).size().unstack(fill_value=0)

    x = np.arange(len(sched_counts.columns))
    colors_sched = [SCHED_COLORS.get(s, '#999') for s in sched_counts.columns]

    ax3.bar(x - width/2, sched_counts.loc['lassen'] if 'lassen' in sched_counts.index else [0]*len(x),
            width, label='Lassen', color=C_LASSEN, alpha=0.85)
    ax3.bar(x + width/2, sched_counts.loc['frontier'] if 'frontier' in sched_counts.index else [0]*len(x),
            width, label='Frontier', color=C_FRONTIER, alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels([s.upper() for s in sched_counts.columns])
    ax3.set_ylabel('Experiments Run')
    ax3.set_title('Use Case 3: Job Scheduling (incl. RL)', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)

    # Highlight RL
    if 'rl' in sched_counts.columns:
        rl_idx = list(sched_counts.columns).index('rl')
        ax3.axvspan(rl_idx - 0.5, rl_idx + 0.5, alpha=0.1, color='red')
        ax3.annotate('RL Agent', xy=(rl_idx, ax3.get_ylim()[1]*0.9), ha='center', fontsize=9, color='red')

    # ========== Use Case 4: Power Analysis ==========
    ax4 = fig.add_subplot(gs[1, 1])
    power_df = df[df['use_case'] == 'power']

    power_counts = power_df.groupby(['system', 'scenario']).size().unstack(fill_value=0)

    x = np.arange(len(power_counts.columns))
    ax4.bar(x - width/2, power_counts.loc['lassen'] if 'lassen' in power_counts.index else [0]*len(x),
            width, label='Lassen', color=C_LASSEN, alpha=0.85)
    ax4.bar(x + width/2, power_counts.loc['frontier'] if 'frontier' in power_counts.index else [0]*len(x),
            width, label='Frontier', color=C_FRONTIER, alpha=0.85)

    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('_', '\n').title() for s in power_counts.columns], fontsize=9)
    ax4.set_ylabel('Experiments Run')
    ax4.set_title('Use Case 4: Power & Carbon Analysis', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)

    # Main title
    fig.suptitle('Figure 1: Digital Twin Successfully Simulates All HPC Use Cases',
                 fontsize=14, fontweight='bold', y=0.98)

    # Summary text
    total_exp = len(df)
    n_routing = len(routing_df)
    n_place = len(place_df)
    n_sched = len(sched_df)
    n_power = len(power_df)

    summary = f"Total: {total_exp} experiments | Routing: {n_routing} | Placement: {n_place} | Scheduling: {n_sched} | Power: {n_power}"
    fig.text(0.5, 0.02, summary, ha='center', fontsize=10, style='italic')

    plt.savefig(OUTPUT_DIR / "fig1_simulation_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig1_simulation_overview.png")


def fig2_routing_placement_insights(df):
    """
    Figure 2: Network Optimization Insights

    Shows: Use Case 1 (Routing) and Use Case 2 (Placement) results
    Insight: Dragonfly has lower latency; Locality-aware placement reduces cost
    """
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # ========== (a) Routing Latency Comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    routing_df = df[df['use_case'] == 'routing']

    data = routing_df.groupby(['system', 'algorithm'])['latency_us'].mean().unstack()

    x = np.arange(len(data.columns))
    width = 0.35

    bars1 = ax1.bar(x - width/2, data.loc['lassen'] if 'lassen' in data.index else [0]*len(x),
                    width, label='Lassen (Fat-Tree)', color=C_LASSEN, alpha=0.85)
    bars2 = ax1.bar(x + width/2, data.loc['frontier'] if 'frontier' in data.index else [0]*len(x),
                    width, label='Frontier (Dragonfly)', color=C_FRONTIER, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in data.columns])
    ax1.set_ylabel('Latency (μs)')
    ax1.set_xlabel('Routing Algorithm')
    ax1.set_title('(a) Network Latency', fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

    # Add insight annotation
    if 'lassen' in data.index and 'frontier' in data.index:
        lassen_avg = data.loc['lassen'].mean()
        frontier_avg = data.loc['frontier'].mean()
        improvement = (lassen_avg - frontier_avg) / lassen_avg * 100
        ax1.annotate(f'Dragonfly: {improvement:.0f}% lower latency',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', fontsize=10, color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # ========== (b) Placement Cost Reduction ==========
    ax2 = fig.add_subplot(gs[0, 1])
    place_df = df[df['use_case'] == 'placement']

    data = place_df.groupby(['system', 'strategy'])['cost_reduction'].mean().unstack() * 100

    x = np.arange(len(data.columns))

    ax2.bar(x - width/2, data.loc['lassen'] if 'lassen' in data.index else [0]*len(x),
            width, label='Lassen', color=C_LASSEN, alpha=0.85)
    ax2.bar(x + width/2, data.loc['frontier'] if 'frontier' in data.index else [0]*len(x),
            width, label='Frontier', color=C_FRONTIER, alpha=0.85)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in data.columns])
    ax2.set_ylabel('Cost Reduction (%)')
    ax2.set_xlabel('Placement Strategy')
    ax2.set_title('(b) Placement Optimization', fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

    # Find best strategy
    if 'locality' in data.columns:
        best_reduction = data['locality'].max()
        ax2.annotate(f'Locality: up to {best_reduction:.0f}% cost reduction',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', fontsize=10, color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # ========== (c) Throughput vs Latency Trade-off ==========
    ax3 = fig.add_subplot(gs[0, 2])

    for system in ['lassen', 'frontier']:
        sys_df = routing_df[routing_df['system'] == system]
        for algo in sys_df['algorithm'].unique():
            algo_df = sys_df[sys_df['algorithm'] == algo]
            lat = algo_df['latency_us'].mean()
            thr = algo_df['throughput_gbps'].mean()

            color = C_LASSEN if system == 'lassen' else C_FRONTIER
            marker = 'o' if system == 'lassen' else 's'
            ax3.scatter(lat, thr, s=150, c=color, marker=marker, alpha=0.8,
                       edgecolors='white', linewidth=1.5)
            ax3.annotate(algo.upper(), (lat, thr), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

    ax3.set_xlabel('Latency (μs)')
    ax3.set_ylabel('Throughput (Gbps)')
    ax3.set_title('(c) Latency-Throughput Trade-off', fontweight='bold')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_LASSEN, markersize=10, label='Lassen'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_FRONTIER, markersize=10, label='Frontier'),
    ]
    ax3.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

    # Main title
    fig.suptitle('Figure 2: Network Topology Significantly Impacts Performance',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(OUTPUT_DIR / "fig2_routing_placement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig2_routing_placement.png")


def fig3_scheduling_power_insights(df):
    """
    Figure 3: Scheduling and Power Optimization Insights

    Shows: Use Case 3 (Scheduling) and Use Case 4 (Power)
    Insight: RL scheduler competitive; Job packing most efficient
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # ========== (a) Scheduler Makespan Comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    sched_df = df[df['use_case'] == 'scheduling']

    data = sched_df.groupby('scheduler')['makespan'].mean() / 3600  # Convert to hours
    colors = [SCHED_COLORS.get(s, '#999') for s in data.index]

    bars = ax1.bar(range(len(data)), data.values, color=colors, alpha=0.85)
    ax1.set_xticks(range(len(data)))
    ax1.set_xticklabels([s.upper() for s in data.index])
    ax1.set_ylabel('Makespan (hours)')
    ax1.set_title('(a) Job Completion Time', fontweight='bold')

    # Highlight RL
    if 'rl' in data.index:
        rl_idx = list(data.index).index('rl')
        bars[rl_idx].set_edgecolor('red')
        bars[rl_idx].set_linewidth(2)

    # ========== (b) System Utilization ==========
    ax2 = fig.add_subplot(gs[0, 1])

    data = sched_df.groupby(['scheduler', 'system'])['utilization'].mean().unstack()

    x = np.arange(len(data.index))
    width = 0.35

    ax2.bar(x - width/2, data['lassen'] if 'lassen' in data.columns else [0]*len(x),
            width, label='Lassen', color=C_LASSEN, alpha=0.85)
    ax2.bar(x + width/2, data['frontier'] if 'frontier' in data.columns else [0]*len(x),
            width, label='Frontier', color=C_FRONTIER, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in data.index])
    ax2.set_ylabel('Utilization (%)')
    ax2.set_title('(b) Resource Utilization', fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)

    # ========== (c) Power Efficiency (GFLOPS/W) ==========
    ax3 = fig.add_subplot(gs[1, 0])
    power_df = df[df['use_case'] == 'power']

    data = power_df.groupby(['scenario', 'system'])['efficiency_gflops_w'].mean().unstack()

    x = np.arange(len(data.index))

    ax3.bar(x - width/2, data['lassen'] / 1000 if 'lassen' in data.columns else [0]*len(x),
            width, label='Lassen', color=C_LASSEN, alpha=0.85)
    ax3.bar(x + width/2, data['frontier'] / 1000 if 'frontier' in data.columns else [0]*len(x),
            width, label='Frontier', color=C_FRONTIER, alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels([s.replace('_', '\n').title() for s in data.index], fontsize=9)
    ax3.set_ylabel('Efficiency (TFLOPS/W)')
    ax3.set_title('(c) Compute Efficiency', fontweight='bold')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)

    # Insight annotation
    if 'job_packing' in data.index:
        best_eff = data.loc['job_packing'].max() / 1000
        ax3.annotate(f'Job Packing: {best_eff:.1f} TFLOPS/W',
                    xy=(0.5, 0.92), xycoords='axes fraction',
                    ha='center', fontsize=10, color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # ========== (d) Carbon Footprint ==========
    ax4 = fig.add_subplot(gs[1, 1])

    data = power_df.groupby(['scenario', 'system'])['co2_kg'].mean().unstack()

    x = np.arange(len(data.index))

    ax4.bar(x - width/2, data['lassen'] if 'lassen' in data.columns else [0]*len(x),
            width, label='Lassen', color=C_LASSEN, alpha=0.85)
    ax4.bar(x + width/2, data['frontier'] if 'frontier' in data.columns else [0]*len(x),
            width, label='Frontier', color=C_FRONTIER, alpha=0.85)

    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('_', '\n').title() for s in data.index], fontsize=9)
    ax4.set_ylabel('CO₂ Emissions (kg)')
    ax4.set_title('(d) Carbon Footprint', fontweight='bold')
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)

    # Main title
    fig.suptitle('Figure 3: Scheduling and Power Management Trade-offs',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / "fig3_scheduling_power.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig3_scheduling_power.png")


def fig4_sim2real_gap(df):
    """
    Figure 4: Sim2Real Gap Analysis

    Shows: Real traces vs Synthetic patterns comparison
    Story: Understanding the gap between simulation and reality
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    real_df = df[df['data_type'] == 'real']
    synth_df = df[df['data_type'] == 'synthetic']

    # ========== (a) Latency: Real vs Synthetic ==========
    ax1 = fig.add_subplot(gs[0, 0])

    routing_df = df[df['use_case'] == 'routing']
    real_lat = routing_df[routing_df['data_type'] == 'real'].groupby('system')['latency_us'].mean()
    synth_lat = routing_df[routing_df['data_type'] == 'synthetic'].groupby('system')['latency_us'].mean()

    x = np.arange(2)
    width = 0.35

    ax1.bar(x - width/2, [real_lat.get('lassen', 0), real_lat.get('frontier', 0)],
            width, label='Real Traces', color=C_REAL, alpha=0.85)
    ax1.bar(x + width/2, [synth_lat.get('lassen', 0), synth_lat.get('frontier', 0)],
            width, label='Synthetic', color=C_SYNTH, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(['Lassen', 'Frontier'])
    ax1.set_ylabel('Latency (μs)')
    ax1.set_title('(a) Network Latency', fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)

    # Calculate gap
    gaps = []
    for sys in ['lassen', 'frontier']:
        if sys in real_lat.index and sys in synth_lat.index:
            gap = abs(real_lat[sys] - synth_lat[sys]) / real_lat[sys] * 100
            gaps.append(gap)
    if gaps:
        avg_gap = np.mean(gaps)
        ax1.annotate(f'Avg Gap: {avg_gap:.1f}%', xy=(0.5, 0.92), xycoords='axes fraction',
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # ========== (b) Makespan: Real vs Synthetic ==========
    ax2 = fig.add_subplot(gs[0, 1])

    sched_df = df[df['use_case'] == 'scheduling']
    real_mk = sched_df[sched_df['data_type'] == 'real'].groupby('system')['makespan'].mean() / 3600
    synth_mk = sched_df[sched_df['data_type'] == 'synthetic'].groupby('system')['makespan'].mean() / 3600

    ax2.bar(x - width/2, [real_mk.get('lassen', 0), real_mk.get('frontier', 0)],
            width, label='Real Traces', color=C_REAL, alpha=0.85)
    ax2.bar(x + width/2, [synth_mk.get('lassen', 0), synth_mk.get('frontier', 0)],
            width, label='Synthetic', color=C_SYNTH, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Lassen', 'Frontier'])
    ax2.set_ylabel('Makespan (hours)')
    ax2.set_title('(b) Job Completion Time', fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)

    # ========== (c) Communication Pattern Impact ==========
    ax3 = fig.add_subplot(gs[1, 0])

    # Compare patterns for synthetic data
    synth_routing = routing_df[routing_df['data_type'] == 'synthetic']
    if 'pattern' in synth_routing.columns:
        pattern_lat = synth_routing.groupby('pattern')['latency_us'].mean()
        colors = ALGO_COLORS[:len(pattern_lat)]
        bars = ax3.bar(range(len(pattern_lat)), pattern_lat.values, color=colors, alpha=0.85)
        ax3.set_xticks(range(len(pattern_lat)))
        ax3.set_xticklabels([p.replace('-', '\n') for p in pattern_lat.index], fontsize=9)
        ax3.set_ylabel('Latency (μs)')
        ax3.set_title('(c) Pattern Impact on Latency', fontweight='bold')

        # Highlight variation
        lat_range = pattern_lat.max() - pattern_lat.min()
        ax3.annotate(f'Variation: {lat_range:.0f}μs ({lat_range/pattern_lat.mean()*100:.0f}%)',
                    xy=(0.5, 0.92), xycoords='axes fraction', ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # ========== (d) Sim2Real Summary ==========
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate gaps for each metric
    metrics = []
    gaps = []

    # Latency gap
    if len(real_lat) > 0 and len(synth_lat) > 0:
        lat_gap = abs(real_lat.mean() - synth_lat.mean()) / real_lat.mean() * 100
        metrics.append('Latency')
        gaps.append(lat_gap)

    # Makespan gap
    if len(real_mk) > 0 and len(synth_mk) > 0:
        mk_gap = abs(real_mk.mean() - synth_mk.mean()) / real_mk.mean() * 100
        metrics.append('Makespan')
        gaps.append(mk_gap)

    # Power gap
    power_df = df[df['use_case'] == 'power']
    real_pwr = power_df[power_df['data_type'] == 'real']['total_power_mw'].mean()
    synth_pwr = power_df[power_df['data_type'] == 'synthetic']['total_power_mw'].mean()
    if real_pwr > 0 and synth_pwr > 0:
        pwr_gap = abs(real_pwr - synth_pwr) / real_pwr * 100
        metrics.append('Power')
        gaps.append(pwr_gap)

    # Placement gap
    place_df = df[df['use_case'] == 'placement']
    real_cost = place_df[place_df['data_type'] == 'real']['communication_cost'].mean()
    synth_cost = place_df[place_df['data_type'] == 'synthetic']['communication_cost'].mean()
    if real_cost > 0 and synth_cost > 0:
        cost_gap = abs(real_cost - synth_cost) / real_cost * 100
        metrics.append('Comm Cost')
        gaps.append(min(cost_gap, 200))  # Cap at 200% for visualization

    if metrics:
        colors_gap = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][:len(metrics)]
        bars = ax4.barh(range(len(metrics)), gaps, color=colors_gap, alpha=0.85)
        ax4.set_yticks(range(len(metrics)))
        ax4.set_yticklabels(metrics)
        ax4.set_xlabel('Sim2Real Gap (%)')
        ax4.set_title('(d) Simulation Accuracy by Metric', fontweight='bold')
        ax4.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='Target: <20%')
        ax4.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Acceptable: <50%')
        ax4.legend(loc='lower right', fontsize=9)

        # Add gap values
        for i, (bar, gap) in enumerate(zip(bars, gaps)):
            ax4.annotate(f'{gap:.1f}%', xy=(bar.get_width() + 2, bar.get_y() + bar.get_height()/2),
                        va='center', fontsize=9)

    # Main title
    fig.suptitle('Figure 4: Sim2Real Gap Analysis - Validating Digital Twin Accuracy',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / "fig4_sim2real_gap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_sim2real_gap.png")


def main():
    print("=" * 60)
    print("SC26 HPC Digital Twin - Final Visualization")
    print("=" * 60)
    print("\nGenerating 4 figures covering:")
    print("  1. Simulation runs successfully")
    print("  2. Insights from results")
    print("  3. Sim2Real gap analysis")
    print()

    df = load_data()

    fig1_simulation_overview(df)
    fig2_routing_placement_insights(df)
    fig3_scheduling_power_insights(df)
    fig4_sim2real_gap(df)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
