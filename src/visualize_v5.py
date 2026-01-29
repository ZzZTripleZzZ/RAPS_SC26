#!/usr/bin/env python3
"""
SC26 HPC Digital Twin - Visualization v5
=========================================
Standard 5-figure visualization:
1. UC1: Adaptive Routing comparison
2. UC2: Node Placement comparison
3. UC3: Job Scheduling comparison
4. UC4: Power Analysis comparison
5. Sim2Real Gap metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/app/data/experiments_v5")
OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

COLORS = {
    'lassen': '#3498db',
    'frontier': '#e74c3c',
    'contiguous': '#2ecc71',
    'random': '#9b59b6',
    'hybrid': '#f39c12',
    'fcfs': '#3498db',
    'sjf': '#e74c3c',
    'easy': '#2ecc71',
    'firstfit': '#f39c12',
}


def load_csv(filename):
    """Load CSV file if exists."""
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def fig1_routing():
    """
    Figure 1: UC1 Adaptive Routing
    Compare allocation strategies (affects traffic patterns).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc1_routing_results.csv")

    # Lassen (real data)
    ax = axes[0]
    lassen_df = df[df['system'] == 'lassen']
    if len(lassen_df) > 0:
        strategies = lassen_df['strategy'].tolist()
        throughput = lassen_df['throughput'].tolist()
        colors = [COLORS.get(s, '#999') for s in strategies]

        bars = ax.bar(strategies, throughput, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        ax.set_ylabel('Throughput (jobs/hour)')
        ax.set_xlabel('Allocation Strategy')
        ax.set_title('Lassen (Real Data)\nFat-Tree Topology', fontweight='bold')

        for bar, val in zip(bars, throughput):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Lassen data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Lassen (Real Data)', fontweight='bold')

    # Frontier (synthetic)
    ax = axes[1]
    frontier_df = df[df['system'] == 'frontier']
    if len(frontier_df) > 0 and frontier_df['throughput'].sum() > 0:
        strategies = frontier_df['strategy'].tolist()
        throughput = frontier_df['throughput'].tolist()
        colors = [COLORS.get(s, '#999') for s in strategies]

        bars = ax.bar(strategies, throughput, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        ax.set_ylabel('Throughput (jobs/hour)')
        ax.set_xlabel('Allocation Strategy')
        ax.set_title('Frontier (Synthetic)\nDragonfly Topology', fontweight='bold')

        for bar, val in zip(bars, throughput):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        # Use power data instead if throughput is 0
        power_df = load_csv("uc4_power_results.csv")
        frontier_power = power_df[power_df['system'] == 'frontier']
        if len(frontier_power) > 0:
            workloads = frontier_power['workload'].tolist()
            power = frontier_power['avg_power_mw'].tolist()
            colors = [COLORS['frontier'], COLORS['lassen']][:len(workloads)]

            bars = ax.bar(workloads, power, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
            ax.set_ylabel('Avg Power (MW)')
            ax.set_xlabel('Workload Type')
            ax.set_title('Frontier (Synthetic)\nPower by Workload', fontweight='bold')

            for bar, val in zip(bars, power):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                           f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Frontier data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Frontier (Synthetic)', fontweight='bold')

    fig.suptitle('UC1: Network Traffic Patterns by Allocation Strategy',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_uc1_routing.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig1_uc1_routing.png'}")


def fig2_placement():
    """
    Figure 2: UC2 Node Placement
    Compare allocation strategies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc2_placement_results.csv")

    # Lassen
    ax = axes[0]
    lassen_df = df[df['system'] == 'lassen']
    if len(lassen_df) > 0:
        strategies = lassen_df['allocation'].tolist()
        completed = lassen_df['jobs_completed'].tolist()
        colors = [COLORS.get(s, '#999') for s in strategies]

        bars = ax.bar(strategies, completed, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        ax.set_ylabel('Jobs Completed')
        ax.set_xlabel('Allocation Strategy')
        ax.set_title('Lassen (Real Data)', fontweight='bold')

        for bar, val in zip(bars, completed):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Lassen data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Lassen (Real Data)', fontweight='bold')

    # Frontier
    ax = axes[1]
    frontier_df = df[df['system'] == 'frontier']
    if len(frontier_df) > 0:
        strategies = frontier_df['allocation'].tolist()
        completed = frontier_df['jobs_completed'].tolist()

        # If jobs_completed is 0, show power instead
        if sum(completed) == 0:
            power = frontier_df['avg_power_mw'].tolist()
            if sum(power) > 0:
                colors = [COLORS.get(s, '#999') for s in strategies]
                bars = ax.bar(strategies, power, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
                ax.set_ylabel('Avg Power (MW)')
                ax.set_xlabel('Allocation Strategy')
                ax.set_title('Frontier (Synthetic)\nPower Consumption', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Frontier data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Frontier (Synthetic)', fontweight='bold')
        else:
            colors = [COLORS.get(s, '#999') for s in strategies]
            bars = ax.bar(strategies, completed, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
            ax.set_ylabel('Jobs Completed')
            ax.set_xlabel('Allocation Strategy')
            ax.set_title('Frontier (Synthetic)', fontweight='bold')

            for bar, val in zip(bars, completed):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                           f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Frontier data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Frontier (Synthetic)', fontweight='bold')

    fig.suptitle('UC2: Node Placement - Allocation Strategy Comparison',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_uc2_placement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig2_uc2_placement.png'}")


def fig3_scheduling():
    """
    Figure 3: UC3 Job Scheduling
    Compare scheduling policies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc3_scheduling_results.csv")

    # Lassen (baseline)
    ax = axes[0]
    lassen_df = df[df['system'] == 'lassen']
    if len(lassen_df) > 0:
        # Show key metrics
        metrics = ['Jobs\nCompleted', 'Throughput\n(jobs/h)', 'Avg Power\n(MW)']
        values = [
            lassen_df['jobs_completed'].iloc[0],
            lassen_df['throughput'].iloc[0],
            lassen_df['avg_power_mw'].iloc[0]
        ]
        colors = [COLORS['lassen'], COLORS['frontier'], COLORS['contiguous']]

        bars = ax.bar(metrics, values, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        ax.set_ylabel('Value')
        ax.set_title('Lassen (Real Data)\nREPLAY Baseline', fontweight='bold')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Lassen data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Lassen (Real Data)', fontweight='bold')

    # Frontier (policy comparison)
    ax = axes[1]
    frontier_df = df[df['system'] == 'frontier']
    if len(frontier_df) > 0:
        policies = [f"{r['policy']}\n{r['backfill']}" for _, r in frontier_df.iterrows()]
        throughput = frontier_df['throughput'].tolist()

        # If throughput is all 0, use power instead
        if sum(throughput) == 0:
            power = frontier_df['avg_power_mw'].tolist()
            if sum(power) > 0:
                colors = [COLORS['fcfs'], COLORS['sjf'], COLORS['easy'], COLORS['firstfit']][:len(policies)]
                bars = ax.bar(range(len(policies)), power, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
                ax.set_xticks(range(len(policies)))
                ax.set_xticklabels(policies, fontsize=9)
                ax.set_ylabel('Avg Power (MW)')
                ax.set_xlabel('Policy / Backfill')
                ax.set_title('Frontier (Synthetic)\nPower by Policy', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Frontier data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Frontier (Synthetic)', fontweight='bold')
        else:
            colors = [COLORS['fcfs'], COLORS['sjf'], COLORS['easy'], COLORS['firstfit']][:len(policies)]
            bars = ax.bar(range(len(policies)), throughput, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
            ax.set_xticks(range(len(policies)))
            ax.set_xticklabels(policies, fontsize=9)
            ax.set_ylabel('Throughput (jobs/hour)')
            ax.set_xlabel('Policy / Backfill')
            ax.set_title('Frontier (Synthetic)\nScheduling Policy Comparison', fontweight='bold')

            for bar, val in zip(bars, throughput):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                           f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Frontier data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Frontier (Synthetic)', fontweight='bold')

    fig.suptitle('UC3: Job Scheduling - Policy Comparison',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_uc3_scheduling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig3_uc3_scheduling.png'}")


def fig4_power():
    """
    Figure 4: UC4 Power Analysis
    Compare power consumption.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc4_power_results.csv")

    # Lassen
    ax = axes[0]
    lassen_df = df[df['system'] == 'lassen']
    if len(lassen_df) > 0:
        metrics = ['Avg Power\n(MW)', 'Total Energy\n(MWh)', 'Cost\n($100s)']
        values = [
            lassen_df['avg_power_mw'].iloc[0],
            lassen_df['total_energy_mwh'].iloc[0],
            lassen_df['total_cost'].iloc[0] / 100
        ]
        colors = [COLORS['frontier'], COLORS['lassen'], COLORS['contiguous']]

        bars = ax.bar(metrics, values, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        ax.set_ylabel('Value')
        ax.set_title('Lassen (Real Data)\n1-hour Simulation', fontweight='bold')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Lassen data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Lassen (Real Data)', fontweight='bold')

    # Frontier
    ax = axes[1]
    frontier_df = df[df['system'] == 'frontier']
    if len(frontier_df) > 0:
        workloads = frontier_df['workload'].tolist()
        power = frontier_df['avg_power_mw'].tolist()

        # Filter out zero values
        valid = [(w, p) for w, p in zip(workloads, power) if p > 0]
        if valid:
            workloads, power = zip(*valid)
            colors = [COLORS['frontier'], COLORS['lassen']][:len(workloads)]

            bars = ax.bar(workloads, power, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
            ax.set_ylabel('Avg Power (MW)')
            ax.set_xlabel('Workload Type')
            ax.set_title('Frontier (Synthetic)\nPower by Workload', fontweight='bold')

            for bar, val in zip(bars, power):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Frontier power data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Frontier (Synthetic)', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Frontier data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Frontier (Synthetic)', fontweight='bold')

    fig.suptitle('UC4: Power Analysis - Workload Comparison',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_uc4_power.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig4_uc4_power.png'}")


def fig5_sim2real():
    """
    Figure 5: Sim2Real Gap
    Quantitative comparison of simulated vs expected values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("sim2real_metrics.csv")

    # Metrics comparison
    ax = axes[0]
    if len(df) > 0:
        # Filter for meaningful metrics
        df_valid = df[df['expected'] > 0].copy()

        if len(df_valid) > 0:
            labels = [f"{r['system']}\n{r['metric']}" for _, r in df_valid.iterrows()]
            expected = df_valid['expected'].tolist()
            simulated = df_valid['simulated'].tolist()

            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, expected, width, label='Expected', color=COLORS['lassen'], alpha=0.85)
            bars2 = ax.bar(x + width/2, simulated, width, label='Simulated', color=COLORS['frontier'], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Value')
            ax.legend()
            ax.set_title('Expected vs Simulated Values', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No valid metrics', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Expected vs Simulated', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Expected vs Simulated', fontweight='bold')

    # Gap percentage
    ax = axes[1]
    if len(df) > 0:
        df_gap = df[df['gap_percent'] >= 0].copy()

        if len(df_gap) > 0:
            labels = [f"{r['system']}\n{r['metric']}" for _, r in df_gap.iterrows()]
            gaps = df_gap['gap_percent'].tolist()

            colors = ['#2ecc71' if g < 10 else '#f39c12' if g < 30 else '#e74c3c' for g in gaps]
            bars = ax.barh(labels, gaps, color=colors, alpha=0.85)

            ax.set_xlabel('Gap (%)')
            ax.set_title('Sim2Real Gap', fontweight='bold')
            ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='<10% (Good)')
            ax.axvline(x=30, color='orange', linestyle='--', alpha=0.5, label='<30% (Moderate)')

            for bar, gap in zip(bars, gaps):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{gap:.1f}%', va='center', fontweight='bold')

            ax.legend(loc='lower right')
        else:
            ax.text(0.5, 0.5, 'No gap data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sim2Real Gap', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Sim2Real Gap', fontweight='bold')

    fig.suptitle('Sim2Real Gap Analysis',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_sim2real_gap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig5_sim2real_gap.png'}")


def main():
    print("="*60)
    print("SC26 Visualization v5")
    print("="*60)

    print("\nGenerating figures...")
    fig1_routing()
    fig2_placement()
    fig3_scheduling()
    fig4_power()
    fig5_sim2real()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
