#!/usr/bin/env python3
"""
SC26 HPC Digital Twin - Visualization v6
=========================================
Standard 5-figure visualization with meaningful comparisons.

Insights from each figure:
1. UC1 Routing: How allocation strategy affects network locality and power
2. UC2 Placement: How workload patterns affect resource utilization
3. UC3 Scheduling: How scheduling policies affect throughput and completion
4. UC4 Power: Power consumption across workload intensities
5. Sim2Real: Validation of simulation accuracy vs published specifications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/app/data/experiments_v6")
OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Color palettes
SYSTEM_COLORS = {'lassen': '#3498db', 'frontier': '#e74c3c'}
STRATEGY_COLORS = {'contiguous': '#2ecc71', 'random': '#9b59b6', 'hybrid': '#f39c12'}
POLICY_COLORS = {'fcfs': '#3498db', 'sjf': '#e74c3c', 'easy': '#2ecc71', 'firstfit': '#f39c12'}
WORKLOAD_COLORS = {'idle': '#95a5a6', 'random': '#3498db', 'randomAI': '#9b59b6',
                   'peak': '#e74c3c', 'benchmark': '#f39c12'}


def load_csv(filename):
    """Load CSV file if exists."""
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def fig1_routing():
    """
    Figure 1: UC1 Allocation Strategy Comparison

    Insight: Shows how contiguous vs random allocation affects:
    - Job throughput (network locality matters)
    - Power consumption (idle vs active nodes)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc1_routing_results.csv")

    if len(df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(OUTPUT_DIR / "fig1_uc1_routing.png", dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Panel 1: Throughput by strategy
    ax = axes[0]
    for i, system in enumerate(['lassen', 'frontier']):
        sys_df = df[df['system'] == system]
        if len(sys_df) > 0:
            x = np.arange(len(sys_df))
            bars = ax.bar(x + i*0.35, sys_df['throughput'], width=0.3,
                         label=f"{system.capitalize()}",
                         color=SYSTEM_COLORS[system], alpha=0.85, edgecolor='white')

            for bar, val in zip(bars, sys_df['throughput']):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    strategies = df['strategy'].unique()
    ax.set_xticks(np.arange(len(strategies)) + 0.175)
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel('Throughput (jobs/hour)')
    ax.set_xlabel('Allocation Strategy')
    ax.set_title('Job Throughput by Allocation Strategy', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(df['throughput'].max() * 1.2, 1))

    # Panel 2: Power by strategy
    ax = axes[1]
    for i, system in enumerate(['lassen', 'frontier']):
        sys_df = df[df['system'] == system]
        if len(sys_df) > 0:
            x = np.arange(len(sys_df))
            bars = ax.bar(x + i*0.35, sys_df['avg_power_mw'], width=0.3,
                         label=f"{system.capitalize()}",
                         color=SYSTEM_COLORS[system], alpha=0.85, edgecolor='white')

            for bar, val in zip(bars, sys_df['avg_power_mw']):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(np.arange(len(strategies)) + 0.175)
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel('Average Power (MW)')
    ax.set_xlabel('Allocation Strategy')
    ax.set_title('Power Consumption by Allocation Strategy', fontweight='bold')
    ax.legend(loc='upper right')

    fig.suptitle('UC1: Allocation Strategy Impact on Throughput and Power',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_uc1_routing.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig1_uc1_routing.png'}")


def fig2_placement():
    """
    Figure 2: UC2 Workload Pattern Comparison

    Insight: Shows how different workload types affect:
    - Resource utilization patterns
    - Power consumption profiles
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc2_placement_results.csv")

    if len(df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(OUTPUT_DIR / "fig2_uc2_placement.png", dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Panel 1: Lassen workloads
    ax = axes[0]
    lassen_df = df[df['system'] == 'lassen']
    if len(lassen_df) > 0:
        workloads = lassen_df['workload'].tolist()
        power = lassen_df['avg_power_mw'].tolist()
        colors = [WORKLOAD_COLORS.get(w, '#999') for w in workloads]

        bars = ax.bar(workloads, power, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        for bar, val in zip(bars, power):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Average Power (MW)')
        ax.set_xlabel('Workload Type')
        ax.set_title('Lassen (Fat-Tree)\n4,626 nodes', fontweight='bold')
        ax.axhline(y=9.2, color='red', linestyle='--', alpha=0.5, label='Peak Power (9.2 MW)')
        ax.legend(loc='upper right', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Lassen data', ha='center', va='center', transform=ax.transAxes)

    # Panel 2: Frontier workloads
    ax = axes[1]
    frontier_df = df[df['system'] == 'frontier']
    if len(frontier_df) > 0:
        workloads = frontier_df['workload'].tolist()
        power = frontier_df['avg_power_mw'].tolist()
        colors = [WORKLOAD_COLORS.get(w, '#999') for w in workloads]

        bars = ax.bar(workloads, power, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        for bar, val in zip(bars, power):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Average Power (MW)')
        ax.set_xlabel('Workload Type')
        ax.set_title('Frontier (Dragonfly)\n9,408 nodes', fontweight='bold')
        ax.axhline(y=21.1, color='red', linestyle='--', alpha=0.5, label='Peak Power (21.1 MW)')
        ax.legend(loc='upper right', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Frontier data', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle('UC2: Workload Pattern Impact on Power Consumption',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_uc2_placement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig2_uc2_placement.png'}")


def fig3_scheduling():
    """
    Figure 3: UC3 Scheduling Policy Comparison

    Insight: Shows how scheduling policies affect:
    - Job completion rates
    - FCFS vs SJF vs Backfill strategies
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc3_scheduling_results.csv")

    if len(df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(OUTPUT_DIR / "fig3_uc3_scheduling.png", dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Create policy labels
    df['policy_label'] = df.apply(
        lambda r: f"{r['policy'].upper()}" if r['backfill'] == 'none'
                  else f"{r['policy'].upper()}+\n{r['backfill'].upper()}", axis=1)

    for idx, (system, title) in enumerate([('lassen', 'Lassen'), ('frontier', 'Frontier')]):
        ax = axes[idx]
        sys_df = df[df['system'] == system]

        if len(sys_df) > 0:
            labels = sys_df['policy_label'].tolist()
            throughput = sys_df['throughput'].tolist()

            # Color by backfill type
            colors = []
            for _, row in sys_df.iterrows():
                if row['backfill'] == 'none':
                    colors.append(POLICY_COLORS.get(row['policy'], '#999'))
                else:
                    colors.append(POLICY_COLORS.get(row['backfill'], '#999'))

            bars = ax.bar(range(len(labels)), throughput, color=colors, alpha=0.85,
                         edgecolor='white', linewidth=2)

            for bar, val in zip(bars, throughput):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Throughput (jobs/hour)')
            ax.set_xlabel('Scheduling Policy')
            ax.set_title(f'{title}', fontweight='bold')

            max_val = max(throughput) if throughput else 1
            ax.set_ylim(0, max_val * 1.3)
        else:
            ax.text(0.5, 0.5, f'No {system} data', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle('UC3: Scheduling Policy Impact on Job Throughput',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_uc3_scheduling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig3_uc3_scheduling.png'}")


def fig4_power():
    """
    Figure 4: UC4 Power Analysis

    Insight: Shows power consumption range from idle to peak:
    - Validates power model against specifications
    - Shows dynamic range across workload intensities
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("uc4_power_results.csv")

    if len(df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(OUTPUT_DIR / "fig4_uc4_power.png", dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Define workload order for display
    workload_order = ['idle', 'random', 'randomAI', 'peak']

    for idx, (system, peak_power) in enumerate([('lassen', 9.2), ('frontier', 21.1)]):
        ax = axes[idx]
        sys_df = df[df['system'] == system].copy()

        if len(sys_df) > 0:
            # Sort by workload order
            sys_df['sort_order'] = sys_df['workload'].apply(
                lambda x: workload_order.index(x) if x in workload_order else 99)
            sys_df = sys_df.sort_values('sort_order')

            workloads = sys_df['workload'].tolist()
            power = sys_df['avg_power_mw'].tolist()
            colors = [WORKLOAD_COLORS.get(w, '#999') for w in workloads]

            bars = ax.bar(workloads, power, color=colors, alpha=0.85, edgecolor='white', linewidth=2)

            for bar, val in zip(bars, power):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                           f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

            ax.axhline(y=peak_power, color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'Spec Peak ({peak_power} MW)')

            ax.set_ylabel('Average Power (MW)')
            ax.set_xlabel('Workload Intensity')
            ax.set_title(f'{system.capitalize()}\nIdle to Peak Power Range', fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.set_ylim(0, max(power + [peak_power]) * 1.15)
        else:
            ax.text(0.5, 0.5, f'No {system} data', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle('UC4: Power Consumption Across Workload Intensities',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_uc4_power.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig4_uc4_power.png'}")


def fig5_sim2real():
    """
    Figure 5: Sim2Real Gap Analysis

    Insight: Validates simulation accuracy:
    - Compares simulated power to published specifications
    - Shows where model is accurate vs needs improvement
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = load_csv("sim2real_metrics.csv")

    if len(df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(OUTPUT_DIR / "fig5_sim2real_gap.png", dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Filter out node_count (always exact match)
    df_power = df[df['unit'] == 'MW'].copy()

    # Panel 1: Expected vs Simulated comparison
    ax = axes[0]
    if len(df_power) > 0:
        labels = [f"{r['system'].capitalize()}\n{r['metric'].replace('_', ' ').title()}"
                 for _, r in df_power.iterrows()]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, df_power['expected'], width, label='Published Spec',
                      color=SYSTEM_COLORS['lassen'], alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + width/2, df_power['simulated'], width, label='RAPS Simulated',
                      color=SYSTEM_COLORS['frontier'], alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Power (MW)')
        ax.legend(loc='upper right')
        ax.set_title('Expected vs Simulated Power', fontweight='bold')

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    # Panel 2: Gap percentage
    ax = axes[1]
    if len(df_power) > 0:
        labels = [f"{r['system'].capitalize()}\n{r['metric'].replace('_', ' ').title()}"
                 for _, r in df_power.iterrows()]
        gaps = df_power['gap_percent'].tolist()

        # Color by gap magnitude
        colors = ['#2ecc71' if g < 10 else '#f39c12' if g < 25 else '#e74c3c' for g in gaps]

        bars = ax.barh(labels, gaps, color=colors, alpha=0.85, edgecolor='white')

        ax.axvline(x=10, color='green', linestyle='--', alpha=0.7, linewidth=2, label='<10% (Excellent)')
        ax.axvline(x=25, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='<25% (Good)')

        for bar, gap in zip(bars, gaps):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{gap:.1f}%', va='center', fontweight='bold')

        ax.set_xlabel('Gap (%)')
        ax.set_title('Simulation Accuracy (Sim2Real Gap)', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(0, max(gaps) * 1.3)

    fig.suptitle('Sim2Real Validation: RAPS vs Published Specifications',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_sim2real_gap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig5_sim2real_gap.png'}")


def main():
    print("="*60)
    print("SC26 Visualization v6")
    print("="*60)
    print("\nGenerating 5 standard figures...")

    fig1_routing()
    fig2_placement()
    fig3_scheduling()
    fig4_power()
    fig5_sim2real()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\nFigure Insights:")
    print("  Fig1: Allocation strategy impact (contiguous vs random vs hybrid)")
    print("  Fig2: Workload pattern impact on power consumption")
    print("  Fig3: Scheduling policy comparison (FCFS, SJF, Backfill)")
    print("  Fig4: Power range from idle to peak workloads")
    print("  Fig5: Simulation accuracy vs published specifications")


if __name__ == "__main__":
    main()
