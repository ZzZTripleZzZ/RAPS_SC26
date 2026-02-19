#!/usr/bin/env python3
"""
Comprehensive Visualization of All RAPS Experiments
=====================================================
Generates publication-quality figures for:
  1. Scaling benchmark results (Lassen + Frontier, 100/1000/10000 nodes)
  2. Use case results (UC1-UC4 for both systems)

Usage:
    python src/plot_all_results.py
    python src/plot_all_results.py --scaling-only
    python src/plot_all_results.py --uc-only
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALING_CSV = PROJECT_ROOT / "output" / "frontier_scaling" / "results.csv"
UC_DIR = PROJECT_ROOT / "output" / "use_cases"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

# Color palettes
SYSTEM_COLORS = {'lassen': '#2196F3', 'frontier': '#FF5722'}
NODE_COLORS = {100: '#4CAF50', 1000: '#FF9800', 10000: '#E91E63'}
DT_COLORS = {60.0: '#1976D2', 10.0: '#388E3C', 1.0: '#F57C00', 0.1: '#D32F2F'}

# Common styling
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_scaling_data():
    """Load and preprocess scaling benchmark results."""
    if not SCALING_CSV.exists():
        print(f"  [WARN] Scaling CSV not found: {SCALING_CSV}")
        return None
    df = pd.read_csv(SCALING_CSV)
    df = df[df['status'] == 'OK'].copy()
    if df.empty:
        return None

    # Compute true simulated time for 10k experiments that exit early
    # For dt>=1, time_unit=1s, ticks = simulated seconds
    # For dt=0.1, time_unit=0.1s, ticks = simulated seconds / 0.1
    df['sim_seconds'] = df.apply(
        lambda r: r['ticks'] * 0.1 if r['delta_t'] == 0.1 else r['ticks'],
        axis=1
    )
    df['true_speedup'] = df['sim_seconds'] / df['sim_wall_s']
    df['per_tick_us'] = df['sim_wall_s'] / df['ticks'] * 1e6  # microseconds

    return df


def load_uc_data(system):
    """Load use case CSVs for a given system."""
    # Try both n1000 and n10000 directories
    for node_count in [1000, 10000]:
        uc_dir = UC_DIR / f"{system}_n{node_count}"
        if uc_dir.exists():
            results = {}
            for i in range(1, 5):
                csv_names = {
                    1: "uc1_routing_results.csv",
                    2: "uc2_scheduling_results.csv",
                    3: "uc3_placement_results.csv",
                    4: "uc4_energy_results.csv",
                }
                csv_path = uc_dir / csv_names[i]
                if csv_path.exists():
                    try:
                        results[f"UC{i}"] = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"  [WARN] Failed to read {csv_path}: {e}")
            if results:
                return results
    return {}


# ============================================================
# FIGURE 1: Scaling Overview (both systems)
# ============================================================
def plot_scaling_overview(df):
    """
    3-panel figure:
      (a) True speedup vs delta_t (log-log), grouped by system & node count
      (b) Per-tick cost vs node count for each delta_t
      (c) Simulation wall time breakdown (init vs sim)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- (a) True speedup vs delta_t ---
    ax = axes[0]
    for system in df['system'].unique():
        for nc in sorted(df['node_count'].unique()):
            sub = df[(df['system'] == system) & (df['node_count'] == nc)]
            if sub.empty:
                continue
            means = sub.groupby('delta_t')['true_speedup'].mean()
            stds = sub.groupby('delta_t')['true_speedup'].std().fillna(0)
            marker = 'o' if system == 'lassen' else 's'
            ls = '-' if system == 'lassen' else '--'
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker=marker, ls=ls, capsize=3, markersize=6,
                        label=f"{system} {nc:,}n",
                        color=NODE_COLORS.get(nc, '#999'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time Quantum dt (s)')
    ax.set_ylabel('True Speedup (sim_time / wall_time)')
    ax.set_title('(a) Simulation Speedup')
    ax.legend(fontsize=8, ncol=2)
    ax.invert_xaxis()

    # --- (b) Per-tick cost vs node count ---
    ax = axes[1]
    for dt in sorted(df['delta_t'].unique(), reverse=True):
        for system in df['system'].unique():
            sub = df[(df['system'] == system) & (df['delta_t'] == dt)]
            if sub.empty:
                continue
            means = sub.groupby('node_count')['per_tick_us'].mean()
            marker = 'o' if system == 'lassen' else 's'
            ls = '-' if system == 'lassen' else '--'
            ax.plot(means.index, means.values, marker=marker, ls=ls,
                    markersize=6, label=f"{system} dt={dt}s",
                    color=DT_COLORS.get(dt, '#999'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Node Count')
    ax.set_ylabel('Per-Tick Cost (us)')
    ax.set_title('(b) Per-Tick Computational Cost')
    ax.legend(fontsize=7, ncol=2)

    # --- (c) Wall time breakdown ---
    ax = axes[2]
    systems = df['system'].unique()
    node_counts = sorted(df['node_count'].unique())
    labels = []
    init_times = []
    sim_times = []
    for system in systems:
        for nc in node_counts:
            sub = df[(df['system'] == system) & (df['node_count'] == nc) & (df['delta_t'] == 1.0)]
            if sub.empty:
                continue
            labels.append(f"{system}\n{nc:,}n")
            init_times.append(sub['engine_init_s'].mean())
            sim_times.append(sub['sim_wall_s'].mean())

    if labels:
        x = np.arange(len(labels))
        w = 0.5
        ax.bar(x, init_times, w, label='Engine Init', color='#42A5F5', alpha=0.8)
        ax.bar(x, sim_times, w, bottom=init_times, label='Simulation', color='#EF5350', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Wall Time (s)')
        ax.set_title('(c) Time Breakdown (dt=1s)')
        ax.legend()

    fig.suptitle('RAPS Scaling Benchmark Results', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_scaling_overview.png')
    print(f"  Saved fig1_scaling_overview.png")
    plt.close()


# ============================================================
# FIGURE 2: Speedup Heatmaps (Lassen + Frontier side by side)
# ============================================================
def plot_speedup_heatmaps(df):
    """
    Side-by-side heatmaps of true speedup for each system,
    with node count on Y-axis and delta_t on X-axis.
    """
    systems = sorted(df['system'].unique())
    ncols = len(systems)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, system in zip(axes, systems):
        sub = df[df['system'] == system]
        dt_vals = sorted(sub['delta_t'].unique())
        nc_vals = sorted(sub['node_count'].unique())

        matrix = np.full((len(nc_vals), len(dt_vals)), np.nan)
        for i, nc in enumerate(nc_vals):
            for j, dt in enumerate(dt_vals):
                vals = sub[(sub['node_count'] == nc) & (sub['delta_t'] == dt)]['true_speedup']
                if len(vals) > 0:
                    matrix[i, j] = vals.mean()

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                       norm=LogNorm(vmin=max(1, np.nanmin(matrix)),
                                    vmax=np.nanmax(matrix)))

        # Annotate cells
        for i in range(len(nc_vals)):
            for j in range(len(dt_vals)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val > np.nanmedian(matrix) else 'black'
                    ax.text(j, i, f'{val:.0f}x', ha='center', va='center',
                            fontsize=10, fontweight='bold', color=color)

        ax.set_xticks(range(len(dt_vals)))
        ax.set_xticklabels([f'{dt}s' for dt in dt_vals])
        ax.set_yticks(range(len(nc_vals)))
        ax.set_yticklabels([f'{nc:,}' for nc in nc_vals])
        ax.set_xlabel('Time Quantum (dt)')
        ax.set_ylabel('Node Count')
        ax.set_title(f'{system.capitalize()} — True Speedup', fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Speedup')

    fig.suptitle('Simulation Speedup Heatmap', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_speedup_heatmap.png')
    print(f"  Saved fig2_speedup_heatmap.png")
    plt.close()


# ============================================================
# FIGURE 3: Node Scaling Analysis (10k deep dive)
# ============================================================
def plot_10k_analysis(df):
    """
    Focus on 10k-node experiments:
      (a) Wall time vs dt (bar chart per system)
      (b) Jobs completed (showing early-exit behavior)
      (c) Per-tick cost comparison across all node counts
    """
    df_10k = df[df['node_count'] == 10000]
    if df_10k.empty:
        print("  [SKIP] No 10k experiments yet")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- (a) Wall time vs dt ---
    ax = axes[0]
    for system in df_10k['system'].unique():
        sub = df_10k[df_10k['system'] == system]
        means = sub.groupby('delta_t')['sim_wall_s'].mean()
        stds = sub.groupby('delta_t')['sim_wall_s'].std().fillna(0)
        color = SYSTEM_COLORS.get(system, '#999')
        ax.bar(
            [f'dt={dt}s' for dt in means.index],
            means.values, yerr=stds.values,
            alpha=0.7, color=color, label=system,
            capsize=4, width=0.35,
            align='edge' if system == 'lassen' else 'center'
        )
    ax.set_ylabel('Simulation Wall Time (s)')
    ax.set_title('(a) 10K Nodes: Wall Time by dt')
    ax.legend()

    # --- (b) Simulated duration (early exit) ---
    ax = axes[1]
    for system in df_10k['system'].unique():
        sub = df_10k[df_10k['system'] == system]
        means = sub.groupby('delta_t')['sim_seconds'].mean()
        color = SYSTEM_COLORS.get(system, '#999')
        bars = ax.bar(
            [f'dt={dt}s' for dt in means.index],
            means.values / 3600,
            alpha=0.7, color=color, label=system,
            width=0.35,
            align='edge' if system == 'lassen' else 'center'
        )
    ax.axhline(y=12, color='red', ls='--', alpha=0.5, label='Target 12h')
    ax.set_ylabel('Simulated Duration (hours)')
    ax.set_title('(b) 10K Nodes: Simulated Time')
    ax.legend()

    # --- (c) Per-tick cost across all node counts ---
    ax = axes[2]
    for dt in [1.0]:  # Focus on dt=1 for clean comparison
        for system in df['system'].unique():
            sub = df[(df['system'] == system) & (df['delta_t'] == dt)]
            if sub.empty:
                continue
            means = sub.groupby('node_count')['per_tick_us'].mean() / 1000  # to ms
            color = SYSTEM_COLORS.get(system, '#999')
            marker = 'o' if system == 'lassen' else 's'
            ax.plot(means.index, means.values, marker=marker, ls='-',
                    markersize=8, color=color, label=f'{system} (dt=1s)',
                    linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Node Count')
    ax.set_ylabel('Per-Tick Cost (ms)')
    ax.set_title('(c) Per-Tick Cost Scaling (dt=1s)')
    ax.legend()

    fig.suptitle('10,000-Node Experiment Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_10k_analysis.png')
    print(f"  Saved fig3_10k_analysis.png")
    plt.close()


# ============================================================
# FIGURE 4: System Comparison Table (text figure)
# ============================================================
def plot_comparison_table(df):
    """
    Summary table comparing Lassen (Fat-tree) vs Frontier (Dragonfly).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Pivot data
    rows = []
    for system in sorted(df['system'].unique()):
        for nc in sorted(df['node_count'].unique()):
            for dt in sorted(df['delta_t'].unique(), reverse=True):
                sub = df[(df['system'] == system) &
                         (df['node_count'] == nc) &
                         (df['delta_t'] == dt)]
                if sub.empty:
                    continue
                mean_speedup = sub['true_speedup'].mean()
                mean_wall = sub['sim_wall_s'].mean()
                mean_tick = sub['per_tick_us'].mean()
                mean_init = sub['engine_init_s'].mean()
                mean_jobs = sub['jobs_completed'].mean()
                mean_total_jobs = sub['jobs_total'].mean()
                rows.append([
                    system.capitalize(),
                    f'{nc:,}',
                    f'{dt}s',
                    f'{mean_speedup:.1f}x',
                    f'{mean_wall:.1f}s',
                    f'{mean_tick:.0f}us',
                    f'{mean_init:.1f}s',
                    f'{mean_jobs:.0f}/{mean_total_jobs:.0f}',
                ])

    if not rows:
        plt.close()
        return

    headers = ['System', 'Nodes', 'dt', 'Speedup', 'Sim Time', 'Per-Tick', 'Init', 'Jobs']
    table = ax.table(cellText=rows, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Color header
    for j, header in enumerate(headers):
        table[0, j].set_facecolor('#1976D2')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#E3F2FD')

    ax.set_title('RAPS Scaling Benchmark — All Configurations',
                 fontsize=14, fontweight='bold', pad=20)
    fig.savefig(FIGURES_DIR / 'fig4_comparison_table.png')
    print(f"  Saved fig4_comparison_table.png")
    plt.close()


# ============================================================
# FIGURE 5: Use Case Results (UC1-UC4 combined dashboard)
# ============================================================
def plot_use_cases(all_uc_data):
    """
    2x4 grid: Top row = Lassen, Bottom row = Frontier
    Each column = one use case
    """
    systems = sorted(all_uc_data.keys())
    if not systems:
        print("  [SKIP] No use case data available")
        return

    uc_names = {
        'UC1': 'Adaptive Routing',
        'UC2': 'Scheduler Policy',
        'UC3': 'Node Placement',
        'UC4': 'Energy Cost',
    }
    uc_metrics = {
        'UC1': ('avg_congestion', 'Avg Network Congestion'),
        'UC2': ('avg_wait_time', 'Avg Wait Time (s)'),
        'UC3': ('avg_congestion', 'Avg Network Congestion'),
        'UC4': ('dilated_pct', 'Jobs Slowed by Network (%)'),
    }
    uc_variant_cols = {
        'UC1': 'routing',
        'UC2': 'label',
        'UC3': 'allocation',
        'UC4': 'label',
    }

    nrows = len(systems)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, system in enumerate(systems):
        uc_data = all_uc_data[system]
        for col_idx, uc_key in enumerate(['UC1', 'UC2', 'UC3', 'UC4']):
            ax = axes[row_idx, col_idx]
            if uc_key not in uc_data:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(f'{uc_key}: {uc_names[uc_key]}')
                continue

            df_uc = uc_data[uc_key]
            metric_col, metric_label = uc_metrics[uc_key]
            variant_col = uc_variant_cols[uc_key]

            if metric_col not in df_uc.columns:
                ax.text(0.5, 0.5, f'Missing: {metric_col}', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
                ax.set_title(f'{uc_key}: {uc_names[uc_key]}')
                continue

            # Get variant labels and values
            if variant_col in df_uc.columns:
                variants = df_uc[variant_col].tolist()
                # Strip UC prefix from label column (e.g. 'UC2_fcfs+firstfit' -> 'fcfs+firstfit')
                if variant_col == 'label':
                    variants = [v.replace(f'{uc_key}_', '', 1) for v in variants]
            else:
                variants = [f'Config {i}' for i in range(len(df_uc))]

            values = df_uc[metric_col].tolist()
            if uc_key == 'UC4' and metric_col == 'total_energy_joules':
                values = [v / 1e6 for v in values]  # Convert J to MJ

            colors = plt.cm.Set2(np.linspace(0, 0.5, len(variants)))
            bars = ax.bar(variants, values, color=colors, edgecolor='gray', alpha=0.85)

            # Use log scale if range spans more than 10x (e.g. Lassen routing congestion)
            nonzero = [v for v in values if v > 0]
            if nonzero and max(nonzero) / min(nonzero) > 10:
                ax.set_yscale('log')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                        f'{val:.1f}' if val < 100 else f'{val:.0f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.set_ylabel(metric_label)
            ax.set_title(f'{uc_key}: {uc_names[uc_key]}')
            if row_idx == 0:
                ax.set_title(f'{uc_key}: {uc_names[uc_key]}', fontsize=12, fontweight='bold')

        # Row label
        axes[row_idx, 0].set_ylabel(f'{system.capitalize()}\n{uc_metrics["UC1"][1]}',
                                     fontsize=11)

    fig.suptitle('Use Case Evaluation: Lassen (Fat-tree) vs Frontier (Dragonfly)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_use_cases.png')
    print(f"  Saved fig5_use_cases.png")
    plt.close()


# ============================================================
# FIGURE 6: Use Case Detailed Metrics
# ============================================================
def plot_uc_detailed(all_uc_data):
    """
    Multi-metric comparison per use case across both systems.
    """
    systems = sorted(all_uc_data.keys())
    if not systems:
        return

    # UC comparison: speedup, jobs completed, dilated percentage
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    uc_keys = ['UC1', 'UC2', 'UC3', 'UC4']
    uc_titles = ['UC1: Routing', 'UC2: Scheduling', 'UC3: Placement', 'UC4: Energy']
    compare_metrics = [
        ('speedup', 'Speedup (x)'),
        ('jobs_completed', 'Jobs Completed'),
        ('dilated_pct', 'Dilated Jobs (%)'),
        ('wall_time', 'Wall Time (s)'),
    ]

    for idx, (metric, ylabel) in enumerate(compare_metrics):
        ax = axes[idx // 2, idx % 2]
        x_labels = []
        x_positions = []
        pos = 0

        for system in systems:
            uc_data = all_uc_data[system]
            for uc_key in uc_keys:
                if uc_key not in uc_data:
                    continue
                df_uc = uc_data[uc_key]
                if metric not in df_uc.columns:
                    continue

                variant_col = {'UC1': 'routing', 'UC2': 'policy',
                               'UC3': 'allocation', 'UC4': 'label'}.get(uc_key, 'label')

                for _, row in df_uc.iterrows():
                    variant = row.get(variant_col, '?')
                    val = row[metric]
                    color = SYSTEM_COLORS.get(system, '#999')
                    ax.bar(pos, val, color=color, alpha=0.7, edgecolor='gray')
                    x_labels.append(f'{uc_key}\n{variant}')
                    x_positions.append(pos)
                    pos += 1
                pos += 0.5  # gap between UCs

        if x_positions:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    # Legend
    patches = [mpatches.Patch(color=c, label=s.capitalize())
               for s, c in SYSTEM_COLORS.items() if s in systems]
    axes[0, 0].legend(handles=patches)

    fig.suptitle('Use Case Detailed Metrics', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_uc_detailed.png')
    print(f"  Saved fig6_uc_detailed.png")
    plt.close()


# ============================================================
# FIGURE 7: Topology Comparison (Lassen fat-tree vs Frontier dragonfly)
# ============================================================
def plot_topology_comparison(df):
    """
    Compare how the two network topologies affect performance
    at different scales.
    """
    systems = sorted(df['system'].unique())
    if len(systems) < 2:
        print("  [SKIP] Need both systems for topology comparison")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- (a) Speedup ratio (frontier/lassen) ---
    ax = axes[0]
    dt_vals = sorted(df['delta_t'].unique(), reverse=True)
    nc_vals = sorted(df['node_count'].unique())

    for dt in dt_vals:
        ratios = []
        ncs = []
        for nc in nc_vals:
            lassen = df[(df['system'] == 'lassen') & (df['node_count'] == nc) & (df['delta_t'] == dt)]
            frontier = df[(df['system'] == 'frontier') & (df['node_count'] == nc) & (df['delta_t'] == dt)]
            if not lassen.empty and not frontier.empty:
                ratio = frontier['true_speedup'].mean() / lassen['true_speedup'].mean()
                ratios.append(ratio)
                ncs.append(nc)
        if ratios:
            ax.plot(ncs, ratios, marker='o', label=f'dt={dt}s',
                    color=DT_COLORS.get(dt, '#999'), linewidth=2)

    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Node Count')
    ax.set_ylabel('Speedup Ratio (Frontier / Lassen)')
    ax.set_title('(a) Relative Performance')
    ax.legend()

    # --- (b) Init time comparison ---
    ax = axes[1]
    for system in systems:
        sub = df[(df['system'] == system) & (df['delta_t'] == 1.0)]
        if sub.empty:
            continue
        means = sub.groupby('node_count')['engine_init_s'].mean()
        color = SYSTEM_COLORS.get(system, '#999')
        ax.plot(means.index, means.values, marker='o', label=system.capitalize(),
                color=color, linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Node Count')
    ax.set_ylabel('Engine Init Time (s)')
    ax.set_title('(b) Initialization Overhead')
    ax.legend()

    # --- (c) Network metrics from use cases: one bar per routing per system ---
    ax = axes[2]
    uc_data_all = {}
    for system in systems:
        uc_data_all[system] = load_uc_data(system)

    # Collect all (system, routing, congestion) tuples separately (systems have
    # different routing algorithms, so we list them independently)
    bar_labels = []
    bar_values = []
    bar_colors = []
    for system in systems:
        if 'UC1' in uc_data_all.get(system, {}):
            df_uc1 = uc_data_all[system]['UC1']
            if 'avg_congestion' in df_uc1.columns and 'routing' in df_uc1.columns:
                color = SYSTEM_COLORS.get(system, '#999')
                for _, row in df_uc1.iterrows():
                    bar_labels.append(f"{system[:2].upper()}\n{row['routing']}")
                    bar_values.append(row['avg_congestion'])
                    bar_colors.append(color)

    has_data = bool(bar_labels)
    if has_data:
        x = np.arange(len(bar_labels))
        ax.bar(x, bar_values, color=bar_colors, alpha=0.8, edgecolor='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        # Log scale if range spans >10x
        nonzero = [v for v in bar_values if v > 0]
        if nonzero and max(nonzero) / min(nonzero) > 10:
            ax.set_yscale('log')
        ax.set_ylabel('Avg Link Overload Ratio (log)')
        ax.set_title('(c) Routing Congestion (UC1)')
        # System legend
        patches = [mpatches.Patch(color=SYSTEM_COLORS[s], label=s.capitalize())
                   for s in systems if s in SYSTEM_COLORS]
        ax.legend(handles=patches)
    else:
        ax.text(0.5, 0.5, 'Waiting for UC results',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('(c) Network Congestion')

    fig.suptitle('Topology Comparison: Fat-tree (Lassen) vs Dragonfly (Frontier)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig7_topology_comparison.png')
    print(f"  Saved fig7_topology_comparison.png")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Plot all RAPS experiment results')
    parser.add_argument('--scaling-only', action='store_true')
    parser.add_argument('--uc-only', action='store_true')
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RAPS Results Visualization")
    print("=" * 60)

    # --- Scaling benchmarks ---
    if not args.uc_only:
        print("\n[1/4] Loading scaling data...")
        df = load_scaling_data()
        if df is not None:
            n_ok = len(df)
            systems = df['system'].unique().tolist()
            print(f"  {n_ok} OK experiments across systems: {systems}")
            print(f"  Node counts: {sorted(df['node_count'].unique())}")
            print(f"  Time quanta: {sorted(df['delta_t'].unique())}")

            print("\n[2/4] Generating scaling figures...")
            plot_scaling_overview(df)
            plot_speedup_heatmaps(df)
            plot_10k_analysis(df)
            plot_comparison_table(df)
            if len(systems) >= 2:
                plot_topology_comparison(df)
        else:
            print("  No scaling data available yet.")

    # --- Use cases ---
    if not args.scaling_only:
        print("\n[3/4] Loading use case data...")
        all_uc_data = {}
        for system in ['lassen', 'frontier']:
            uc = load_uc_data(system)
            if uc:
                all_uc_data[system] = uc
                print(f"  {system}: {list(uc.keys())}")
            else:
                print(f"  {system}: no results yet")

        if all_uc_data:
            print("\n[4/4] Generating use case figures...")
            plot_use_cases(all_uc_data)
            plot_uc_detailed(all_uc_data)
        else:
            print("  No use case data available yet.")

    # Summary
    print("\n" + "=" * 60)
    figs = list(FIGURES_DIR.glob('fig*.png'))
    print(f"Generated {len(figs)} figures in {FIGURES_DIR}")
    for f in sorted(figs):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == '__main__':
    main()
