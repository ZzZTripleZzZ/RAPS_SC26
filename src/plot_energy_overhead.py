#!/usr/bin/env python3
"""
Energy Overhead of Network Congestion  (SC-paper main figures)
==============================================================
Loads UC4 energy results from standard (200 jobs) and heavy (300 jobs)
experiments and visualises the energy cost of routing choices vs the
no-congestion ideal baseline.

Produces two single-column PNG files:
  energy_overhead_bars.png   — per-routing energy overhead % (std vs heavy)
  energy_overhead_epj.png    — energy-per-job normalised to ideal baseline

Usage:
    python src/plot_energy_overhead.py
    python src/plot_energy_overhead.py --no-copy   # skip copying to main/
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
UC_BASE = ROOT / "output" / "use_cases"
OUT_DIR = ROOT / "output" / "figures"
MAIN    = OUT_DIR / "main"

# ── Style ────────────────────────────────────────────────────────────────────
FIGW = 3.5   # single-column width (inches)
FIGH = 2.3
DPI  = 300

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':          9,
    'axes.labelsize':    10,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    7.5,
    'legend.framealpha':  0.85,
    'legend.edgecolor':  '#cccccc',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.axisbelow':     True,
    'figure.dpi':         DPI,
    'savefig.dpi':        DPI,
})

# ColorBrewer Dark2 – matches existing UC figures
SYS_COLOR  = {'frontier': '#D95F02', 'lassen': '#1B9E77'}
SYS_LABEL  = {'frontier': 'Frontier (dragonfly)', 'lassen': 'Lassen (fat-tree)'}

# Per-routing display name and colour
ROUTE_LABEL = {
    'minimal':  'Minimal',
    'ugal':     'UGAL',
    'valiant':  'Valiant',
    'ecmp':     'ECMP',
    'adaptive': 'Adaptive',
}
ROUTE_COLOR = {
    'minimal':  '#7570B3',
    'ugal':     '#D95F02',
    'valiant':  '#E7298A',
    'ecmp':     '#1B9E77',
    'adaptive': '#E7298A',
}

LOAD_HATCH = {'standard': '', 'heavy': '//'}
LOAD_ALPHA = {'standard': 0.85, 'heavy': 0.65}


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or 'label' not in df.columns:
            return None
        return df
    except Exception:
        return None


def load_uc4(system: str) -> dict:
    """Return {'standard': df | None, 'heavy': df | None}."""
    fname = 'uc4_energy_results.csv'
    return {
        'standard': _load_csv(UC_BASE / f'{system}_n1000'       / fname),
        'heavy':    _load_csv(UC_BASE / f'{system}_n1000_heavy'  / fname),
    }


def compute_overhead(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute energy overhead % and normalised EPJ relative to no_congestion."""
    if df is None:
        return None
    base_row = df[df['label'].str.contains('no_congestion', case=False)]
    if base_row.empty:
        return None
    e_base   = float(base_row.iloc[0]['total_energy_joules'])
    epj_base = float(base_row.iloc[0]['energy_per_completed_job'])
    if e_base <= 0 or epj_base <= 0:
        return None

    rows = []
    for _, row in df.iterrows():
        lbl = str(row['label'])
        if 'no_congestion' in lbl:
            continue
        routing = str(row.get('routing', '')).lower()
        rows.append({
            'routing':         routing,
            'energy_overhead': (float(row['total_energy_joules']) - e_base) / e_base * 100,
            'epj_norm':        float(row['energy_per_completed_job']) / epj_base,
            'dilated_pct':     float(row.get('dilated_pct', 0)),
            'avg_stall_ratio': float(row.get('avg_stall_ratio', 0)),
            'makespan_s':      float(row.get('simulated_seconds', 0)),
        })
    return pd.DataFrame(rows) if rows else None


# ── Figure 1: Energy overhead % grouped by routing, std vs heavy ─────────────

def fig_energy_overhead_bars(save_dir: Path):
    """
    Grouped bar chart: energy overhead (%) for each routing algorithm.
    Within each routing group, standard and heavy bars appear side-by-side.
    One panel per system.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIGW * 2, FIGH),
                             sharey=False, constrained_layout=True)

    any_data = False
    for ax, system in zip(axes, ['frontier', 'lassen']):
        dfs = load_uc4(system)
        oh  = {load: compute_overhead(df) for load, df in dfs.items()}

        # Collect all routing algorithms present in any load variant
        routings = []
        for df_oh in oh.values():
            if df_oh is not None:
                for r in df_oh['routing'].tolist():
                    if r not in routings:
                        routings.append(r)

        if not routings:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='#aaa')
            ax.set_title(SYS_LABEL[system], fontsize=9, pad=4)
            ax.set_xlabel('Routing', fontsize=9)
            ax.set_ylabel('Energy overhead vs ideal (%)', fontsize=9)
            continue

        any_data = True
        loads_present = [l for l in ['standard', 'heavy'] if oh[l] is not None]
        n_loads = len(loads_present)
        bw = 0.35 if n_loads == 2 else 0.52
        x  = np.arange(len(routings))

        for li, load in enumerate(loads_present):
            df_oh = oh[load]
            if df_oh is None:
                continue
            offset = (li - (n_loads - 1) / 2) * bw
            vals = []
            for r in routings:
                row = df_oh[df_oh['routing'] == r]
                vals.append(float(row['energy_overhead'].iloc[0]) if not row.empty else 0.0)

            bars = ax.bar(
                x + offset, vals, width=bw * 0.9,
                color=SYS_COLOR[system],
                alpha=LOAD_ALPHA[load],
                hatch=LOAD_HATCH[load],
                edgecolor='white' if load == 'standard' else SYS_COLOR[system],
                linewidth=0.6,
                label=f'{load.capitalize()} load',
            )
            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f'{v:.1f}%', ha='center', va='bottom',
                            fontsize=6.5, color='#333')

        ax.set_xticks(x)
        ax.set_xticklabels([ROUTE_LABEL.get(r, r) for r in routings],
                           rotation=20, ha='right')
        ax.set_ylabel('Energy overhead vs ideal (%)', fontsize=9)
        ax.set_title(SYS_LABEL[system], fontsize=9, pad=4)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.7)
        if n_loads > 1:
            ax.legend(fontsize=7, loc='upper left')

    out = save_dir / 'energy_overhead_bars.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')
    return any_data


# ── Figure 2: Normalised energy-per-job (EPJ) ────────────────────────────────

def fig_energy_per_job(save_dir: Path):
    """
    Normalised energy-per-job relative to no-congestion ideal.
    Side-by-side bars: routing algorithms × systems.
    Both standard and heavy load shown per routing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIGW * 2, FIGH),
                             sharey=False, constrained_layout=True)

    for ax, system in zip(axes, ['frontier', 'lassen']):
        dfs = load_uc4(system)
        oh  = {load: compute_overhead(df) for load, df in dfs.items()}

        routings = []
        for df_oh in oh.values():
            if df_oh is not None:
                for r in df_oh['routing'].tolist():
                    if r not in routings:
                        routings.append(r)

        if not routings:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='#aaa')
            ax.set_title(SYS_LABEL[system], fontsize=9, pad=4)
            ax.set_xlabel('Routing', fontsize=9)
            ax.set_ylabel('Energy/job (normalised to ideal)', fontsize=9)
            continue

        loads_present = [l for l in ['standard', 'heavy'] if oh[l] is not None]
        n_loads = len(loads_present)
        bw = 0.35 if n_loads == 2 else 0.52
        x  = np.arange(len(routings))

        for li, load in enumerate(loads_present):
            df_oh = oh[load]
            if df_oh is None:
                continue
            offset = (li - (n_loads - 1) / 2) * bw
            vals = []
            for r in routings:
                row = df_oh[df_oh['routing'] == r]
                vals.append(float(row['epj_norm'].iloc[0]) if not row.empty else 1.0)

            bars = ax.bar(
                x + offset, vals, width=bw * 0.9,
                color=SYS_COLOR[system],
                alpha=LOAD_ALPHA[load],
                hatch=LOAD_HATCH[load],
                edgecolor='white' if load == 'standard' else SYS_COLOR[system],
                linewidth=0.6,
                label=f'{load.capitalize()} load',
            )
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{v:.2f}×', ha='center', va='bottom',
                        fontsize=6.5, color='#333')

        # Ideal baseline line
        ax.axhline(1.0, color='#555', linestyle='--', linewidth=0.8, alpha=0.6,
                   label='Ideal (no congestion)')

        ax.set_xticks(x)
        ax.set_xticklabels([ROUTE_LABEL.get(r, r) for r in routings],
                           rotation=20, ha='right')
        ax.set_ylabel('Energy/job (normalised to ideal)', fontsize=9)
        ax.set_title(SYS_LABEL[system], fontsize=9, pad=4)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.7)
        ax.legend(fontsize=7, loc='upper left')

    out = save_dir / 'energy_overhead_epj.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ── Figure 3: Single-column summary (main-ready) ─────────────────────────────

def fig_main_summary(save_dir: Path):
    """
    Single-column (3.5 × 2.3 in) figure for main/.
    Shows energy overhead % for all routing options across both systems,
    with standard/heavy as paired bars. Skips missing data gracefully.
    """
    # Gather all data
    all_entries = []   # list of (system, load, routing, energy_overhead)
    for system in ['frontier', 'lassen']:
        dfs = load_uc4(system)
        for load, df in dfs.items():
            oh = compute_overhead(df)
            if oh is None:
                continue
            for _, row in oh.iterrows():
                all_entries.append({
                    'system':  system,
                    'load':    load,
                    'routing': row['routing'],
                    'overhead': row['energy_overhead'],
                    'epj_norm': row['epj_norm'],
                })

    if not all_entries:
        print('  No data available yet — skipping main summary figure.')
        return

    data = pd.DataFrame(all_entries)

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    # One x-group per (system, routing) pair
    groups = (data.groupby(['system', 'routing'])
              .size().reset_index(name='n')[['system', 'routing']])
    # Order: frontier first, then lassen; within each: canonical routing order
    canonical = ['minimal', 'ugal', 'valiant', 'ecmp', 'adaptive']
    groups['sys_ord'] = groups['system'].map({'frontier': 0, 'lassen': 1})
    groups['rt_ord']  = groups['routing'].map(
        {r: i for i, r in enumerate(canonical)}).fillna(99)
    groups = groups.sort_values(['sys_ord', 'rt_ord']).reset_index(drop=True)

    n_groups  = len(groups)
    loads     = ['standard', 'heavy']
    bw        = 0.35
    x         = np.arange(n_groups)

    for li, load in enumerate(loads):
        sub = data[data['load'] == load]
        if sub.empty:
            continue
        offset = (li - 0.5) * bw
        vals   = []
        colors = []
        for _, g in groups.iterrows():
            match = sub[(sub['system'] == g['system']) &
                        (sub['routing'] == g['routing'])]
            vals.append(float(match['overhead'].iloc[0]) if not match.empty else None)
            colors.append(SYS_COLOR[g['system']])

        for xi, (v, c) in enumerate(zip(vals, colors)):
            if v is None:
                continue
            bar = ax.bar(xi + offset, v, width=bw * 0.88,
                         color=c, alpha=LOAD_ALPHA[load],
                         hatch=LOAD_HATCH[load],
                         edgecolor='white' if load == 'standard' else c,
                         linewidth=0.5,
                         label=f'{load.capitalize()} load' if xi == 0 else '_')
            if v > 0.1:
                ax.text(xi + offset, v + 0.15, f'{v:.1f}%',
                        ha='center', va='bottom', fontsize=6, color='#333')

    # Separator line between frontier and lassen groups
    frontier_indices = [i for i, (_, g) in enumerate(groups.iterrows())
                        if g['system'] == 'frontier']
    lassen_indices   = [i for i, (_, g) in enumerate(groups.iterrows())
                        if g['system'] == 'lassen']
    if frontier_indices and lassen_indices:
        sep_x = (frontier_indices[-1] + lassen_indices[0]) / 2
        ax.axvline(sep_x, color='#bbb', linewidth=0.8, linestyle=':')

    # Compute y_top from actual data to position system labels
    all_vals = data['overhead'].values
    y_top = float(all_vals.max()) * 1.18 if len(all_vals) > 0 else 10.0
    ax.set_ylim(bottom=0, top=y_top)

    _bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc',
                 alpha=0.85, linewidth=0.6)
    if frontier_indices:
        ax.text(np.mean(frontier_indices), y_top * 0.97,
                'Frontier', ha='center', va='top',
                fontsize=7, color=SYS_COLOR['frontier'], fontweight='bold',
                bbox=_bbox)
    if lassen_indices:
        ax.text(np.mean(lassen_indices), y_top * 0.97,
                'Lassen', ha='center', va='top',
                fontsize=7, color=SYS_COLOR['lassen'], fontweight='bold',
                bbox=_bbox)

    xlabels = [ROUTE_LABEL.get(g['routing'], g['routing'])
               for _, g in groups.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Energy overhead vs ideal (%)', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.7)

    # Legend: load level + system colour patches
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_items = [
        Patch(facecolor='#888', alpha=LOAD_ALPHA['standard'],
              label='Standard load (200 jobs)'),
        Patch(facecolor='#888', alpha=LOAD_ALPHA['heavy'],
              hatch='//', label='Heavy load (300 jobs)'),
        Patch(facecolor=SYS_COLOR['frontier'], label='Frontier'),
        Patch(facecolor=SYS_COLOR['lassen'],   label='Lassen'),
    ]
    ax.legend(handles=legend_items, fontsize=6.5, loc='upper left',
              ncol=2, handlelength=1.2, columnspacing=0.8,
              handletextpad=0.4, borderpad=0.5)

    fig.tight_layout(pad=0.5)
    out = save_dir / 'energy_overhead_main.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')
    return out


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary():
    print()
    print('=' * 60)
    print('ENERGY OVERHEAD SUMMARY  (vs no-congestion ideal)')
    print('=' * 60)
    for system in ['frontier', 'lassen']:
        dfs = load_uc4(system)
        print(f'\n{SYS_LABEL[system]}:')
        for load in ['standard', 'heavy']:
            df = dfs[load]
            oh = compute_overhead(df)
            if oh is None:
                print(f'  [{load:8s}]  no data')
                continue
            print(f'  [{load:8s}]  n_jobs='
                  f'{int(df[~df["label"].str.contains("no_cong")].iloc[0]["num_jobs"]) if df is not None else "?"}')
            for _, row in oh.iterrows():
                print(f'    {row["routing"]:12s}  overhead={row["energy_overhead"]:+6.2f}%  '
                      f'epj_norm={row["epj_norm"]:.3f}×  '
                      f'dilated={row["dilated_pct"]:.1f}%  '
                      f'stall={row["avg_stall_ratio"]:.3f}')
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot UC4 energy overhead figures')
    parser.add_argument('--no-copy', action='store_true',
                        help='Do not copy output to main/')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)

    print_summary()

    print('Generating figures...')
    fig_energy_overhead_bars(OUT_DIR)
    fig_energy_per_job(OUT_DIR)
    main_fig = fig_main_summary(OUT_DIR)

    if not args.no_copy and main_fig and main_fig.exists():
        dst = MAIN / 'energy_overhead.png'
        shutil.copy2(main_fig, dst)
        print(f'  Copied {main_fig.name} → main/')

    print(f'\nDone. Figures in {OUT_DIR}')


if __name__ == '__main__':
    main()
