#!/usr/bin/env python3
"""
NRAPS Fidelity Validation: Bully-Victim Interference (E2)
==========================================================

Generates comparison figures showing victim job slowdown as a function of
bully job size (load fraction), comparing NRAPS against SST-Macro.

Panel A: Dragonfly (Frontier) — NRAPS minimal/valiant + SST-Macro
Panel B: Fat-tree (Lassen)    — NRAPS minimal/ecmp   + SST-Macro
Panel C: Torus-3D (Blue Waters) — NRAPS dor_xyz      + SST-Macro

Figure size: 7.0 × 2.3 in (IEEE double-column, 3 panels), fonts 8/7pt.

Usage:
    python src/plot_interference_validation.py
    python src/plot_interference_validation.py --output-dir output/figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERFERENCE_DIR = PROJECT_ROOT / "output" / "interference"
SSTMACRO_DIR = PROJECT_ROOT / "Baseline" / "sst-macro" / "multi_job" / "output"
OUT_DIR = PROJECT_ROOT / "output" / "validation_figures"

# ── SC paper style ──────────────────────────────────────────────────────────
FIGW = 7.0    # double-column for 3-panel
FIGH = 2.3
DPI = 300

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          8,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '#cccccc',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.axisbelow':     True,
    'figure.dpi':         DPI,
    'savefig.dpi':        DPI,
})

# ColorBrewer Dark2
COLORS = {
    'minimal':   '#1B9E77',
    'ecmp':      '#1B9E77',
    'valiant':   '#D95F02',
    'adaptive':  '#E7298A',
    'dor_xyz':   '#1B9E77',
    'sst_macro': '#7570B3',
    'baseline':  '#AAAAAA',
}
MARKERS = {
    'minimal':   'o',
    'ecmp':      'o',
    'valiant':   's',
    'adaptive':  'D',
    'dor_xyz':   'o',
    'sst_macro': '^',
}
LABELS = {
    'minimal':   'NRAPS minimal',
    'ecmp':      'NRAPS ecmp',
    'valiant':   'NRAPS valiant',
    'adaptive':  'NRAPS adaptive',
    'dor_xyz':   'NRAPS dor-xyz',
    'sst_macro': 'SST-Macro',
}

# System → SST-Macro output subdirectory
SYSTEM_TO_SST_SUBDIR = {
    'frontier':   'dragonfly',
    'lassen':     'fattree',
    'bluewaters': 'torus3d',
}


def load_raps_data(system: str) -> pd.DataFrame:
    """Load NRAPS bully-victim sweep results for a system."""
    sys_dir = INTERFERENCE_DIR / system
    if not sys_dir.exists():
        print(f"  [WARN] No NRAPS dir for {system}: {sys_dir}")
        return pd.DataFrame()
    matches = list(sys_dir.glob("bully_sweep_*.csv"))
    if not matches:
        print(f"  [WARN] No NRAPS CSV for {system}")
        return pd.DataFrame()
    csv_path = matches[0]
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} NRAPS rows for {system} from {csv_path.name}")
    return df


def load_sstmacro_data(system: str) -> pd.DataFrame:
    """Load SST-Macro multi-job bully-victim results for a system."""
    import json
    subdir = SYSTEM_TO_SST_SUBDIR.get(system, system)
    out_dir = SSTMACRO_DIR / subdir
    if not out_dir.exists():
        print(f"  [WARN] No SST-Macro data for {system}: {out_dir}")
        return pd.DataFrame()

    rows = []
    for json_file in sorted(out_dir.glob("bully_*.json")):
        try:
            with open(json_file) as f:
                d = json.load(f)
            if d.get('status') == 'ok' and d.get('avg_victim_slowdown') is not None:
                rows.append({
                    'bully_nodes': d.get('bully_nodes', 0),
                    'load_fraction': d.get('load_fraction', 0.0),
                    'avg_victim_slowdown': d.get('avg_victim_slowdown', 1.0),
                    'max_victim_slowdown': d.get('max_victim_slowdown', 1.0),
                })
        except Exception as e:
            print(f"  [WARN] Failed to parse {json_file.name}: {e}")

    if rows:
        df = pd.DataFrame(rows).sort_values('bully_nodes')
        print(f"  Loaded {len(df)} SST-Macro rows for {system}")
        return df
    print(f"  [WARN] No completed SST-Macro results for {system}")
    return pd.DataFrame()


def plot_panel(ax, raps_df: pd.DataFrame, sstmacro_df: pd.DataFrame,
               system: str, routings: list, title: str):
    """Plot one panel: slowdown vs load_fraction for given system."""
    has_data = False
    baseline_plotted = False

    # NRAPS curves
    if not raps_df.empty:
        for routing in routings:
            label_key = routing if routing else 'dor_xyz'
            subset = raps_df[raps_df['routing_label'] == (routing or 'dor_xyz')]
            if subset.empty and 'routing' in raps_df.columns:
                subset = raps_df[raps_df['routing'] == routing]
            if subset.empty:
                continue
            subset = subset.sort_values('load_fraction')
            baseline = subset[subset['bully_nodes'] == 0]
            main = subset[subset['bully_nodes'] > 0]
            if main.empty:
                continue
            color = COLORS.get(label_key, '#333333')
            marker = MARKERS.get(label_key, 'o')
            lbl = LABELS.get(label_key, f'NRAPS {label_key}')
            ax.plot(main['load_fraction'], main['avg_victim_slowdown'],
                    color=color, marker=marker, markersize=4,
                    linewidth=1.5, label=lbl)
            if not baseline.empty and not baseline_plotted:
                ax.axhline(y=float(baseline['avg_victim_slowdown'].iloc[0]),
                           color=COLORS['baseline'], linewidth=0.8,
                           linestyle='--', label='No bully')
                baseline_plotted = True
            has_data = True

    # SST-Macro curve
    if not sstmacro_df.empty:
        sst = sstmacro_df.sort_values('load_fraction')
        ax.plot(sst['load_fraction'], sst['avg_victim_slowdown'],
                color=COLORS['sst_macro'], marker=MARKERS['sst_macro'],
                markersize=4, linewidth=1.5, linestyle='--',
                label=LABELS['sst_macro'])
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, f'No data\n({system})',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='gray')

    ax.set_xlabel("Bully load fraction")
    ax.set_ylabel("Avg victim slowdown")
    ax.set_title(title, fontsize=8)
    ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle=':')
    if has_data:
        ax.legend(loc='upper left', fontsize=6.5)


def fig_three_panel(output_dir: Path):
    """Main 3-panel figure: dragonfly + fat-tree + torus3d."""
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, FIGH))
    fig.subplots_adjust(wspace=0.35)

    panels = [
        ('frontier',   ['minimal', 'valiant'], '(a) Dragonfly (Frontier)'),
        ('lassen',     ['minimal', 'ecmp'],    '(b) Fat-tree (Lassen)'),
        ('bluewaters', [None],                 '(c) Torus-3D (Blue Waters)'),
    ]

    for ax, (system, routings, title) in zip(axes, panels):
        raps_df = load_raps_data(system)
        sst_df = load_sstmacro_data(system)
        plot_panel(ax, raps_df, sst_df,
                   system=system, routings=routings, title=title)

    out_path = output_dir / "interference_validation.pdf"
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def fig_stall_ratio(output_dir: Path):
    """3-panel stall-ratio comparison across all systems."""
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, FIGH))
    fig.subplots_adjust(wspace=0.35)

    panels = [
        ('frontier',   ['minimal', 'valiant'], '(a) Dragonfly stall ratio'),
        ('lassen',     ['minimal', 'ecmp'],    '(b) Fat-tree stall ratio'),
        ('bluewaters', [None],                 '(c) Torus-3D stall ratio'),
    ]

    for ax, (system, routings, title) in zip(axes, panels):
        raps_df = load_raps_data(system)
        if raps_df.empty:
            ax.text(0.5, 0.5, f'No data ({system})', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(title, fontsize=8)
            continue
        for routing in routings:
            label_key = routing if routing else 'dor_xyz'
            subset = raps_df[raps_df['routing_label'] == (routing or 'dor_xyz')]
            if subset.empty:
                continue
            subset = subset[subset['bully_nodes'] > 0].sort_values('load_fraction')
            if subset.empty:
                continue
            color = COLORS.get(label_key, '#333333')
            marker = MARKERS.get(label_key, 'o')
            ax.plot(subset['load_fraction'], subset['avg_victim_stall_ratio'],
                    color=color, marker=marker, markersize=4,
                    linewidth=1.5, label=LABELS.get(label_key, label_key))
        ax.set_xlabel("Bully load fraction")
        ax.set_ylabel("Avg victim stall ratio")
        ax.set_title(title, fontsize=8)
        ax.legend(loc='upper left', fontsize=6.5)

    out_path = output_dir / "interference_stall_ratio.pdf"
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot NRAPS bully-victim interference validation figures")
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR,
                        help="Output directory for figures")
    parser.add_argument('--stall-ratio', action='store_true',
                        help="Also generate stall ratio figure")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating interference validation figures...")
    print(f"  NRAPS data   : {INTERFERENCE_DIR}")
    print(f"  SST-Macro data: {SSTMACRO_DIR}")

    # Main 3-panel figure
    fig_three_panel(args.output_dir)

    # Optional stall ratio figure
    if args.stall_ratio:
        fig_stall_ratio(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
