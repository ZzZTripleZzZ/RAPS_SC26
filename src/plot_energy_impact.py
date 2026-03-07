#!/usr/bin/env python3
"""
Energy Impact Visualization  (SC-paper single-column style)
============================================================
Generates one figure per metric per system — no subfigures.
Figure size: 3.5 × 2.0 in  (~7:4 ratio, single IEEE/ACM column).

Usage:
    python src/plot_energy_impact.py
    python src/plot_energy_impact.py --systems frontier lassen
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UC_BASE = PROJECT_ROOT / "output" / "use_cases"
OUT_DIR  = PROJECT_ROOT / "output" / "figures"

# ── SC paper style ─────────────────────────────────────────────────────────
FIGW = 3.5   # single-column width  (inches)
FIGH = 2.3   # slightly taller for rotated x-tick labels
DPI  = 300

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

# ColorBrewer Dark2 — colorblind-friendly
ROUTE_PAL  = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A']
PLACE_PAL  = ['#1B9E77', '#D95F02', '#7570B3']
SCHED_PAL  = ['#D95F02', '#1B9E77', '#7570B3']
ENERGY_PAL = ['#4DAF4A', '#1B9E77', '#D95F02', '#7570B3']  # green = ideal

BAR_W = 0.52

# System colors and labels for multi-system comparison figures
SYS_COLORS_CMP = {
    'frontier':   '#D95F02',   # orange
    'lassen':     '#1B9E77',   # green
    'bluewaters': '#7570B3',   # purple
}
SYS_SHORT = {
    'frontier':   'Frontier',
    'lassen':     'Lassen',
    'bluewaters': 'Blue Waters',
}

_SYS_TAG = {
    'frontier':   'Frontier (dragonfly)',
    'lassen':     'Lassen (fat-tree)',
    'bluewaters': 'Blue Waters (torus3d)',
}

# Track files written in the current run (used by save_main_figures).
_generated_this_run: set = set()


# ── data loading ────────────────────────────────────────────────────────────

def load_uc(system: str, uc_num: int):
    """Load UC results, preferring heavy dir if complete, else falling back to standard."""
    csv_names = {
        1: "uc1_routing_results.csv",
        2: "uc2_scheduling_results.csv",
        3: "uc3_placement_results.csv",
        4: "uc4_energy_results.csv",
    }
    # Minimum expected rows per UC (all configs must be present to use that file).
    # UC4 fat-tree (lassen): 4 variants (no_cong + minimal + ecmp + adaptive).
    # UC4 dragonfly (frontier): 4 variants (no_cong + minimal + ugal + valiant).
    # UC4 torus3d (bluewaters): 2 variants (no_cong + dor_xyz).
    # UC1 torus3d: 1 variant (dor_xyz only).
    _FAT_TREE_SYSTEMS = {'lassen', 'mit_supercloud', 'setonix', 'marconi100'}
    _TORUS3D_SYSTEMS  = {'bluewaters'}
    if system in _TORUS3D_SYSTEMS:
        uc4_rows = 2
        uc1_rows = 1
    elif system in _FAT_TREE_SYSTEMS:
        uc4_rows = 4
        uc1_rows = 3
    else:
        uc4_rows = 4
        uc1_rows = 3
    min_rows = {1: uc1_rows, 2: 3, 3: 3, 4: uc4_rows}
    csv_file = csv_names[uc_num]
    for suffix in [f"{system}_n1000_heavy", f"{system}_n1000"]:
        d = UC_BASE / suffix
        p = d / csv_file
        if p.exists():
            try:
                df = pd.read_csv(p)
                # Keep last occurrence per label (handles duplicate rows from
                # multiple job runs appending to the same CSV).
                df = df.drop_duplicates(subset=['label'], keep='last')
                if len(df) < min_rows[uc_num]:
                    print(f"  [SKIP-INCOMPLETE] {suffix}/{csv_file}: "
                          f"{len(df)}/{min_rows[uc_num]} rows, trying next dir")
                    continue
                df['_source'] = d.name
                return df
            except Exception as e:
                print(f"  [WARN] cannot read {p}: {e}")
    return None


def strip_prefix(labels, prefix):
    return [l.replace(f"{prefix}_", "", 1) for l in labels]


def _energy_per_job(df):
    completed = df['jobs_completed'].replace(0, np.nan)
    return (df['total_energy_joules'] / 1e6) / completed


# ── single-panel bar figure ─────────────────────────────────────────────────

def _auto_fmt(v):
    """Adaptive annotation format: enough precision to distinguish nearby values.

    Boundaries:
      >= 100 → integer        e.g. "1760"   (makespan in minutes)
      >= 10  → 1 decimal      e.g. "22.1"   (energy MJ)
      >= 2   → 1 decimal      e.g. "3.5"    (overhead %, congestion > 2)
      >= 1   → 3 decimals     e.g. "1.071"  (slowdown factor near 1.0)
      < 1    → 3 decimals     e.g. "0.322"  (congestion ratio < 1)
    """
    if v == 0:
        return '0'
    abs_v = abs(v)
    if abs_v >= 100:
        return f'{v:.0f}'
    if abs_v >= 10:
        return f'{v:.1f}'
    if abs_v >= 2:
        return f'{v:.1f}'
    if abs_v >= 1:
        return f'{v:.3f}'   # slowdown / small overhead — preserve 3 sig figs above 1
    return f'{v:.3f}'


_LABEL_ABBREV = {
    'no_congestion': 'no-cong',
    'fcfs+firstfit': 'fcfs+ff',
    'adaptive': 'macro adaptive',
}


def _shorten_labels(labels):
    return [_LABEL_ABBREV.get(l, l) for l in labels]


def _bar_fig(labels, values, colors, ylabel, fname,
             baseline=None, yscale='linear', ymin=None,
             xlabel=None, sys_tag=None):
    """Save one bar chart as a single-panel SC-paper figure.

    sys_tag: short system identifier shown as top-right annotation,
             e.g. 'Frontier (dragonfly)' or 'Lassen (fat-tree)'.
    xlabel:  optional x-axis label (e.g. 'Routing algorithm').
    """
    labels = _shorten_labels(labels)
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor='none',
                  width=BAR_W, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    if baseline is not None:
        ax.axhline(baseline, color='#444', ls='--', lw=1.3, alpha=0.8)

    # Value annotations above each bar
    v_max = max(values) if values else 1.0
    ymin_val = ymin if ymin is not None else 0.0
    v_range = v_max - ymin_val
    for b, v in zip(bars, values):
        if yscale == 'log':
            ypos = v * 1.20
        else:
            ypos = v + v_range * 0.04
        ax.text(b.get_x() + b.get_width() / 2, ypos,
                _auto_fmt(v), ha='center', va='bottom', fontsize=7)

    # Explicit y-axis limits so annotations are never clipped
    if yscale == 'linear':
        top = v_max + v_range * 0.22
        ax.set_ylim(top=top)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
    elif yscale == 'log':
        ax.set_ylim(top=v_max * 1.6)

    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_yscale(yscale)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')

    # System identifier tag — top-right corner in italic
    if sys_tag:
        ax.text(0.98, 0.97, sys_tag, transform=ax.transAxes,
                ha='right', va='top', fontsize=6.5, style='italic',
                color='#333',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc',
                          alpha=0.80, linewidth=0.6))

    fig.tight_layout(pad=0.4)
    out = OUT_DIR / Path(fname).with_suffix('.png').name
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    _generated_this_run.add(out)
    print(f"  Saved {out.name}")


# ── UC1: Routing Impact ──────────────────────────────────────────────────────

def fig_routing(systems=('frontier', 'lassen')):
    for sys in systems:
        df = load_uc(sys, 1)
        if df is None:
            print(f"  [SKIP] UC1 {sys}: no data")
            continue

        labels = strip_prefix(df['label'].tolist(), 'UC1')
        pal = ROUTE_PAL[:len(labels)]
        tag = _SYS_TAG.get(sys, sys)
        xlab = 'Routing algorithm'

        epj = _energy_per_job(df).tolist()
        _bar_fig(labels, epj, pal,
                 'Energy / job (MJ)', f'uc1_{sys}_epj.png',
                 ymin=min(epj) * 0.98, xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['avg_job_slowdown'].tolist(), pal,
                 'Avg job slowdown', f'uc1_{sys}_slowdown.png',
                 baseline=1.0, ymin=0.98, xlabel=xlab, sys_tag=tag)

        yscale = 'log' if df['avg_congestion'].max() > 3 else 'linear'
        _bar_fig(labels, df['avg_congestion'].tolist(), pal,
                 'Avg link overload ratio', f'uc1_{sys}_congestion.png',
                 yscale=yscale, xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['dilated_pct'].tolist(), pal,
                 'Jobs slowed (%)', f'uc1_{sys}_dilated.png',
                 xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['avg_stall_ratio'].tolist(), pal,
                 'Avg stall/pkt ratio', f'uc1_{sys}_stall.png',
                 xlabel=xlab, sys_tag=tag)


# ── UC3: Placement Impact ────────────────────────────────────────────────────

def fig_placement(systems=('frontier', 'lassen')):
    for sys in systems:
        df = load_uc(sys, 3)
        if df is None:
            print(f"  [SKIP] UC3 {sys}: no data")
            continue

        labels = strip_prefix(df['label'].tolist(), 'UC3')
        pal = PLACE_PAL[:len(labels)]
        tag = _SYS_TAG.get(sys, sys)
        xlab = 'Placement strategy'

        epj3 = _energy_per_job(df).tolist()
        _bar_fig(labels, epj3, pal,
                 'Energy / job (MJ)', f'uc3_{sys}_epj.png',
                 ymin=min(epj3) * 0.98, xlabel=xlab, sys_tag=tag)

        gl_ratio = df['global_local_ratio'].tolist()
        if any(v > 0 for v in gl_ratio):
            _bar_fig(labels, gl_ratio, pal,
                     'Global traffic fraction', f'uc3_{sys}_locality.png',
                     xlabel=xlab, sys_tag=tag)
        else:
            _bar_fig(labels, df['avg_hop_count'].tolist(), pal,
                     'Avg hop count', f'uc3_{sys}_locality.png',
                     xlabel=xlab, sys_tag=tag)

        yscale = 'log' if df['avg_congestion'].max() > 3 else 'linear'
        _bar_fig(labels, df['avg_congestion'].tolist(), pal,
                 'Avg link overload ratio', f'uc3_{sys}_congestion.png',
                 yscale=yscale, xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['dilated_pct'].tolist(), pal,
                 'Jobs slowed (%)', f'uc3_{sys}_dilated.png',
                 xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['avg_stall_ratio'].tolist(), pal,
                 'Avg stall/pkt ratio', f'uc3_{sys}_stall.png',
                 xlabel=xlab, sys_tag=tag)


# ── UC2: Scheduling Impact ───────────────────────────────────────────────────

def fig_scheduling(systems=('frontier', 'lassen')):
    for sys in systems:
        df = load_uc(sys, 2)
        if df is None:
            print(f"  [SKIP] UC2 {sys}: no data")
            continue

        labels = strip_prefix(df['label'].tolist(), 'UC2')
        pal = SCHED_PAL[:len(labels)]
        tag = _SYS_TAG.get(sys, sys)
        xlab = 'Scheduling policy'

        epj2 = _energy_per_job(df).tolist()
        _bar_fig(labels, epj2, pal,
                 'Energy / job (MJ)', f'uc2_{sys}_epj.png',
                 ymin=min(epj2) * 0.98, xlabel=xlab, sys_tag=tag)

        sld2 = df['avg_job_slowdown'].tolist()
        _bar_fig(labels, sld2, pal,
                 'Avg job slowdown', f'uc2_{sys}_slowdown.png',
                 baseline=1.0, ymin=0.98, xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['avg_wait_time'].tolist(), pal,
                 'Avg wait time (s)', f'uc2_{sys}_wait.png',
                 xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['jobs_completed'].tolist(), pal,
                 'Jobs completed', f'uc2_{sys}_throughput.png',
                 xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['avg_stall_ratio'].tolist(), pal,
                 'Avg stall/pkt ratio', f'uc2_{sys}_stall.png',
                 xlabel=xlab, sys_tag=tag)


# ── UC4: Energy Baseline ─────────────────────────────────────────────────────

def fig_uc4_baseline(systems=('frontier', 'lassen')):
    for sys in systems:
        df = load_uc(sys, 4)
        if df is None:
            print(f"  [SKIP] UC4 {sys}: no data")
            continue

        labels = strip_prefix(df['label'].tolist(), 'UC4')
        pal = ENERGY_PAL[:len(labels)]
        tag = _SYS_TAG.get(sys, sys)
        xlab = 'Routing algorithm'
        energy_mj = df['total_energy_joules'].values / 1e6
        baseline_e = float(energy_mj[0])

        _bar_fig(labels, energy_mj.tolist(), pal,
                 'Total energy (MJ)', f'uc4_{sys}_energy.png',
                 baseline=baseline_e, xlabel=xlab, sys_tag=tag)

        makespan_min = (df['simulated_seconds'].values / 60).tolist()
        # Zoom y-axis: differences are small relative to absolute scale
        ms_min = min(makespan_min)
        ms_zoom = ms_min * 0.98   # show 2% below minimum
        _bar_fig(labels, makespan_min, pal,
                 'Makespan (min)', f'uc4_{sys}_makespan.png',
                 baseline=float(makespan_min[0]), ymin=ms_zoom,
                 xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['dilated_pct'].tolist(), pal,
                 'Jobs slowed (%)', f'uc4_{sys}_dilated.png',
                 xlabel=xlab, sys_tag=tag)

        overhead = ((energy_mj - baseline_e) / baseline_e * 100).tolist()
        _bar_fig(labels, overhead, pal,
                 'Energy overhead (%)', f'uc4_{sys}_overhead.png',
                 xlabel=xlab, sys_tag=tag)

        _bar_fig(labels, df['avg_stall_ratio'].tolist(), pal,
                 'Avg stall/pkt ratio', f'uc4_{sys}_stall.png')


# ── Multi-system comparison figures ──────────────────────────────────────────

def _get_val(df, uc_prefix, label_key, col):
    """Return a scalar from df where label == '{uc_prefix}_{label_key}', or None."""
    full = f"{uc_prefix}_{label_key}"
    row = df[df['label'] == full]
    if row.empty:
        return None
    return float(row.iloc[0][col])


def _cmp_grouped_bar(groups, systems, values_by_sys, ylabel, fname,
                     baseline=None, ymin=None, yscale='auto'):
    """
    Grouped bar chart: x-axis = groups (policies/strategies),
    bars within each group = systems (one bar per system, system color).
    yscale: 'auto' uses log when values span >10×, 'log', or 'linear'.
    """
    n_groups = len(groups)
    n_sys    = len([s for s in systems if values_by_sys.get(s)])
    w        = 0.22
    pad      = 0.20   # gap between groups
    fig, ax  = plt.subplots(figsize=(FIGW, FIGH))
    x        = np.arange(n_groups) * (n_sys * w + pad)

    all_vals = [v for vv in values_by_sys.values() if vv for v in vv if v is not None and v > 0]
    v_max    = max(all_vals) if all_vals else 1.0
    v_min    = min(all_vals) if all_vals else 1.0
    use_log  = (yscale == 'log') or (yscale == 'auto' and v_max / max(v_min, 1e-9) > 10)

    drawn = 0
    for sys in systems:
        vals = values_by_sys.get(sys)
        if not vals:
            continue
        offset = (drawn - (n_sys - 1) / 2) * w
        color  = SYS_COLORS_CMP[sys]
        bars   = ax.bar(x + offset, vals, width=w, color=color,
                        edgecolor='none', alpha=0.88, label=SYS_SHORT[sys])
        ymin_v = ymin if ymin is not None else 0.0
        v_rng  = v_max - ymin_v
        for b, v in zip(bars, vals):
            if v is None:
                continue
            ypos = v * 1.25 if use_log else v + v_rng * 0.04
            ax.text(b.get_x() + b.get_width() / 2, ypos,
                    _auto_fmt(v), ha='center', va='bottom', fontsize=6)
        drawn += 1

    ax.set_xticks(x)
    ax.set_xticklabels(_shorten_labels(groups), rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    if baseline is not None:
        ax.axhline(baseline, color='#444', ls='--', lw=1.2, alpha=0.8)

    if use_log:
        ax.set_yscale('log')
        ax.set_ylim(bottom=v_min * 0.4, top=v_max * 4.0)
    else:
        ymin_v = ymin if ymin is not None else 0.0
        v_rng  = v_max - ymin_v
        ax.set_ylim(top=v_max + v_rng * 0.28)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)

    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.85,
              handlelength=1.0, handletextpad=0.4, borderpad=0.4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')
    fig.tight_layout(pad=0.4)
    out = OUT_DIR / fname
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    _generated_this_run.add(out)
    print(f"  Saved {Path(fname).name}")


def fig_comparison_routing(systems):
    """
    UC1: avg_job_slowdown for all routing variants, grouped by system.
    Same layout as fig_comparison_energy — flat bars colored by system,
    vertical separators between system groups.
    Two metrics in two separate figures: slowdown and avg_congestion.
    """
    for metric, ylabel, fname_suffix in [
        ('avg_job_slowdown',  'Avg job slowdown',       'slowdown'),
        ('avg_congestion',    'Avg link overload ratio', 'congestion'),
    ]:
        entries    = []   # (sys, routing_label, value)
        sys_spans  = {}   # sys → (start, end)
        for sys in systems:
            df = load_uc(sys, 1)
            if df is None:
                continue
            start = len(entries)
            for _, row in df.iterrows():
                routing = row['label'].replace('UC1_', '')
                entries.append((sys, routing, float(row[metric])))
            sys_spans[sys] = (start, len(entries))

        if not entries:
            print(f"  [SKIP] UC1 comparison {metric}: no data")
            continue

        fig, ax = plt.subplots(figsize=(FIGW, FIGH))
        x      = np.arange(len(entries))
        colors = [SYS_COLORS_CMP[e[0]] for e in entries]
        vals   = [e[2] for e in entries]
        xlbls  = [_LABEL_ABBREV.get(e[1], e[1]) for e in entries]

        # Baseline reference at 1.0 for slowdown
        if metric == 'avg_job_slowdown':
            ax.axhline(1.0, color='#444', ls='--', lw=1.1, alpha=0.7)

        bars = ax.bar(x, vals, color=colors, edgecolor='none', alpha=0.88, width=0.55)

        v_max = max(vals) if vals else 1.0
        v_min = min(v for v in vals if v > 0) if vals else 1.0
        # Use log scale when values span more than 5× (avoids tiny invisible bars)
        use_log = (metric == 'avg_congestion') and (v_max / max(v_min, 1e-9) > 5)
        ymin  = 0.98 if metric == 'avg_job_slowdown' else 0.0

        for b, v in zip(bars, vals):
            if use_log:
                ypos = v * 1.25
            else:
                ypos = v + (v_max - ymin) * 0.04
            ax.text(b.get_x() + b.get_width() / 2, ypos,
                    _auto_fmt(v), ha='center', va='bottom', fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale('log')
            ax.set_ylim(bottom=v_min * 0.5, top=v_max * 3.5)
        else:
            ax.set_ylim(bottom=ymin, top=v_max + (v_max - ymin) * 0.28)

        # System group labels and separators
        for sys, (s, e) in sys_spans.items():
            mid = (s + e - 1) / 2
            label_y = (v_max * 2.8) if use_log else (v_max + (v_max - ymin) * 0.20)
            ax.text(mid, label_y, SYS_SHORT[sys],
                    ha='center', va='top', fontsize=6.5,
                    color=SYS_COLORS_CMP[sys], fontweight='bold')
            if e < len(entries):
                ax.axvline(e - 0.5, color='#bbb', lw=0.8, ls='--')

        ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')
        fig.tight_layout(pad=0.4)
        fname = f'cmp_uc1_routing_{fname_suffix}.png'
        out = OUT_DIR / fname
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        _generated_this_run.add(out)
        print(f"  Saved {fname}")


def fig_comparison_scheduling(systems):
    """UC2: avg_job_slowdown by scheduling policy, grouped bars per system."""
    policies   = ['fcfs', 'fcfs+firstfit', 'sjf']
    values_by_sys = {}
    for sys in systems:
        df = load_uc(sys, 2)
        if df is None:
            continue
        vals = [_get_val(df, 'UC2', p, 'avg_job_slowdown') for p in policies]
        if any(v is not None for v in vals):
            values_by_sys[sys] = vals

    if not values_by_sys:
        print("  [SKIP] UC2 comparison: no data")
        return

    _cmp_grouped_bar(policies, systems, values_by_sys,
                     'Avg job slowdown', 'cmp_uc2_scheduling_slowdown.png',
                     baseline=1.0, ymin=0.98)


def fig_comparison_placement(systems):
    """UC3: avg_stall_ratio by placement strategy, grouped bars per system."""
    strategies    = ['contiguous', 'random', 'hybrid']
    values_by_sys = {}
    for sys in systems:
        df = load_uc(sys, 3)
        if df is None:
            continue
        vals = [_get_val(df, 'UC3', s, 'avg_stall_ratio') for s in strategies]
        if any(v is not None for v in vals):
            values_by_sys[sys] = vals

    if not values_by_sys:
        print("  [SKIP] UC3 comparison: no data")
        return

    _cmp_grouped_bar(strategies, systems, values_by_sys,
                     'Avg stall/pkt ratio', 'cmp_uc3_placement_stall.png')


def fig_comparison_energy(systems):
    """
    UC4: energy overhead % for all routing variants, grouped by system.
    For each variant, shows standard load (lighter) and heavy load (solid) side by side.
    """
    UC4_CSV = "uc4_energy_results.csv"

    def _load_raw(system, suffix):
        p = UC_BASE / suffix / UC4_CSV
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p)
            return df if len(df) >= 2 else None
        except Exception:
            return None

    def _overheads(df):
        if df is None:
            return {}
        labels  = df['label'].tolist()
        energies = df['total_energy_joules'].values
        base_e  = energies[0]
        return {lbl.replace('UC4_', ''): (e - base_e) / base_e * 100
                for lbl, e in zip(labels[1:], energies[1:])}

    # Collect (sys, routing, std_overhead, heavy_overhead) in order
    entries   = []
    sys_spans = {}

    for sys in systems:
        std_data   = _overheads(_load_raw(sys, f"{sys}_n1000"))
        heavy_data = _overheads(_load_raw(sys, f"{sys}_n1000_heavy"))
        routings = list(heavy_data.keys()) if heavy_data else list(std_data.keys())
        if not routings:
            continue
        start = len(entries)
        for r in routings:
            entries.append((sys, r, std_data.get(r), heavy_data.get(r)))
        sys_spans[sys] = (start, len(entries))

    if not entries:
        print("  [SKIP] UC4 energy comparison: no data")
        return

    all_vals = [v for _, _, s, h in entries for v in (s, h) if v is not None]
    v_max = max(all_vals)

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    n = len(entries)
    x = np.arange(n)
    w = 0.33

    for i, (sys, routing, std_v, heavy_v) in enumerate(entries):
        col = SYS_COLORS_CMP[sys]
        if std_v is not None:
            ax.bar(x[i] - w / 2, std_v, width=w, color=col,
                   edgecolor='none', alpha=0.40)
        if heavy_v is not None:
            ax.bar(x[i] + w / 2, heavy_v, width=w, color=col,
                   edgecolor='none', alpha=0.88)
        # Smart label placement: offset when std and heavy are close
        gap = v_max * 0.025
        if std_v is not None and heavy_v is not None:
            if abs(heavy_v - std_v) < v_max * 0.06:
                # Values too close — stack labels vertically
                lo, hi = min(std_v, heavy_v), max(heavy_v, std_v)
                ax.text(x[i] - w / 2, lo + gap,
                        _auto_fmt(std_v) + '%', ha='center', va='bottom',
                        fontsize=5.5, color='#666')
                ax.text(x[i] + w / 2, hi + gap + v_max * 0.055,
                        _auto_fmt(heavy_v) + '%', ha='center', va='bottom',
                        fontsize=5.5)
            else:
                ax.text(x[i] - w / 2, std_v + gap,
                        _auto_fmt(std_v) + '%', ha='center', va='bottom',
                        fontsize=5.5, color='#666')
                ax.text(x[i] + w / 2, heavy_v + gap,
                        _auto_fmt(heavy_v) + '%', ha='center', va='bottom',
                        fontsize=5.5)
        else:
            if std_v is not None:
                ax.text(x[i] - w / 2, std_v + gap,
                        _auto_fmt(std_v) + '%', ha='center', va='bottom',
                        fontsize=5.5, color='#666')
            if heavy_v is not None:
                ax.text(x[i] + w / 2, heavy_v + gap,
                        _auto_fmt(heavy_v) + '%', ha='center', va='bottom',
                        fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels([_LABEL_ABBREV.get(e[1], e[1]) for e in entries],
                       rotation=30, ha='right')
    ax.set_ylabel('Energy overhead vs no-congestion (%)')
    ax.set_ylim(top=v_max * 1.40)

    # Vertical separators and system labels
    for sys, (s, e) in sys_spans.items():
        mid = (s + e - 1) / 2
        ax.text(mid, v_max * 1.32, SYS_SHORT[sys],
                ha='center', va='top', fontsize=6.5,
                color=SYS_COLORS_CMP[sys], fontweight='bold')
        if e < len(entries):
            ax.axvline(e - 0.5, color='#bbb', lw=0.8, ls='--')

    # Legend for load types — place inside plot below the Frontier bars
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#888', alpha=0.40, label='Standard load'),
        Patch(facecolor='#888', alpha=0.88, label='Heavy load'),
    ], fontsize=6, loc='center left')

    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.97)  # ensure right-side system label not clipped
    out = OUT_DIR / 'cmp_uc4_energy_overhead.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    _generated_this_run.add(out)
    print(f"  Saved cmp_uc4_energy_overhead.png")


def fig_comparison(systems):
    """Generate all multi-system comparison figures."""
    print("\nMulti-system comparison figures:")
    fig_comparison_routing(systems)
    fig_comparison_scheduling(systems)
    fig_comparison_placement(systems)
    fig_comparison_energy(systems)


# ── main figures curation ────────────────────────────────────────────────────

# Each entry: (source_filename, dest_filename)
# Preferred system order: frontier first (always available), then lassen if present.
# When lassen data is available, the lassen versions are preferred for UC1/UC3/UC4
# because fat-tree + high congestion shows more dramatic effects.
_MAIN_FIGURES = [
    # UC1: Routing — job slowdown (baseline=1.0 reference line) + Cassini stall ratio
    # lassen preferred (fat-tree high congestion); frontier shown separately (dragonfly)
    ('uc1_{sys}_slowdown.png',  'uc1_routing_slowdown.png'),
    ('uc1_{sys}_stall.png',     'uc1_routing_stall.png'),
    # UC1 frontier (dragonfly: minimal/ugal/valiant) — pairs with chord diagram
    ('uc1_frontier_slowdown.png', 'uc1_routing_slowdown_frontier.png'),
    ('uc1_frontier_stall.png',    'uc1_routing_stall_frontier.png'),
    # UC2: Scheduling — wait time (core scheduling metric) + stall ratio (SJF paradox)
    ('uc2_{sys}_wait.png',      'uc2_scheduling_wait.png'),
    ('uc2_{sys}_stall.png',     'uc2_scheduling_stall.png'),
    # UC3: Placement — stall ratio (11× gap contiguous vs random) + locality mechanism
    ('uc3_{sys}_stall.png',     'uc3_placement_stall.png'),
    ('uc3_{sys}_locality.png',  'uc3_placement_locality.png'),
    # UC3 frontier (dragonfly: contiguous/random/hybrid) — now complete with heavy data
    ('uc3_frontier_stall.png',    'uc3_placement_stall_frontier.png'),
    ('uc3_frontier_locality.png', 'uc3_placement_locality_frontier.png'),
    # UC4: Energy — overhead % (the energy-tax thesis) + makespan extension
    ('uc4_{sys}_overhead.png',  'uc4_energy_overhead.png'),
    ('uc4_{sys}_makespan.png',  'uc4_energy_makespan.png'),
    # Multi-system comparison figures (all 3 topologies)
    ('cmp_uc1_routing_slowdown.png',    'cmp_uc1_routing_slowdown.png'),
    ('cmp_uc1_routing_congestion.png',  'cmp_uc1_routing_congestion.png'),
    ('cmp_uc2_scheduling_slowdown.png', 'cmp_uc2_scheduling_slowdown.png'),
    ('cmp_uc3_placement_stall.png',     'cmp_uc3_placement_stall.png'),
    ('cmp_uc4_energy_overhead.png',     'cmp_uc4_energy_overhead.png'),
]


def save_main_figures(systems):
    """Copy the curated set of figures into output/figures/main/.

    For each slot, tries each system in order (lassen preferred for high-congestion
    drama, frontier as fallback) and copies the first file that exists.
    """
    main_dir = OUT_DIR / 'main'
    main_dir.mkdir(parents=True, exist_ok=True)

    # Prefer lassen then frontier for UC1/UC3/UC4 (higher congestion contrast);
    # prefer frontier for UC2 (lassen UC2 may not be available yet).
    sys_order = {
        'uc1': ['lassen', 'frontier'],
        'uc2': ['frontier', 'lassen'],
        'uc3': ['lassen', 'frontier'],
        'uc4': ['lassen', 'frontier'],
    }

    print("\nCurating main figures →", main_dir)
    copied = 0
    for src_template, dest_name in _MAIN_FIGURES:
        chosen = None
        if '{sys}' not in src_template:
            # Explicit filename — no system substitution needed
            src = OUT_DIR / src_template
            if src in _generated_this_run:
                chosen = (src, src_template.split('_')[1])  # e.g. 'frontier'
        else:
            uc_key = src_template.split('_')[0]   # 'uc1', 'uc2', ...
            order = sys_order.get(uc_key, systems)
            for sys in order:
                if sys not in systems:
                    continue
                src = OUT_DIR / src_template.format(sys=sys)
                # Only use files generated in this run (avoids stale partial figures).
                if src in _generated_this_run:
                    chosen = (src, sys)
                    break
        if chosen is None:
            print(f"  [SKIP] {dest_name}: not generated this run")
            continue
        src, sys = chosen
        dst = main_dir / dest_name
        shutil.copy2(src, dst)
        print(f"  {src.name} ({sys}) → main/{dest_name}")
        copied += 1

    print(f"  {copied}/{len(_MAIN_FIGURES)} figures saved to main/")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Energy impact figures (SC single-column style)')
    parser.add_argument('--systems', nargs='+', default=['frontier', 'lassen'])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    systems = args.systems

    print("=" * 60)
    print("Energy Impact Visualization  (SC paper style)")
    print("=" * 60)
    print()

    fig_routing(systems)
    fig_placement(systems)
    fig_scheduling(systems)
    fig_uc4_baseline(systems)
    fig_comparison(systems)

    save_main_figures(systems)

    print()
    print("Done. Figures written to:", OUT_DIR)


if __name__ == '__main__':
    main()
