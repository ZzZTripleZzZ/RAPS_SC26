#!/usr/bin/env python3
"""
Benchmark comparison: RAPS vs GPCNeT real measurements
=======================================================
Generates two panels for the SC26 paper:
  1. Simulation speedup: RAPS wall-clock time vs real time (log scale)
  2. Network accuracy: RAPS predicted congestion vs GPCNeT measured BW degradation

Usage:
    python src/plot_benchmark_comparison.py
    python src/plot_benchmark_comparison.py --no-gpcnet   # skip if GPCNeT not yet run

Inputs:
    output/frontier_scaling/results.csv   — RAPS benchmark data (already complete)
    output/gpcnet/network_test_*.out      — GPCNeT isolated baseline (after job runs)
    output/gpcnet/network_load_test_*.out — GPCNeT under congestion (after job runs)
    output/sstmacro/sstmacro_run*.out     — SST-Macro timing (optional)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAPS_CSV     = PROJECT_ROOT / "output" / "frontier_scaling" / "results.csv"
GPCNET_DIR   = PROJECT_ROOT / "output" / "gpcnet"
SSTMACRO_DIR = PROJECT_ROOT / "output" / "sstmacro"
OUT_DIR      = PROJECT_ROOT / "output" / "figures"

# ── Style ────────────────────────────────────────────────────────────────────
FIGW, FIGH = 3.5, 2.2
DPI = 300
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.framealpha': 0.85,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.axisbelow': True, 'figure.dpi': DPI, 'savefig.dpi': DPI,
})
C_RAPS     = '#D95F02'   # orange — Frontier/RAPS (consistent with other figures)
C_GPCNET   = '#E6AB02'   # amber  — GPCNeT real measurement
C_SSTMACRO = '#555555'   # dark grey — SST-Macro
C_THEORY   = '#888888'   # grey   — M/D/1 theory line


# ── RAPS data ─────────────────────────────────────────────────────────────────

def load_raps():
    df = pd.read_csv(RAPS_CSV)
    df = df[df['status'] == 'OK'].copy()
    # Keep last occurrence per (system, node_count, delta_t, repeat) to handle
    # duplicate rows from multiple job runs appending to the same CSV.
    df = df.drop_duplicates(subset=['system', 'node_count', 'delta_t', 'repeat'], keep='last')
    df['sim_hours'] = df['ticks'] * df['delta_t'] / 3600
    return df


def raps_speedup_table(df, system='frontier'):
    """Return mean speedup for dt=1 (12h simulated) for the given system."""
    sel = df[(df['system'] == system) & (df['delta_t'] == 1.0)]
    tbl = sel.groupby('node_count').agg(
        speedup_mean=('speedup', 'mean'),
        speedup_std=('speedup', 'std'),
        wall_mean=('sim_wall_s', 'mean'),
    ).reset_index()
    return tbl


def raps_congestion_table(df):
    """Return mean avg_congestion for Frontier dt=1 runs."""
    sel = df[(df['system'] == 'frontier') & (df['delta_t'] == 1.0)]
    tbl = sel.groupby('node_count').agg(
        cong_mean=('avg_congestion', 'mean'),
        cong_std=('avg_congestion', 'std'),
    ).reset_index()
    return tbl


# ── GPCNeT parsing ────────────────────────────────────────────────────────────

_TABLE_ROW = re.compile(
    r'\|\s+(.+?)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+(\S+)\s+\|'
)

def _parse_gpcnet_table(text):
    """Parse a GPCNeT results table from stdout text."""
    rows = []
    for m in _TABLE_ROW.finditer(text):
        name, avg, p99, unit = m.groups()
        name = name.strip()
        if name in ('Name', '---', '') or name.startswith('-'):
            continue
        rows.append({'name': name, 'avg': float(avg), 'p99': float(p99), 'unit': unit})
    return pd.DataFrame(rows)


def load_gpcnet_isolated(node_count):
    """Parse network_test output for the given node count."""
    candidates = sorted(GPCNET_DIR.glob(f"network_test_{node_count}n.out"))
    if not candidates:
        return None
    text = candidates[-1].read_text()
    return _parse_gpcnet_table(text)


def load_gpcnet_congested(node_count):
    """
    Parse network_load_test output.
    Returns dict with baseline and loaded BW for key tests, plus degradation ratio.
    """
    candidates = sorted(GPCNET_DIR.glob(f"network_load_test_{node_count}n.out"))
    if not candidates:
        return None
    text = candidates[-1].read_text()

    # network_load_test prints two table sections:
    # "Isolated Network Tests" then a loaded section whose header varies:
    #   "Network Tests with Congestors" (older GPCNeT)
    #   "Network Tests running with Congestion Tests" (newer GPCNeT / Frontier runs)
    _LOADED_HDR = r'Network Tests\s+(?:with\s+Congestors|running\s+with\s+Congestion\s+Tests)'
    isolated_match = re.search(
        r'Isolated Network Tests.*?(?=' + _LOADED_HDR + r'|\Z)',
        text, re.DOTALL | re.IGNORECASE)
    loaded_match = re.search(
        _LOADED_HDR + r'.*',
        text, re.DOTALL | re.IGNORECASE)

    if not isolated_match or not loaded_match:
        # Fallback: parse everything and split by index
        return None

    df_iso  = _parse_gpcnet_table(isolated_match.group())
    df_load = _parse_gpcnet_table(loaded_match.group())

    if df_iso.empty or df_load.empty:
        return None

    # Match rows by name (same order expected)
    results = []
    for _, row_iso in df_iso.iterrows():
        name = row_iso['name']
        match = df_load[df_load['name'] == name]
        if match.empty:
            continue
        row_load = match.iloc[0]
        bw_base  = row_iso['avg']
        bw_load  = row_load['avg']
        if bw_base > 0 and 'BW' in name:
            degradation = (bw_base - bw_load) / bw_base * 100
        elif bw_base > 0 and 'Lat' in name:
            degradation = (bw_load - bw_base) / bw_base * 100  # latency increase
        else:
            degradation = 0.0
        results.append({
            'name': name, 'unit': row_iso['unit'],
            'baseline': bw_base, 'loaded': bw_load,
            'degradation_pct': degradation,
        })
    return pd.DataFrame(results)


# ── M/D/1 theory ─────────────────────────────────────────────────────────────

def md1_bw_degradation(rho):
    """
    Fractional throughput loss from M/D/1 queuing.
    rho: link utilization (0 < rho < 1).
    Returns degradation as percentage (0-100).
    """
    rho = np.asarray(rho, dtype=float)
    rho = np.clip(rho, 0.0, 0.999)
    slowdown = 1 + rho**2 / (2 * (1 - rho))
    return (1 - 1.0 / slowdown) * 100


# ── SST-Macro timing ──────────────────────────────────────────────────────────

# GPCNeT network_test simulated duration (approximate, measured from real runs)
GPCNET_TRACE_DURATION_S = 30.0

def _parse_timing_file(path):
    """Parse a timing_Xn.txt file. Returns wall_sec of the first successful run."""
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    for line in reversed(lines):
        m = re.search(r'wall_sec=(\d+).*rc=0', line)
        if m:
            return int(m.group(1))
    return None


def load_sstmacro_timings():
    """
    Returns dict {node_count: wall_sec} for SST-Macro runs that completed.
    Keys: 100, 1000
    """
    results = {}
    for nc in (100, 1000):
        wall = _parse_timing_file(SSTMACRO_DIR / f"timing_{nc}n.txt")
        if wall is not None:
            results[nc] = wall
    return results


# ── Figure 1: Simulation Speedup ──────────────────────────────────────────────

C_LASSEN     = '#1B9E77'   # green  — Lassen (fat-tree)
C_BLUEWATERS = '#7570B3'   # purple — Blue Waters (torus3d)


def fig_speedup(df_raps, sstmacro_timings=None):
    """
    Grouped bar chart: RAPS speedup for Frontier, Lassen, and Blue Waters, dt=1.
    """
    tbls = {
        'frontier':   (raps_speedup_table(df_raps, system='frontier'),   C_RAPS,        'Frontier (dragonfly)'),
        'lassen':     (raps_speedup_table(df_raps, system='lassen'),     C_LASSEN,      'Lassen (fat-tree)'),
        'bluewaters': (raps_speedup_table(df_raps, system='bluewaters'), C_BLUEWATERS,  'Blue Waters (torus3d)'),
    }

    # Node counts present in all systems
    nc_sets = [set(t.get('node_count', [])) for t, _, _ in tbls.values()]
    common_nc = sorted(set.intersection(*nc_sets)) if all(nc_sets) else sorted(nc_sets[0])

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    x = np.arange(len(common_nc))
    n_sys = len(tbls)
    w = 0.25
    offsets = np.linspace(-(n_sys-1)/2 * w, (n_sys-1)/2 * w, n_sys)

    for (sys, (tbl, color, label)), offset in zip(tbls.items(), offsets):
        tbl_filt = tbl[tbl['node_count'].isin(common_nc)].set_index('node_count').reindex(common_nc)
        means = tbl_filt['speedup_mean'].values
        stds  = tbl_filt['speedup_std'].fillna(0).values
        bars = ax.bar(x + offset, means, yerr=stds, width=w,
                      color=color, edgecolor='none', alpha=0.88, capsize=2, label=label)
        for b, v, std in zip(bars, means, stds):
            if np.isnan(v):
                continue
            ypos = (v + std) * 1.12 if v > 0 else v * 1.12
            ax.text(b.get_x() + b.get_width()/2, ypos,
                    f'{v:.0f}×', ha='center', va='bottom', fontsize=6, color='#222')

    # SST-Macro reference — shown as text annotation rather than a barely-visible line
    if sstmacro_timings:
        sst_x, sst_y = [], []
        for i, nc in enumerate(common_nc):
            if nc in sstmacro_timings:
                speedup = GPCNET_TRACE_DURATION_S / sstmacro_timings[nc]
                sst_x.append(i)
                sst_y.append(speedup)
        if sst_x:
            ax.plot(sst_x, sst_y, 's--', color=C_SSTMACRO, lw=1.3, ms=7,
                    label='SST-Macro (measured)')
    else:
        ax.axhline(0.03, color='#444444', ls='--', lw=1.3, alpha=0.85,
                   label='SST-Macro (ref: ≈0.03×)')
        # Also label the line at the right edge to avoid confusion
        ax.text(len(common_nc) - 0.45, 0.035, '≈0.03×\nSST-Macro',
                color='#333333', fontsize=6.5, va='bottom', ha='right', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n:,}' for n in common_nc])
    ax.set_xlabel('Node count (12h simulated, Δt=1s)')
    ax.set_ylabel('Speedup over real time (×)')
    ax.set_yscale('log')
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7)
    # Place legend at lower-left to avoid overlapping bar annotations
    ax.legend(loc='lower left', fontsize=7)

    fig.tight_layout(pad=0.5)
    out = OUT_DIR / 'benchmark_speedup.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 2: Network Accuracy ────────────────────────────────────────────────

def fig_accuracy():
    """
    RAPS predicted M/D/1 BW degradation for a single RR job vs GPCNeT 0%.

    Same metric: BW degradation (%) under RR traffic.
    At Frontier n=100 (closest to GPCNeT's 128-node test), RAPS predicts
    <0.2% degradation — consistent with GPCNeT Impact Factor = 1.0× (0%).
    Blue Waters shows higher values because torus3d paths are longer.

    Data: output/rr_validation.json (from src/_run_rr_validation.py)
    """
    import json
    rr_json = PROJECT_ROOT / "output" / "rr_validation.json"
    if not rr_json.exists():
        print("  WARNING: rr_validation.json not found; run src/_run_rr_validation.py first")
        return

    data = json.loads(rr_json.read_text())

    node_counts = [100, 1000, 10000]
    systems = [
        ('frontier',   C_RAPS,        'Frontier (dragonfly)'),
        ('lassen',     C_LASSEN,      'Lassen (fat-tree)'),
        ('bluewaters', C_BLUEWATERS,  'Blue Waters (torus3d)'),
    ]

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    x = np.arange(len(node_counts))
    n_sys = len(systems)
    w = 0.22
    offsets = np.linspace(-(n_sys - 1) / 2 * w, (n_sys - 1) / 2 * w, n_sys)

    for (sys, color, label), offset in zip(systems, offsets):
        sys_data = data.get(sys, {})
        vals = np.array([sys_data.get(str(nc), {}).get("bw_deg_pct", np.nan)
                         for nc in node_counts])
        valid = ~np.isnan(vals)

        bars = ax.bar(x[valid] + offset, vals[valid], width=w,
                      color=color, edgecolor='none', alpha=0.88, label=label)
        for b, v in zip(bars, vals[valid]):
            ypos = v + 0.3 if v >= 0.5 else 0.5
            ax.text(b.get_x() + b.get_width() / 2, ypos,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=6, color=color)

    # GPCNeT reference: Frontier ~128 nodes, RR, Impact Factor = 1.0× → 0% BW degradation
    ax.axhline(0.0, color=C_GPCNET, ls='--', lw=1.4, alpha=0.9, zorder=2,
               label='GPCNeT')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n:,}' for n in node_counts])
    ax.set_xlabel('Node count')
    ax.set_ylabel('Predicted BW degradation (%)')
    ax.set_ylim(-0.5, 14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=6.5,
              handlelength=1.2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7)

    fig.tight_layout(pad=0.5)
    out = OUT_DIR / 'benchmark_accuracy.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 3: RAPS speedup across dt values (existing benchmark sweep) ────────

def fig_speedup_vs_dt(df_raps, sstmacro_timings=None):
    """
    Line plot: RAPS speedup vs delta_t for Frontier and Lassen.
    Frontier: solid lines; Lassen: dashed lines.
    """
    node_colors = ['#1B9E77', '#7570B3', '#E7298A']   # shared by node count

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    # Identify single-repeat points (only 1 OK run → no std, mark specially)
    repeat_counts = df_raps.groupby(['system', 'node_count', 'delta_t']).size()

    for sys, linestyle, marker in [('frontier', '-', 'o'), ('lassen', '--', 's'),
                                    ('bluewaters', ':', '^')]:
        sub_df = df_raps[df_raps['system'] == sys].copy()
        if sub_df.empty:
            continue
        tbl = sub_df.groupby(['node_count', 'delta_t']).agg(
            speedup=('speedup', 'mean')).reset_index()
        node_counts_sorted = sorted(tbl['node_count'].unique())

        sys_label = {'frontier': 'Frontier', 'lassen': 'Lassen',
                     'bluewaters': 'Blue Waters'}.get(sys, sys.capitalize())
        for nc, color in zip(node_counts_sorted, node_colors):
            sub = tbl[tbl['node_count'] == nc].sort_values('delta_t')
            label = f'{sys_label} {nc:,}n'
            ax.plot(sub['delta_t'], sub['speedup'], f'{marker}{linestyle}',
                    color=color, label=label, lw=1.4, ms=4, alpha=0.9)

            # Annotate single-repeat points (incomplete data)
            for _, row in sub.iterrows():
                key = (sys, nc, row['delta_t'])
                if key in repeat_counts and repeat_counts[key] == 1:
                    ax.annotate('(r0)', xy=(row['delta_t'], row['speedup']),
                                xytext=(4, 4), textcoords='offset points',
                                fontsize=6, color=color, alpha=0.8)

            # SST-Macro overlay (Frontier only)
            if sys == 'frontier' and sstmacro_timings and nc in sstmacro_timings:
                sst_speedup = GPCNET_TRACE_DURATION_S / sstmacro_timings[nc]
                min_dt = sub['delta_t'].min()
                ax.plot(min_dt * 0.3, sst_speedup, 'D', color='#333333',
                        ms=7, zorder=5,
                        label=f'SST-Macro {nc:,}n ({sst_speedup:.3f}×)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Simulation Δt (seconds)')
    ax.set_ylabel('Speedup over real time (×)')
    ax.legend(loc='upper left', fontsize=6, ncol=2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7)
    ax.xaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7)

    fig.tight_layout(pad=0.5)
    out = OUT_DIR / 'benchmark_speedup_vs_dt.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark comparison figures')
    parser.add_argument('--no-gpcnet', action='store_true',
                        help='Skip GPCNeT data (not yet available)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Benchmark Comparison Plots")
    print("=" * 60)

    df_raps = load_raps()
    print(f"  RAPS: {len(df_raps)} completed runs loaded")

    sstmacro_timings = load_sstmacro_timings()
    if sstmacro_timings:
        for nc, ws in sorted(sstmacro_timings.items()):
            speedup = GPCNET_TRACE_DURATION_S / ws
            print(f"  SST-Macro {nc}n: wall={ws}s, speedup={speedup:.4f}×")
    else:
        print("  SST-Macro: no timing data yet (using published reference)")

    print("\nGenerating figures...")
    fig_speedup(df_raps, sstmacro_timings)
    fig_accuracy()
    fig_speedup_vs_dt(df_raps, sstmacro_timings)

    # Copy to main/
    main_dir = OUT_DIR / 'main'
    main_dir.mkdir(exist_ok=True)
    import shutil
    for f in ['benchmark_speedup.png', 'benchmark_accuracy.png',
              'benchmark_speedup_vs_dt.png']:
        src = OUT_DIR / f
        if src.exists():
            shutil.copy2(src, main_dir / f)
            print(f"  Copied {f} → main/")

    print(f"\nDone. Figures in {OUT_DIR}")


if __name__ == '__main__':
    main()
