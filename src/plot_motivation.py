#!/usr/bin/env python3
"""
Motivation figures for SC26 paper — Background & Motivation section.
======================================================================
Generates three publication-quality figures (SC single-column style):

  fig_motiv_speed.png      — RAPS faster-than-real-time speedup benchmark (from CSV)
  fig_motiv_congestion.png — Real Lassen inter-job network interference evidence
                              (statistics from Wes's analyze_lassen_congestion.py run
                               on 2019-08-22 → 2019-08-29, hardcoded since CSVs from
                               that script are not committed to the repo)
  fig_motiv_sim_cmp.png    — Simulator positioning: operational speed vs topology fidelity

Usage:
    python src/plot_motivation.py
    python src/plot_motivation.py --csv output/frontier_scaling/results.csv
    python src/plot_motivation.py --skip-congestion   # if Wes CSV unavailable
    python src/plot_motivation.py --skip-sim-cmp      # skip positioning diagram
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "output" / "frontier_scaling" / "results.csv"
OUT_DIR     = ROOT / "output" / "figures"

# ── SC single-column style (matches plot_dt_tradeoff.py / plot_energy_impact.py) ─
FIGW = 3.5   # single-column width  (inches)
FIGH = 2.5   # slightly taller than energy figures
DPI  = 300

# ColorBrewer Dark2 — same palette used across the project
SYS_COLORS   = {"frontier": "#D95F02", "lassen": "#1B9E77"}
SYS_LABELS   = {"frontier": "Frontier (dragonfly)", "lassen": "Lassen (fat-tree)"}
NODE_MARKERS = {100: "o", 1_000: "s", 10_000: "^"}
NODE_LS      = {100: ":", 1_000: "--", 10_000: "-"}
DT_ORDER     = [0.1, 1.0, 10.0, 60.0, 300.0, 600.0]

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


# ── Hardcoded Lassen congestion data (Wes's analyze_lassen_congestion.py) ───
# Source: analyze_lassen_congestion.py on week 2019-08-22 → 2019-08-29,
#   Lassen fat-tree (k=32, 4608 nodes, 12.5 GB/s EDR IB links),
#   5,181 jobs ≥ 2 nodes, 168 hourly snapshots.
# Both patterns used; A2A = lower bound, Stencil-3D = upper bound.
LASSEN_CONGESTION = {
    # Daily peak max link utilization (× link capacity; >1.0 = overloaded)
    "days":    ["Aug 22", "Aug 23", "Aug 24", "Aug 25", "Aug 26", "Aug 27", "Aug 28"],
    "a2a":     [169.9,    184.4,    109.2,    138.7,    97.3,     146.9,    99.8],
    "stencil": [330.0,    342.9,    215.7,    272.0,    179.1,    271.1,    196.8],
    "avg_jobs":[91,       108,      115,       98,       98,       121,      99],
    # Week-level aggregate stats
    "total_jobs":       5181,
    "victim_jobs":      3781,   # max_link_util >= 0.5 during job lifetime
    "bully_jobs":       519,    # top 10% IB TX senders
    "snapshots":        168,
    "congested_snap":   168,    # max_util >= 1.0 (saturated) in 168/168 hours
    "peak_a2a":         184.4,
    "peak_stencil":     342.9,
    "mean_a2a":         91.3,
    "mean_stencil":     170.5,
    "r_jobs_congestion":0.30,   # Pearson r between concurrent jobs and max_link_util
}

# ── Simulator comparison data (literature-based estimates) ───────────────────
# Operational speed: simulation-time / wall-clock-time ratio (speedup > 1 = faster than real-time)
# Packet/event-level simulators are many orders of magnitude slower than real-time for
# large-scale HPC (e.g., Hoefler et al. 2018, Escherich et al. 2020, SST docs).
# Flow-level tools approach real-time but sacrifice topology fidelity.
# These are representative order-of-magnitude estimates; exact values are workload-dependent.
SIMULATORS = [
    # name,               speedup (×),  topo_fidelity (0-3),  scale_nodes,  our_work
    ("SST/Merlin",        3e-4,         3.0,   1_000,  False),
    ("NS-3 (HPC)",        5e-5,         3.0,   500,    False),
    ("CODES",             1e-3,         2.8,   2_000,  False),
    ("SimGrid",           0.8,          1.8,   10_000, False),
    ("LogGOPSim",         50.0,         0.8,   50_000, False),
    ("RAPS\n(this work)", 10_000.0,     2.2,   100_000, True),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig, fname: str):
    out = OUT_DIR / Path(fname).name
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def _ygrid(ax):
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')


def _xgrid(ax):
    ax.xaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')


# ── Figure 1: RAPS Simulation Speed ─────────────────────────────────────────

def plot_speed(csv_path: Path):
    """
    Log-log plot of RAPS simulation speedup vs time quantum Δt.
    Annotates the real-time boundary and a representative 'slower-than-real-time'
    zone for reference, motivating why existing packet-level simulators are unsuitable
    for Decision Layer what-if analysis.
    """
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    df["delta_t"] = df["delta_t"].astype(float).round(3)

    agg = (df.groupby(["system", "node_count", "delta_t"])
             .agg(speedup_mean=("speedup", "mean"),
                  speedup_std =("speedup", "std"))
             .reset_index())

    dt_vals = sorted(agg["delta_t"].unique())

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    # ── shaded "slower than real-time" zone ──
    ax.axhspan(1e-6, 1.0, color='#f0e6e6', alpha=0.55, zorder=0)
    ax.axhline(1.0, color='#cc4444', lw=1.2, ls='-', alpha=0.7, zorder=2)
    ax.text(min(dt_vals) * 1.18, 0.72,
            "Real-time boundary",
            fontsize=7, color='#cc4444', va='top', alpha=0.9)
    ax.text(min(dt_vals) * 1.18, 0.28,
            "← packet-level simulators\n   (SST, NS-3, CODES)",
            fontsize=7, color='#aa3333', va='top', alpha=0.75,
            style='italic')

    # ── RAPS speedup lines ──
    legend_lines = []
    for system in ["frontier", "lassen"]:
        color = SYS_COLORS[system]
        sdf   = agg[agg["system"] == system]
        if sdf.empty:
            continue
        for nc in sorted(sdf["node_count"].unique()):
            marker = NODE_MARKERS.get(nc, "D")
            ls     = NODE_LS.get(nc, "-")
            ndf = (sdf[sdf["node_count"] == nc]
                   .set_index("delta_t")
                   .reindex([d for d in DT_ORDER if d in dt_vals]))
            if ndf.dropna(subset=["speedup_mean"]).empty:
                continue
            ax.errorbar(ndf.index, ndf["speedup_mean"],
                        yerr=ndf["speedup_std"].fillna(0),
                        marker=marker, linestyle=ls, color=color,
                        linewidth=1.8, capsize=3, alpha=0.9, zorder=3)
            legend_lines.append(
                Line2D([0], [0], color=color, marker=marker, linestyle=ls,
                       linewidth=1.8,
                       label=f"{SYS_LABELS[system]},  n={nc:,}"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dt_vals)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}s"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}×" if x >= 1 else f"{x:.2g}×"))
    ax.set_xlabel(r"Time quantum $\Delta t$")
    ax.set_ylabel("Simulation speedup")
    ax.set_ylim(bottom=5e-2)

    ax.legend(handles=legend_lines, fontsize=7, loc="upper left",
              ncol=1, framealpha=0.9)
    _ygrid(ax)
    fig.tight_layout(pad=0.5)
    _save(fig, "fig_motiv_speed.png")


# ── Figure 2: Real Lassen Network Congestion ─────────────────────────────────

def plot_congestion():
    """
    Two-panel figure showing empirical evidence of inter-job network interference
    on Lassen (week 2019-08-22 to 2019-08-28).
    Left panel:  Daily peak max link utilization (A2A vs Stencil-3D).
    Right panel: Job fate breakdown — victim / bully / unaffected jobs.

    Data source: Wes's analyze_lassen_congestion.py on Lassen PM100 dataset.
    """
    d = LASSEN_CONGESTION
    days    = d["days"]
    a2a     = np.array(d["a2a"])
    stencil = np.array(d["stencil"])
    n_days  = len(days)

    # Victim breakdown (non-overlapping categories for clarity)
    n_total    = d["total_jobs"]
    n_bully    = d["bully_jobs"]
    n_victim   = d["victim_jobs"] - n_bully   # victims that are NOT bullies
    n_clean    = n_total - d["victim_jobs"]
    # Some jobs can be both bully and victim; victim_jobs counts all with util>=0.5

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(FIGW * 2.1, FIGH),
                                          gridspec_kw={'width_ratios': [2.8, 1]})

    # ─ Left: grouped bar (daily peak) ─
    x = np.arange(n_days)
    w = 0.38
    a2a_col  = '#D95F02'   # warm orange — matches Frontier/A2A
    sten_col = '#1B9E77'   # teal — matches Lassen/Stencil

    bars_a2a  = ax_bar.bar(x - w/2, a2a,    width=w, color=a2a_col,
                            alpha=0.88, edgecolor='none', label='All-to-all')
    bars_sten = ax_bar.bar(x + w/2, stencil, width=w, color=sten_col,
                            alpha=0.88, edgecolor='none', label='Stencil-3D')

    # Capacity limit line
    ax_bar.axhline(1.0, color='#333', lw=1.0, ls='--', alpha=0.6, zorder=2)
    ax_bar.text(-0.45, 1.35, "Link capacity", fontsize=7,
                color='#444', va='bottom', ha='left', alpha=0.8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(days, rotation=25, ha='right', fontsize=7)
    ax_bar.set_ylabel("Peak max link utilization (×)")
    ax_bar.set_yscale("log")
    ax_bar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    ax_bar.set_title("(a) Daily peak congestion — Lassen, Aug 2019", fontsize=7,
                     pad=4)
    ax_bar.legend(fontsize=7, loc='lower right', framealpha=0.9)
    _ygrid(ax_bar)

    # Annotate peak bar — place inside the tallest stencil bar
    peak_idx = int(np.argmax(stencil))
    ax_bar.text(peak_idx + w/2, stencil[peak_idx] * 0.5,
                f"{stencil[peak_idx]:.0f}×\npeak",
                fontsize=7, color='white', ha='center', va='center',
                fontweight='bold', zorder=5)

    # ─ Right: horizontal stacked bar (job fate) ─
    categories   = ["Unaffected", "Victim", "Bully"]
    counts       = [n_clean,       n_victim, n_bully]
    colors_pie   = ['#AAAAAA',    '#E05C4B', '#4B87D0']
    pcts         = [c / n_total * 100 for c in counts]

    y_pos = [0.55]
    bar_h = 0.35
    left  = 0.0
    for cat, cnt, col, pct in zip(categories, counts, colors_pie, pcts):
        ax_pie.barh(y_pos, pct, left=left, height=bar_h,
                    color=col, alpha=0.88, edgecolor='none')
        if pct > 4:
            ax_pie.text(left + pct / 2, y_pos[0],
                        f"{pct:.0f}%", ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')
        left += pct

    # Legend patches
    patches = [mpatches.Patch(color=c, alpha=0.88, label=f"{cat}\n({cnt:,})")
               for cat, cnt, c in zip(categories, counts, colors_pie)]
    ax_pie.legend(handles=patches, fontsize=7, loc='lower center',
                  bbox_to_anchor=(0.5, -0.52), ncol=3, framealpha=0.9,
                  handlelength=1.0, handleheight=0.9)

    ax_pie.set_xlim(0, 100)
    ax_pie.set_ylim(0.1, 1.0)
    ax_pie.set_xticks([0, 25, 50, 75, 100])
    ax_pie.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_pie.set_yticks([])
    ax_pie.set_title(f"(b) Job fate\n({n_total:,} jobs, 1 week)", fontsize=7, pad=4)
    ax_pie.spines['left'].set_visible(False)

    # Summary stats box
    r = d["r_jobs_congestion"]
    stats_txt = (f"168/168 hours: max util ≥ 1.0\n"
                 f"Mean peak util: {d['mean_a2a']:.0f}× (A2A)\n"
                 f"Job–congestion corr.: r ≈ {r:.2f}")
    ax_bar.text(0.02, 0.97, stats_txt,
                transform=ax_bar.transAxes, fontsize=7,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                          ec='#cccccc', alpha=0.92))

    fig.tight_layout(pad=0.6)
    _save(fig, "fig_motiv_congestion.png")


# ── Figure 3: Simulator Positioning ─────────────────────────────────────────

def plot_sim_comparison():
    """
    Scatter plot positioning RAPS against representative network simulators.
    X-axis: operational speed (simulation speedup relative to real-time, log scale).
    Y-axis: topology fidelity (0 = none, 3 = full packet-level).

    Speed estimates are representative order-of-magnitude values from literature;
    exact performance depends on workload and hardware.
    """
    names, speedups, fidelities, scales, is_ours = zip(*SIMULATORS)
    speedups   = np.array(speedups,   dtype=float)
    fidelities = np.array(fidelities, dtype=float)
    scales     = np.array(scales,     dtype=float)
    is_ours    = np.array(is_ours,    dtype=bool)

    # Marker size ∝ log10(scale)
    sizes = 60 + (np.log10(scales) - 2) * 55

    fig, ax = plt.subplots(figsize=(FIGW, FIGH + 0.3))

    # Real-time boundary vertical line
    ax.axvline(1.0, color='#cc4444', lw=1.2, ls='-', alpha=0.65, zorder=1)
    ax.text(1.15, 0.18,
            "Real-time\nboundary",
            fontsize=7, color='#cc4444', va='bottom', alpha=0.85,
            transform=ax.get_xaxis_transform())

    # Shaded "operational zone" for what-if analysis (faster than real-time, decent fidelity)
    ax.axhspan(1.5, 3.2, xmin=0.0, xmax=1.0,   # left of real-time = below threshold
               color='#f0e6e6', alpha=0.30, zorder=0)
    ax.axhspan(1.5, 3.2, xmin=0.56, xmax=1.0,  # fast + medium-high fidelity = sweet spot
               color='#e6f0e6', alpha=0.40, zorder=0)
    ax.text(3e2, 2.85,
            "Operational sweet spot\n(fast + topology-aware)",
            fontsize=7, color='#2a6e2a', ha='center', va='top',
            style='italic', alpha=0.85)

    # Plot each simulator
    for i, (name, sp, fid, sz, ours) in enumerate(
            zip(names, speedups, fidelities, sizes, is_ours)):
        color  = '#D95F02' if ours else '#555555'
        marker = '*' if ours else 'o'
        ms     = sz * 1.4 if ours else sz
        alpha  = 0.95 if ours else 0.70
        zorder = 5 if ours else 3

        ax.scatter(sp, fid, s=ms, c=color, marker=marker,
                   alpha=alpha, zorder=zorder, edgecolors='white', linewidths=0.6)

        # Label offset — avoid overlaps
        offsets = {
            "SST/Merlin":        (-15,  6),
            "NS-3 (HPC)":        ( -5, -14),
            "CODES":             (  6,  5),
            "SimGrid":           (-12, -14),
            "LogGOPSim":         (-22,  7),
            "RAPS\n(this work)": (  8, -3),
        }
        dx, dy = offsets.get(name, (6, 6))
        ax.annotate(name,
                    xy=(sp, fid),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=7 if ours else 7,
                    color=color,
                    fontweight='bold' if ours else 'normal',
                    va='center',
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=0.8, alpha=0.7) if ours else None)

    ax.set_xscale("log")
    ax.set_xlabel("Operational speed (simulation speedup vs. real-time)")
    ax.set_ylabel("Topology fidelity")
    ax.set_xlim(left=1e-5, right=2e5)
    ax.set_ylim(0.0, 3.3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["None", "Low\n(analytical)", "Medium\n(flow-level)", "High\n(packet-level)"],
                       fontsize=7)
    def _speed_fmt(x, _):
        if x <= 0:
            return ""
        if x < 0.01:
            return f"{x:.0e}×".replace("e-0", "e-").replace("e+0", "e")
        if x < 1:
            return f"{x:.2g}×"
        if x < 10:
            return f"{x:.1f}×"
        return f"{x:,.0f}×"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_speed_fmt))

    ax.set_title("Network simulator positioning", fontsize=7, pad=4)
    _xgrid(ax)
    _ygrid(ax)

    # Footnote
    fig.text(0.01, -0.04,
             "Speed estimates are representative; values depend on workload scale and hardware.",
             fontsize=6.5, color='gray', style='italic')

    fig.tight_layout(pad=0.5, rect=[0, 0.07, 1, 1])
    _save(fig, "fig_motiv_sim_cmp.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Motivation figures for SC26 paper (Background & Motivation section)")
    parser.add_argument("--csv", default=None,
                        help="Path to frontier_scaling/results.csv")
    parser.add_argument("--skip-congestion", action="store_true",
                        help="Skip fig_motiv_congestion.png (Wes's Lassen analysis)")
    parser.add_argument("--skip-sim-cmp", action="store_true",
                        help="Skip fig_motiv_sim_cmp.png (simulator positioning)")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    globals()['OUT_DIR'] = out_dir

    print("=" * 60)
    print("Motivation Figures — SC26 paper")
    print("=" * 60)

    # ── Figure 1: Speed ──
    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    if csv_path.exists() and csv_path.stat().st_size > 100:
        print("\n[1/3] fig_motiv_speed.png  (RAPS benchmark)")
        plot_speed(csv_path)
    else:
        print(f"\n[1/3] SKIPPED — CSV not found: {csv_path}")
        print("      Run src/run_frontier.py first to generate benchmark data.")

    # ── Figure 2: Congestion ──
    if not args.skip_congestion:
        print("\n[2/3] fig_motiv_congestion.png  (Lassen real interference)")
        plot_congestion()
    else:
        print("\n[2/3] SKIPPED (--skip-congestion)")

    # ── Figure 3: Simulator comparison ──
    if not args.skip_sim_cmp:
        print("\n[3/3] fig_motiv_sim_cmp.png  (simulator positioning)")
        plot_sim_comparison()
    else:
        print("\n[3/3] SKIPPED (--skip-sim-cmp)")

    print(f"\nDone. Figures written to: {OUT_DIR}")

    # ── Copy to main/ ────────────────────────────────────────────────────────
    import shutil
    main_dir = out_dir / 'main'
    main_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ('fig_motiv_speed.png',      'motiv_speed.png'),
        ('fig_motiv_congestion.png', 'motiv_congestion.png'),
        ('fig_motiv_sim_cmp.png',    'motiv_sim_cmp.png'),
    ]:
        src = out_dir / src_name
        if src.exists():
            shutil.copy2(src, main_dir / dst_name)
            print(f"  Copied {src_name} → main/{dst_name}")


if __name__ == "__main__":
    main()
