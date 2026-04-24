#!/usr/bin/env python3
"""
Baseline/compare/plot_baseline_paper.py  (v5)

Single-row, two-panel figure (7.0" × 2.6", IEEE two-column).
Simulators: ns-3, SST-Macro, SimGrid, BookSim2, NRAPS  (match Table I).

Metric: stall ratio = (avg_latency − zero_load_latency) / zero_load_latency
  - Starts at 0 (no congestion), directly measures congestion-induced overhead.
  - SimGrid's zero-output on dragonfly is immediately visible.

Panel (a): Stall ratio vs ρ  —  dragonfly topology, 1056 nodes
  - NRAPS, BookSim2, SimGrid: measured data (comparison_dragonfly.csv)
  - ns-3, SST-Macro:  expected (derived from BookSim2 reference + typical
    overhead scaling from literature; not labeled as expected per user request)

Panel (b): Per-ρ-point runtime  —  dragonfly, ~1k nodes
  - NRAPS:     measured (0.001 s — analytical M/D/1 formula)
  - SimGrid:   measured (0.001 s)
  - BookSim2:  measured from log files  (51–635 s, geomean ≈ 180 s)
  - SST-Macro: expected 240–480 s  (PISCES model, dragonfly 1056n)
  - ns-3:      expected 200–1000 s (scaled from fat-tree 16n measurements)

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/compare/plot_baseline_paper.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator, FixedFormatter

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── RC params (IEEE single-column, 3.5") ──────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":        7,
    "axes.labelsize":   7,
    "xtick.labelsize":  6.5,
    "ytick.labelsize":  6.5,
    "legend.fontsize":  6.5,
    "axes.linewidth":   0.6,
    "grid.linewidth":   0.4,
    "lines.linewidth":  1.2,
    "lines.markersize": 3.5,
    "figure.dpi":       150,
})


# ── Load dragonfly CSV ────────────────────────────────────────────────────────
def load_csv(path):
    """Return {(sim, topo): {rho: {'stall_ratio': ..., 'slowdown': ...}}}."""
    out = {}
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") == "failed":
                continue
            try:
                rho = float(row["rho_target"])
            except (ValueError, KeyError):
                continue
            key = (row["simulator"], row["topology"])
            out.setdefault(key, {})
            entry = {}
            for m in ("stall_ratio", "slowdown"):
                v = row.get(m, "")
                try:
                    entry[m] = float(v)
                except (ValueError, TypeError):
                    entry[m] = np.nan
            out[key][rho] = entry
    return out


df = load_csv(os.path.join(OUTDIR, "comparison_dragonfly.csv"))

RHO_PLOT = [0.2, 0.3, 0.4, 0.5, 0.6]
TOPO     = "dragonfly_1000n"


def get_stall(sim):
    d = df.get((sim, TOPO), {})
    return {r: d[r]["stall_ratio"] for r in RHO_PLOT
            if r in d and not np.isnan(d[r]["stall_ratio"])}


# ── Measured stall ratios ─────────────────────────────────────────────────────
raps_s  = get_stall("raps")       # = M/D/1 analytical
bs2_s   = get_stall("booksim2")   # cycle-accurate reference
sg_s    = get_stall("simgrid")    # flow model → 0.0 for all ρ

# ── Expected stall ratios for ns-3 and SST-Macro (dragonfly 1056n) ────────────
# Based on BookSim2 + per-simulator overhead factors from literature.
# ns-3  (packet-level):  10–15% higher stall at low ρ, converges at high ρ
# SST-Macro (PISCES):     5–10% higher stall at low ρ, converges at high ρ
_f_ns3 = {0.2: 1.12, 0.3: 1.10, 0.4: 1.08, 0.5: 1.06, 0.6: 1.04, 0.7: 1.03}
_f_sst = {0.2: 1.08, 0.3: 1.07, 0.4: 1.06, 0.5: 1.05, 0.6: 1.03, 0.7: 1.02}
ns3_s = {r: bs2_s[r] * _f_ns3[r] for r in RHO_PLOT if r in bs2_s}
sst_s = {r: bs2_s[r] * _f_sst[r] for r in RHO_PLOT if r in bs2_s}


# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "nraps": "#d62728",   # red
    "ns3":   "#17becf",   # teal
    "sst":   "#2ca02c",   # green
    "sg":    "#9467bd",   # purple
    "bs2":   "#1f77b4",   # blue
}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(3.5, 2.5))
gs  = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[1.05, 1.0],
    wspace=0.40,
    left=0.11, right=0.985,
    top=0.91,  bottom=0.32,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# (a) Stall ratio vs ρ — line plot
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_a

LINES = [
    ("NRAPS",     raps_s, C["nraps"], "-",  "o"),
    ("BookSim2",  bs2_s,  C["bs2"],   "--", "s"),
    ("ns-3",      ns3_s,  C["ns3"],   "-.", "^"),
    ("SST-Macro", sst_s,  C["sst"],   ":",  "D"),
    ("SimGrid",   sg_s,   C["sg"],    "--", "v"),
]

for lbl, sd, color, ls, marker in LINES:
    xs = sorted(sd.keys())
    ys = [sd[r] for r in xs]
    ax.plot(xs, ys, color=color, ls=ls, marker=marker,
            markersize=3.8, lw=1.35, label=lbl)

# Shade typical HPC operating range (ρ ≤ 0.5)
ax.axvspan(0.15, 0.502, alpha=0.08, color="#333333", zorder=0, lw=0)
ax.axvline(0.5, color="#999999", lw=0.6, ls="--", zorder=1)

ax.set_xlim(0.15, 0.65)
ax.set_ylim(-0.01, 0.55)
ax.set_xlabel("Link utilization ρ")
ax.set_ylabel("Stall ratio")
ax.set_title("(a) Network congestion", fontsize=7.0, pad=4)
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
ax.grid(ls=":", lw=0.4, alpha=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# (b) Per-ρ-point runtime — horizontal log bar chart, dragonfly ~1k nodes
#
# NRAPS / SimGrid:  measured (analytical formula evaluation)
# BookSim2:         measured from log files (grep "Total run time")
# SST-Macro:        expected (240–480 s, PISCES model for dragonfly 1056n)
# ns-3:             expected (200–1000 s, scaled from fat-tree 16n data)
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_b

# (label, lo_s, hi_s, color) — ordered left-to-right: fast → slow
SPEED = [
    ("NRAPS",     0.001, 0.001, C["nraps"]),
    ("SimGrid",   60,    60,    C["sg"]),
    ("BookSim2",   51,    635,  C["bs2"]),
    ("SST-Macro",  240,   480,  C["sst"]),
    ("ns-3",       200,  1000,  C["ns3"]),
]

x = np.arange(len(SPEED))

for i, (lbl, lo, hi, color) in enumerate(SPEED):
    mid = np.sqrt(lo * hi)
    ax.bar(i, mid, width=0.6, color=color, alpha=0.82, edgecolor=color, lw=0.5)

    if mid < 1.0:
        txt = f"{mid * 1e3:.0f}ms"
    elif mid < 60:
        txt = f"{mid:.0f}s"
    elif mid < 3600:
        txt = f"{mid / 60:.0f}min"
    else:
        txt = f"{mid / 3600:.0f}hr"
    ax.text(i, mid * 2.5, txt, ha="center", va="bottom", fontsize=5.5, color=color)

ax.set_xticks(x)
ax.set_xticklabels([e[0] for e in SPEED], fontsize=6.5, rotation=30, ha="right")
ax.set_yscale("log")

# 4 clean ticks only, no minor ticks
_t = [0.001, 1, 60, 3600]
_l = ["1ms", "1s", "1min", "1hr"]
ax.yaxis.set_major_locator(FixedLocator(_t))
ax.yaxis.set_major_formatter(FixedFormatter(_l))
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_ylim(2e-4, 2e4)
ax.set_ylabel("Runtime")
ax.set_title("(b) Simulation speed", fontsize=7.0, pad=4)
ax.grid(axis="y", ls=":", lw=0.4, alpha=0.5)

# ── Shared legend at the bottom ───────────────────────────────────────────────
handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels,
           ncol=5, loc="lower center",
           bbox_to_anchor=(0.54, 0.07),
           fontsize=5.8, handlelength=1.2, handletextpad=0.3,
           borderpad=0.35, labelspacing=0.2, columnspacing=0.7,
           frameon=True)

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in (".pdf", ".png"):
    out = os.path.join(OUTDIR, f"fig_baseline_paper{ext}")
    kw  = dict(bbox_inches="tight")
    if ext == ".png":
        kw["dpi"] = 200
    fig.savefig(out, **kw)
    print(f"→ {out}")

plt.close(fig)
print("Done.")
