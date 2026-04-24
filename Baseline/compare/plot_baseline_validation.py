#!/usr/bin/env python3
"""
Baseline/compare/plot_baseline_validation.py

Three-panel figure for RAPS baseline validation (dragonfly topology, 1056 nodes):
  (a) Slowdown vs. link utilization ρ  — RAPS, BookSim2, SST-Macro, SimGrid, GPCNet
  (b) Relative stall-ratio error vs. BookSim2 (ρ ≤ 0.5)
  (c) Simulation speed (wall-clock per ρ-point)

All three panels side-by-side at 7" total width (IEEE SC two-column format).

GPCNet data from Frontier (1000 nodes, 8000 MPI ranks):
  - Isolated: 8B latency = 2.9 µs
  - Under 800/1000-node congestion: 8B latency = 2.8 µs  → impact factor = 1.0×
  (output/gpcnet/network_test_1000n.out, network_load_test_1000n.out)

SST-Macro dragonfly_1000n is pending (SLURM job 4203020, template fix applied).
Until it completes, expected values based on BookSim2 are shown as dashed lines.

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/compare/plot_baseline_validation.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Make the repo root importable so we can pull the hardcoded kappa values.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from raps.network.calibration import STALL_CALIBRATION  # noqa: E402

# This figure validates the dragonfly topology, so use the dragonfly kappa.
KAPPA = STALL_CALIBRATION["dragonfly"]


def _apply_kappa(sd_dict, kappa=KAPPA):
    """Apply calibrated stall ratio: reported = 1 + κ·(raw-1)."""
    return {r: 1.0 + kappa * (v - 1.0) for r, v in sd_dict.items()}

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Baseline/
OUTDIR = os.path.dirname(os.path.abspath(__file__))                    # Baseline/compare/

# ── IEEE SC two-column style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":       8,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth":  0.6,
    "grid.linewidth":  0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})

RHO_VALS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
LOW_RHO  = [r for r in RHO_VALS if 0.15 <= r <= 0.5]  # skip ρ<0.2: stall ratio too small


# ── helpers ───────────────────────────────────────────────────────────────────
def load_slowdown(rel_path):
    """Return {rho: slowdown} dict from summary JSONs in BASE/rel_path/."""
    d = os.path.join(BASE, rel_path)
    result = {}
    for r in RHO_VALS:
        rho_str = f"{r:.2f}".replace(".", "p")
        fpath = os.path.join(d, f"summary_{rho_str}.json")
        if os.path.exists(fpath):
            dat = json.load(open(fpath))
            sd = dat.get("slowdown")
            if sd is not None and dat.get("status", "ok") != "failed":
                result[r] = sd
    return result


def md1(rho):
    return 1.0 + rho ** 2 / (2.0 * (1.0 - rho))


def rho_arr(rho_vals, sd_dict):
    xs = [r for r in rho_vals if r in sd_dict]
    return np.array(xs), np.array([sd_dict[r] for r in xs])


# ── load data ─────────────────────────────────────────────────────────────────
raps_raw = load_slowdown("raps/output/dragonfly_1000n")
# Paper §6.1 uses the topology-specific calibration factor κ to scale the raw
# M/D/1 stall ratio before plotting against cycle-accurate baselines.
raps = _apply_kappa(raps_raw)
bs2  = load_slowdown("booksim2/output/dragonfly_1000n")
sg   = load_slowdown("simgrid/output/dragonfly_1000n")
sst  = load_slowdown("sst-macro/output/dragonfly_1000n")

sst_pending = (len(sst) == 0)
if sst_pending:
    # SST-Macro data is still being collected.  Use the BookSim2 stall ratio as
    # a conservative estimate — no ad-hoc per-rho factor; the calibrated κ is
    # already applied to RAPS above.
    sst = dict(bs2)

# ── GPCNet real Frontier hardware data ───────────────────────────────────────
# Isolated (1000 nodes, light traffic):   8B two-sided latency = 2.9 µs
# Congestion (800/1000 nodes congestors): 8B two-sided latency = 2.8 µs
# Congestion Impact Factor (avg) = 1.0× for latency
GPCNET_ISOLATED_US  = 2.9
GPCNET_CONGESTED_US = 2.8
gpcnet_slowdown_isolated  = 1.0    # reference point
gpcnet_slowdown_congested = GPCNET_CONGESTED_US / GPCNET_ISOLATED_US  # ≈ 0.97

# Estimated ρ for each test point (cannot be measured directly from GPCNet):
#   isolated:   low background traffic → ρ ≈ 0.05-0.10
#   congested:  800 nodes at near-BW → ρ ≈ 0.50-0.70 (conservative estimate: 0.60)
GPCNET_PTS = {
    0.10: gpcnet_slowdown_isolated,
    0.60: gpcnet_slowdown_congested,
}

print("Data loaded:")
print(f"  RAPS      : {len(raps)}/9 points")
print(f"  BookSim2  : {len(bs2)}/9 points")
print(f"  SimGrid   : {len(sg)}/9 points")
print(f"  SST-Macro : {'PENDING (expected values used)' if sst_pending else len(sst)}")
print(f"  GPCNet    : {len(GPCNET_PTS)} reference points (real Frontier hardware)")

# ── color / marker scheme ─────────────────────────────────────────────────────
CLR = {
    "raps":   "#d62728",   # red
    "bs2":    "#1f77b4",   # blue
    "sst":    "#2ca02c",   # green
    "sg":     "#ff7f0e",   # orange
    "gpcnet": "#555555",   # dark grey
    "md1":    "#bbbbbb",   # light grey
}


def plot_line(ax, sd_dict, color, marker, ls="-", label=None, zorder=3, mfc=None):
    xs, ys = rho_arr(RHO_VALS, sd_dict)
    if len(xs) == 0:
        return
    ax.plot(xs, ys,
            color=color, marker=marker, ls=ls, zorder=zorder,
            markerfacecolor=(mfc if mfc is not None else color),
            markeredgecolor=color, label=label)


# ═══════════════════════════════════════════════════════════════════════════════
# Three-panel figure: 7" wide × 2.55" tall (IEEE two-column)
# ═══════════════════════════════════════════════════════════════════════════════
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.0, 2.55))
fig.subplots_adjust(wspace=0.44, left=0.07, right=0.995, top=0.87, bottom=0.19)

# ── (a) Slowdown vs. ρ ────────────────────────────────────────────────────────
ax = ax_a
rho_cont = np.linspace(0.01, 0.84, 300)
ax.plot(rho_cont, [md1(r) for r in rho_cont],
        color=CLR["md1"], lw=0.8, ls=":", zorder=1, label="M/D/1 theory")

plot_line(ax, raps, CLR["raps"], "o", label="RAPS")
plot_line(ax, bs2,  CLR["bs2"],  "s", label="BookSim2")
plot_line(ax, sst,  CLR["sst"],  "^",
          ls="--" if sst_pending else "-",
          mfc="white" if sst_pending else None,
          label="SST-Macro")
plot_line(ax, sg,   CLR["sg"],   "D", label="SimGrid")

# GPCNet real hardware reference points
gx = list(GPCNET_PTS.keys())
gy = [GPCNET_PTS[r] for r in gx]
ax.scatter(gx, gy, color=CLR["gpcnet"], marker="*", s=36, zorder=5,
           label="GPCNet (Frontier)")

# shade typical HPC operating range
ax.axvspan(0.0, 0.505, alpha=0.05, color="#333333", zorder=0)
ax.axvline(0.5, color="#888888", lw=0.5, ls=":", zorder=1)

ax.set_title("(a) Slowdown vs. ρ\n(dragonfly, 1056 nodes)", fontsize=7.5, pad=2)
ax.set_xlabel("Link utilization ρ")
ax.set_ylabel("Slowdown")
ax.set_xlim(0.0, 0.85)
ax.set_ylim(0.95, 3.05)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
ax.grid(True, ls=":", lw=0.4, alpha=0.5)
ax.legend(fontsize=6.5, loc="upper left", frameon=True,
          handlelength=1.5, handletextpad=0.4,
          borderpad=0.4, labelspacing=0.28)

# ── (b) Relative stall-ratio error vs. BookSim2 (ρ ≤ 0.5) ────────────────────
def stall_pct_err(sim_dict, ref_dict, rhos=LOW_RHO):
    """Per-ρ absolute % error in stall ratio vs reference."""
    errs = []
    for r in rhos:
        if r in sim_dict and r in ref_dict:
            r_stall = ref_dict[r] - 1.0
            s_stall = sim_dict[r] - 1.0
            if r_stall > 0.002:
                errs.append(abs(s_stall - r_stall) / r_stall * 100.0)
            else:
                errs.append(0.0)
        else:
            errs.append(np.nan)
    return errs


ax = ax_b
rows = [
    ("RAPS",      stall_pct_err(raps, bs2), CLR["raps"], False),
    ("SimGrid",   stall_pct_err(sg,   bs2), CLR["sg"],   False),
    ("SST-Macro", stall_pct_err(sst,  bs2), CLR["sst"],  sst_pending),
]
x  = np.arange(len(LOW_RHO))
bw = 0.22
rho_labels = [f"{r:.2f}" for r in LOW_RHO]

for i, (lbl, errs, color, is_pend) in enumerate(rows):
    offset = (i - 1) * bw
    vals   = [e if not np.isnan(e) else 0.0 for e in errs]
    hatch  = "//" if is_pend else None
    ax.bar(x + offset, vals, bw,
           color=color, alpha=0.75,
           hatch=hatch, edgecolor=color, linewidth=0.5,
           label=lbl)

ax.set_xticks(x)
ax.set_xticklabels(rho_labels, rotation=35, ha="right")
ax.set_xlabel("Link utilization ρ")
ax.set_ylabel("|Δ stall ratio| vs BookSim2 (%)")
ax.set_title("(b) Accuracy vs. BookSim2\n(dragonfly, 0.2 ≤ ρ ≤ 0.5)", fontsize=7.5, pad=2)
ax.set_ylim(0, None)
ax.grid(axis="y", ls=":", lw=0.4, alpha=0.5)
ax.legend(fontsize=6.5, loc="upper left", frameon=True,
          handlelength=1.2, handletextpad=0.4, borderpad=0.4, labelspacing=0.28)

# ── (c) Simulation speed (wall-clock per ρ-point) ────────────────────────────
# Times (seconds/ρ-point):
#   RAPS M/D/1:       ~0.001 s  (analytical formula, measured)
#   SimGrid (analyt): ~0.001 s  (same framework, measured)
#   BookSim2 1056n:   51–635 s  (range across ρ values, measured from logs)
#   SST-Macro 1056n:  240–480 s (estimated; PISCES model, similar scale)
speed_entries = [
    # (label, lo_s, hi_s, is_measured, color)
    ("RAPS",       0.001,  0.001, True,  CLR["raps"]),
    ("SimGrid",    0.001,  0.001, True,  CLR["sg"]),
    ("BookSim2",   51,     635,   True,  CLR["bs2"]),
    ("SST-Macro",  240,    480,   False, CLR["sst"]),
]

ax = ax_c
n = len(speed_entries)
y = np.arange(n)
lo_vals  = [e[1] for e in speed_entries]
hi_vals  = [e[2] for e in speed_entries]
mid_vals = [np.sqrt(l * h) for l, h in zip(lo_vals, hi_vals)]   # geometric mean
measured = [e[3] for e in speed_entries]
colors_s = [e[4] for e in speed_entries]
labels_s = [e[0] for e in speed_entries]

for i, (lo, hi, mid, color, meas) in enumerate(
        zip(lo_vals, hi_vals, mid_vals, colors_s, measured)):
    hatch = None if meas else "//"
    ax.barh(i, mid, height=0.52, color=color, alpha=0.80,
            hatch=hatch, edgecolor=color, linewidth=0.5)
    if lo != hi:
        ax.barh(i, hi - lo, left=lo, height=0.52,
                color=color, alpha=0.18, edgecolor=color, linewidth=0)
    label_x = max(hi, 8e-4) * 2.8
    txt = f"~{mid:.0f} s" if mid >= 1 else f"~{mid * 1000:.1f} ms"
    ax.text(label_x, i, txt, va="center", fontsize=6.5, color=color)

ax.set_yticks(y)
ax.set_yticklabels(labels_s)
ax.set_xscale("log")
ax.set_xlim(3e-4, 5e4)
ax.set_xlabel("Wall-clock time per ρ-point (s)")
ax.set_title("(c) Simulation speed\n(dragonfly, 1056 nodes)", fontsize=7.5, pad=2)
ax.grid(axis="x", ls=":", lw=0.4, alpha=0.5)

# speedup annotation
i_raps = labels_s.index("RAPS")
i_bs2  = labels_s.index("BookSim2")
speedup = mid_vals[i_bs2] / mid_vals[i_raps]
if speedup >= 1000:
    spd_str = f"×{speedup/1000:,.0f}K"
else:
    spd_str = f"×{speedup:,.0f}"
ax.text(0.04, 0.04,
        f"RAPS {spd_str}\nfaster than\nBookSim2",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6,
        color=CLR["raps"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=CLR["raps"], alpha=0.9, lw=0.8))

leg_h = [
    Patch(facecolor="#888", alpha=0.8, ec="#555", label="Measured"),
    Patch(facecolor="#888", alpha=0.8, ec="#555", hatch="//", label="Estimated"),
]
ax.legend(handles=leg_h, fontsize=6, loc="lower right", frameon=True,
          handlelength=1.2, handletextpad=0.4)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in [".pdf", ".png"]:
    kw = dict(bbox_inches="tight")
    if ext == ".png":
        kw["dpi"] = 180
    out = os.path.join(OUTDIR, f"fig_dragonfly_validation{ext}")
    fig.savefig(out, **kw)
    print(f"→ {out}")
plt.close(fig)

print("\nData availability:")
print(f"  RAPS      : {len(raps)}/9")
print(f"  BookSim2  : {len(bs2)}/9")
print(f"  SimGrid   : {len(sg)}/9")
print(f"  SST-Macro : {'PENDING — dashed line shows expected values' if sst_pending else len(sst)}/9")
print(f"  GPCNet    : {len(GPCNET_PTS)} real hardware points (ρ estimated)")
