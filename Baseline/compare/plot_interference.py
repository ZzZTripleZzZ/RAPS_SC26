#!/usr/bin/env python3
"""
Plot inter-job interference comparison: RAPS (M/D/1) vs SST-Macro (SNAPPR).

Generates a multi-panel figure showing:
  - Panel (a-c): Slowdown vs bully_nx for each topology
  - Panel (d): Correlation between RAPS traffic intensity ratio and SST-Macro slowdown

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/compare/plot_interference.py
"""

import json
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAPS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(RAPS_ROOT, "main")
os.makedirs(OUT_DIR, exist_ok=True)

# Allow importing repo-level modules.
if RAPS_ROOT not in sys.path:
    sys.path.insert(0, RAPS_ROOT)
from raps.network.calibration import (  # noqa: E402
    STALL_CALIBRATION,
    LAYER2_POWERLAW_FIT,
    layer2_calibrated_slowdown,
)

# Map the interference-experiment topology labels (dragonfly / torus / fattree)
# to the NetworkModel topology keys used for calibration lookup.
_CALIB_KEYS = {"dragonfly": "dragonfly", "torus": "torus3d", "fattree": "fat-tree"}


def _kappa_for(topo_label):
    return STALL_CALIBRATION.get(_CALIB_KEYS.get(topo_label, topo_label), 1.0)


def _powerlaw_for(topo_label):
    return LAYER2_POWERLAW_FIT.get(_CALIB_KEYS.get(topo_label, topo_label))

TOPOLOGIES = ["dragonfly", "torus", "fattree"]
BULLY_NX_VALUES = [0, 50, 100, 150, 200, 300, 400]

# Scale-dependent config
SCALE_CONFIG = {
    "small": {
        "sst_dir": os.path.join(RAPS_ROOT, "Baseline", "sst-macro", "multi_job", "output", "interference"),
        "raps_dir": os.path.join(RAPS_ROOT, "Baseline", "raps", "output", "interference"),
        "labels": {
            "dragonfly": "Dragonfly (72 hosts)",
            "torus": "3D Torus (1024 hosts)",
            "fattree": "Fat-Tree (16 hosts)",
        },
        "suffix": "",
    },
    "large": {
        "sst_dir": os.path.join(RAPS_ROOT, "Baseline", "sst-macro", "multi_job", "output", "interference_large"),
        "raps_dir": os.path.join(RAPS_ROOT, "Baseline", "raps", "output", "interference_large"),
        "labels": {
            "dragonfly": "Dragonfly (1056 hosts)",
            "torus": "3D Torus (1024 hosts)",
            "fattree": "Fat-Tree (1024 hosts)",
        },
        "suffix": "_large",
    },
}

# Default: use whichever scale has data (prefer large)
def _pick_scale():
    for s in ("large", "small"):
        sst = SCALE_CONFIG[s]["sst_dir"]
        if os.path.isdir(sst) and any(
            os.path.isdir(os.path.join(sst, t)) for t in TOPOLOGIES
        ):
            return s
    return "small"

_SCALE = _pick_scale()
SST_DIR = SCALE_CONFIG[_SCALE]["sst_dir"]
RAPS_DIR = SCALE_CONFIG[_SCALE]["raps_dir"]
TOPO_LABELS = SCALE_CONFIG[_SCALE]["labels"]

# IEEE SC two-column figure style
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def load_sst_data(topo):
    """Load SST-Macro results for a topology."""
    topo_dir = os.path.join(SST_DIR, topo)
    if not os.path.isdir(topo_dir):
        return None
    data = {}
    baseline_avg = None
    for bully_nx in BULLY_NX_VALUES:
        jp = os.path.join(topo_dir, f"{topo}_bully_nx{bully_nx}.json")
        if not os.path.exists(jp):
            continue
        d = json.load(open(jp))
        if d.get("status") != "ok":
            continue
        avg = d.get("victim_avg_iter_us", 0)
        if bully_nx == 0:
            baseline_avg = avg
        data[bully_nx] = {
            "avg_iter_us": avg,
            "total_us": d.get("victim_total_us", 0),
            "wall_time_s": d.get("wall_time_s", 0),
        }
    if baseline_avg:
        for bx, dd in data.items():
            dd["slowdown"] = dd["avg_iter_us"] / baseline_avg
    return data


def load_raps_data(topo):
    """Load RAPS analytical results for a topology."""
    topo_dir = os.path.join(RAPS_DIR, topo)
    if not os.path.isdir(topo_dir):
        return None
    data = {}
    for bully_nx in BULLY_NX_VALUES:
        jp = os.path.join(topo_dir, f"{topo}_bully_nx{bully_nx}.json")
        if not os.path.exists(jp):
            continue
        d = json.load(open(jp))
        if d.get("status") != "ok":
            continue
        data[bully_nx] = {
            "traffic_intensity_ratio": d.get("traffic_intensity_ratio", 1.0),
            "effective_tir": d.get("effective_tir", 1.0),
            "temporal_overlap": d.get("temporal_overlap", 1.0),
            # ρ-based metrics: raps_raw_combined_rho = max_combined_rr / victim_tx
            # Differentiates BOTH topology (fattree >> dragonfly >> torus) and bully_nx
            "raps_victim_rho": d.get("raps_victim_rho", 0.0),
            "raps_combined_rho": d.get("raps_combined_rho", 0.0),
            "raps_raw_combined_rho": d.get("raps_raw_combined_rho", 0.0),
            "raps_md1_slowdown": d.get("raps_md1_slowdown", 1.0),
        }
    return data


def plot_interference_comparison():
    """Main comparison figure: 2×2 panels."""
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    colors_sst = {"dragonfly": "#2196F3", "torus": "#4CAF50", "fattree": "#FF9800"}
    markers_sst = {"dragonfly": "o", "torus": "s", "fattree": "^"}

    all_sst = {}
    all_raps = {}
    for topo in TOPOLOGIES:
        sst = load_sst_data(topo)
        raps = load_raps_data(topo)
        if sst:
            all_sst[topo] = sst
        if raps:
            all_raps[topo] = raps

    # ─── Panels (a-c): Slowdown vs bully_nx per topology ───
    for idx, topo in enumerate(TOPOLOGIES):
        if idx < 2:
            ax = fig.add_subplot(gs[0, idx])
        else:
            ax = fig.add_subplot(gs[1, 0])

        sst = all_sst.get(topo, {})
        raps = all_raps.get(topo, {})

        # SST-Macro slowdown
        sst_nx = sorted(sst.keys())
        sst_sd = [sst[nx].get("slowdown", 1.0) for nx in sst_nx]

        if sst_nx and sst_sd:
            ax.plot(sst_nx, sst_sd, 'o-',
                    color=colors_sst[topo], markersize=4, linewidth=1.5,
                    label="SST-Macro", zorder=3)

        # RAPS ρ_combined (right y-axis) — raw combined load / victim_tx
        if raps:
            raps_nx = sorted(raps.keys())
            raps_rho = [raps[nx].get("raps_raw_combined_rho", 1.0) for nx in raps_nx]

            ax2 = ax.twinx()
            ax2.plot(raps_nx, raps_rho, 's--',
                     color="#E91E63", markersize=3, linewidth=1.0, alpha=0.7,
                     label=r"NRAPS $\rho_{comb}$")
            ax2.set_ylabel(r"NRAPS $\rho_{comb}$", fontsize=7, color="#E91E63")
            ax2.tick_params(axis='y', labelcolor="#E91E63", labelsize=6)
            ax2.set_yscale("log")
            # Overlay M/D/1 slowdown prediction on left axis (all topologies)
            raps_md1 = [raps[nx].get("raps_md1_slowdown", 1.0) for nx in raps_nx]
            ax.plot(raps_nx, raps_md1, '^:',
                    color="#9C27B0", markersize=3, linewidth=1.0, alpha=0.8,
                    label="NRAPS M/D/1")

        ax.set_xlabel("Bully nx")
        ax.set_ylabel("Victim Slowdown")
        ax.set_title(f"({chr(97+idx)}) {TOPO_LABELS[topo]}", fontsize=8)
        ax.set_ylim(bottom=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=6)

    # ─── Panel (d): Correlation scatter ───
    ax_corr = fig.add_subplot(gs[1, 1])

    # Skip bully_nx=50 (parse artifact: victim appears faster than baseline)
    VALID_NX = [nx for nx in BULLY_NX_VALUES if nx != 50 and nx > 0]

    for topo in TOPOLOGIES:
        sst = all_sst.get(topo, {})
        raps = all_raps.get(topo, {})
        if not sst or not raps:
            continue

        common_nx = sorted(set(sst.keys()) & set(raps.keys()))
        common_nx = [nx for nx in common_nx if nx in VALID_NX]

        if not common_nx:
            continue

        rho_vals = [raps[nx].get("raps_raw_combined_rho", 1.0) for nx in common_nx]
        sd_vals = [sst[nx]["slowdown"] for nx in common_nx]

        ax_corr.scatter(rho_vals, sd_vals,
                        marker=markers_sst[topo], color=colors_sst[topo],
                        s=30, label=topo, zorder=3)

    # Fit power law: slowdown = a * rho^alpha
    all_rho, all_sd = [], []
    for topo in TOPOLOGIES:
        sst = all_sst.get(topo, {})
        raps = all_raps.get(topo, {})
        if not sst or not raps:
            continue
        common_nx = [nx for nx in sorted(set(sst.keys()) & set(raps.keys()))
                     if nx in VALID_NX]
        for nx in common_nx:
            rv = raps[nx].get("raps_raw_combined_rho", 1.0)
            if rv > 0:
                all_rho.append(rv)
                all_sd.append(sst[nx]["slowdown"])

    if all_rho and all_sd:
        log_rho = np.log(all_rho)
        log_sd = np.log(all_sd)
        alpha, beta = np.polyfit(log_rho, log_sd, 1)

        std_rho = np.std(log_rho)
        std_sd = np.std(log_sd)
        if std_rho > 0 and std_sd > 0:
            corr = np.corrcoef(log_rho, log_sd)[0, 1]
        else:
            corr = float('nan')

        rho_range = np.exp(np.linspace(np.log(min(all_rho)), np.log(max(all_rho)), 100))
        sd_fit = np.exp(beta) * rho_range ** alpha
        ax_corr.plot(rho_range, sd_fit, 'k--', linewidth=1.0, alpha=0.5,
                     label=f"$r = {corr:.3f}$")

    ax_corr.set_xlabel(r"NRAPS $\rho_{comb}$ (link load / victim tx)")
    ax_corr.set_ylabel("SST-Macro Slowdown")
    ax_corr.set_title("(d) NRAPS vs SST-Macro", fontsize=8)
    ax_corr.set_xscale("log")
    ax_corr.grid(True, alpha=0.3)
    ax_corr.legend(fontsize=6)

    out_path = os.path.join(OUT_DIR, "interference_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Print summary table
    print("\n  Summary: SST-Macro slowdown vs RAPS metrics")
    print(f"  {'topo':>10s}  {'bnx':>4s}  {'SST_sd':>7s}  {'rho_raw':>9s}  {'md1':>6s}")
    print(f"  {'-'*42}")
    for topo in TOPOLOGIES:
        sst = all_sst.get(topo, {})
        raps = all_raps.get(topo, {})
        for nx in BULLY_NX_VALUES:
            ssd = sst.get(nx, {}).get("slowdown", "")
            rrho = raps.get(nx, {}).get("raps_raw_combined_rho", "")
            md1 = raps.get(nx, {}).get("raps_md1_slowdown", "")
            if ssd and rrho:
                print(f"  {topo:>10s}  {nx:>4d}  {ssd:>7.3f}  {rrho:>9.3f}  {md1:>6.3f}")


def plot_simple_comparison():
    """Simpler single-panel comparison for each topology (Option B).

    Three curves per topology:
      1. SST-Macro actual victim slowdown (ground truth)
      2. RAPS M/D/1 (raw, direct model prediction — no calibration)
      3. RAPS calibrated power-law fit to SST data (for reference)
    Panel title includes per-topology Pearson r (log-log) between
    raps_raw_combined_rho and SST-Macro slowdown.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))

    colors_sst = {"dragonfly": "#2196F3", "torus": "#4CAF50", "fattree": "#FF9800"}

    # bully_nx=50 excluded (SST-Macro shows slowdown<1 — phase-measurement artifact)
    VALID_NX_NONZERO = [nx for nx in BULLY_NX_VALUES if nx != 50 and nx > 0]

    for idx, topo in enumerate(TOPOLOGIES):
        ax = axes[idx]
        sst = load_sst_data(topo)
        raps = load_raps_data(topo)

        # ── (1) SST-Macro ground truth ──────────────────────────────────
        if sst:
            sst_nx = sorted(sst.keys())
            sst_sd = [sst[nx].get("slowdown", 1.0) for nx in sst_nx]
            ax.plot(sst_nx, sst_sd, 'o-',
                    color=colors_sst[topo], markersize=4, linewidth=1.5,
                    label="SST-Macro")

        # ── (2) RAPS raw M/D/1 prediction (no calibration) ──────────────
        if raps:
            raps_nx = sorted(raps.keys())
            raps_md1 = [raps[nx].get("raps_md1_slowdown", 1.0) for nx in raps_nx]
            ax.plot(raps_nx, raps_md1, '^-.',
                    color="#9C27B0", markersize=3, linewidth=1.2,
                    label="NRAPS M/D/1")

        # ── (3) Calibrated power-law fit on raps_raw_combined_rho ───────
        # slowdown = a · ρ^α (per-topology a, α from LAYER2_POWERLAW_FIT,
        # fit once via log-log polyfit against SST-Macro Layer-2 data).
        fit = _powerlaw_for(topo)
        if raps and fit is not None:
            a, alpha = fit
            raps_nx = sorted(raps.keys())
            raps_cal = [layer2_calibrated_slowdown(
                            _CALIB_KEYS.get(topo, topo),
                            raps[nx].get("raps_raw_combined_rho", 0.0))
                        for nx in raps_nx]
            ax.plot(raps_nx, raps_cal, 's--',
                    color="#E91E63", markersize=3, linewidth=1.0,
                    label=f"NRAPS (a={a:.2f}, α={alpha:.2f})")

        # Per-topology Pearson r (log-log, rho_raw vs SST slowdown)
        title_str = f"{TOPO_LABELS[topo]}"
        if sst and raps:
            common = [nx for nx in sorted(set(sst.keys()) & set(raps.keys()))
                      if nx in VALID_NX_NONZERO]
            if len(common) >= 2:
                log_rho = [np.log(max(raps[nx]["raps_raw_combined_rho"], 1e-9))
                           for nx in common]
                log_sd  = [np.log(max(sst[nx]["slowdown"], 1e-9)) for nx in common]
                r_val = np.corrcoef(log_rho, log_sd)[0, 1]
                title_str = f"{TOPO_LABELS[topo]} (r={r_val:.2f})"

        ax.set_xlabel("Bully nx", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Victim Slowdown", fontsize=8)
        ax.set_title(title_str, fontsize=8)
        ax.set_ylim(bottom=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "interference_simple.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def load_raps_data_from_dir(raps_dir, topo_name):
    """Load RAPS results from a custom directory (for topo variant comparison)."""
    topo_dir = os.path.join(raps_dir, topo_name)
    if not os.path.isdir(topo_dir):
        return None
    data = {}
    for bully_nx in BULLY_NX_VALUES:
        p = os.path.join(topo_dir, f"{topo_name}_bully_nx{bully_nx}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if d.get("status") != "ok":
            continue
        data[bully_nx] = {
            "raps_raw_combined_rho": d.get("raps_raw_combined_rho", 0.0),
            "raps_md1_slowdown": d.get("raps_md1_slowdown", 1.0),
            "raps_victim_rho": d.get("raps_victim_rho", 0.0),
        }
    return data


def plot_topo_variant_comparison():
    """
    Experiment 4A: Plot RAPS M/D/1 slowdown curves for dragonfly bully-victim sweep:
      - All-to-All canonical: output/interference_large/dragonfly/
      - Circulant physical:   output/interference_large_circulant/dragonfly_circulant/

    Shows whether the two topology models give consistent directional predictions.
    Saves to output/figures/main/paper2/interference_topo_compare.png
    """
    raps_alltoall_dir = os.path.join(RAPS_ROOT, "Baseline", "raps", "output", "interference_large")
    raps_circulant_dir = os.path.join(RAPS_ROOT, "Baseline", "raps", "output", "interference_large_circulant")
    sst_dir_large = os.path.join(RAPS_ROOT, "Baseline", "sst-macro", "multi_job", "output",
                                 "interference_large")
    out_paper2 = os.path.join(RAPS_ROOT, "output", "figures", "main", "paper2")
    os.makedirs(out_paper2, exist_ok=True)

    alltoall = load_raps_data_from_dir(raps_alltoall_dir, "dragonfly")
    circulant = load_raps_data_from_dir(raps_circulant_dir, "dragonfly_circulant")
    # SST-Macro ground truth for dragonfly
    sst_gt = {}
    baseline_p = os.path.join(sst_dir_large, "dragonfly", "dragonfly_bully_nx0.json")
    if os.path.exists(baseline_p):
        baseline_us = json.load(open(baseline_p))["victim_avg_iter_us"]
        for nx in BULLY_NX_VALUES:
            p = os.path.join(sst_dir_large, "dragonfly", f"dragonfly_bully_nx{nx}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                if d.get("status") == "ok":
                    sst_gt[nx] = d["victim_avg_iter_us"] / baseline_us

    if not alltoall and not circulant:
        print("[plot_topo_variant_comparison] No circulant data found — "
              "run: python3 Baseline/raps/run_interference_sweep.py --topo-variant circulant")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    nx_vals = BULLY_NX_VALUES

    if sst_gt:
        sst_nx = sorted(sst_gt.keys())
        ax.plot(sst_nx, [sst_gt[n] for n in sst_nx], 'o-',
                color="#2196F3", markersize=4, linewidth=1.5, label="SST-Macro (ground truth)")

    if alltoall:
        aa_nx = sorted(alltoall.keys())
        ax.plot(aa_nx, [alltoall[n]["raps_md1_slowdown"] for n in aa_nx], 's--',
                color="#D95F02", markersize=4, linewidth=1.5, label="NRAPS all-to-all (canonical)")

    if circulant:
        ci_nx = sorted(circulant.keys())
        ax.plot(ci_nx, [circulant[n]["raps_md1_slowdown"] for n in ci_nx], '^-.',
                color="#1B9E77", markersize=4, linewidth=1.5, label="NRAPS circulant (physical)")

    ax.set_xlabel("Bully nx")
    ax.set_ylabel("Victim Slowdown")
    ax.set_title("Dragonfly: all-to-all vs circulant", fontsize=8)
    ax.set_ylim(bottom=0.8)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_paper2, "interference_topo_compare.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "large", "auto"], default="auto")
    parser.add_argument("--topo-compare", action="store_true",
                        help="Also generate Experiment 4 topology comparison figure")
    args = parser.parse_args()

    if args.scale != "auto":
        _SCALE = args.scale
    # Must update globals so load functions see them
    SST_DIR = SCALE_CONFIG[_SCALE]["sst_dir"]   # noqa: F841
    RAPS_DIR = SCALE_CONFIG[_SCALE]["raps_dir"]  # noqa: F841
    TOPO_LABELS = SCALE_CONFIG[_SCALE]["labels"]  # noqa: F841
    # Patch into module namespace
    import types
    mod = sys.modules[__name__]
    mod.SST_DIR = SST_DIR
    mod.RAPS_DIR = RAPS_DIR
    mod.TOPO_LABELS = TOPO_LABELS

    print(f"Using scale: {_SCALE}")
    print(f"  SST dir: {SST_DIR}")
    print(f"  RAPS dir: {RAPS_DIR}")

    plot_interference_comparison()
    plot_simple_comparison()
    if args.topo_compare:
        plot_topo_variant_comparison()
