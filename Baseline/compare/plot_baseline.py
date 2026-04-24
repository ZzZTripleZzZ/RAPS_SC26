#!/usr/bin/env python3
"""
Baseline/compare/plot_baseline.py — Side-by-side simulator validation figures.

Reads comparison CSVs from aggregate.py and produces:
  - Baseline/compare/fig_baseline_dragonfly.pdf  (3-panel: util, stall, latency vs ρ)
  - Baseline/compare/fig_baseline_fattree.pdf
  - Baseline/compare/fig_baseline_combined.pdf   (2×3 panels, both topologies)

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/compare/aggregate.py   # first!
    .venv/bin/python3 Baseline/compare/plot_baseline.py

Figure layout (per topology):
  Panel (a): Mean link utilization vs ρ
  Panel (b): Stall ratio vs ρ  (log scale)
  Panel (c): Slowdown factor vs ρ
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style settings (IEEE two-column, 3.5" figures)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.dpi": 150,
})

_COMPARE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_COMPARE_DIR)

# ---------------------------------------------------------------------------
# Simulator display config
# ---------------------------------------------------------------------------
SIM_STYLES = {
    "raps":         dict(color="#e74c3c", marker="o",  linestyle="-",  label="RAPS (M/D/1)", zorder=5),
    "booksim2":     dict(color="#2980b9", marker="s",  linestyle="--", label="BookSim2",      zorder=4),
    "sst-macro":    dict(color="#27ae60", marker="^",  linestyle="-.", label="SST-Macro",     zorder=4),
    "simgrid":      dict(color="#8e44ad", marker="D",  linestyle=":",  label="SimGrid",       zorder=4),
    "simgrid_analytical": dict(color="#8e44ad", marker="d", linestyle=":", label="SimGrid (approx)", zorder=3),
    "codes":        dict(color="#e67e22", marker="v",  linestyle="--", label="CODES",         zorder=4),
    "ns3":          dict(color="#16a085", marker="<",  linestyle="-.", label="ns-3",          zorder=3),
    "md1_theory":   dict(color="#2c3e50", marker="",   linestyle="-",  label="M/D/1 theory",  zorder=6, linewidth=1.0, alpha=0.5),
}

TOPOLOGY_LABELS = {
    "dragonfly": "Dragonfly (9g×4r×2h, 25 GB/s)",
    "fattree":   "Fat-tree (k=4, 12.5 GB/s)",
    "torus3d":   "3D Torus (8³×2, 9.6 GB/s)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_comparison(csv_path: str) -> dict:
    """Load comparison CSV into {simulator: {metric: [values sorted by rho]}}."""
    rows = []
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found. Run aggregate.py first.")
        return {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    data = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: (r.get("simulator",""), float(r.get("rho_target",0) or 0))):
        sim = row.get("simulator", "unknown")
        try:
            rho = float(row["rho_target"])
        except (KeyError, ValueError, TypeError):
            continue

        data[sim]["rho"].append(rho)
        for field in ["mean_utilization", "max_utilization", "stall_ratio", "slowdown", "avg_latency_ns", "md1_stall_ratio", "md1_slowdown"]:
            val = row.get(field)
            try:
                data[sim][field].append(float(val) if val not in (None, "", "None") else np.nan)
            except (ValueError, TypeError):
                data[sim][field].append(np.nan)

    return dict(data)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_metric(ax, data: dict, metric: str, ylabel: str, title: str,
                 show_legend: bool = False, logy: bool = False):
    """Plot one metric across all simulators."""
    rho_theory = np.linspace(0.01, 0.99, 200)

    # M/D/1 theory curve (background reference)
    if metric == "stall_ratio":
        theory_vals = rho_theory**2 / (2 * (1 - rho_theory))
        ax.plot(rho_theory, theory_vals, **{**SIM_STYLES["md1_theory"], "zorder": 1})
    elif metric == "slowdown":
        theory_vals = 1 + rho_theory**2 / (2 * (1 - rho_theory))
        ax.plot(rho_theory, theory_vals, **{**SIM_STYLES["md1_theory"], "zorder": 1})
    elif metric == "mean_utilization":
        # Ideal: utilization = rho
        ax.plot(rho_theory, rho_theory, **{**SIM_STYLES["md1_theory"], "label": "ideal (util=ρ)", "zorder": 1})

    # Per-simulator data
    for sim, sim_data in sorted(data.items(), key=lambda kv: kv[0]):
        if sim == "md1_theory":
            continue
        style = SIM_STYLES.get(sim, dict(color="gray", marker="x", linestyle="-", label=sim))
        rhos = np.array(sim_data.get("rho", []))
        vals = np.array(sim_data.get(metric, []))
        if len(rhos) == 0 or np.all(np.isnan(vals)):
            continue

        mask = ~np.isnan(vals)
        ax.plot(rhos[mask], vals[mask],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                label=style["label"],
                zorder=style.get("zorder", 3))

    ax.set_xlabel("Injection rate ρ")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, 0.85)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if logy:
        ax.set_yscale("log")
    if show_legend:
        ax.legend(loc="upper left", framealpha=0.9)


def plot_topology(topo: str, data: dict, out_path: str):
    """3-panel figure for one topology."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    fig.subplots_adjust(wspace=0.40, left=0.08, right=0.98, bottom=0.22, top=0.88)

    topo_label = TOPOLOGY_LABELS.get(topo, topo)
    fig.suptitle(f"Simulator Validation — {topo_label}", fontsize=8, fontweight="bold")

    _plot_metric(axes[0], data, "mean_utilization",
                 ylabel="Mean link util.", title="(a) Link utilization",
                 show_legend=True)
    _plot_metric(axes[1], data, "stall_ratio",
                 ylabel="Stall ratio", title="(b) Stall ratio", logy=True)
    _plot_metric(axes[2], data, "slowdown",
                 ylabel="Slowdown factor", title="(c) Slowdown")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined(dragonfly_data: dict, fattree_data: dict, out_path: str,
                  torus3d_data: dict = None):
    """3×3 panel figure (dragonfly / fat-tree / torus3d rows, util/stall/slowdown cols)."""
    n_rows = 3 if torus3d_data else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(7.0, n_rows * 2.1))
    fig.subplots_adjust(wspace=0.42, hspace=0.52,
                        left=0.08, right=0.98, bottom=0.08, top=0.93)

    fig.suptitle("RAPS Accuracy Validation: Comparison with Packet-level Simulators",
                 fontsize=8.5, fontweight="bold")

    # Row 0: dragonfly
    _plot_metric(axes[0, 0], dragonfly_data, "mean_utilization",
                 ylabel="Mean link util.", title="(a) Dragonfly: link util.", show_legend=True)
    _plot_metric(axes[0, 1], dragonfly_data, "stall_ratio",
                 ylabel="Stall ratio", title="(b) Dragonfly: stall ratio", logy=True)
    _plot_metric(axes[0, 2], dragonfly_data, "slowdown",
                 ylabel="Slowdown", title="(c) Dragonfly: slowdown")

    # Row 1: fat-tree
    _plot_metric(axes[1, 0], fattree_data, "mean_utilization",
                 ylabel="Mean link util.", title="(d) Fat-tree: link util.")
    _plot_metric(axes[1, 1], fattree_data, "stall_ratio",
                 ylabel="Stall ratio", title="(e) Fat-tree: stall ratio", logy=True)
    _plot_metric(axes[1, 2], fattree_data, "slowdown",
                 ylabel="Slowdown", title="(f) Fat-tree: slowdown")

    # Row 2: torus3d (if data provided)
    if torus3d_data and n_rows == 3:
        _plot_metric(axes[2, 0], torus3d_data, "mean_utilization",
                     ylabel="Mean link util.", title="(g) 3D Torus: link util.")
        _plot_metric(axes[2, 1], torus3d_data, "stall_ratio",
                     ylabel="Stall ratio", title="(h) 3D Torus: stall ratio", logy=True)
        _plot_metric(axes[2, 2], torus3d_data, "slowdown",
                     ylabel="Slowdown", title="(i) 3D Torus: slowdown")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    compare_dir = _COMPARE_DIR

    print("[plot_baseline] Loading comparison data...")
    df_csv = os.path.join(compare_dir, "comparison_dragonfly.csv")
    ft_csv = os.path.join(compare_dir, "comparison_fattree.csv")
    t3d_csv = os.path.join(compare_dir, "comparison_torus3d.csv")

    df_data  = load_comparison(df_csv)
    ft_data  = load_comparison(ft_csv)
    t3d_data = load_comparison(t3d_csv)

    if not df_data and not ft_data and not t3d_data:
        print("ERROR: No comparison data found. Run aggregate.py first.")
        sys.exit(1)

    print(f"  Dragonfly simulators: {', '.join(s for s in df_data if s != 'md1_theory') or 'none'}")
    print(f"  Fat-tree simulators:  {', '.join(s for s in ft_data if s != 'md1_theory') or 'none'}")
    print(f"  Torus3d simulators:   {', '.join(s for s in t3d_data if s != 'md1_theory') or 'none'}")

    print("\n[plot_baseline] Generating figures...")

    if df_data:
        plot_topology("dragonfly", df_data,
                      os.path.join(compare_dir, "fig_baseline_dragonfly.pdf"))

    if ft_data:
        plot_topology("fattree", ft_data,
                      os.path.join(compare_dir, "fig_baseline_fattree.pdf"))

    if t3d_data:
        plot_topology("torus3d", t3d_data,
                      os.path.join(compare_dir, "fig_baseline_torus3d.pdf"))

    if df_data or ft_data or t3d_data:
        plot_combined(df_data or {}, ft_data or {},
                      os.path.join(compare_dir, "fig_baseline_combined.pdf"),
                      torus3d_data=t3d_data or {})

    print("\n[plot_baseline] Done.")
    print(f"  Output directory: {compare_dir}/")


if __name__ == "__main__":
    main()
