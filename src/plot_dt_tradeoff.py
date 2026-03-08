#!/usr/bin/env python3
"""
Speed–Accuracy Tradeoff vs Time Quantum (Δt) — SC paper single-column figures.

Each logical panel is saved as an independent figure (3.5 × 2.5 in, 7:5 ratio).
Five output files:
  fig_dt_speedup.png       — speedup vs Δt (all systems & node counts)
  fig_dt_cost.png          — per-tick computation cost vs Δt
  fig_dt_tradeoff.png      — dual-axis speedup + accuracy error vs Δt
  fig_dt_accuracy.png      — per-metric accuracy breakdown (bar)
  fig_dt_efficiency.png    — normalised efficiency score vs Δt

Usage:
    python src/plot_dt_tradeoff.py
    python src/plot_dt_tradeoff.py --csv output/frontier_scaling/results.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths & constants ────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "output/frontier_scaling/results.csv"
OUT_DIR     = ROOT / "output/figures"

DT_ORDER      = [0.1, 1.0, 10.0, 60.0, 300.0, 600.0]
SYS_LABELS    = {"frontier": "Frontier (dragonfly)", "lassen": "Lassen (fat-tree)",
                  "bluewaters": "Blue Waters (torus3d)"}
SYS_COLORS    = {"frontier": "#D95F02", "lassen": "#1B9E77", "bluewaters": "#7570B3"}
NODE_MARKERS  = {100: "o", 1_000: "s", 10_000: "^"}
NODE_LS       = {100: ":", 1_000: "--", 10_000: "-"}

# ── SC paper rcParams ────────────────────────────────────────────────────────
FIGW = 3.5   # single-column width (inches)
FIGH = 2.5   # slightly taller for multi-line + legend
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


# ── Data helpers ─────────────────────────────────────────────────────────────

def rel_error(val, ref, eps=1e-10):
    return abs(val - ref) / max(abs(ref), eps) * 100.0


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    df["delta_t"] = df["delta_t"].astype(float).round(3)
    return df


def determine_reference_dt(df: pd.DataFrame) -> float:
    avail = sorted(df["delta_t"].unique())
    ref = avail[0]
    if ref != 0.1:
        print(f"  Note: Δt=0.1 not available; using Δt={ref} as reference.")
    return ref


def compute_tradeoff(df: pd.DataFrame, reference_dt: float) -> pd.DataFrame:
    rows = []
    for (system, nc, rep), grp in df.groupby(["system", "node_count", "repeat"]):
        ref_rows = grp[grp["delta_t"] == reference_dt]
        if ref_rows.empty:
            continue
        ref = ref_rows.iloc[0]
        for _, row in grp.iterrows():
            cng_err = rel_error(row["avg_congestion"], ref["avg_congestion"])
            sld_err = rel_error(row["avg_slowdown"],   ref["avg_slowdown"])
            rows.append({
                "system":           system,
                "node_count":       nc,
                "delta_t":          row["delta_t"],
                "repeat":           rep,
                "speedup":          row["speedup"],
                "per_tick_ms":      row["per_tick_ms"],
                "congestion_error": cng_err,
                "slowdown_error":   sld_err,
                "combined_error":   np.mean([cng_err, sld_err]),
            })
    return pd.DataFrame(rows)


def aggregate(tradeoff: pd.DataFrame) -> pd.DataFrame:
    return (tradeoff
            .groupby(["system", "node_count", "delta_t"])
            .agg(
                speedup_mean        = ("speedup",              "mean"),
                speedup_std         = ("speedup",              "std"),
                per_tick_ms_mean    = ("per_tick_ms",          "mean"),
                per_tick_ms_std     = ("per_tick_ms",          "std"),
                congestion_err_mean = ("congestion_error",     "mean"),
                slowdown_err_mean   = ("slowdown_error",       "mean"),
                combined_err_mean   = ("combined_error",       "mean"),
                combined_err_std    = ("combined_error",       "std"),
            )
            .reset_index())


def _save(fig, fname):
    out = OUT_DIR / Path(fname).with_suffix('.png').name
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def _xticks_log(ax, dt_vals):
    ax.set_xscale("log")
    ax.set_xticks(dt_vals)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}s"))


def _ygrid(ax):
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')


def _recommended_xi(err_vals, eps=4.0):
    """Return index of recommended Δt using the accuracy-cliff criterion.

    Recommended Δt = coarsest Δt where going to the next-finer Δt would
    improve accuracy by >= eps%.  Rationale: beyond this point, using a
    coarser Δt costs real accuracy; finer than this gives no additional gain.

    If no such cliff exists (e.g., all errors already 0%), default to the
    coarsest Δt (highest index).
    """
    n = len(err_vals)
    result = n - 1                        # default: coarsest
    for i in range(n - 1, 0, -1):        # search from coarsest inward
        if err_vals[i] - err_vals[i - 1] >= eps:
            result = i
            break
    return result


# ── Figure 1: Speedup vs Δt ──────────────────────────────────────────────────

def plot_speedup(agg: pd.DataFrame, out_dir: Path, reference_dt: float):
    dt_in_data = sorted(agg["delta_t"].unique())
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    legend_lines = []
    for system in ["frontier", "lassen", "bluewaters"]:
        color = SYS_COLORS[system]
        sdf   = agg[agg["system"] == system]
        if sdf.empty:
            continue
        for nc in sorted(sdf["node_count"].unique()):
            marker = NODE_MARKERS.get(nc, "D")
            ls     = NODE_LS.get(nc, "-")
            ndf = (sdf[sdf["node_count"] == nc]
                   .set_index("delta_t")
                   .reindex([d for d in DT_ORDER if d in dt_in_data]))
            if ndf.empty:
                continue
            ax.errorbar(ndf.index, ndf["speedup_mean"],
                        yerr=ndf["speedup_std"].fillna(0),
                        marker=marker, linestyle=ls, color=color,
                        linewidth=1.8, capsize=3, alpha=0.9)
            legend_lines.append(
                Line2D([0], [0], color=color, marker=marker, linestyle=ls,
                       linewidth=1.8, label=f"{SYS_LABELS[system]},  n={nc:,}"))

    # Δt-linear reference line
    ref_sp = agg[agg["delta_t"] == reference_dt]["speedup_mean"].mean()
    if np.isfinite(ref_sp):
        dt_range = np.array([min(dt_in_data), max(dt_in_data)])
        ax.plot(dt_range, ref_sp * (dt_range / reference_dt),
                "k--", linewidth=1.0, alpha=0.35)
        legend_lines.append(Line2D([0], [0], color="k", linestyle="--",
                                   linewidth=1.0, alpha=0.45,
                                   label=r"$\Delta t$-linear (theory)"))

    _xticks_log(ax, dt_in_data)
    ax.set_yscale("log")
    ax.set_xlabel(r"Time quantum $\Delta t$  (s)")
    ax.set_ylabel("Simulation speedup")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}×"))
    ax.legend(handles=legend_lines, fontsize=7, loc="upper left",
              ncol=1, framealpha=0.88)
    _ygrid(ax)
    fig.tight_layout(pad=0.5)
    _save(fig, "fig_dt_speedup.png")


# ── Figure 2: Per-tick cost vs Δt ────────────────────────────────────────────

def plot_cost_per_tick(agg: pd.DataFrame, out_dir: Path, reference_dt: float):
    dt_in_data = sorted(agg["delta_t"].unique())
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    for system in ["frontier", "lassen", "bluewaters"]:
        color = SYS_COLORS[system]
        sdf   = agg[agg["system"] == system]
        if sdf.empty:
            continue
        for nc in sorted(sdf["node_count"].unique()):
            marker = NODE_MARKERS.get(nc, "D")
            ls     = NODE_LS.get(nc, "-")
            ndf = (sdf[sdf["node_count"] == nc]
                   .set_index("delta_t")
                   .reindex([d for d in DT_ORDER if d in dt_in_data]))
            if ndf.empty:
                continue
            ax.errorbar(ndf.index, ndf["per_tick_ms_mean"],
                        yerr=ndf["per_tick_ms_std"].fillna(0),
                        marker=marker, linestyle=ls, color=color,
                        linewidth=1.8, capsize=3, alpha=0.9,
                        label=f"{SYS_LABELS[system]},  n={nc:,}")

    _xticks_log(ax, dt_in_data)
    ax.set_yscale("log")
    ax.set_xlabel(r"Time quantum $\Delta t$  (s)")
    ax.set_ylabel("Cost per tick (ms)")
    ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.88)
    _ygrid(ax)
    fig.tight_layout(pad=0.5)
    _save(fig, "fig_dt_cost.png")


# ── Figure 3: Speedup & accuracy vs Δt (dual-axis) ──────────────────────────

def plot_speedup_accuracy(agg: pd.DataFrame, out_dir: Path, reference_dt: float):
    dt_in_data = sorted(agg["delta_t"].unique())
    x_pos   = np.arange(len(dt_in_data))          # evenly spaced categorical axis
    x_labels = [f"{dt:g}s" for dt in dt_in_data]

    fig, ax_sp = plt.subplots(figsize=(FIGW, 2.3))  # match UC figure dimensions
    ax_err = ax_sp.twinx()

    # Pre-compute per-system data and efficiency-optimal index
    sys_data = {}
    for system in ["frontier", "lassen", "bluewaters"]:
        sdf = agg[agg["system"] == system]
        if sdf.empty:
            continue
        mdt = (sdf.groupby("delta_t")
               .agg(speedup=("speedup_mean", "mean"),
                    err    =("combined_err_mean", "mean"))
               .reindex([d for d in DT_ORDER if d in dt_in_data]))
        sys_data[system] = {"mdt": mdt,
                            "opt_xi": _recommended_xi(mdt["err"].values)}

    FS = 6.5    # uniform small font for all text in this figure

    for system in ["frontier", "lassen", "bluewaters"]:
        if system not in sys_data:
            continue
        color = SYS_COLORS[system]
        mdt   = sys_data[system]["mdt"]
        oi    = sys_data[system]["opt_xi"]

        # Short label: system name only; line style is self-evident from dual axes
        short = SYS_LABELS[system].split(' (')[0]
        ax_sp.plot(x_pos, mdt["speedup"].values, "o-",
                   color=color, linewidth=1.3, markersize=3.5, alpha=0.9,
                   label=short)
        ax_err.plot(x_pos, mdt["err"].values, "s--",
                    color=color, linewidth=1.1, markersize=3, alpha=0.55)

        # ★ at efficiency-optimal; color identifies system
        ax_sp.plot(x_pos[oi], mdt["speedup"].values[oi],
                   "*", color=color, markersize=8, zorder=5)

    ax_sp.set_xticks(x_pos)
    ax_sp.set_xticklabels(x_labels, fontsize=FS)
    ax_sp.set_yscale("log")
    ax_sp.set_yticks([100, 1000, 10000])       # evenly spaced on log scale
    ax_sp.set_ylim(50, 25000)
    ax_sp.set_xlabel(r"Time quantum $\Delta t$", fontsize=FS + 1)
    ax_sp.set_ylabel("Simulation speedup", fontsize=FS + 1)
    ax_sp.tick_params(axis='y', labelsize=FS)
    ax_sp.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}×"))
    ax_err.set_ylabel(f"Error vs Δt={reference_dt:g}s (%)", color="dimgray",
                      fontsize=FS + 2)
    ax_err.set_yticks([0, 2, 4, 6])
    ax_err.tick_params(axis='y', labelcolor="dimgray", labelsize=FS + 1)
    ax_err.set_ylim(-0.3, 7.5)

    ax_sp.legend(fontsize=FS, loc="upper left", framealpha=0.88,
                 handlelength=1.5, handletextpad=0.4, borderpad=0.4)
    ax_sp.yaxis.grid(True, linestyle='--', alpha=0.2, linewidth=0.6, color='#888')
    fig.tight_layout(pad=0.4)
    _save(fig, "fig_dt_tradeoff.png")


# ── Figure 4: Per-metric accuracy breakdown ──────────────────────────────────

def plot_accuracy_breakdown(agg: pd.DataFrame, out_dir: Path, reference_dt: float):
    dt_in_data = sorted(agg["delta_t"].unique())
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    w = 0.15
    metric_colors = ["#D95F02", "#7570B3"]
    metric_names  = ["Avg congestion", "Avg slowdown"]
    metric_keys   = ["cong_err", "slow_err"]

    for si, system in enumerate(["frontier", "lassen", "bluewaters"]):
        color = SYS_COLORS[system]
        sdf   = agg[agg["system"] == system]
        if sdf.empty:
            continue
        mdt = (sdf.groupby("delta_t")
               .agg(cong_err=("congestion_err_mean", "mean"),
                    slow_err=("slowdown_err_mean",   "mean"))
               .reindex([d for d in DT_ORDER if d in dt_in_data]))

        x = np.arange(len(mdt))
        offset = si * (2 * w + 0.06)
        for mi, (mkey, mname, mcol) in enumerate(
                zip(metric_keys, metric_names, metric_colors)):
            ax.bar(x + offset + mi * w, mdt[mkey].fillna(0), width=w,
                   color=mcol, alpha=0.78,
                   label=mname if si == 0 else "_nolegend_")

        for xi in x:
            ax.text(xi + offset + w, -0.5, system.capitalize(),
                    ha="center", va="top", fontsize=7, color=color, alpha=0.8)

    ax.set_xticks(np.arange(len(dt_in_data)) + 0.3)
    ax.set_xticklabels([f"Δt={dt:g}s" for dt in dt_in_data], rotation=15, ha='right')
    ax.set_ylabel(f"Accuracy error vs Δt={reference_dt:g}s  (%)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper left")
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.7, color='#888')
    fig.tight_layout(pad=0.5)
    _save(fig, "fig_dt_accuracy.png")


# ── Figure 5: Efficiency score vs Δt ────────────────────────────────────────

def plot_efficiency_score(agg: pd.DataFrame, out_dir: Path, reference_dt: float):
    dt_in_data = sorted(agg["delta_t"].unique())
    x_pos = np.arange(len(dt_in_data))          # evenly spaced categorical positions
    x_labels = [f"{dt:g}s" for dt in dt_in_data]

    fig, ax = plt.subplots(figsize=(FIGW, 2.3))  # match UC figure dimensions

    best_dts = {}
    for system in ["frontier", "lassen", "bluewaters"]:
        color = SYS_COLORS[system]
        sdf   = agg[agg["system"] == system]
        if sdf.empty:
            continue
        mdt = (sdf.groupby("delta_t")
               .agg(speedup      =("speedup_mean",      "mean"),
                    combined_err =("combined_err_mean", "mean"))
               .reindex([d for d in DT_ORDER if d in dt_in_data]))

        eff = mdt["speedup"] * (1 - mdt["combined_err"] / 100).clip(lower=0)
        eff_norm = eff / eff.max() if eff.max() > 0 else eff

        ax.plot(x_pos, eff_norm.values, "o-", color=color, linewidth=1.8,
                markersize=5, label=SYS_LABELS[system])

        # ★ at the accuracy-cliff recommended Δt
        best_xi = _recommended_xi(mdt["combined_err"].values)
        ax.plot(x_pos[best_xi], float(eff_norm.iloc[best_xi]),
                "*", color=color, markersize=11, zorder=5)
        best_dts[system] = best_xi

    # Single vline per unique best Δt index
    for xi in sorted(set(best_dts.values())):
        ax.axvline(xi, color='#999', linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_xlabel(r"Time quantum $\Delta t$", fontsize=8)
    ax.set_ylabel("Normalized efficiency\n(speedup × accuracy)", fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(0, 1.22)
    ax.legend(fontsize=7, loc="upper left")
    _ygrid(ax)
    fig.tight_layout(pad=0.5)
    _save(fig, "fig_dt_efficiency.png")


# ── Console summary ──────────────────────────────────────────────────────────

def print_summary(agg: pd.DataFrame, reference_dt: float):
    print()
    print("=" * 72)
    print("SPEED–ACCURACY TRADEOFF SUMMARY")
    print(f"  Reference: Δt={reference_dt:g}s  (error=0% by definition)")
    print("=" * 72)
    for system in ["frontier", "lassen", "bluewaters"]:
        sdf = agg[agg["system"] == system]
        if sdf.empty:
            continue
        print(f"\n{SYS_LABELS[system]}")
        print(f"  {'Δt (s)':>8}  {'Speedup (mean)':>15}  {'Error (%)':>10}")
        print("  " + "-" * 40)
        mdt = (sdf.groupby("delta_t")
               .agg(sp_mean=("speedup_mean", "mean"),
                    err    =("combined_err_mean", "mean"))
               .reindex([d for d in DT_ORDER if d in sdf["delta_t"].values]))
        for dt_val, row in mdt.iterrows():
            print(f"  {dt_val:>8.1f}  {row.sp_mean:>15,.0f}×  {row.err:>10.2f}%")
        rec_xi = _recommended_xi(mdt["err"].values, eps=4.0)
        rec_dt = mdt.index[rec_xi]
        print(f"\n  ★ Recommended Δt={rec_dt:g}s  "
              f"→  speedup={mdt.loc[rec_dt,'sp_mean']:,.0f}×,  "
              f"error={mdt.loc[rec_dt,'err']:.2f}%")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot Δt speed-accuracy tradeoff (SC single-column figures)")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--reference-dt", type=float, default=None)
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    elif DEFAULT_CSV.exists() and DEFAULT_CSV.stat().st_size > 100:
        csv_path = DEFAULT_CSV
        print(f"Using results CSV: {csv_path}")
    else:
        print("No results CSV found. Run src/run_frontier.py first.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    print(f"Loaded {len(df)} experiments from {csv_path.name}")
    print(f"  Systems:     {sorted(df['system'].unique())}")
    print(f"  Node counts: {sorted(df['node_count'].unique())}")
    print(f"  Δt values:   {sorted(df['delta_t'].unique())}")

    if df.empty:
        print("No OK experiments found.")
        sys.exit(1)

    reference_dt = args.reference_dt or determine_reference_dt(df)
    tradeoff = compute_tradeoff(df, reference_dt)
    agg      = aggregate(tradeoff)

    print_summary(agg, reference_dt)

    plot_speedup(agg, out_dir, reference_dt)
    plot_cost_per_tick(agg, out_dir, reference_dt)
    plot_speedup_accuracy(agg, out_dir, reference_dt)
    plot_accuracy_breakdown(agg, out_dir, reference_dt)
    plot_efficiency_score(agg, out_dir, reference_dt)

    # ── Copy key figure to main/ ─────────────────────────────────────────────
    import shutil
    main_dir = out_dir / 'main'
    main_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ('fig_dt_tradeoff.png',   'benchmark_dt_tradeoff.png'),
        ('fig_dt_efficiency.png', 'benchmark_dt_efficiency.png'),
        ('fig_dt_tradeoff.png',   'dt_tradeoff.png'),
        ('fig_dt_efficiency.png', 'dt_efficiency.png'),
    ]:
        src = out_dir / src_name
        if src.exists():
            shutil.copy2(src, main_dir / dst_name)
            print(f"  Copied {src_name} → main/{dst_name}")


if __name__ == "__main__":
    main()
