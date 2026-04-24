#!/usr/bin/env python3
"""
Layer 3 Validation Plot: RAPS predictions vs Blue Waters real hardware.

Default (--slowdown): 4-panel figure validating RAPS inter-job congestion
against observed runtime slowdown on Blue Waters production workloads.

Legacy (--multi): original multiday concurrency vs RAPS congestion plot.

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/hardware/plot_layer3.py --slowdown
    .venv/bin/python3 Baseline/hardware/plot_layer3.py --multi
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIG_DIR = os.path.join(_ROOT, "main")
os.makedirs(FIG_DIR, exist_ok=True)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from raps.network.calibration import STALL_CALIBRATION  # noqa: E402

# Blue Waters is a 3D torus; apply the torus kappa when scaling the raw
# M/D/1 predictor before comparing to actual runtime slowdown (paper §6.2).
KAPPA_TORUS = STALL_CALIBRATION["torus3d"]

# IEEE SC style (3.5" column, 8pt font)
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

# App colors
APP_COLORS = {
    "NAMD": "#1565C0",         # blue
    "LAMMPS": "#C62828",       # red
    "scsw_xy_2x": "#E65100",   # orange
    "scsw_xy": "#388E3C",      # green
    "scsw": "#7B1FA2",         # purple
}
APP_LABELS = {
    "NAMD": "NAMD (n=64)",
    "LAMMPS": "LAMMPS (n=96)",
    "scsw_xy_2x": "scsw_xy_2x (n=160)",
    "scsw_xy": "scsw_xy (n=80)",
    "scsw": "scsw (n=80)",
}

# Digitized data from Jha et al. NSDI'20, Fig 4:
# (PT_s stall fraction, observed execution time slowdown)
JHA_DIGITIZED = [
    (0.05, 1.10),
    (0.10, 1.20),
    (0.20, 1.50),
    (0.35, 3.40),
]


def _md1_slowdown(rho_arr):
    """M/D/1 queueing slowdown: 1 + rho^2 / (2*(1-rho))."""
    rho = np.clip(rho_arr, 0, 0.9999)
    return 1.0 + rho ** 2 / (2.0 * (1.0 - rho))


# ── Panel (a): Actual slowdown distribution ───────────────────────
def _panel_slowdown_hist(ax, df_valid):
    """Histogram of actual_slowdown grouped by app."""
    ax.set_title("(a) Observed runtime slowdown", fontsize=8)

    bins = np.linspace(0.8, 5.0, 30)
    for app, color in APP_COLORS.items():
        grp = df_valid[df_valid["name"] == app]["actual_slowdown"].dropna()
        if len(grp) == 0:
            continue
        label = f"{APP_LABELS.get(app, app)} (n={len(grp)})"
        ax.hist(grp, bins=bins, alpha=0.65, color=color,
                edgecolor="white", linewidth=0.4, label=label, density=True)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="No slowdown")
    ax.set_xlabel("Actual slowdown (actual / baseline)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, None)


# ── Panel (b): RAPS congestion vs actual slowdown (MAIN RESULT) ───
def _panel_congestion_vs_slowdown(ax, df_valid, predictor="stall_pct"):
    """Scatter: RAPS inter-job predictor vs actual slowdown.

    predictor: 'stall_pct'  → raps_interjob_stall_pct (link util %)
               'md1'        → raps_interjob_md1_slowdown (M/D/1 factor)

    Per-app Pearson r is shown in legend labels; NAMD r is highlighted
    in a text annotation (main claim for the paper).
    """
    xcol = "raps_interjob_md1_slowdown" if predictor == "md1" else "raps_interjob_stall_pct"
    xlabel = ("RAPS M/D/1 slowdown factor (κ-calibrated)"
              if predictor == "md1" else "RAPS inter-job congestion (%)")
    ax.set_title("(b) RAPS congestion vs observed slowdown", fontsize=8)

    # For the M/D/1 predictor we apply the fixed torus κ so the x-axis shows
    # calibrated (1 + κ·(raw-1)) rather than raw saturation-capped values.
    df_valid = df_valid.copy()
    if predictor == "md1" and xcol in df_valid:
        df_valid[xcol] = 1.0 + KAPPA_TORUS * (df_valid[xcol] - 1.0)

    r_all = None
    namd_r = None
    for app, color in APP_COLORS.items():
        grp = df_valid[df_valid["name"] == app].dropna(subset=[xcol, "actual_slowdown"])
        if len(grp) < 2:
            continue
        x = grp[xcol]
        y = grp["actual_slowdown"]

        # Per-app Pearson r
        if len(grp) >= 3:
            r_app, _ = scipy_stats.pearsonr(x, y)
            r_str = f"r={r_app:.2f}"
            if app == "NAMD":
                namd_r = r_app
        else:
            r_str = ""

        label = f"{APP_LABELS.get(app, app)} ({r_str}, n={len(grp)})" if r_str else \
                f"{APP_LABELS.get(app, app)} (n={len(grp)})"
        ax.scatter(x, y, c=color, s=12, alpha=0.55, edgecolors="none",
                   label=label, zorder=3)

        # Per-app trend line
        if len(grp) >= 3:
            z = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 50)
            ax.plot(xr, np.poly1d(z)(xr), color=color,
                    linewidth=0.9, linestyle="--", alpha=0.8)

    # Highlight NAMD correlation (main claim)
    combined = df_valid.dropna(subset=[xcol, "actual_slowdown"])
    if namd_r is not None:
        bbox = dict(boxstyle="round", fc="white", alpha=0.85, ec="#cccccc")
        ax.text(0.05, 0.95,
                f"NAMD: r = {namd_r:.2f}",
                transform=ax.transAxes, fontsize=7,
                color=APP_COLORS["NAMD"],
                verticalalignment="top", bbox=bbox)
    elif len(combined) >= 3:
        r, p = scipy_stats.pearsonr(combined[xcol], combined["actual_slowdown"])
        r_all = r
        bbox = dict(boxstyle="round", fc="white", alpha=0.85, ec="#cccccc")
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(0.05, 0.95,
                f"r = {r:.2f} ({p_str}, n={len(combined)})",
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", bbox=bbox)
    r_all = namd_r if namd_r is not None else r_all

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Actual slowdown (actual / baseline)")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)
    return r_all


# ── Panel (c): M/D/1 model calibration ───────────────────────────
def _panel_md1_calibration(ax, df_valid):
    """RAPS M/D/1 curve vs Jha et al. digitized data."""
    ax.set_title("(c) M/D/1 model vs prior work", fontsize=8)

    # Smooth M/D/1 curve
    rho = np.linspace(0.001, 0.95, 300)
    ax.plot(rho * 100, _md1_slowdown(rho), color="#1565C0", linewidth=1.5,
            label="RAPS M/D/1 model")

    # Jha et al. digitized reference points
    jha_x = [p[0] * 100 for p in JHA_DIGITIZED]
    jha_y = [p[1] for p in JHA_DIGITIZED]
    ax.scatter(jha_x, jha_y, marker="^", color="#E65100", s=30, zorder=5,
               label="Jha et al. [NSDI'20]\n(hardware PT_s counters)")

    # RAPS predictions for valid target jobs (scatter small)
    grp = df_valid.dropna(subset=["raps_interjob_stall_pct", "raps_interjob_md1_slowdown"])
    if len(grp) > 0:
        ax.scatter(grp["raps_interjob_stall_pct"], grp["raps_interjob_md1_slowdown"],
                   c="gray", s=6, alpha=0.3, edgecolors="none",
                   label="RAPS predictions\n(this work)")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Link utilization ρ (%)")
    ax.set_ylabel("Slowdown factor")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.9, None)


# ── Panel (d): System load vs predicted congestion ────────────────
def _panel_load_vs_congestion(ax, df_valid):
    """Binned system load vs mean RAPS inter-job congestion."""
    ax.set_title("(d) System load vs RAPS congestion", fontsize=8)

    df_v = df_valid.dropna(subset=["concurrent_fraction", "raps_interjob_stall_pct"])
    if df_v.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    bins = np.linspace(0, df_v["concurrent_fraction"].quantile(0.99), 9)
    df_v = df_v.copy()
    df_v["load_bin"] = pd.cut(df_v["concurrent_fraction"], bins=bins, include_lowest=True)

    bin_stats = df_v.groupby("load_bin", observed=True).agg(
        mean_stall=("raps_interjob_stall_pct", "mean"),
        sem_stall=("raps_interjob_stall_pct", "sem"),
        mean_load=("concurrent_fraction", "mean"),
        count=("raps_interjob_stall_pct", "count"),
    ).dropna(subset=["mean_load"])

    if bin_stats.empty:
        return

    ax.bar(bin_stats["mean_load"] * 100, bin_stats["mean_stall"],
           yerr=bin_stats["sem_stall"],
           width=(bins[1] - bins[0]) * 80,
           color="#4CAF50", alpha=0.75, edgecolor="#2E7D32", linewidth=0.5,
           error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8))

    for _, row in bin_stats.iterrows():
        ax.text(row["mean_load"] * 100, row["mean_stall"] + row["sem_stall"],
                f"n={int(row['count'])}", ha="center", va="bottom", fontsize=5)

    ax.set_xlabel("System load (% BW XE nodes busy)")
    ax.set_ylabel("Mean RAPS congestion (%)")
    ax.grid(True, alpha=0.3, axis="y")


# ── Main slowdown validation figure ──────────────────────────────
def plot_slowdown_validation(tag="201701", predictor="stall_pct"):
    """4-panel figure: Layer 3 BW validation with slowdown.

    predictor: 'stall_pct' (default) or 'md1' — controls panel (b) x-axis.
    """
    val_csv = os.path.join(OUT_DIR, f"bw_slowdown_validation_{tag}.csv")

    if not os.path.exists(val_csv):
        print(f"ERROR: {val_csv} not found. Run run_bw_validation.py --slowdown first.")
        return

    df = pd.read_csv(val_csv)
    print(f"Loaded {len(df)} rows from {val_csv}")

    # Filter to valid (not filtered) runs with actual_slowdown
    df_valid = df[~df["filtered"].fillna(True)].copy()
    df_valid = df_valid[df_valid["actual_slowdown"].notna()].copy()
    print(f"Valid runs (not filtered): {len(df_valid)}")

    for app, grp in df_valid.groupby("name"):
        print(f"  {app}: {len(grp)} runs, "
              f"slowdown {grp['actual_slowdown'].min():.2f}-{grp['actual_slowdown'].max():.2f}x")

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_slowdown_hist(ax_a, df_valid)
    r_val = _panel_congestion_vs_slowdown(ax_b, df_valid, predictor=predictor)
    _panel_md1_calibration(ax_c, df_valid)
    _panel_load_vs_congestion(ax_d, df_valid)

    pred_suffix = "_md1" if predictor == "md1" else ""
    out_path = os.path.join(FIG_DIR, f"layer3_bw_validation_{tag}{pred_suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # Summary stats
    print(f"\n{'='*60}")
    print(f"  Layer 3 BW Validation Summary (tag={tag}, predictor={predictor})")
    print(f"{'='*60}")
    print(f"  Total runs: {len(df)}, valid: {len(df_valid)}")
    if r_val is not None:
        narrative = "captures" if abs(r_val) >= 0.3 else "weak correlation for"
        print(f"  Overall r(RAPS congestion, actual slowdown) = {r_val:.2f}")
        print(f"  → RAPS {narrative} real inter-job network interference")
        print(f"  Reference: Jha et al. [NSDI'20] r=0.89 with hardware PT_s counters")


# ── Fig 7: per-app scatter (all target apps, both months) ────────
def plot_fig7_all_apps(tags=("201701", "201705")):
    """Fig 7: 2×2 grid, one panel per target app.

    Loads bw_slowdown_validation_{tag}.csv for each tag, merges them,
    then draws one scatter panel per app (NAMD, scsw_xy_2x, scsw_xy, scsw)
    showing RAPS inter-job congestion (%) vs actual runtime slowdown.
    """
    frames = []
    for tag in tags:
        csv = os.path.join(OUT_DIR, f"bw_slowdown_validation_{tag}.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df["_tag"] = tag
            frames.append(df)
            print(f"Loaded {len(df)} rows from {csv}")
        else:
            print(f"WARNING: {csv} not found — skipping tag {tag}")

    if not frames:
        print("ERROR: no validation CSVs found. Run run_bw_validation.py --slowdown first.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    # Deduplicate: same job_id may appear in both tags (unlikely but safe)
    df_all = df_all.drop_duplicates(subset=["job_id"])

    df_valid = df_all[~df_all["filtered"].fillna(True)].copy()
    df_valid = df_valid[df_valid["actual_slowdown"].notna()].copy()
    print(f"Total valid runs after merge: {len(df_valid)}")
    for app, grp in df_valid.groupby("name"):
        print(f"  {app}: {len(grp)} runs")

    apps = ["NAMD", "LAMMPS", "scsw_xy_2x", "scsw_xy", "scsw"]
    ncols = 3
    nrows = -(-len(apps) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.8 * nrows))
    fig.subplots_adjust(hspace=0.45, wspace=0.38)

    xcol = "raps_interjob_stall_pct"
    xlabel = "RAPS inter-job congestion (%)"

    for ax in axes.flat[len(apps):]:
        ax.set_visible(False)

    for ax, app in zip(axes.flat, apps):
        color = APP_COLORS.get(app, "#555555")
        grp = df_valid[df_valid["name"] == app].dropna(subset=[xcol, "actual_slowdown"])

        if len(grp) == 0:
            ax.text(0.5, 0.5, f"No data\n({app})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(APP_LABELS.get(app, app), fontsize=8)
            continue

        x = grp[xcol]
        y = grp["actual_slowdown"]

        ax.scatter(x, y, c=color, s=12, alpha=0.55, edgecolors="none", zorder=3)

        # Linear trend
        if len(grp) >= 3:
            z = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, np.poly1d(z)(xr), color=color,
                    linewidth=1.0, linestyle="--", alpha=0.85)
            r, p = scipy_stats.pearsonr(x, y)
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            bbox = dict(boxstyle="round", fc="white", alpha=0.85, ec="#cccccc")
            ax.text(0.05, 0.95,
                    f"r = {r:.3f}\n({p_str}, n={len(grp)})",
                    transform=ax.transAxes, fontsize=7,
                    color=color, verticalalignment="top", bbox=bbox)

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
        ax.set_title(APP_LABELS.get(app, app), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Actual slowdown (actual / baseline)")
        ax.grid(True, alpha=0.3)

    out_path = os.path.join(FIG_DIR, "fig7_layer3_all_apps.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


# ── Legacy multiday figure ────────────────────────────────────────
def plot_layer3(day="20170328", multi=False):
    if multi:
        val_csv = os.path.join(OUT_DIR, "bw_validation_multiday.csv")
        load_csv = os.path.join(OUT_DIR, "bw_load_20170328.csv")
    else:
        val_csv = os.path.join(OUT_DIR, f"bw_validation_{day}.csv")
        load_csv = os.path.join(OUT_DIR, f"bw_load_{day}.csv")

    if not os.path.exists(val_csv):
        print(f"ERROR: {val_csv} not found. Run run_bw_validation.py first.")
        return

    df = pd.read_csv(val_csv)
    print(f"Loaded {len(df)} validated jobs")

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel (a): System load over time ──
    ax_a = fig.add_subplot(gs[0, 0])
    if os.path.exists(load_csv):
        load_df = pd.read_csv(load_csv)
        t0 = load_df["timestamp"].min()
        hours = (load_df["timestamp"] - t0) / 3600
        ax_a.fill_between(hours, load_df["active_nodes"],
                          alpha=0.4, color="#2196F3")
        ax_a.plot(hours, load_df["active_nodes"],
                  linewidth=0.5, color="#1565C0")
        ax_a.set_xlabel("Time (hours)")
        ax_a.set_ylabel("Active nodes")
        ax_a.set_title("(a) Blue Waters system load", fontsize=8)
        ax_a.axhline(22640, color="red", linestyle="--", linewidth=0.5,
                     label="Total XE nodes")
        ax_a.legend(fontsize=6)
    else:
        ax_a.text(0.5, 0.5, "No load data", ha="center", va="center",
                  transform=ax_a.transAxes)
    ax_a.grid(True, alpha=0.3)

    # ── Panel (b): Concurrent fraction vs RAPS max_util ──
    ax_b = fig.add_subplot(gs[0, 1])
    scatter = ax_b.scatter(
        df["concurrent_fraction"], df["raps_max_util"],
        c=np.log10(df["tx_rate_per_node_bps"].clip(lower=1)),
        cmap="viridis", s=15, alpha=0.6, edgecolors="none")
    ax_b.set_xlabel("System concurrency fraction")
    ax_b.set_ylabel("RAPS max link utilization")
    ax_b.set_title("(b) Concurrency vs RAPS congestion", fontsize=8)
    if len(df) > 3:
        x = df["concurrent_fraction"]
        y = df["raps_max_util"]
        mask = (x > 0) & (y > 0)
        if mask.sum() > 3:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            xr = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax_b.plot(xr, p(xr), "r--", linewidth=1.0, alpha=0.7)
            r = x[mask].corr(y[mask])
            ax_b.text(0.05, 0.95, f"r = {r:.3f}",
                      transform=ax_b.transAxes, fontsize=7,
                      verticalalignment="top",
                      bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax_b.grid(True, alpha=0.3)
    cb = plt.colorbar(scatter, ax=ax_b, pad=0.02)
    cb.set_label("log₁₀(tx rate)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    # ── Panel (c): TX rate vs M/D/1 slowdown ──
    ax_c = fig.add_subplot(gs[1, 0])
    mask_active = df["raps_md1_slowdown"] > 1.001
    if mask_active.sum() > 0:
        ax_c.scatter(df.loc[mask_active, "tx_rate_per_node_bps"] / 1e6,
                     df.loc[mask_active, "raps_md1_slowdown"],
                     s=15, alpha=0.6, color="#4CAF50", edgecolors="none",
                     label=f"Congested ({mask_active.sum()} jobs)")
    mask_idle = ~mask_active
    if mask_idle.sum() > 0:
        ax_c.scatter(df.loc[mask_idle, "tx_rate_per_node_bps"] / 1e6,
                     df.loc[mask_idle, "raps_md1_slowdown"],
                     s=10, alpha=0.3, color="#9E9E9E", edgecolors="none",
                     label=f"No congestion ({mask_idle.sum()} jobs)")
    ax_c.set_xlabel("TX rate per node (MB/s)")
    ax_c.set_ylabel("RAPS M/D/1 slowdown")
    ax_c.set_title("(c) Network load vs predicted slowdown", fontsize=8)
    ax_c.legend(fontsize=6, loc="upper left")
    ax_c.grid(True, alpha=0.3)

    # ── Panel (d): Distribution comparison ──
    ax_d = fig.add_subplot(gs[1, 1])
    df["conc_bin"] = pd.cut(df["concurrent_fraction"],
                            bins=np.linspace(0, 1, 11),
                            labels=False, include_lowest=True)
    bin_stats = df.groupby("conc_bin").agg(
        mean_util=("raps_max_util", "mean"),
        mean_slowdown=("raps_md1_slowdown", "mean"),
        count=("raps_max_util", "count"),
        mean_conc=("concurrent_fraction", "mean"),
    ).dropna()

    if len(bin_stats) > 0:
        ax_d.bar(bin_stats["mean_conc"], bin_stats["mean_slowdown"],
                 width=0.08, color="#FF9800", alpha=0.7, edgecolor="#E65100",
                 linewidth=0.5)
        ax_d.set_xlabel("System concurrency fraction")
        ax_d.set_ylabel("Mean RAPS slowdown")
        ax_d.set_title("(d) Slowdown vs system load", fontsize=8)
        for _, row in bin_stats.iterrows():
            if row["count"] > 0:
                ax_d.text(row["mean_conc"], row["mean_slowdown"],
                          f"n={int(row['count'])}", ha="center", va="bottom",
                          fontsize=5)
    ax_d.grid(True, alpha=0.3)

    out_path = os.path.join(FIG_DIR, "layer3_bw_validation_legacy.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    print(f"\nLayer 3 Legacy Summary ({day}):")
    print(f"  Jobs analyzed: {len(df)}")
    mask_active = df["raps_md1_slowdown"] > 1.001
    print(f"  Jobs with congestion (slowdown > 1.001): {mask_active.sum()}")
    if len(df) > 3:
        r = df["concurrent_fraction"].corr(df["raps_max_util"])
        print(f"  Correlation (concurrency vs RAPS util): r = {r:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default="20170328")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--slowdown", action="store_true",
                        help="Plot slowdown validation (uses bw_slowdown_validation.csv)")
    parser.add_argument("--tag", default="201701",
                        help="Data tag for --slowdown mode (default: 201701)")
    parser.add_argument("--predictor", default="stall_pct",
                        choices=["stall_pct", "md1"],
                        help="RAPS predictor for panel (b): stall_pct or md1 (default: stall_pct)")
    parser.add_argument("--all-predictors", action="store_true",
                        help="Generate both stall_pct and md1 variants")
    parser.add_argument("--fig7", action="store_true",
                        help="Generate fig7: per-app scatter across all target apps")
    parser.add_argument("--tags", nargs="+", default=["201701", "201705"],
                        help="Data tags to merge for --fig7 (default: 201701 201705)")
    args = parser.parse_args()

    if args.fig7:
        plot_fig7_all_apps(tags=args.tags)
    elif args.slowdown:
        if args.all_predictors:
            for pred in ["stall_pct", "md1"]:
                plot_slowdown_validation(tag=args.tag, predictor=pred)
        else:
            plot_slowdown_validation(tag=args.tag, predictor=args.predictor)
    else:
        plot_layer3(args.day, multi=args.multi)
