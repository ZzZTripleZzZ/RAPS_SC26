#!/usr/bin/env python3
"""
Compute quantitative error metrics comparing NRAPS predictions to SST-Macro
for the Layer 2 bully-victim interference experiment (large scale).

Two prediction models are evaluated:
  1. Raw M/D/1 (eff_tir fallback for saturated links)
  2. Calibrated power-law: slowdown = exp(β) * ρ_raw_combined^α
     (same calibration used in interference_simple.png)

Metrics reported per topology:
  - MAPE: Mean Absolute Percentage Error over bully_nx ∈ {100,150,200,300,400}
  - Max relative error and corresponding bully_nx
  (bully_nx=50 excluded: parsing artifact causing victim_avg_iter_us < baseline)

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    python3 Baseline/compare/compute_mape.py
"""

import json
import math
import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SST_DIR  = os.path.join(_ROOT, "Baseline", "sst-macro", "multi_job", "output", "interference_large")
RAPS_DIR = os.path.join(_ROOT, "Baseline", "raps", "output", "interference_large")

# bully_nx=50 excluded (parsing artifact: victim appears faster than no-bully baseline)
# bully_nx=0  excluded (baseline, slowdown=1.0 by definition — not a prediction target)
BULLY_NX_EVAL = [100, 150, 200, 300, 400]

TOPO_MAP = {
    "dragonfly": ("dragonfly", "dragonfly"),
    "fattree":   ("fattree",   "fattree"),
    "torus3d":   ("torus",     "torus"),
}


def load_sst_slowdown(topo_key):
    """Load SST-Macro slowdown values normalized to no-bully baseline."""
    sst_subdir, _ = TOPO_MAP[topo_key]
    base_path = os.path.join(SST_DIR, sst_subdir, f"{sst_subdir}_bully_nx0.json")
    if not os.path.exists(base_path):
        return None
    baseline_us = json.load(open(base_path))["victim_avg_iter_us"]

    slowdowns = {}
    for nx in BULLY_NX_EVAL:
        p = os.path.join(SST_DIR, sst_subdir, f"{sst_subdir}_bully_nx{nx}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if d.get("status") != "ok":
            continue
        slowdowns[nx] = d["victim_avg_iter_us"] / baseline_us
    return slowdowns


def load_raps_data(topo_key):
    """Load NRAPS M/D/1 slowdown and raw combined rho for each bully_nx."""
    _, raps_subdir = TOPO_MAP[topo_key]

    md1      = {}
    rho_raw  = {}
    saturated = {}
    for nx in BULLY_NX_EVAL:
        p = os.path.join(RAPS_DIR, raps_subdir, f"{raps_subdir}_bully_nx{nx}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if d.get("status") != "ok":
            continue
        md1[nx]       = d["raps_md1_slowdown"]
        rho_raw[nx]   = d["raps_raw_combined_rho"]
        saturated[nx] = d.get("raps_victim_rho", 0.0) >= 1.0
    return md1, rho_raw, saturated


def fit_calibration(sst_sd, rho_raw):
    """
    Fit power-law calibration in log-log space:
        log(SST_slowdown) = α * log(ρ_raw) + β
    Returns (α, β, r_pearson) using the common evaluation points.
    """
    common = sorted(set(sst_sd) & set(rho_raw))
    if len(common) < 2:
        return None, None, None

    log_rho = [math.log(max(rho_raw[nx], 1e-9)) for nx in common]
    log_sd  = [math.log(max(sst_sd[nx],  1e-9)) for nx in common]

    alpha, beta = np.polyfit(log_rho, log_sd, 1)
    r_val = float(np.corrcoef(log_rho, log_sd)[0, 1])
    return alpha, beta, r_val


def apply_calibration(rho_raw, alpha, beta):
    """Apply calibrated power-law: slowdown = exp(β) * ρ_raw^α."""
    return {nx: math.exp(beta) * max(rho_raw[nx], 1e-9) ** alpha
            for nx in rho_raw}


def compute_metrics(sst_sd, pred_sd):
    """Return (MAPE%, max_err%, bully_nx_at_max) for overlapping bully_nx values."""
    common = sorted(set(sst_sd) & set(pred_sd))
    if not common:
        return None, None, None

    errors = [(nx, abs(pred_sd[nx] - sst_sd[nx]) / sst_sd[nx] * 100.0)
              for nx in common if sst_sd[nx] > 0]
    if not errors:
        return None, None, None

    mape = sum(e for _, e in errors) / len(errors)
    max_nx, max_err = max(errors, key=lambda x: x[1])
    return mape, max_err, max_nx


def main():
    print("=" * 80)
    print("NRAPS vs SST-Macro: Quantitative Error Metrics (Layer 2 Validation)")
    print(f"Evaluation points: bully_nx ∈ {BULLY_NX_EVAL}")
    print("(bully_nx=50 excluded — parsing artifact)")
    print("=" * 80)

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("Model: Raw M/D/1 (eff_tir fallback for dragonfly/fat-tree)")
    print(f"  {'Topology':<12} {'MAPE':>8} {'Max Err':>10} {'At nx':>7}  Notes")
    print("  " + "-" * 60)
    for topo_key in ["dragonfly", "fattree", "torus3d"]:
        sst_sd = load_sst_slowdown(topo_key)
        md1, rho_raw, saturated = load_raps_data(topo_key)
        if sst_sd is None or not md1:
            print(f"  {topo_key:<12}  DATA MISSING")
            continue
        mape, max_err, max_nx = compute_metrics(sst_sd, md1)
        if mape is None:
            print(f"  {topo_key:<12}  NO OVERLAP")
            continue
        n_sat = sum(1 for v in saturated.values() if v)
        note = f"eff_tir fallback ({n_sat}/{len(saturated)} pts)" if n_sat else "M/D/1 sub-sat"
        print(f"  {topo_key:<12} {mape:>7.1f}% {max_err:>9.1f}% {max_nx:>6}   {note}")

    print()
    print("Model: Calibrated power-law  slowdown = exp(β)·ρ_raw^α  (fitted on same points)")
    print(f"  {'Topology':<12} {'MAPE':>8} {'Max Err':>10} {'At nx':>7}  {'α':>6}  {'r':>6}")
    print("  " + "-" * 65)
    for topo_key in ["dragonfly", "fattree", "torus3d"]:
        sst_sd = load_sst_slowdown(topo_key)
        md1, rho_raw, saturated = load_raps_data(topo_key)
        if sst_sd is None or not rho_raw:
            print(f"  {topo_key:<12}  DATA MISSING")
            continue
        alpha, beta, r_val = fit_calibration(sst_sd, rho_raw)
        if alpha is None:
            print(f"  {topo_key:<12}  INSUFFICIENT DATA")
            continue
        cal_sd = apply_calibration(rho_raw, alpha, beta)
        mape, max_err, max_nx = compute_metrics(sst_sd, cal_sd)
        if mape is None:
            print(f"  {topo_key:<12}  NO OVERLAP")
            continue
        print(f"  {topo_key:<12} {mape:>7.1f}% {max_err:>9.1f}% {max_nx:>6}  {alpha:>5.2f}  {r_val:>5.3f}")

    # ── Per-point breakdown ────────────────────────────────────────────────────
    print()
    print("Per-point breakdown (SST | raw M/D/1 | calibrated):")
    print()
    for topo_key in ["dragonfly", "fattree", "torus3d"]:
        sst_sd = load_sst_slowdown(topo_key)
        md1, rho_raw, saturated = load_raps_data(topo_key)
        if sst_sd is None or not rho_raw:
            continue
        alpha, beta, r_val = fit_calibration(sst_sd, rho_raw)
        cal_sd = apply_calibration(rho_raw, alpha, beta) if alpha is not None else {}
        common = sorted(set(sst_sd) & set(md1))
        print(f"  {topo_key}  (α={alpha:.2f}, r={r_val:.3f}):")
        print(f"    {'nx':>5}  {'SST':>7}  {'M/D/1':>7}  {'err':>7}  {'calib':>7}  {'err_c':>7}  flag")
        for nx in common:
            s  = sst_sd[nx]
            r  = md1[nx]
            c  = cal_sd.get(nx, float("nan"))
            e1 = abs(r - s) / s * 100.0 if s > 0 else float("nan")
            ec = abs(c - s) / s * 100.0 if s > 0 else float("nan")
            flag = " [sat]" if saturated.get(nx) else ""
            print(f"    {nx:>5}  {s:>7.3f}  {r:>7.3f}  {e1:>6.1f}%  {c:>7.3f}  {ec:>6.1f}%{flag}")
        print()


if __name__ == "__main__":
    main()
