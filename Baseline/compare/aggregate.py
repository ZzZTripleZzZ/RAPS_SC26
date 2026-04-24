#!/usr/bin/env python3
"""
Baseline/compare/aggregate.py — Merge all simulator outputs into a comparison DataFrame.

Reads summary_rho_X.json from each simulator's output directory and produces:
  - Baseline/compare/comparison_dragonfly.csv
  - Baseline/compare/comparison_fattree.csv

Usage:
    cd /lustre/orion/gen053/scratch/zhangzifan/RAPS_SC26
    .venv/bin/python3 Baseline/compare/aggregate.py

Output columns:
    simulator, topology, rho_target, mean_utilization, max_utilization,
    stall_ratio, slowdown, avg_latency_ns, status
"""

import csv
import glob
import json
import os

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Map: (simulator_name, topology_dir_key, topology_label, topo_category)
# topo_category: "dragonfly" or "fattree" — controls which comparison CSV
SIMULATOR_ENTRIES = [
    # RAPS analytical (M/D/1 formula)
    ("raps",           "raps/output/dragonfly",          "dragonfly",      "dragonfly"),
    ("raps",           "raps/output/dragonfly_1000n",    "dragonfly_1000n","dragonfly"),
    ("raps",           "raps/output/fattree",             "fattree",        "fattree"),
    # RAPS engine (simulate_network_utilization code path)
    ("raps_engine",    "raps/output/dragonfly_72n_engine",   "dragonfly",      "dragonfly"),
    ("raps_engine",    "raps/output/dragonfly_1000n_engine", "dragonfly_1000n","dragonfly"),
    # BookSim2 cycle-accurate
    ("booksim2",       "booksim2/output/dragonfly",      "dragonfly",      "dragonfly"),
    ("booksim2",       "booksim2/output/dragonfly_1000n","dragonfly_1000n","dragonfly"),
    ("booksim2",       "booksim2/output/fattree",         "fattree",        "fattree"),
    # SST-Macro PISCES (1056-node only; no 72-node SST-Macro run)
    ("sst-macro",      "sst-macro/output/dragonfly_1000n","dragonfly_1000n","dragonfly"),
    # SimGrid flow-level (max-min fairness)
    ("simgrid",        "simgrid/output/dragonfly",         "dragonfly",      "dragonfly"),
    ("simgrid",        "simgrid/output/dragonfly_1000n",   "dragonfly_1000n","dragonfly"),
    ("simgrid",        "simgrid/output/fattree",            "fattree",        "fattree"),
    # CODES packet-level
    ("codes",          "codes/output/dragonfly",          "dragonfly",      "dragonfly"),
    # ns-3 excluded: different rho definition (injection rate vs link util),
    #   UDP+DropTail congestion collapse at rho>=0.5, startup burst artifact
    # Torus3d (BlueWaters 8×8×8×2 = 1024 hosts)
    ("raps",           "raps/output/torus3d",              "torus3d",        "torus3d"),
    ("simgrid",        "simgrid/output/torus3d",           "torus3d",        "torus3d"),
    ("booksim2",       "booksim2/output/torus3d",          "torus3d",        "torus3d"),
]

# Fields to extract from each summary JSON
FIELDS = [
    "rho_target",
    "mean_utilization",
    "max_utilization",
    "stall_ratio",
    "slowdown",
    "avg_latency_ns",
]

# M/D/1 theoretical values for comparison
def md1_theory(rho):
    if rho <= 0:
        return {"stall_ratio": 0.0, "slowdown": 1.0}
    if rho >= 1.0:
        return {"stall_ratio": float("inf"), "slowdown": float("inf")}
    stall = rho**2 / (2.0 * (1.0 - rho))
    return {"stall_ratio": stall, "slowdown": 1.0 + stall}


def load_simulator_results(sim_name: str, topo: str, out_dir: str) -> list[dict]:
    # Also check for sweep_results.csv (from run_engine_sweep.py)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    if os.path.exists(csv_path) and not glob.glob(os.path.join(out_dir, "summary_*.json")):
        return _load_from_sweep_csv(sim_name, topo, csv_path)
    return _load_from_jsons(sim_name, topo, out_dir)


def _load_from_sweep_csv(sim_name: str, topo: str, csv_path: str) -> list[dict]:
    """Load from run_engine_sweep.py's sweep_results.csv."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {
                "simulator": sim_name,
                "topology": topo,
                "status": "ok",
            }
            for field in FIELDS:
                val = row.get(field)
                try:
                    result[field] = float(val) if val not in (None, "", "None") else np.nan
                except (ValueError, TypeError):
                    result[field] = np.nan
            rho = result.get("rho_target")
            if rho is not None and not np.isnan(rho):
                theory = md1_theory(rho)
                result["md1_stall_ratio"] = theory["stall_ratio"]
                result["md1_slowdown"] = theory["slowdown"]
                if result.get("stall_ratio") is not None and theory["stall_ratio"] > 0:
                    result["stall_ratio_vs_md1"] = result["stall_ratio"] / theory["stall_ratio"]
            rows.append(result)
    print(f"  {sim_name}/{topo}: {len(rows)} ρ values loaded (from sweep_results.csv)")
    return rows


def _load_from_jsons(sim_name: str, topo: str, out_dir: str) -> list[dict]:
    """Load all summary JSON files for one (simulator, topology) pair."""
    rows = []
    pattern = os.path.join(out_dir, "summary_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"  WARNING: No summary JSON files found in {out_dir}/")
        return rows

    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARNING: Could not read {path}: {e}")
            continue

        # Skip placeholder "not_attempted" entries
        status = data.get("status", "ok")
        if status in ("not_attempted", "not_installed"):
            print(f"  SKIP: {sim_name}/{topo} rho={data.get('rho_target','?')} — {status}")
            continue

        row = {
            "simulator": sim_name,
            "topology": topo,
            "status": status,
        }
        for field in FIELDS:
            row[field] = data.get(field, None)

        # Add M/D/1 theory columns for comparison
        rho = row.get("rho_target")
        if rho is not None:
            theory = md1_theory(rho)
            row["md1_stall_ratio"] = theory["stall_ratio"]
            row["md1_slowdown"] = theory["slowdown"]
            if row.get("stall_ratio") is not None and theory["stall_ratio"] > 0:
                row["stall_ratio_vs_md1"] = row["stall_ratio"] / theory["stall_ratio"]
            else:
                row["stall_ratio_vs_md1"] = None

        rows.append(row)

    print(f"  {sim_name}/{topo}: {len(rows)} ρ values loaded")
    return rows


def write_csv(rows: list[dict], out_path: str):
    if not rows:
        print(f"  No data to write to {out_path}")
        return
    all_keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {out_path}")


def main():
    compare_dir = os.path.dirname(os.path.abspath(__file__))

    print("[aggregate] Loading simulator results...")
    dragonfly_rows = []
    fattree_rows = []
    torus3d_rows = []

    for sim_name, rel_dir, topo_label, topo_cat in SIMULATOR_ENTRIES:
        out_dir = os.path.join(_BASE, rel_dir)
        rows = load_simulator_results(sim_name, topo_label, out_dir)
        if topo_cat == "dragonfly":
            dragonfly_rows.extend(rows)
        elif topo_cat == "torus3d":
            torus3d_rows.extend(rows)
        else:
            fattree_rows.extend(rows)

    # Add M/D/1 theoretical reference rows
    rho_vals = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    for rho in rho_vals:
        theory = md1_theory(rho)
        md1_row = {
            "simulator": "md1_theory",
            "rho_target": rho,
            "stall_ratio": theory["stall_ratio"],
            "slowdown": theory["slowdown"],
            "mean_utilization": rho,
            "max_utilization": rho,
            "avg_latency_ns": None,
            "status": "theory",
        }
        dragonfly_rows.append({**md1_row, "topology": "dragonfly"})
        fattree_rows.append({**md1_row, "topology": "fattree"})
        torus3d_rows.append({**md1_row, "topology": "torus3d"})

    # Sort by (simulator, rho)
    dragonfly_rows.sort(key=lambda r: (r.get("simulator",""), r.get("rho_target", 0)))
    fattree_rows.sort(key=lambda r: (r.get("simulator",""), r.get("rho_target", 0)))
    torus3d_rows.sort(key=lambda r: (r.get("simulator",""), r.get("rho_target", 0)))

    print("\n[aggregate] Writing comparison CSVs...")
    write_csv(dragonfly_rows, os.path.join(compare_dir, "comparison_dragonfly.csv"))
    write_csv(fattree_rows,   os.path.join(compare_dir, "comparison_fattree.csv"))
    write_csv(torus3d_rows,   os.path.join(compare_dir, "comparison_torus3d.csv"))

    # Print summary table
    print("\n=== Stall Ratio Comparison (vs M/D/1 theory) ===")
    print(f"{'Simulator':<15} {'Topology':<12} {'ρ':>6} {'stall_ratio':>12} {'md1_theory':>12} {'ratio':>8}")
    print("-" * 72)
    for row in dragonfly_rows + fattree_rows + torus3d_rows:
        if row.get("simulator") == "md1_theory":
            continue
        if row.get("stall_ratio") is None:
            continue
        print(f"{row['simulator']:<15} {row['topology']:<12} "
              f"{row['rho_target']:>6.2f} "
              f"{row['stall_ratio']:>12.4f} "
              f"{row.get('md1_stall_ratio', 0):>12.4f} "
              f"{row.get('stall_ratio_vs_md1', 0):>8.3f}")

    print("\n[aggregate] Done.")


if __name__ == "__main__":
    main()
