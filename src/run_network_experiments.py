#!/usr/bin/env python3
"""
SC26 HPC Digital Twin - Network Communication Experiments
==========================================================
Tests different communication patterns and allocation strategies.

Uses RAPS workloads that have actual network traces:
- allocation_test: Bully (high-comm) vs Victim (low-comm) jobs
- inter_job_congestion: Cross-group vs intra-group communication
"""

import subprocess
import json
import pandas as pd
from pathlib import Path

RAPS_DIR = Path("/app/extern/raps")
DATA_DIR = Path("/app/data/experiments_network")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_raps_network(system, workload, output_dir, allocation="contiguous", numjobs=30):
    """Run RAPS with network simulation enabled."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "main.py", "run",
        "--system", system,
        "--workload", workload,
        "--time", "30m",  # Shorter simulation
        "--allocation", allocation,
        "--net",  # Enable network simulation
        "--numjobs", str(numjobs),
        "--output", str(output_dir),
        "--noui",
    ]

    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(RAPS_DIR), timeout=300)

    if result.returncode != 0:
        print(f"Error: {result.stderr[:500]}")
        return None

    print(result.stdout[-1500:])

    # Parse stats.out
    stats_file = output_dir / "stats.out"
    metrics = {'jobs_completed': 0, 'avg_power_mw': 0, 'throughput': 0}

    if stats_file.exists():
        try:
            content = stats_file.read_text()
            parts = content.split('}{')
            if len(parts) >= 2:
                power_stats = json.loads(parts[0] + '}')
                job_stats = json.loads('{' + parts[1].split('}')[0] + '}')

                metrics['avg_power_mw'] = power_stats.get('average_power', 0)
                metrics['jobs_completed'] = job_stats.get('jobs_completed', 0)
                metrics['throughput'] = job_stats.get('throughput', 0)
        except Exception as e:
            print(f"Parse error: {e}")

    return metrics


def run_allocation_test_experiments():
    """
    Compare allocation strategies with network-intensive workload.
    allocation_test has high-comm (bully) and low-comm (victim) jobs.
    """
    print("\n" + "="*60)
    print("Experiment: Allocation Strategy with Network Traffic")
    print("Workload: allocation_test (Bully vs Victim jobs)")
    print("="*60)

    results = []

    for system in ['lassen', 'frontier']:
        for allocation in ['contiguous', 'random', 'hybrid']:
            print(f"\n--- {system} {allocation} ---")
            output_dir = DATA_DIR / f"alloc_{system}_{allocation}"

            metrics = run_raps_network(
                system=system,
                workload="allocation_test",
                output_dir=output_dir,
                allocation=allocation,
                numjobs=30
            )

            if metrics:
                results.append({
                    'system': system,
                    'allocation': allocation,
                    'workload': 'allocation_test',
                    **metrics
                })

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "allocation_test_results.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'allocation_test_results.csv'}")
    print(df.to_string())
    return df


def run_inter_job_congestion_experiments():
    """
    Test inter-job network congestion with different topologies.
    """
    print("\n" + "="*60)
    print("Experiment: Inter-Job Network Congestion")
    print("Workload: inter_job_congestion (cross-group traffic)")
    print("="*60)

    results = []

    for system in ['lassen', 'frontier']:
        print(f"\n--- {system} ---")
        output_dir = DATA_DIR / f"congestion_{system}"

        metrics = run_raps_network(
            system=system,
            workload="inter_job_congestion",
            output_dir=output_dir,
            allocation="contiguous",
            numjobs=60
        )

        if metrics:
            topology = 'fat-tree' if system == 'lassen' else 'dragonfly'
            results.append({
                'system': system,
                'topology': topology,
                'workload': 'inter_job_congestion',
                **metrics
            })

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "congestion_results.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'congestion_results.csv'}")
    print(df.to_string())
    return df


def main():
    print("="*60)
    print("SC26 Network Communication Experiments")
    print("="*60)
    print("\nThese experiments use workloads WITH network traces:")
    print("- allocation_test: High-comm (bully) vs Low-comm (victim) jobs")
    print("- inter_job_congestion: Cross-group vs intra-group traffic")

    try:
        run_allocation_test_experiments()
    except Exception as e:
        print(f"allocation_test failed: {e}")

    try:
        run_inter_job_congestion_experiments()
    except Exception as e:
        print(f"inter_job_congestion failed: {e}")

    print("\n" + "="*60)
    print("Network experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()
