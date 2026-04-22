#!/usr/bin/env python3
"""
RAPS Allocation Strategy Test (Bully Effect Study)
===================================================

This script tests different node allocation strategies and their impact on
network congestion, based on the "Bully" phenomenon from:
"Watch Out for the Bully! Job Interference Study on Dragonfly Network"
(Yang et al., SC16)

It bypasses the full RAPS simulation for fast iteration, directly computing
network congestion for jobs placed using different allocation strategies.

Usage:
    python scripts/run_allocation_test.py --config config/lassen.yaml

Example comparing all strategies:
    python scripts/run_allocation_test.py --config config/lassen.yaml --compare -v

Example single strategy:
    python scripts/run_allocation_test.py --config config/lassen.yaml --allocation random -v
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List

from raps.job import Job
from raps.system_config import get_system_config
from raps.policy import AllocationStrategy
from raps.network import (
    NetworkModel,
    simulate_inter_job_congestion,
)
from raps.workloads.allocation_test import generate_allocation_test_jobs


def allocate_nodes_to_jobs(
    jobs: List[Job],
    available_nodes: List[int],
    strategy: AllocationStrategy,
    hybrid_threshold: float = 0.5,
) -> List[Job]:
    """
    Apply allocation strategy to assign nodes to jobs.

    This simulates what the resource manager would do, but for a static
    snapshot where all jobs run concurrently.
    """
    import numpy as np

    # Sort available nodes for contiguous allocation
    available = sorted(available_nodes.copy())

    for job in jobs:
        n = job.nodes_required

        if n > len(available):
            print(f"[WARN] Job {job.id} needs {n} nodes but only {len(available)} available, skipping")
            job.scheduled_nodes = []
            continue

        # Determine effective strategy for this job
        if strategy == AllocationStrategy.HYBRID:
            # Compute communication intensity from network traces
            intensity = _compute_intensity(job)
            effective_strategy = (
                AllocationStrategy.RANDOM if intensity >= hybrid_threshold
                else AllocationStrategy.CONTIGUOUS
            )
        else:
            effective_strategy = strategy

        # Apply allocation
        if effective_strategy == AllocationStrategy.CONTIGUOUS:
            job.scheduled_nodes = available[:n]
        else:  # RANDOM
            job.scheduled_nodes = random.sample(available, n)

        # Remove allocated nodes from available pool
        available = [node for node in available if node not in job.scheduled_nodes]

    return jobs


def _compute_intensity(job: Job) -> float:
    """Compute normalized communication intensity for hybrid strategy."""
    import numpy as np

    ntx = getattr(job, 'ntx_trace', None)
    nrx = getattr(job, 'nrx_trace', None)

    total = 0.0
    count = 0

    for trace in [ntx, nrx]:
        if trace is not None:
            if isinstance(trace, (list, np.ndarray)) and len(trace) > 0:
                total += np.mean(trace)
                count += 1
            elif isinstance(trace, (int, float)):
                total += trace
                count += 1

    if count == 0:
        return 0.0

    avg_network = total / count
    # Normalize based on typical bandwidth values
    # This threshold should be tuned based on your workload
    intensity = min(1.0, avg_network / 1e11)  # Normalize to ~100GB/s reference
    return intensity


def run_allocation_test(
    legacy_cfg: dict,
    strategy: AllocationStrategy,
    num_jobs: int = 60,
    seed: int = 42,
    hybrid_threshold: float = 0.5,
    verbose: bool = False,
    debug: bool = False,
) -> dict:
    """
    Run allocation test for a single strategy.

    Returns congestion statistics.
    """
    random.seed(seed)

    # Generate jobs (without node assignments)
    jobs = generate_allocation_test_jobs(
        legacy_cfg=legacy_cfg,
        num_jobs=num_jobs,
        seed=seed,
    )

    # Get available nodes
    total_nodes = legacy_cfg["TOTAL_NODES"]
    down_nodes = set(legacy_cfg.get("DOWN_NODES", []))
    available_nodes = [i for i in range(total_nodes) if i not in down_nodes]

    # Apply allocation strategy
    jobs = allocate_nodes_to_jobs(
        jobs=jobs,
        available_nodes=available_nodes,
        strategy=strategy,
        hybrid_threshold=hybrid_threshold,
    )

    # Filter out jobs that couldn't be allocated
    jobs = [j for j in jobs if j.scheduled_nodes]

    if verbose:
        print(f"\n[{strategy.value.upper()}] Allocated {len(jobs)} jobs")
        # Show some example allocations
        for job in jobs[:5]:
            nodes_preview = job.scheduled_nodes[:5]
            suffix = "..." if len(job.scheduled_nodes) > 5 else ""
            print(f"  Job {job.id} ({job.name}): {len(job.scheduled_nodes)} nodes -> {nodes_preview}{suffix}")

    # Initialize network model
    net = NetworkModel(
        config=legacy_cfg,
        available_nodes=available_nodes,
        debug=debug,
    )

    # Simulate congestion
    stats = simulate_inter_job_congestion(net, jobs, legacy_cfg, debug=debug)

    return stats


def print_comparison_table(results: dict):
    """Print a comparison table of all strategies."""
    print("\n" + "=" * 70)
    print("ALLOCATION STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Max Cong':>12} {'Mean Cong':>12} {'Std Dev':>12}")
    print("-" * 70)

    for strategy, stats in results.items():
        print(f"{strategy:<15} {stats['max']:>12.2f} {stats['mean']:>12.2f} {stats['std_dev']:>12.2f}")

    print("=" * 70)

    # Analysis
    strategies = list(results.keys())
    if len(strategies) >= 2:
        max_congs = {s: results[s]['max'] for s in strategies}
        best = min(max_congs, key=max_congs.get)
        worst = max(max_congs, key=max_congs.get)

        improvement = (max_congs[worst] - max_congs[best]) / max_congs[worst] * 100
        print(f"\nBest strategy: {best} (lowest max congestion)")
        print(f"Improvement over {worst}: {improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Test allocation strategies and their impact on network congestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", required=True, help="Path to system YAML (e.g., config/lassen.yaml)")
    parser.add_argument("--allocation", choices=["contiguous", "random", "hybrid"],
                        default="contiguous", help="Allocation strategy to test")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all allocation strategies")
    parser.add_argument("--jobs", type=int, default=60, help="Number of synthetic jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--hybrid-threshold", type=float, default=0.5,
                        help="Threshold for hybrid strategy (0-1)")
    parser.add_argument("--debug", action="store_true", help="Enable network debug output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed statistics")
    args = parser.parse_args()

    # Load config
    sys_cfg = get_system_config(args.config)
    legacy = sys_cfg.get_legacy()

    topology = legacy.get("TOPOLOGY", "").lower()
    if not topology:
        raise ValueError(f"Could not infer topology from {args.config}")

    print(f"[INFO] System: {args.config}")
    print(f"[INFO] Topology: {topology}")
    print(f"[INFO] Total nodes: {legacy['TOTAL_NODES']}")
    print(f"[INFO] Jobs: {args.jobs}")

    if args.compare:
        # Compare all strategies
        results = {}
        for strategy in AllocationStrategy:
            print(f"\n{'='*50}")
            print(f"Testing {strategy.value.upper()} allocation...")
            print('='*50)

            stats = run_allocation_test(
                legacy_cfg=legacy,
                strategy=strategy,
                num_jobs=args.jobs,
                seed=args.seed,
                hybrid_threshold=args.hybrid_threshold,
                verbose=args.verbose,
                debug=args.debug,
            )
            results[strategy.value] = stats

            print(f"[RESULT] {strategy.value}: max_congestion={stats['max']:.2f}, "
                  f"mean={stats['mean']:.2f}")

        print_comparison_table(results)
    else:
        # Single strategy test
        strategy = AllocationStrategy(args.allocation)

        stats = run_allocation_test(
            legacy_cfg=legacy,
            strategy=strategy,
            num_jobs=args.jobs,
            seed=args.seed,
            hybrid_threshold=args.hybrid_threshold,
            verbose=args.verbose,
            debug=args.debug,
        )

        print(f"\n[RESULT] strategy={args.allocation}, max_congestion={stats['max']:.2f}, "
              f"mean={stats['mean']:.2f}")

        if args.verbose:
            print("\n--- Detailed Network Congestion Stats ---")
            print(f"  Max Congestion (Worst Link): {stats['max']:.2f}")
            print(f"  Mean Link Congestion:        {stats['mean']:.2f}")
            print(f"  Min Link Congestion:         {stats['min']:.2f}")
            print(f"  Std Dev of Congestion:       {stats['std_dev']:.2f}")
            if 'top_links' in stats:
                print("\n  Top 10 Most Congested Links:")
                for (link, congestion) in stats['top_links'][:10]:
                    print(f"    - Link {link}: {congestion:.2f}")


if __name__ == "__main__":
    main()
