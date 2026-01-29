#!/usr/bin/env python3
"""
SC26 Complete Experiment Runner
================================

Main entry point for running the complete SC26 experiment pipeline:

1. Parse mini-app MPI traces to traffic matrices
2. Run RAPS simulations for all use cases
3. Generate visualization figures

Usage:
    python run_sc26_complete.py --all           # Run everything
    python run_sc26_complete.py --parse         # Only parse traces
    python run_sc26_complete.py --simulate      # Only run simulations
    python run_sc26_complete.py --visualize     # Only generate figures
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_traces():
    """Parse MPI traces into traffic matrices."""
    print("\n" + "="*70)
    print("Step 1: Parsing MPI Traces")
    print("="*70 + "\n")

    from parse_traces import main as parse_main

    try:
        parse_main()
    except Exception as e:
        print(f"Warning: Trace parsing had issues: {e}")
        print("Continuing with synthetic workloads...")


def run_simulations(use_synthetic: bool = True):
    """Run RAPS simulations for all use cases."""
    print("\n" + "="*70)
    print("Step 2: Running RAPS Simulations")
    print("="*70 + "\n")

    from sc26_pipeline import run_complete_pipeline

    results = run_complete_pipeline(use_synthetic=use_synthetic)
    return results


def generate_figures():
    """Generate all visualization figures."""
    print("\n" + "="*70)
    print("Step 3: Generating Figures")
    print("="*70 + "\n")

    from sc26_visualize import generate_all_figures

    generate_all_figures()


def print_workflow():
    """Print the detailed workflow documentation."""
    workflow = """
================================================================================
SC26 EXPERIMENT WORKFLOW
================================================================================

This pipeline integrates mini-app communication patterns with RAPS network
simulation to evaluate four use cases across two HPC systems.

--------------------------------------------------------------------------------
SYSTEMS
--------------------------------------------------------------------------------
1. Lassen (LLNL)
   - Topology: Fat-Tree (k=32)
   - Network: InfiniBand EDR (100 Gbps)
   - Routing: Minimal, ECMP, Adaptive

2. Frontier (ORNL)
   - Topology: Dragonfly (d=48, a=48, p=4)
   - Network: Slingshot (200 Gbps)
   - Routing: Minimal, UGAL, Valiant

--------------------------------------------------------------------------------
MINI-APPS & COMMUNICATION PATTERNS
--------------------------------------------------------------------------------
1. LULESH - Stencil-3D pattern (6-neighbor exchange)
   - Structured mesh hydrodynamics
   - Regular, localized communication

2. CoMD - Neighbor exchange pattern (~Stencil)
   - Molecular dynamics
   - Similar to stencil but with dynamic neighbors

3. HPGMG - Hierarchical pattern (approximated as all-to-all)
   - Multigrid solver
   - Multi-level communication

4. CoSP2 - All-to-all pattern
   - Sparse matrix operations
   - Global communication

--------------------------------------------------------------------------------
USE CASES
--------------------------------------------------------------------------------

UC1: ADAPTIVE ROUTING
   Question: Which routing algorithm performs best for each communication pattern?

   Experiments:
   - Lassen: Compare Minimal vs ECMP vs Adaptive routing
   - Frontier: Compare Minimal vs UGAL vs Valiant routing

   Metrics:
   - Max link utilization (congestion indicator)
   - Average link utilization (load balance)
   - Link utilization std dev (uniformity)

   Key Insight:
   - Stencil patterns work well with minimal routing (localized traffic)
   - All-to-all benefits from UGAL/ECMP (path diversity reduces hotspots)

--------------------------------------------------------------------------------

UC2: NODE PLACEMENT
   Question: How does allocation strategy affect network performance?

   Experiments:
   - Compare CONTIGUOUS vs RANDOM allocation on both systems
   - Test with both stencil and all-to-all patterns

   Metrics:
   - Max link utilization
   - Congestion factor
   - Path length distribution

   Key Insight:
   - Stencil: CONTIGUOUS is better (neighbors are physically adjacent)
   - All-to-all: RANDOM may help (distributes traffic across more paths)

--------------------------------------------------------------------------------

UC3: SCHEDULING (RL vs Traditional)
   Question: Can RL-based scheduling reduce network congestion?

   Experiments:
   - Compare allocation strategies as proxy for scheduling decisions
   - Measure job slowdown due to network congestion

   Metrics:
   - Job slowdown factor
   - Congestion factor distribution
   - System utilization

   Key Insight:
   - Communication-aware scheduling can reduce interference
   - Pattern-aware allocation outperforms random assignment

--------------------------------------------------------------------------------

UC4: POWER CONSUMPTION
   Question: How do communication patterns affect system power?

   Experiments:
   - Measure power under different workload configurations
   - Compare systems with different network topologies

   Metrics:
   - Total power (kW)
   - Power per node
   - Network interface power contribution

   Key Insight:
   - Higher network utilization increases NIC power
   - Congestion-induced job slowdown increases total energy (longer runtime)

--------------------------------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------------------------------
Results are saved to: /app/output/sc26_experiments/

CSV Data:
- uc1_adaptive_routing.csv    : Routing comparison results
- uc2_node_placement.csv      : Allocation strategy results
- uc3_scheduling.csv          : Scheduling comparison results
- uc4_power_consumption.csv   : Power analysis results
- experiment_summary.json     : Aggregated statistics

Figures (in figures/ subdirectory):
- uc1_adaptive_routing.png    : Routing algorithm comparison
- uc2_node_placement.png      : Allocation strategy comparison
- uc3_scheduling.png          : Scheduling impact visualization
- uc4_power_consumption.png   : Power consumption charts
- combined_summary.png        : Overview of all use cases

--------------------------------------------------------------------------------
DATA FLOW
--------------------------------------------------------------------------------

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Mini-Apps     │     │  Traffic Matrix  │     │  RAPS Network   │
│                 │────>│   / Affinity     │────>│   Simulation    │
│ LULESH, CoMD,   │     │     Graph        │     │                 │
│ HPGMG, CoSP2    │     │                  │     │ Link loads,     │
└─────────────────┘     └──────────────────┘     │ Congestion,     │
        │                        │               │ Power           │
        │                        │               └────────┬────────┘
        │  MPI Traces            │  Parsed data           │
        │  (DUMPI/custom)        │  (numpy/JSON)          │
        v                        v                        v
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ data/raw_traces │     │  data/matrices   │     │ output/sc26_*   │
└─────────────────┘     └──────────────────┘     └─────────────────┘

                                                          │
                                                          v
                                                 ┌─────────────────┐
                                                 │   Figures       │
                                                 │   (PNG)         │
                                                 └─────────────────┘

================================================================================
"""
    print(workflow)


def main():
    parser = argparse.ArgumentParser(
        description="SC26 Complete Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sc26_complete.py --all           # Run complete pipeline
  python run_sc26_complete.py --simulate      # Run simulations only
  python run_sc26_complete.py --visualize     # Generate figures only
  python run_sc26_complete.py --workflow      # Print workflow documentation
        """
    )

    parser.add_argument("--all", action="store_true",
                       help="Run complete pipeline (parse + simulate + visualize)")
    parser.add_argument("--parse", action="store_true",
                       help="Parse MPI traces only")
    parser.add_argument("--simulate", action="store_true",
                       help="Run simulations only")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate figures only")
    parser.add_argument("--workflow", action="store_true",
                       help="Print detailed workflow documentation")
    parser.add_argument("--synthetic", action="store_true", default=True,
                       help="Include synthetic workloads (default: True)")

    args = parser.parse_args()

    # If no arguments, print help and workflow
    if not any([args.all, args.parse, args.simulate, args.visualize, args.workflow]):
        parser.print_help()
        print("\n")
        print_workflow()
        return

    if args.workflow:
        print_workflow()
        return

    print("="*70)
    print("SC26 Complete Experiment Pipeline")
    print("="*70)

    if args.all:
        parse_traces()
        run_simulations(use_synthetic=args.synthetic)
        generate_figures()
    else:
        if args.parse:
            parse_traces()
        if args.simulate:
            run_simulations(use_synthetic=args.synthetic)
        if args.visualize:
            generate_figures()

    print("\n" + "="*70)
    print("SC26 Pipeline Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
