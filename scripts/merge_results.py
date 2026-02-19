#!/usr/bin/env python3
"""
Merge Results from Frontier Scaling Experiments
================================================
Combines CSV files from all 4 experiment tiers into a single results file.

Usage:
    python scripts/merge_results.py
    
    # Or specify custom output directory:
    python scripts/merge_results.py --output-dir output/custom_location

Output:
    Creates 'all_results.csv' with all 96 experiments combined.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd


def merge_tier_results(output_dir: Path, output_file: Path):
    """Merge results from all tier subdirectories.
    
    Args:
        output_dir: Base output directory containing tier subdirectories
        output_file: Path for merged output CSV file
    """
    print("=" * 70)
    print("Merging Frontier Scaling Experiment Results")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all result CSV files: try tier subdirectories first, then flat layout
    tier_files = []
    tier_dfs = []

    for tier in range(1, 5):
        tier_dir = output_dir / f"frontier_scaling_tier{tier}"
        csv_path = tier_dir / "results.csv"

        if csv_path.exists():
            print(f"  Found Tier {tier}: {csv_path}")
            df = pd.read_csv(csv_path)
            tier_dfs.append(df)
            tier_files.append(csv_path)
            print(f"  - {len(df)} experiments")
        else:
            print(f"  Missing Tier {tier}: {csv_path}")

    # Fallback: check flat frontier_scaling/results.csv
    flat_csv = output_dir / "frontier_scaling" / "results.csv"
    if flat_csv.exists() and flat_csv not in tier_files:
        print(f"  Found flat results: {flat_csv}")
        df = pd.read_csv(flat_csv)
        tier_dfs.append(df)
        tier_files.append(flat_csv)
        print(f"  - {len(df)} experiments")

    if not tier_dfs:
        print()
        print("ERROR: No result files found!")
        print(f"Expected files in: {output_dir}/frontier_scaling_tier*/results.csv")
        print(f"             or:   {output_dir}/frontier_scaling/results.csv")
        sys.exit(1)
    
    print()
    print(f"Merging {len(tier_dfs)} tier result files...")
    
    # Merge all dataframes
    merged_df = pd.concat(tier_dfs, ignore_index=True)
    
    print(f"Total experiments: {len(merged_df)}")
    print()
    
    # Summary statistics
    print("Summary by system:")
    for system in merged_df['system'].unique():
        system_df = merged_df[merged_df['system'] == system]
        print(f"  {system}: {len(system_df)} experiments")
    
    print()
    print("Summary by status:")
    for status in merged_df['status'].unique():
        status_df = merged_df[merged_df['status'] == status]
        print(f"  {status}: {len(status_df)} experiments")
    
    # Write merged CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    
    print()
    print("=" * 70)
    print(f"Merged results written to: {output_file}")
    print("=" * 70)
    
    # Additional analysis
    if 'status' in merged_df.columns:
        failed = merged_df[merged_df['status'] != 'OK']
        if len(failed) > 0:
            print()
            print("Failed experiments:")
            for _, row in failed.iterrows():
                error_msg = row.get('error', 'unknown error')
                print(f"  - {row['label']}: {error_msg}")
    
    print()
    print("Quick analysis:")
    print(f"  Average speedup: {merged_df['speedup'].mean():.1f}x")
    print(f"  Average per-tick time: {merged_df['per_tick_ms'].mean():.2f} ms")
    
    if 'avg_net_util_pct' in merged_df.columns:
        print(f"  Average network utilization: {merged_df['avg_net_util_pct'].mean():.2f}%")
    
    print()
    print("Next steps:")
    print("  - Analyze results: pandas.read_csv('output/all_results.csv')")
    print("  - Plot scaling curves by system, node count, and time quantum")
    print("  - Compare lassen (real data) vs frontier (synthetic)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Frontier scaling experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory containing tier subdirectories (default: output)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/all_results.csv",
        help="Path for merged output CSV (default: output/all_results.csv)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    output_file = Path(args.output_file).resolve()
    
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)
    
    merge_tier_results(output_dir, output_file)


if __name__ == "__main__":
    main()
