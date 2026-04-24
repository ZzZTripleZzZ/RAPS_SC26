#!/usr/bin/env python3
"""Preprocess LDMS Blue Waters gz files.

Extracts columns 1,2,16,17 (timestamp, node_id, tx_bytes, rx_bytes)
from 18GB daily gz archives into compact 4-column CSV files matching
the existing cray_system_sampler format.

Usage:
    .venv/bin/python3 Baseline/hardware/preprocess_ldms.py
    .venv/bin/python3 Baseline/hardware/preprocess_ldms.py --day 20170112
    .venv/bin/python3 Baseline/hardware/preprocess_ldms.py --start 20170101 --end 20170130
"""

import argparse
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

BW_DATA = "/lustre/orion/gen053/scratch/zhangzifan/bluewaters_data"
OUT_SUBDIR = "cray_system_sampler"


def preprocess_day(args):
    """Worker: extract cols 1,2,16,17 from one gz file using awk."""
    day, force = args
    gz_path = os.path.join(BW_DATA, f"{day}.gz")
    out_dir = os.path.join(BW_DATA, OUT_SUBDIR)
    out_path = os.path.join(out_dir, day)

    if not os.path.exists(gz_path):
        return day, False, f"gz not found"

    if os.path.exists(out_path) and not force:
        size = os.path.getsize(out_path)
        lines = _count_lines(out_path)
        return day, True, f"already exists ({size/1e6:.0f} MB, {lines/1e6:.1f}M rows)"

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    gz_size = os.path.getsize(gz_path)

    # awk extracts 4 columns: timestamp, node_id, tx_bytes, rx_bytes
    cmd = (
        f"zcat '{gz_path}' | "
        f"awk -F',' '{{print $1\",\"$2\",\"$16\",\"$17}}' "
        f"> '{out_path}'"
    )
    ret = subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)

    if ret != 0:
        return day, False, f"command failed (exit {ret})"

    out_size = os.path.getsize(out_path)
    dt = time.time() - t0
    return day, True, f"{gz_size/1e9:.1f} GB → {out_size/1e6:.0f} MB in {dt:.0f}s"


def _count_lines(path):
    try:
        result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
        return int(result.stdout.split()[0])
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LDMS Blue Waters gz files (extract tx/rx cols)"
    )
    parser.add_argument("--day", help="Single day to process (YYYYMMDD)")
    parser.add_argument("--start", default="20170101", help="Start day (YYYYMMDD)")
    parser.add_argument("--end", default="20170130", help="End day (YYYYMMDD)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    args = parser.parse_args()

    if args.day:
        days = [args.day]
    else:
        d = datetime.strptime(args.start, "%Y%m%d")
        end = datetime.strptime(args.end, "%Y%m%d")
        days = []
        while d <= end:
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)

    out_dir = os.path.join(BW_DATA, OUT_SUBDIR)
    print(f"Preprocessing {len(days)} days with {args.workers} parallel workers")
    print(f"  Input:  {BW_DATA}/YYYYMMDD.gz")
    print(f"  Output: {out_dir}/YYYYMMDD")
    print()

    ok, fail, skip = 0, 0, 0
    tasks = [(day, args.force) for day in days]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(preprocess_day, t): t[0] for t in tasks}
        for future in as_completed(futures):
            day = futures[future]
            _, success, msg = future.result()
            if "already exists" in msg:
                print(f"  [SKIP] {day}: {msg}")
                skip += 1
            elif success:
                print(f"  [OK]   {day}: {msg}")
                ok += 1
            else:
                print(f"  [FAIL] {day}: {msg}")
                fail += 1

    print(f"\nDone: {ok} processed, {skip} skipped, {fail} failed")
    if ok + skip > 0:
        print(f"Output in: {out_dir}/")


if __name__ == "__main__":
    main()
