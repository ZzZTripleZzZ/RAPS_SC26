#!/usr/bin/env python3
"""
SC26 Trace Generation Script (v2.0)
===================================
Supports two tracing backends:
1. SST-DUMPI (preferred): Standard HPC trace format, widely compatible
2. Custom libtracer.so (fallback): Lightweight MPI interceptor

Usage:
    python generate_traces.py [--backend dumpi|tracer]
"""

import os
import subprocess
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# ==========================================
# Configuration
# ==========================================
PROJECT_ROOT = Path("/app")
OUTPUT_DIR = Path("/app/data/raw_traces")
TRACER_LIB = PROJECT_ROOT / "src/tracer/libtracer.so"

# SST-DUMPI paths (if installed)
DUMPI_LIB_PATHS = [
    "/usr/local/lib/libdumpi.so",
    "/opt/sst-dumpi/lib/libdumpi.so",
    "/usr/lib/x86_64-linux-gnu/libdumpi.so",
]


def check_dumpi_available():
    """Check if SST-DUMPI is available on the system."""
    # Check for library
    for lib_path in DUMPI_LIB_PATHS:
        if Path(lib_path).exists():
            return lib_path

    # Check if it can be found via ldconfig
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True, text=True, timeout=5
        )
        if "libdumpi" in result.stdout:
            for line in result.stdout.split("\n"):
                if "libdumpi" in line:
                    path = line.split("=>")[-1].strip()
                    if Path(path).exists():
                        return path
    except Exception:
        pass

    return None


def get_experiments():
    """Define experiments to run."""
    experiments = []
    rank_configs = [64]

    for np in rank_configs:
        # 1. LULESH (Stencil pattern - 3D structured grid)
        experiments.append({
            "app": "lulesh",
            "args": ["-s", "15", "-i", "15"],
            "np": np,
            "name": f"lulesh_n{np}",
            "pattern": "stencil_3d"
        })

        # 2. CoMD (Molecular Dynamics - neighbor list communication)
        if np == 64:
            experiments.append({
                "app": "comd",
                "args": ["-i", "4", "-j", "4", "-k", "4", "-x", "40", "-y", "40", "-z", "40"],
                "np": 64,
                "name": "comd_n64",
                "pattern": "neighbor_exchange"
            })

        # 3. HPGMG (Multigrid - hierarchical communication)
        experiments.append({
            "app": "hpgmg",
            "args": ["5", "1"],
            "np": np,
            "name": f"hpgmg_n{np}",
            "pattern": "hierarchical"
        })

        # 4. CoSP2 (Sparse Matrix - irregular all-to-all)
        experiments.append({
            "app": "cosp2",
            "args": [],
            "np": np,
            "name": f"cosp2_n{np}",
            "pattern": "sparse_alltoall"
        })

    return experiments


def run_experiment_dumpi(exp, dumpi_lib_path):
    """Run experiment using SST-DUMPI tracing."""
    app_name = exp["app"]

    binary_map = {
        'hpgmg': '/app/bin/hpgmg-fv',
        'cosp2': '/app/bin/cosp2-bin',
        'lulesh': '/app/bin/lulesh',
        'comd': '/app/bin/comd',
        'qs': '/app/bin/qs'
    }

    binary_path = Path(binary_map.get(app_name, f'/app/bin/{app_name}'))

    if not binary_path.exists():
        print(f"[Skip] Binary not found for {app_name} at {binary_path}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{exp['name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # DUMPI output directory
    dumpi_dir = run_dir / "dumpi"
    dumpi_dir.mkdir(exist_ok=True)

    print(f"--> Running {exp['name']} (np={exp['np']}) with SST-DUMPI...")

    os.environ["OMP_NUM_THREADS"] = "1"

    # Configure DUMPI
    os.environ["DUMPI_OUTPUT"] = str(dumpi_dir / "trace")
    os.environ["DUMPI_ENABLE_IO"] = "0"
    os.environ["DUMPI_TIMESTAMPS"] = "1"

    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-np", str(exp["np"]),
        "-x", f"LD_PRELOAD={dumpi_lib_path}",
        "-x", "DUMPI_OUTPUT",
        "-x", "DUMPI_ENABLE_IO",
        "-x", "DUMPI_TIMESTAMPS",
        str(binary_path)
    ] + exp["args"]

    try:
        with open(run_dir / "stdout.log", "w") as f_out, \
             open(run_dir / "stderr.log", "w") as f_err:
            start_time = time.time()
            subprocess.run(cmd, stdout=f_out, stderr=f_err, check=True,
                         cwd=run_dir, timeout=3600)
            duration = time.time() - start_time

        # Save metadata
        with open(run_dir / "metadata.json", "w") as f:
            import json
            json.dump({
                "app": exp["app"],
                "name": exp["name"],
                "np": exp["np"],
                "pattern": exp.get("pattern", "unknown"),
                "backend": "dumpi",
                "duration_seconds": duration,
                "timestamp": timestamp
            }, f, indent=2)

        print(f"    Done in {duration:.2f}s (DUMPI traces in {dumpi_dir})")
        return run_dir

    except subprocess.CalledProcessError as e:
        print(f"    FAILED! Exit code: {e.returncode}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 3600s!")
        return None


def run_experiment_tracer(exp):
    """Run experiment using custom libtracer.so (fallback)."""
    app_name = exp["app"]

    binary_map = {
        'hpgmg': '/app/bin/hpgmg-fv',
        'cosp2': '/app/bin/cosp2-bin',
        'lulesh': '/app/bin/lulesh',
        'comd': '/app/bin/comd',
        'qs': '/app/bin/qs'
    }

    binary_path = Path(binary_map.get(app_name, f'/app/bin/{app_name}'))

    if not binary_path.exists():
        print(f"[Skip] Binary not found for {app_name} at {binary_path}")
        return None

    if not TRACER_LIB.exists():
        print(f"[Skip] Tracer library not found at {TRACER_LIB}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{exp['name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"--> Running {exp['name']} (np={exp['np']}) with libtracer...")

    os.environ["OMP_NUM_THREADS"] = "1"

    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-np", str(exp["np"]),
        "-x", f"LD_PRELOAD={TRACER_LIB}",
        str(binary_path)
    ] + exp["args"]

    try:
        with open(run_dir / "stdout.log", "w") as f_out, \
             open(run_dir / "stderr.log", "w") as f_err:
            start_time = time.time()
            subprocess.run(cmd, stdout=f_out, stderr=f_err, check=True,
                         cwd=run_dir, timeout=3600)
            duration = time.time() - start_time

        # Save metadata
        with open(run_dir / "metadata.json", "w") as f:
            import json
            json.dump({
                "app": exp["app"],
                "name": exp["name"],
                "np": exp["np"],
                "pattern": exp.get("pattern", "unknown"),
                "backend": "tracer",
                "duration_seconds": duration,
                "timestamp": timestamp
            }, f, indent=2)

        print(f"    Done in {duration:.2f}s")
        return run_dir

    except subprocess.CalledProcessError as e:
        print(f"    FAILED! Exit code: {e.returncode}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 3600s!")
        return None


def install_sst_dumpi():
    """Print instructions for installing SST-DUMPI."""
    print("""
================================================================================
SST-DUMPI Installation Instructions
================================================================================

SST-DUMPI is not found on this system. To install:

Option 1: Using Spack (Recommended)
    spack install sst-dumpi
    spack load sst-dumpi

Option 2: From Source
    git clone https://github.com/sstsimulator/sst-dumpi.git
    cd sst-dumpi
    ./bootstrap.sh
    ./configure --prefix=/opt/sst-dumpi
    make -j$(nproc)
    sudo make install
    export LD_LIBRARY_PATH=/opt/sst-dumpi/lib:$LD_LIBRARY_PATH

Option 3: Using apt (if available)
    sudo apt-get install sst-dumpi

After installation, re-run this script with --backend dumpi
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Generate MPI traces for SC26")
    parser.add_argument("--backend", choices=["dumpi", "tracer", "auto"],
                       default="auto", help="Tracing backend to use")
    parser.add_argument("--install-help", action="store_true",
                       help="Show SST-DUMPI installation instructions")
    args = parser.parse_args()

    if args.install_help:
        install_sst_dumpi()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine backend
    dumpi_lib = check_dumpi_available()

    if args.backend == "dumpi":
        if not dumpi_lib:
            print("ERROR: SST-DUMPI requested but not found!")
            install_sst_dumpi()
            return
        backend = "dumpi"
    elif args.backend == "tracer":
        backend = "tracer"
    else:  # auto
        if dumpi_lib:
            print(f"Found SST-DUMPI at {dumpi_lib}")
            backend = "dumpi"
        elif TRACER_LIB.exists():
            print(f"Using fallback tracer at {TRACER_LIB}")
            print("NOTE: For better compatibility, consider installing SST-DUMPI")
            backend = "tracer"
        else:
            print("ERROR: No tracing backend available!")
            install_sst_dumpi()
            return

    experiments = get_experiments()
    print(f"Running {len(experiments)} experiments with {backend} backend...")

    total_start = time.time()
    successful = []

    for exp in experiments:
        if backend == "dumpi":
            result = run_experiment_dumpi(exp, dumpi_lib)
        else:
            result = run_experiment_tracer(exp)

        if result:
            successful.append(result)

    print(f"\nCompleted {len(successful)}/{len(experiments)} experiments "
          f"in {time.time() - total_start:.2f}s")
    print(f"Traces saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
