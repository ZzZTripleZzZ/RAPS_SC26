"""
Per-topology calibration constants and per-application-class message sizes.

Values are hard-coded (not fit at simulation runtime). Step 2 of the rollout
replaces the 1.0 / default placeholders with values derived once from
Baseline/compare/compute_mape.py against SST-Macro Layer-2 experiments.
"""
from __future__ import annotations


# Per-topology calibration factor kappa applied to the stall ratio before
# reporting.  reported_stall = kappa * raw_stall.  kappa = 1.0 means the raw
# per-link model is reported as-is; kappa != 1 folds in the residual gap between
# the per-link model and the packet-level SST-Macro baseline (MPI-barrier
# amplification / sub-saturation smoothing).
#
# Values derived ONCE from Baseline/compare/compute_mape.py (see Step 2 of the
# rollout plan): minimize MAPE of (1 + kappa * (raw_slowdown - 1)) vs the
# SST-Macro Layer-2 bully-victim slowdowns.  Treated as fixed constants; the
# simulator does not re-fit at runtime.
STALL_CALIBRATION: dict[str, float] = {
    "dragonfly": 0.373,   # MAPE 8.8% vs SST-Macro (saturated raw fallback)
    "fat-tree":  0.151,   # MAPE 36.4% — scalar fit is inherently limited here
                          # because SST slowdowns span 1.02x..4.69x over nx
    "torus3d":   0.415,   # MAPE 6.6% vs SST-Macro
}


# Median message size per proxy-app class, derived from SST-Dumpi traces in
# data/matrices/*_dynamic_meta.json.  Used when a job is constructed without
# an explicit message_size; selected via the leading token of job.name.
APP_CLASS_MSG_SIZE: dict[str, int] = {
    "lulesh":      65536,     # 64 KiB stencil halo
    "comd":        32768,     # 32 KiB
    "hpgmg":      131072,     # 128 KiB multigrid restriction
    "cosp2":       16384,     # 16 KiB sparse
    "quicksilver":  4096,     # 4 KiB Monte Carlo particles
    "default":     65536,
}


def get_stall_kappa(topology: str | None) -> float:
    """Look up kappa for a topology; unknown topologies get 1.0 (no scaling)."""
    if topology is None:
        return 1.0
    return STALL_CALIBRATION.get(topology, 1.0)


def get_app_class_msg_size(job_name: str | None) -> int:
    """Pick a message size from the proxy-app token of job.name."""
    if not job_name:
        return APP_CLASS_MSG_SIZE["default"]
    token = job_name.lower().split("_", 1)[0].split("-", 1)[0]
    return APP_CLASS_MSG_SIZE.get(token, APP_CLASS_MSG_SIZE["default"])
