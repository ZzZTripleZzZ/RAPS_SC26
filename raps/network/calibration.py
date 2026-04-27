"""
Per-topology calibration constants and per-application-class message sizes.

Values are hard-coded (not fit at simulation runtime). Step 2 of the rollout
replaces the 1.0 / default placeholders with values derived once from
Baseline/compare/compute_mape.py against SST-Macro Layer-2 experiments.
"""
from __future__ import annotations


# Per-topology calibration factor kappa applied to the engine's facility-scale
# stall ratio aggregation: reported_stall = kappa * raw_stall, where raw_stall
# is the mean per-job (slowdown - 1) across concurrent jobs in a tick. It folds
# in the residual gap between the per-job M/D/1 dilation model and the observed
# system-level stall ratio (MPI-barrier amplification, etc.).
#
# Note: this is *not* the Layer-2 calibration reported in the paper. The paper's
# Layer-2 MAPE numbers (Sec. IV-B) come from the per-topology power-law fit
# slowdown = a * rho^alpha against raps_raw_combined_rho, captured below in
# LAYER2_POWERLAW_FIT. The two calibrations serve different purposes:
#   - STALL_CALIBRATION: aggregate stall reporting in the running engine
#   - LAYER2_POWERLAW_FIT: offline validation against SST-Macro bully-victim
STALL_CALIBRATION: dict[str, float] = {
    "dragonfly": 0.373,
    "fat-tree":  0.151,
    "torus3d":   0.415,
}


# Per-topology log-log power-law fit slowdown = a * rho^alpha, where rho is
# raps_raw_combined_rho from the bully-victim sweep (un-clamped sum of per-link
# loads / victim_tx). Fit recipe: np.polyfit(log(rho), log(sst_slowdown), 1)
# over Baseline/sst-macro/multi_job/output/interference_large/<topo>/, excluding
# bully_nx in {0, 50} (nx=0 is the interference-free baseline; nx=50 is a known
# parse artifact where SST shows victim faster than baseline).
#
# Reproduces paper Sec. IV-B MAPE figures essentially exactly:
#   dragonfly: a=1.008, alpha=0.136, r=0.951, MAPE 2.93% (paper: 2.9%)
#   fat-tree:  a=0.042, alpha=0.764, r=0.988, MAPE 8.36% (paper: 8.4%)
#   torus3d:   a=1.170, alpha=0.107, r=0.906, MAPE 3.10% (paper: 3.1%)
#
# Use layer2_calibrated_slowdown(topology, raw_rho) to apply the fit.
LAYER2_POWERLAW_FIT: dict[str, tuple[float, float]] = {
    "dragonfly": (1.00817, 0.13598),
    "fat-tree":  (0.04195, 0.76427),
    "torus3d":   (1.16988, 0.10672),
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


def layer2_calibrated_slowdown(topology: str | None, raw_rho: float) -> float:
    """Apply the per-topology Layer-2 power-law fit: slowdown = a * rho^alpha.

    Used for offline validation plots (Baseline/compare/plot_interference.py).
    Unknown topologies fall back to identity (slowdown = 1.0).
    """
    if topology is None or raw_rho <= 0.0:
        return 1.0
    fit = LAYER2_POWERLAW_FIT.get(topology)
    if fit is None:
        return 1.0
    a, alpha = fit
    return float(a * (raw_rho ** alpha))


def get_app_class_msg_size(job_name: str | None) -> int:
    """Pick a message size from the proxy-app token of job.name."""
    if not job_name:
        return APP_CLASS_MSG_SIZE["default"]
    token = job_name.lower().split("_", 1)[0].split("-", 1)[0]
    return APP_CLASS_MSG_SIZE.get(token, APP_CLASS_MSG_SIZE["default"])
