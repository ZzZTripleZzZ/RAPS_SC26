"""
Stage 1 + 2 tests for stall/packet ratio computation.

The stall_ratio is the RAPS analog of the Cassini hardware counter ratio:
    (hni_tx_paused_0 + hni_tx_paused_1) / parbs_tarb_pi_posted_pkts

Derivation: tx_paused_l = (s - 1) * posted_pkts_l  →  stall_ratio = s - 1
"""
import pytest
from unittest.mock import MagicMock
from raps.network.base import (
    compute_stall_ratio,
    apply_job_slowdown,
    compute_link_stall_packet_stats,
    aggregate_link_stall_stats,
)


# ---------------------------------------------------------------------------
# compute_stall_ratio
# ---------------------------------------------------------------------------

def test_stall_ratio_no_congestion():
    """Uncongested job (s=1) yields stall_ratio=0."""
    assert compute_stall_ratio(1.0) == 0.0


def test_stall_ratio_two_x_slowdown():
    """2x slowdown yields stall_ratio=1."""
    assert compute_stall_ratio(2.0) == pytest.approx(1.0)


def test_stall_ratio_frontier_level():
    """Frontier example: s≈6.7 → stall_ratio≈5.7."""
    assert compute_stall_ratio(6.7) == pytest.approx(5.7)


def test_stall_ratio_never_negative():
    """Slowdown values <1 (shouldn't happen in practice) clamp to 0."""
    assert compute_stall_ratio(0.5) == 0.0
    assert compute_stall_ratio(0.0) == 0.0


def test_stall_ratio_exactly_one():
    """s=1 boundary case."""
    assert compute_stall_ratio(1.0) == 0.0


# ---------------------------------------------------------------------------
# apply_job_slowdown sets job.stall_ratio
# ---------------------------------------------------------------------------

def _make_job():
    """Create a minimal mock job for testing apply_job_slowdown."""
    job = MagicMock()
    job.dilated = False
    job.slowdown_factor = 1.0
    job.stall_ratio = 0.0
    return job


def test_apply_job_slowdown_sets_stall_ratio_uncongested():
    """Non-congested path sets stall_ratio=0."""
    job = _make_job()
    result = apply_job_slowdown(
        job=job,
        max_throughput=1000.0,
        net_util=0.5,
        net_cong=0.5,
        net_tx=400.0,
        net_rx=100.0,
    )
    assert result == 1
    assert job.stall_ratio == 0.0


def test_apply_job_slowdown_sets_stall_ratio_congested():
    """Congested path (net_cong>1) sets stall_ratio = slowdown_factor - 1."""
    job = _make_job()
    # net_cong=1.5 → slowdown_factor = net_cong = 1.5 → stall_ratio = 0.5
    result = apply_job_slowdown(
        job=job,
        max_throughput=1000.0,
        net_util=1.5,
        net_cong=1.5,
        net_tx=2000.0,
        net_rx=1000.0,
    )
    assert result == pytest.approx(1.5)
    assert job.stall_ratio == pytest.approx(0.5)   # s - 1


def test_apply_job_slowdown_already_dilated():
    """Already-dilated job still gets stall_ratio set correctly."""
    job = _make_job()
    job.dilated = True   # already dilated — no further apply_dilation call
    result = apply_job_slowdown(
        job=job,
        max_throughput=1000.0,
        net_util=1.5,
        net_cong=1.5,
        net_tx=2000.0,
        net_rx=1000.0,
    )
    assert job.stall_ratio == pytest.approx(result - 1.0)


# ---------------------------------------------------------------------------
# compute_link_stall_packet_stats
# ---------------------------------------------------------------------------

# Slingshot parameters (Frontier)
LINK_BW_BPS = 25e9 * 8      # 200 Gb/s in bits/s
MEAN_PKT_BYTES = 116
DT = 15.0                   # trace_quanta seconds


def test_link_stats_no_load():
    """Zero-load links: posted_pkts=0, tx_paused=0."""
    loads = {('a', 'b'): 0.0, ('b', 'c'): 0.0}
    stats = compute_link_stall_packet_stats(loads, LINK_BW_BPS, MEAN_PKT_BYTES, DT, slowdown_factor=2.0)
    for edge, s in stats.items():
        assert s['posted_pkts'] == pytest.approx(0.0)
        assert s['tx_paused'] == pytest.approx(0.0)
        assert s['stall_ratio'] == pytest.approx(1.0)   # s-1


def test_link_stats_at_capacity():
    """Link carrying exactly BW * dt bytes → rho=1, posted_pkts = max_pkt_rate * dt."""
    byte_load = LINK_BW_BPS / 8 * DT   # exactly one tick at full rate
    loads = {('a', 'b'): byte_load}
    stats = compute_link_stall_packet_stats(loads, LINK_BW_BPS, MEAN_PKT_BYTES, DT, slowdown_factor=1.0)
    s = stats[('a', 'b')]
    expected_posted = (LINK_BW_BPS / (MEAN_PKT_BYTES * 8)) * DT
    assert s['posted_pkts'] == pytest.approx(expected_posted, rel=1e-6)
    assert s['tx_paused'] == pytest.approx(0.0)   # s=1 → no stall
    assert s['utilization'] == pytest.approx(1.0)


def test_link_stats_stall_ratio_equals_s_minus_1():
    """stall_ratio field is always slowdown_factor - 1."""
    loads = {('a', 'b'): 1e6}
    for s in [1.0, 2.0, 5.7, 6.7]:
        stats = compute_link_stall_packet_stats(loads, LINK_BW_BPS, MEAN_PKT_BYTES, DT, slowdown_factor=s)
        assert stats[('a', 'b')]['stall_ratio'] == pytest.approx(max(0.0, s - 1.0))


def test_link_stats_utilization_clamped():
    """Utilization is clamped to 1.0 even when byte_load exceeds link capacity."""
    byte_load = LINK_BW_BPS / 8 * DT * 10  # 10x overload
    loads = {('a', 'b'): byte_load}
    stats = compute_link_stall_packet_stats(loads, LINK_BW_BPS, MEAN_PKT_BYTES, DT, slowdown_factor=1.0)
    assert stats[('a', 'b')]['utilization'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# aggregate_link_stall_stats
# ---------------------------------------------------------------------------

def test_aggregate_empty_stats():
    """Empty stats returns zeros."""
    result = aggregate_link_stall_stats({})
    assert result['total_posted_pkts'] == 0.0
    assert result['total_tx_paused'] == 0.0
    assert result['system_stall_ratio'] == 0.0


def test_aggregate_system_stall_ratio_equals_s_minus_1():
    """System stall ratio = s - 1 (by construction of the weighting)."""
    byte_load = LINK_BW_BPS / 8 * DT * 0.5   # 50% utilization
    loads = {('a', 'b'): byte_load, ('b', 'c'): byte_load}
    slowdown = 3.5
    link_stats = compute_link_stall_packet_stats(loads, LINK_BW_BPS, MEAN_PKT_BYTES, DT, slowdown)
    agg = aggregate_link_stall_stats(link_stats)
    # By derivation: system_stall_ratio = tx_paused / posted_pkts = (s-1)*P / P = s-1
    assert agg['system_stall_ratio'] == pytest.approx(slowdown - 1.0, rel=1e-6)


def test_aggregate_frontier_sanity_check():
    """
    Rough sanity check: stall_ratio = s - 1 = 6.7 - 1 = 5.7 for Frontier-level congestion.
    """
    slowdown = 6.7
    ratio = compute_stall_ratio(slowdown)
    assert ratio == pytest.approx(5.7, abs=0.01)
