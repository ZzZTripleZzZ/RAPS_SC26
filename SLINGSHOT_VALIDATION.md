# Frontier Slingshot Network Telemetry Validation

## Goal
Validate the RAPS network congestion model (`raps/network/base.py`) against real Frontier Slingshot NIC telemetry.

## Telemetry Data Structure

```
slingshot_{metric}/{month}/{date}/{job_id}_{job_name}/{metric}_cassini.parquet
```

Four metrics, each with the same directory structure:
- `slingshot_txBW`         — TX bandwidth per NIC port (bytes/s, instantaneous)
- `slingshot_rxBW`         — RX bandwidth per NIC port (bytes/s, instantaneous)
- `slingshot_rxCongestion` — RX congestion counter (**cumulative** bytes, must diff to get rate)
- `slingshot_idle`         — Link idle percentage (%, e.g. 96–98% typical)

**Parquet schema:** columns are `Timestamp`, then `frontierNNNNNhP` per NIC port
(node NNNNN, port P ∈ {0,1,2,3}; each node has 4 × 200 Gb/s Slingshot ports)
**NaN = no traffic / no change** — treat as 0 for bandwidth, skip for cumulative counters.
**Sampling interval:** ~60 seconds.
**Scale:** ~540 job directories per day (~17–20 nodes/job average on 2025-08-23).

## Key Mapping: Slingshot → RAPS

| Telemetry | RAPS quantity | Formula |
|---|---|---|
| `txBW` sum h0–h3 per node | `net_tx` per node (bytes/s) | fill NaN→0, sum ports |
| `rxBW` sum h0–h3 per node | `net_rx` per node (bytes/s) | fill NaN→0, sum ports |
| `idle` % | `1 - network_utilization` | `util = (100 - idle) / 100` |
| `rxCongestion` delta / rxBW | stall fraction ≈ `stall_ratio` | `Δcongestion / Δtime / rx_bytes` |

RAPS model functions to validate:
- `network_utilization()` in `base.py:146` — compare against `(100-idle)/100`
- `network_slowdown()` in `base.py:155` — compare against rxCongestion delta ratio
- `compute_stall_ratio()` in `base.py:72` — predicts `slowdown_factor - 1`

## Job ID Linkage

The **job_id is the numeric prefix** of the directory name:
```
3691196_AGMNPOEJMA  →  job_id = 3691196
```

This matches `job_id` in the existing `joblive` parquet (Frontier dataloader, `frontier.py:188`).

**Full join:**
```
slingshot dir job_id
    ↕
joblive parquet: job_id → xnames, time_start, time_end, node_count
    ↕
xname_to_index() (frontier.py:558) → RAPS node indices
    ↕
node_index_to_name() (frontier.py:582) → verify against frontierNNNNNhP columns
```

The `frontierNNNNNhP` column names give a second cross-check path (node number → xname),
but the `xname_to_index` / `node_index_to_name` round-trip needs to be verified against
the actual node numbering scheme used in the slingshot data.

## Validation Steps

1. **Parse job_id** from slingshot directory name (split on `_`, take first token)
2. **Look up job in joblive** → get xnames, time_start, time_end, node_count
3. **Load all 4 parquets** for that job, fill NaN→0
4. **Aggregate per-node bandwidth:** sum h0+h1+h2+h3 per node for tx and rx
5. **Compute observed utilization:** `(100 - idle_pct) / 100` per NIC
6. **Compute observed congestion rate:** diff consecutive rxCongestion rows, divide by interval and rxBW
7. **Run RAPS replay** for same job(s) with `--policy replay --net`
8. **Compare:** simulated vs observed utilization, tx/rx bandwidth magnitude, stall_ratio

## Findings from Initial Data Exploration

### Sample jobs (2025-08-23) are lightly loaded
- TX peak: ~27 MB/s on one port = **0.1% of 25 GB/s link capacity**
- RX peak: ~262 KB/s = **0.001% utilization**
- Neither sample job would trigger congestion in the RAPS model
- NaN values mean zero / below reporting threshold (not missing data)
- Port h3 consistently idle across samples — possibly reserved or on an unused path

### rxCongestion counter is cumulative from boot, not per-job
Values ~1.5e12 persist across jobs. Must diff consecutive non-NaN readings within a job
to get the per-job increment. Apparent decreases between readings are likely float
precision artifacts in the parquet storage.

### `max_delta` is not a useful ranking metric
Running `slingshot_find_congested_jobs.py --metric rxCongestion` and sorting by `max_delta` shows
nearly identical values (~6.8e12) for all top-20 jobs regardless of size or duration.
The counter saturates on hot ports, so max_delta just reflects the baseline counter level.
**Use `sum_delta` or `sum_delta / n_nodes` instead.**

### Top congested jobs (by sum_delta, from `slingshot_find_congested_jobs.py`)

| job_id | nodes | duration | sum_delta | congested% | notes |
|---|---|---|---|---|---|
| 3513339 | 1707 | **1 min** | 1.044e16 | 24% | suspicious — very short, huge delta |
| 3687816 | 1920 | 1h | 1.668e16 | 58% | good candidate |
| 3687555 | 1920 | 1h | 2.690e16 | 70% | good candidate |
| 3572284 | **9000** | 41min | 6.575e14 | 12% | best for model validation — spans many groups |
| 3897345 | 16 | 1.25h | 1.274e14 | **70%** | small job, hot links |

Job 3572284 (9000 nodes, 41 min) is the strongest candidate: large enough to span many
dragonfly groups, moderate congestion fraction, enough timestamps for time-series analysis.

### `rxCongestion` alone is insufficient for validation
It confirms congestion occurred but not the utilization that caused it. Need all three:

| Metric | Purpose |
|---|---|
| `txBW` + `rxBW` | Observed link utilization → what model *should* predict |
| `rxCongestion` delta | Ground truth that congestion actually occurred |
| `idle` | Independent utilization cross-check (optional but useful) |

The key validation plot: **utilization (from txBW+rxBW) vs congestion_delta** — checks
whether the model's congestion threshold and slowdown magnitude are correctly calibrated.

### Script: `scripts/slingshot_find_congested_jobs.py`
Crawls all slingshot directories and ranks jobs by congestion or bandwidth.
- Sort by `sum_delta` not `max_delta` for rxCongestion
- Run with `--metric rxCongestion|txBW|rxBW|idle`; results saved in `results_congestion/`

### Script: `scripts/slingshot_analyze_job_metrics.py`
Loads all four metrics for a single job, aligns on Timestamp, and produces a 4-subplot figure:
1. Utilization over time (`(100 - idle%) / 100`)
2. rx/tx bandwidth over time (total bytes/s, all ports)
3. rxCongestion rate over time (`diff(counter) / interval`)
4. **Utilization vs congestion ratio** scatter — key RAPS validation plot

`congestion_ratio = cong_rate / rxBW` is the dimensionless signal to compare against RAPS `stall_ratio`.

```bash
DATA=/lustre/orion/stf218/proj-shared/data/lake/frontier-data-campaign-2026/frontier-interconnect-fabric-telemetry
python scripts/slingshot_analyze_job_metrics.py $DATA --job-id 3691034 --date 2025_08_23 --out results_congestion/
```

## Validation Results (from `results_congestion/summary.csv`)

Generated by `scripts/slingshot_analyze_job_metrics.py --csv` across 8 jobs spanning four size regimes.
Bandwidth columns are **per node** (system total / n_nodes). `cong_onset_util` = mean utilization
at timestamps where any congestion was detected.

| job_id | nodes | duration | peak_util | mean_util | peak_rxBW/node | mean_rxBW/node | frac_congested | cong_onset_util |
|---|---|---|---|---|---|---|---|---|
| 3897345 | 16 | 79 min | 0.108 | 0.099 | 4.6 GB/s | 2.9 GB/s | 0.49 | **0.10** |
| 3691634 | 192 | 59 min | 0.430 | 0.415 | 19.5 GB/s | 11.7 GB/s | 0.47 | **0.42** |
| 3691034 | 1920 | 71 min | 0.713 | 0.408 | 20.5 GB/s | 5.8 GB/s | 0.40 | **0.46** |
| 3688454 | 1920 | 71 min | 0.699 | 0.400 | 20.2 GB/s | 5.6 GB/s | 0.38 | **0.45** |
| 3691160 | 1920 | 71 min | 0.767 | 0.353 | 14.2 GB/s | 4.6 GB/s | 0.42 | **0.39** |
| 3688392 | 1920 | 100 min | 0.712 | 0.283 | 20.4 GB/s | 5.3 GB/s | 0.44 | **0.30** |
| 3689621 | 6750 | 85 min | 0.058 | 0.040 | 0.65 GB/s | 0.24 GB/s | 0.40 | **0.044** |
| 3688655 | 9408 | 18 min | 0.070 | 0.013 | 0.41 GB/s | 0.04 GB/s | 0.26 | **0.024** |

### Key finding: congestion onset threshold depends on job size

`cong_onset_util` is not constant — it varies strongly with node count, consistent with
dragonfly inter-group bottleneck physics:

| regime | nodes | cong_onset_util | bottleneck |
|---|---|---|---|
| intra-rack / small | 16 | 0.10 | local router ports |
| intra-group | 192 | 0.42 | group-local links |
| multi-group | 1920 | 0.30–0.46 | inter-group global links starting to load |
| large multi-group | 6750 | 0.044 | most traffic is inter-group |
| near-full system | 9408 | 0.024 | nearly all links are inter-group |

**The original RAPS `network_slowdown()` triggered only above 100% utilization — it
misses congestion entirely for every job in this table.** A size-dependent threshold is
needed.

### Note on `peak_cong_rate_GBs_per_node`

This column's values exceed link line rate (100 GB/s/node max) for most jobs, confirming
the Cassini `rxCongestion` counter is **not in bytes** — it is likely a stall-cycle or
packet counter. Use `frac_congested` and `cong_onset_util` for quantitative validation;
treat `cong_rate` only as a presence/absence signal.

### Data artifacts to ignore
- Jobs 3678979 and 3690552 have `peak_bw=187096%` in idle file — counter overflow, not real.
- `max_delta` saturates at ~6.78e12 for top congestion jobs — use `sum_delta` for ranking.
- Job 3691634 was previously noted as "zero congestion baseline" — this was wrong; it shows
  `frac_congested=0.47`. The earlier finding was from `slingshot_find_congested_jobs.py` using a
  different threshold.

## Topology Analysis: Why Bandwidth Conservation Doesn't Work

Before calibrating a threshold, it's important to understand what the threshold physically
represents. Analysis of Frontier's dragonfly parameters reveals a key constraint:

**Global link overprovisioning on Frontier (D=32, H=30, P=2, G=74):**
- NICs per group: D×P = 64
- Global link ports per group: D×H = 960 → **15× more capacity than NIC injection rate**

Because aggregate global bandwidth exceeds NIC injection bandwidth by 15×, no bandwidth
conservation model can predict congestion: the worst link utilization (`net_cong`) stays
<< 1 for any realistically achievable NIC utilization. The original `net_cong > 1`
trigger essentially never fired.

The empirically observed low thresholds (e.g. 0.024 at full system) reflect **credit-based
head-of-line (HOL) blocking** — what Cassini `hni_tx_paused` actually measures. When a
large job spans many groups, Slingshot credit pools drain before any link hits 100%
bandwidth utilization. This mechanism is not derivable from bandwidth conservation alone;
it requires Slingshot hardware specs (credit pool size, per-VC buffers, link RTT) that
are not publicly available.

**What IS derivable from topology parameters alone:**

```
threshold = H / ((G-1) × P)
          = dragonfly_inter / ((dragonfly_groups - 1) × dragonfly_p)
          = 30 / (73 × 2) = 0.205   [Frontier]
```

Physical meaning: NIC utilization at which aggregate per-node global link demand equals
per-node global link supply, for a balanced full-system all-to-all job. Derived entirely
from config parameters, no fitted constants.

**Tradeoff**: This is size-independent. The empirical data shows `cong_onset_util` ranging
from 0.024 (n=9408) to 0.42 (n=192) — the topology-derived threshold of 0.205 falls in
the middle of that range. It overestimates congestion for small jobs and underestimates
for large jobs. A power-law fit matches the data better, but requires three empirical
constants and is only calibrated for Frontier 2025-08-23 data.

**Decision**: Use the topology-derived threshold for RAPS. It is physically motivated,
fully reproducible from config, portable to other dragonfly systems (Frontier, Polaris,
etc.), and defensible in a paper without relying on a curve fit.

## Model Update (implemented)

`raps/network/base.py` now has a topology-derived congestion threshold:

```python
def congestion_threshold(config: dict) -> float:
    # For dragonfly: H / ((G-1) * P)
    # For all other topologies: 1.0 (no congestion model)
    H = config['DRAGONFLY_INTER']   # global links per router
    G = config['DRAGONFLY_GROUPS']  # number of groups
    P = config['DRAGONFLY_P']       # hosts per router
    return H / ((G - 1) * P)
```

For Frontier: `30 / (73 × 2) = 0.205`. Computed once at `Engine.__init__` from the
system config and passed as a constant to `apply_job_slowdown` each tick.

- `apply_job_slowdown(threshold=...)` triggers on `net_util > threshold`
- Slowdown magnitude: `net_util / threshold` (linear above onset)
- No hardcoded empirical constants anywhere in the model
- Non-dragonfly topologies get `threshold=1.0` (effectively no congestion)
- **Important**: RAPS is an aggregate bandwidth model, not a flow model. The graph/routing
  layer computes `net_cong` (worst link util) but the slowdown is driven by NIC-aggregate
  `net_util`. Accurate label: "aggregate bandwidth model with topology-derived congestion
  threshold and topology-aware routing for adaptive path selection."

## Next Step

Add `--total-nodes` arg to `slingshot_analyze_job_metrics.py` and output `threshold` +
`pred_frac_congested` columns to compare model vs observed congestion fraction across
all 8 jobs. This validates that the model predicts the right *amount* of congestion,
not just the onset. Note: threshold was calibrated on this same dataset so comparison
is internal consistency, not out-of-sample. For stronger validation: held-out dates
or RAPS replay comparison.

## Open Questions

- Does `frontierNNNNNN` node number correspond 1:1 to xname index? Need to verify mapping.
- Is `rxCongestion` reset at job start or persistent across reboots? (values ~1.5e12 suggest long-running counter)
- Do you have `joblive` parquet for the same dates as the slingshot data (e.g. 2025-08-23)?
- What are the exact units of the Cassini `rxCongestion` counter? (stall cycles? packets? not bytes)
