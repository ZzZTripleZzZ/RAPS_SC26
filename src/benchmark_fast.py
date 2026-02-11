#!/usr/bin/env python3
"""
Fast RAPS Benchmark
===================
Discrete-event simulation benchmark using RAPS Engine with:
- Poisson arrival + heavy-tailed (Weibull) job size distributions
- Real traffic template tiling from mini-app matrices
- Topology and routing comparison (minimal vs adaptive)
- Optional bursty traffic modulation (Poisson / CBR)
- Dilation and bully-effect metrics
- Real-time performance guard
"""
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import timedelta

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.stats import get_engine_stats, get_job_stats, get_network_stats
from raps.job import CommunicationPattern

# Import mini-app templates
try:
    from traffic_integration import load_all_templates, TrafficMatrixTemplate
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    print("Warning: traffic_integration not available")

# Mini-app to CommunicationPattern mapping
MINI_APP_PATTERNS = {
    'lulesh': CommunicationPattern.STENCIL_3D,
    'comd': CommunicationPattern.STENCIL_3D,
    'hpgmg': CommunicationPattern.STENCIL_3D,
    'cosp2': CommunicationPattern.ALL_TO_ALL,
}


class BenchmarkSimConfig(SingleSimConfig):
    """SimConfig for benchmarking - minimal overrides."""
    pass


def load_representative_templates(matrix_dir: Path) -> dict:
    """Load one representative template per mini-app."""
    if not TEMPLATES_AVAILABLE:
        return {}

    all_templates = load_all_templates(matrix_dir)

    # Filter to keep one per mini-app
    patterns = ['lulesh_n64', 'comd_n64', 'hpgmg_n64', 'cosp2_n64']
    templates = {}
    for name, template in all_templates.items():
        for pattern in patterns:
            if pattern in name:
                mini_app = pattern.split('_')[0]
                if mini_app not in templates:
                    templates[mini_app] = template
                    break
    return templates


def select_mini_app(job, mini_apps):
    """Select a mini-app for a job based on its characteristics."""
    # Use job size to bias selection: larger jobs more likely to be stencil
    if job.nodes_required > 64:
        # Prefer stencil patterns for large jobs
        stencil_apps = [a for a in mini_apps if MINI_APP_PATTERNS.get(a) == CommunicationPattern.STENCIL_3D]
        if stencil_apps:
            return stencil_apps[hash(job.id) % len(stencil_apps)]
    return mini_apps[hash(job.id) % len(mini_apps)]


def estimate_traffic_volume(job):
    """Estimate total traffic volume for a job based on size and duration."""
    nodes = max(1, job.nodes_required)
    duration = getattr(job, 'expected_run_time', 3600)
    # Rough estimate: bytes per node per second based on comm pattern
    comm = getattr(job, 'comm_pattern', CommunicationPattern.ALL_TO_ALL)
    if comm == CommunicationPattern.STENCIL_3D:
        bytes_per_node_per_sec = 6 * 150.0  # 6 neighbors, ~150 bytes each
    else:
        bytes_per_node_per_sec = (nodes - 1) * 50.0  # all-to-all
    return nodes * bytes_per_node_per_sec * duration


def modulate_traffic(base_rate, trace_len, mode='constant'):
    """Generate time-varying traffic trace from constant base rate.

    Args:
        base_rate: Average bytes per trace quantum
        trace_len: Number of trace entries
        mode: 'constant', 'poisson', or 'cbr'

    Returns:
        List of traffic values per trace quantum
    """
    if mode == 'poisson':
        # Poisson-modulated bursts
        bursts = np.random.poisson(lam=max(1, base_rate), size=trace_len)
        return bursts.astype(float).tolist()
    elif mode == 'cbr':
        # Constant bit rate with small variance
        noise = np.random.normal(0, base_rate * 0.1, trace_len)
        return (np.full(trace_len, base_rate) + noise).clip(0).tolist()
    else:
        return [base_rate] * trace_len


def assign_mini_app_patterns(jobs, templates: dict, use_tiling: bool = True,
                             traffic_modulation: str = 'constant'):
    """
    Assign mini-app communication patterns to jobs.

    1. Sets job.comm_pattern (STENCIL_3D or ALL_TO_ALL)
    2. If tiling templates available, uses tile_to_size() for realistic traffic
    3. Applies traffic modulation (constant, poisson, cbr)

    Args:
        jobs: Job list
        templates: Mini-app template dict
        use_tiling: Whether to use template tiling for traffic generation
        traffic_modulation: Traffic modulation mode ('constant', 'poisson', 'cbr')
    """
    if not jobs:
        return

    mini_apps = list(templates.keys()) if templates else ['lulesh']

    for i, job in enumerate(jobs):
        # 1. Assign mini-app type
        mini_app = select_mini_app(job, mini_apps)
        job.mini_app = mini_app

        # 2. Set communication pattern
        job.comm_pattern = MINI_APP_PATTERNS.get(mini_app, CommunicationPattern.ALL_TO_ALL)

        # 3. Set network traces
        if job.ntx_trace is None or len(job.ntx_trace) == 0:
            trace_quanta = getattr(job, 'trace_quanta', 15)
            trace_len = max(1, int(job.expected_run_time / trace_quanta))

            if use_tiling and templates and mini_app in templates:
                template = templates[mini_app]
                # Estimate total traffic volume
                total_volume = estimate_traffic_volume(job)
                # Tile the template to job size
                nodes = max(1, job.nodes_required)
                traffic_matrix = template.tile_to_size(nodes, total_volume)
                # Derive per-node TX/RX from the tiled matrix
                row_sums = traffic_matrix.sum(axis=1)  # bytes sent per node
                avg_bytes_per_step = float(row_sums.mean())
                # Apply traffic modulation
                job.ntx_trace = modulate_traffic(avg_bytes_per_step, trace_len, traffic_modulation)
                job.nrx_trace = modulate_traffic(avg_bytes_per_step, trace_len, traffic_modulation)
            else:
                # Fallback to simple estimation
                nodes = max(1, job.nodes_required)
                if job.comm_pattern == CommunicationPattern.STENCIL_3D:
                    traffic_per_step = nodes * 6 * 150.0
                else:
                    traffic_per_step = nodes * (nodes - 1) * 50.0
                job.ntx_trace = modulate_traffic(traffic_per_step, trace_len, traffic_modulation)
                job.nrx_trace = modulate_traffic(traffic_per_step, trace_len, traffic_modulation)


def collect_dilation_metrics(engine):
    """Collect dilation and bully-effect metrics from completed/running jobs.

    Returns:
        Dict with dilation statistics
    """
    all_jobs = engine.jobs
    if not all_jobs:
        return {}

    dilated_jobs = [j for j in all_jobs if getattr(j, 'dilated', False)]
    slowdown_factors = [getattr(j, 'slowdown_factor', 1.0) for j in all_jobs]

    metrics = {
        'total_jobs': len(all_jobs),
        'dilated_jobs': len(dilated_jobs),
        'dilated_pct': len(dilated_jobs) / len(all_jobs) * 100 if all_jobs else 0,
        'avg_slowdown': np.mean(slowdown_factors) if slowdown_factors else 1.0,
        'max_slowdown': max(slowdown_factors) if slowdown_factors else 1.0,
    }

    # Bully effect: jobs that are large and caused congestion while others dilated
    # A "bully" is a large job running concurrently with dilated smaller jobs
    if dilated_jobs:
        # Find jobs that were NOT dilated but were large (potential bullies)
        non_dilated = [j for j in all_jobs if not getattr(j, 'dilated', False)]
        large_non_dilated = [j for j in non_dilated if j.nodes_required > np.median([jj.nodes_required for jj in all_jobs])]
        metrics['potential_bullies'] = len(large_non_dilated)
    else:
        metrics['potential_bullies'] = 0

    # Max congestion from engine history
    if engine.net_congestion_history:
        congestion_values = [c for t, c in engine.net_congestion_history]
        metrics['max_congestion'] = max(congestion_values)
        metrics['avg_congestion'] = np.mean(congestion_values)
    else:
        metrics['max_congestion'] = 0.0
        metrics['avg_congestion'] = 0.0

    return metrics


def print_dilation_metrics(metrics):
    """Print dilation and bully-effect metrics."""
    if not metrics:
        print("  No dilation metrics available")
        return

    print(f"\n  --- Dilation & Bully Effect Metrics ---")
    print(f"  Total jobs processed:    {metrics['total_jobs']}")
    print(f"  Dilated jobs:            {metrics['dilated_jobs']} ({metrics['dilated_pct']:.1f}%)")
    print(f"  Avg slowdown factor:     {metrics['avg_slowdown']:.3f}")
    print(f"  Max slowdown factor:     {metrics['max_slowdown']:.3f}")
    print(f"  Potential bully jobs:    {metrics['potential_bullies']}")
    print(f"  Max congestion:          {metrics['max_congestion']:.4f}")
    print(f"  Avg congestion:          {metrics['avg_congestion']:.4f}")


def override_routing(system_config, routing, ugal_threshold=2.0, valiant_bias=0.0):
    """Override routing algorithm in a SystemConfig.

    Returns a new SystemConfig with the routing changed.
    """
    data = system_config.model_dump(mode='json')
    if 'network' in data and data['network'] is not None:
        data['network']['routing_algorithm'] = routing
        data['network']['ugal_threshold'] = ugal_threshold
        data['network']['valiant_bias'] = valiant_bias
    from raps.system_config import SystemConfig
    return SystemConfig.model_validate(data)


def run_benchmark(
    system: str = 'lassen',
    duration_minutes: int = 10,
    delta_t_seconds: int = 1,
    use_network: bool = True,
    data_path: str = None,
    use_mini_apps: bool = True,
    routing: str = None,
    traffic_modulation: str = 'constant',
):
    """
    Run a single benchmark configuration.

    Uses RAPS native Engine with minimal overhead.
    """
    print(f"\n{'='*60}")
    routing_label = routing or 'default'
    print(f"Config: {system}, {duration_minutes}min, dt={delta_t_seconds}s, "
          f"network={use_network}, routing={routing_label}, traffic={traffic_modulation}")
    print(f"{'='*60}")

    # Prepare configuration
    config_dict = {
        'system': system,
        'time': timedelta(minutes=duration_minutes),
        'time_delta': timedelta(seconds=delta_t_seconds),
        'simulate_network': use_network,
        'cooling': False,
        'uncertainties': False,
        'weather': False,
        'output': 'none',
        'noui': True,
        'verbose': False,
        'debug': False,
    }

    # Load data
    t_start = time.perf_counter()

    if data_path and Path(data_path).exists():
        # Use real data via replay
        config_dict['replay'] = [data_path]
        workload = True
    else:
        # Use synthetic workload with Poisson arrival + Weibull distributions
        config_dict['workload'] = 'synthetic'
        config_dict['policy'] = 'fcfs'
        config_dict['arrival'] = 'poisson'
        config_dict['numjobs'] = 50
        # Heavy-tailed job size (Weibull shape<1 approximates log-normal tail)
        config_dict['jobsize_distribution'] = ['weibull']
        config_dict['jobsize_weibull_scale'] = 64
        config_dict['jobsize_weibull_shape'] = 0.8
        # Heavy-tailed walltime
        config_dict['walltime_distribution'] = ['weibull']
        config_dict['walltime_weibull_scale'] = 1800
        config_dict['walltime_weibull_shape'] = 0.7
        workload = None

    t_load = time.perf_counter()
    print(f"Data load: {t_load - t_start:.2f}s")

    # Load mini-app templates
    templates = {}
    if use_mini_apps and TEMPLATES_AVAILABLE:
        templates = load_representative_templates(Path('data/matrices'))
        print(f"Loaded {len(templates)} mini-app templates")

    # Create Engine
    sim_config = BenchmarkSimConfig(**config_dict)

    # Override routing if requested
    if routing and use_network:
        orig_system_config = sim_config.system_configs[0]
        new_system_config = override_routing(orig_system_config, routing)
        sim_config._system_configs = [new_system_config]

    engine = Engine(sim_config)

    t_engine = time.perf_counter()
    print(f"Engine created: {t_engine - t_load:.2f}s")

    # Assign mini-app patterns (for synthetic workload)
    if workload is None and use_mini_apps:
        assign_mini_app_patterns(engine.jobs, templates,
                                 traffic_modulation=traffic_modulation)

    print(f"Jobs: {len(engine.jobs)}")

    # Run simulation with real-time performance guard
    t_sim_start = time.perf_counter()

    tick_count = 0
    realtime_warnings = 0
    for tick_data in engine.run_simulation():
        tick_count += 1

        # Real-time performance guard: check every 100 ticks
        if tick_count % 100 == 0:
            elapsed = time.perf_counter() - t_sim_start
            simulated = tick_count * delta_t_seconds
            if elapsed > simulated:
                realtime_warnings += 1
                if realtime_warnings <= 3:  # Don't spam warnings
                    print(f"  WARNING: Falling behind real-time "
                          f"({elapsed:.1f}s wall > {simulated:.1f}s sim)")

    t_sim_end = time.perf_counter()

    # Compute results
    sim_time = t_sim_end - t_sim_start
    total_time = t_sim_end - t_start
    simulated_seconds = duration_minutes * 60
    speedup = simulated_seconds / sim_time if sim_time > 0 else float('inf')

    print(f"\nResults:")
    print(f"  Simulation wall time:  {sim_time:.2f}s")
    print(f"  Total wall time:       {total_time:.2f}s")
    print(f"  Ticks:                 {tick_count}")
    print(f"  Per tick:              {sim_time/tick_count:.4f}s" if tick_count > 0 else "  Per tick: N/A")
    print(f"  Speedup:               {speedup:.1f}x")
    if realtime_warnings:
        print(f"  Real-time warnings:    {realtime_warnings}")

    # Network statistics
    if use_network:
        net_stats = get_network_stats(engine)
        print(f"  Avg Network Util:      {net_stats.get('avg_network_util', 0):.2f}%")
        print(f"  Avg Job Slowdown:      {net_stats.get('avg_per_job_slowdown', 1.0):.3f}")
        print(f"  Max Job Slowdown:      {net_stats.get('max_per_job_slowdown', 1.0):.3f}")
        print(f"  Avg Inter-Job Cong:    {net_stats.get('avg_inter_job_congestion', 0):.4f}")
        print(f"  Max Inter-Job Cong:    {net_stats.get('max_inter_job_congestion', 0):.4f}")

    # Dilation metrics
    dilation_metrics = collect_dilation_metrics(engine)
    print_dilation_metrics(dilation_metrics)

    return {
        'system': system,
        'duration_min': duration_minutes,
        'delta_t': delta_t_seconds,
        'network': use_network,
        'routing': routing or 'default',
        'traffic_modulation': traffic_modulation,
        'sim_time': sim_time,
        'total_time': total_time,
        'ticks': tick_count,
        'speedup': speedup,
        'jobs': len(engine.jobs),
        'realtime_warnings': realtime_warnings,
        'dilation_metrics': dilation_metrics,
    }


def run_routing_comparison(system, duration_minutes, delta_t_seconds, data_path,
                           use_mini_apps, traffic_modulation):
    """Run benchmark with minimal vs adaptive routing and compare results."""
    print("\n" + "#" * 60)
    print("# ROUTING COMPARISON: minimal vs adaptive")
    print("#" * 60)

    results = {}
    for routing in ['minimal', 'adaptive']:
        results[routing] = run_benchmark(
            system=system,
            duration_minutes=duration_minutes,
            delta_t_seconds=delta_t_seconds,
            use_network=True,
            data_path=data_path,
            use_mini_apps=use_mini_apps,
            routing=routing,
            traffic_modulation=traffic_modulation,
        )

    # Print comparison table
    print("\n" + "=" * 60)
    print("ROUTING COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Minimal':>12} {'Adaptive':>12}")
    print("-" * 54)

    for key in ['sim_time', 'speedup', 'ticks', 'jobs', 'realtime_warnings']:
        m_val = results['minimal'].get(key, 'N/A')
        a_val = results['adaptive'].get(key, 'N/A')
        if isinstance(m_val, float):
            print(f"{key:<30} {m_val:>12.2f} {a_val:>12.2f}")
        else:
            print(f"{key:<30} {str(m_val):>12} {str(a_val):>12}")

    # Dilation comparison
    print(f"\n{'Dilation Metrics':<30} {'Minimal':>12} {'Adaptive':>12}")
    print("-" * 54)
    for key in ['dilated_jobs', 'dilated_pct', 'avg_slowdown', 'max_slowdown',
                'potential_bullies', 'max_congestion', 'avg_congestion']:
        m_val = results['minimal'].get('dilation_metrics', {}).get(key, 'N/A')
        a_val = results['adaptive'].get('dilation_metrics', {}).get(key, 'N/A')
        if isinstance(m_val, float):
            print(f"{key:<30} {m_val:>12.3f} {a_val:>12.3f}")
        else:
            print(f"{key:<30} {str(m_val):>12} {str(a_val):>12}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Fast RAPS Benchmark')
    parser.add_argument('--system', default='lassen', choices=['lassen', 'frontier'])
    parser.add_argument('--duration', type=int, default=10, help='Duration in minutes')
    parser.add_argument('--delta-t', type=int, default=1, help='Time quantum in seconds')
    parser.add_argument('--no-network', action='store_true', help='Disable network simulation')
    parser.add_argument('--no-mini-apps', action='store_true', help='Disable mini-app patterns')
    parser.add_argument('--data-path', type=str, default='data/lassen/repo/Lassen-Supercomputer-Job-Dataset')
    parser.add_argument('--routing', type=str, default=None,
                        choices=['minimal', 'adaptive', 'ecmp', 'ugal', 'valiant'],
                        help='Override routing algorithm')
    parser.add_argument('--compare-routing', action='store_true',
                        help='Run minimal vs adaptive routing comparison')
    parser.add_argument('--traffic-modulation', type=str, default='constant',
                        choices=['constant', 'poisson', 'cbr'],
                        help='Traffic modulation mode')

    args = parser.parse_args()

    print("=" * 60)
    print("Fast RAPS Benchmark")
    print("Poisson arrival, Weibull job sizes, template tiling")
    print("=" * 60)

    if args.compare_routing:
        run_routing_comparison(
            system=args.system,
            duration_minutes=args.duration,
            delta_t_seconds=args.delta_t,
            data_path=args.data_path,
            use_mini_apps=not args.no_mini_apps,
            traffic_modulation=args.traffic_modulation,
        )
    else:
        run_benchmark(
            system=args.system,
            duration_minutes=args.duration,
            delta_t_seconds=args.delta_t,
            use_network=not args.no_network,
            data_path=args.data_path,
            use_mini_apps=not args.no_mini_apps,
            routing=args.routing,
            traffic_modulation=args.traffic_modulation,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
