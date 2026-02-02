from .utils import ValueComparableEnum


class PolicyType(ValueComparableEnum):
    """Supported scheduling policies."""
    REPLAY = 'replay'  # Default is specified in each scheduler!
    FCFS = 'fcfs'
    PRIORITY = 'priority'
    SJF = 'sjf'
    LJF = 'ljf'


class BackfillType(ValueComparableEnum):
    """Supported backfilling policies."""
    NONE = None
    FIRSTFIT = 'firstfit'
    BESTFIT = 'bestfit'
    GREEDY = 'greedy'
    EASY = 'easy'  # Earliest Available Start Time Yielding
    CONSERVATIVE = 'conservative'


class AllocationStrategy(ValueComparableEnum):
    """Supported node allocation strategies.

    Based on job placement policies from:
    "Watch Out for the Bully! Job Interference Study on Dragonfly Network"
    (Yang et al., SC16)

    CONTIGUOUS: Nodes assigned consecutively, filling groups/racks first.
                Minimizes network resource sharing between jobs.
    RANDOM: Nodes randomly selected from available pool.
            Distributes traffic uniformly, enables load balancing.
    HYBRID: Communication-intensive jobs get random allocation,
            less intensive jobs get contiguous allocation.
    """
    CONTIGUOUS = 'contiguous'
    RANDOM = 'random'
    HYBRID = 'hybrid'


class RoutingAlgorithm(ValueComparableEnum):
    """Supported network routing algorithms for HPC topologies.

    Based on routing algorithms from:
    "Study of Workload Interference with Intelligent Routing on Dragonfly"
    (Kang et al., SC22)

    Dragonfly algorithms:
    MINIMAL: Always use shortest/minimal path routing.
             For Dragonfly: at most 3 hops (local-global-local).
    VALIANT: Valiant load balancing - route via random intermediate group.
             Configurable bias parameter controls minimal vs non-minimal ratio.
             valiant_bias=0.05 means 5% non-minimal, 95% minimal.
    UGAL: Universal Globally-Adaptive Load-balanced routing.
          Dynamically chooses minimal or non-minimal based on congestion.
          Uses threshold comparison: if min_latency < threshold * nonmin_latency,
          use minimal path; otherwise use non-minimal.

    Fat-tree algorithms:
    ECMP: Equal-Cost Multi-Path routing. Randomly selects among all
          shortest paths between source and destination.
    ADAPTIVE: Adaptive ECMP routing (InfiniBand AR). Selects the least
              congested path among all equal-cost shortest paths.
    """
    MINIMAL = 'minimal'
    VALIANT = 'valiant'
    UGAL = 'ugal'
    ECMP = 'ecmp'
    ADAPTIVE = 'adaptive'
