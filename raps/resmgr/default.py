import random

from raps.job import JobState
from raps.policy import PolicyType, AllocationStrategy


class ExclusiveNodeResourceManager:
    """
    Exclusive-node resource manager: allocates and frees full nodes.

    Supports different allocation strategies based on:
    "Watch Out for the Bully! Job Interference Study on Dragonfly Network"
    (Yang et al., SC16)
    """

    def __init__(self, total_nodes, down_nodes, config=None,
                 allocation_strategy=AllocationStrategy.CONTIGUOUS,
                 hybrid_threshold=None):
        """
        Initialize the resource manager.

        Parameters:
        - total_nodes: Total number of nodes in the system
        - down_nodes: Set of node IDs that are down/unavailable
        - config: Configuration dictionary
        - allocation_strategy: Node allocation strategy (CONTIGUOUS, RANDOM, HYBRID)
        - hybrid_threshold: For HYBRID strategy, jobs with communication intensity
                           above this threshold get RANDOM allocation, below get CONTIGUOUS.
                           If None, defaults to 0.5 (median).
        """
        self.total_nodes = total_nodes
        self.down_nodes = set(down_nodes)
        self.config = config or {}
        self.allocation_strategy = allocation_strategy
        self.hybrid_threshold = hybrid_threshold if hybrid_threshold is not None else 0.5

        # Determine per-node capacities
        cfg = self.config
        if 'CPUS_PER_NODE' in cfg and 'CORES_PER_CPU' in cfg:
            total_cpu = cfg['CPUS_PER_NODE'] * cfg['CORES_PER_CPU']
        else:
            total_cpu = cfg.get('CORES_PER_NODE', cfg.get('CPUS_PER_NODE', 1))
        total_gpu = cfg.get('GPUS_PER_NODE', 0)

        # Build unified node list so engine can inspect resource_manager.nodes
        self.nodes = []
        for i in range(self.total_nodes):
            is_down = i in self.down_nodes
            self.nodes.append({
                'id': i,
                'total_cpu_cores':     total_cpu,
                'available_cpu_cores': 0 if is_down else total_cpu,
                'total_gpu_units':     total_gpu,
                'available_gpu_units': 0 if is_down else total_gpu,
                'is_down':             is_down
            })

        # Available nodes list for allocation/frees
        self.available_nodes = [n['id'] for n in self.nodes if not n['is_down']]
        # System utilization history (time, util%)
        self.sys_util_history = []

    def assign_nodes_to_job(self, job, current_time, policy, node_id=None):
        """Assigns full nodes to a job using the configured allocation strategy.

        Allocation strategies:
        - CONTIGUOUS: Take first N available nodes (sequential allocation)
        - RANDOM: Randomly sample N nodes from available pool
        - HYBRID: Use job's communication intensity to decide strategy
        """
        # Ensure enough free nodes
        if len(self.available_nodes) < job.nodes_required:
            raise ValueError(f"Not enough available nodes to schedule job {job.id}",
                             f"{len(self.available_nodes)} < {job.nodes_required}")

        if policy == PolicyType.REPLAY and job.scheduled_nodes:
            # Telemetry replay: use the exact nodes from trace
            self.available_nodes = [n for n in self.available_nodes if n not in job.scheduled_nodes]
        else:
            # Apply allocation strategy
            job.scheduled_nodes = self._allocate_nodes(job)
            self.available_nodes = [n for n in self.available_nodes if n not in job.scheduled_nodes]

        # Mark job running
        job.start_time = current_time
        if job.expected_run_time:
            job.end_time = current_time + job.expected_run_time  # This may be an assumption!
        job.current_state = JobState.RUNNING

    def _allocate_nodes(self, job):
        """Select nodes based on allocation strategy.

        Returns:
            List of node IDs allocated to the job.
        """
        n = job.nodes_required
        strategy = self._get_effective_strategy(job)

        if strategy == AllocationStrategy.CONTIGUOUS:
            # Take first N available nodes (maintains locality)
            return self.available_nodes[:n]
        elif strategy == AllocationStrategy.RANDOM:
            # Randomly sample N nodes (distributes traffic)
            return random.sample(self.available_nodes, n)
        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")

    def _get_effective_strategy(self, job):
        """Determine effective strategy, handling HYBRID logic.

        For HYBRID: high communication intensity -> RANDOM, low -> CONTIGUOUS
        """
        if self.allocation_strategy != AllocationStrategy.HYBRID:
            return self.allocation_strategy

        # HYBRID: decide based on job's communication intensity
        intensity = self._compute_communication_intensity(job)
        if intensity >= self.hybrid_threshold:
            return AllocationStrategy.RANDOM
        else:
            return AllocationStrategy.CONTIGUOUS

    def _compute_communication_intensity(self, job):
        """Compute normalized communication intensity for a job.

        Uses network TX/RX traces to estimate how communication-intensive
        the job is. Returns value in [0, 1] range.
        """
        import numpy as np

        # Get network traces
        ntx = getattr(job, 'ntx_trace', None)
        nrx = getattr(job, 'nrx_trace', None)

        # Compute average network activity
        total = 0.0
        count = 0

        for trace in [ntx, nrx]:
            if trace is not None:
                if isinstance(trace, (list, np.ndarray)) and len(trace) > 0:
                    total += np.mean(trace)
                    count += 1
                elif isinstance(trace, (int, float)):
                    total += trace
                    count += 1

        if count == 0:
            # No network data available, default to contiguous (conservative)
            return 0.0

        avg_network = total / count

        # Normalize: this is a simple heuristic, can be tuned
        # Assumes network values are in some reasonable range
        # For now, use a simple sigmoid-like normalization
        # Values > 1.0 are considered "high intensity"
        intensity = min(1.0, avg_network / 1.0) if avg_network > 0 else 0.0

        return intensity

    def free_nodes_from_job(self, job):
        """Frees the full nodes previously allocated to a job."""
        if getattr(job, 'scheduled_nodes', None):
            for n in job.scheduled_nodes:
                if n not in self.available_nodes:
                    self.available_nodes.append(n)
                else:
                    # Already free — log instead of raising
                    print(f"[WARN] Tried to free node {n}, but it was already available")
                    print(f"Atempting to free node {n} after completion of job {job.id}. " +
                                     "Node is already free (in available nodes)!")
            self.available_nodes = sorted(self.available_nodes)

    def update_system_utilization(self, current_time, running_jobs):
        """
        Computes system utilization as percentage of non-down nodes that are active.

        Parameters:
        - current_time: simulation time
        - running_jobs: list of currently running Job objects
        """
        # Number of active nodes is length of running_jobs
        num_active = len(running_jobs)
        total_operational = self.total_nodes - len(self.down_nodes)
        util = (num_active / total_operational) * 100 if total_operational else 0
        self.sys_util_history.append((current_time, util))
        return util
        # """
        # Computes system utilization as percentage of non-down nodes that are active.
        # """
        # total_operational = self.total_nodes - len(self.down_nodes)
        # util = (num_active_nodes / total_operational) * 100 if total_operational else 0
        # self.sys_util_history.append((current_time, util))
        # return util

    def node_failure(self, mtbf):
        return []
        # Node failure not working!
        #  """Simulate node failure using Weibull distribution."""
        #  shape_parameter = 1.5
        #  scale_parameter = mtbf * 3600  # Convert to seconds

        #  # Create a NumPy array of node indices, excluding down nodes
        #  all_nodes = np.array(sorted(set(range(self.total_nodes)) - set(self.down_nodes)))

        #  # Sample the Weibull distribution for all nodes at once
        #  random_values = weibull_min.rvs(shape_parameter, scale=scale_parameter, size=all_nodes.size)

        #  # Identify nodes that have failed
        #  failure_threshold = 0.1
        #  failed_nodes_mask = random_values < failure_threshold
        #  newly_downed_nodes = all_nodes[failed_nodes_mask]

        #  # Update available and down nodes
        #  for node_index in newly_downed_nodes:
        #      if node_index in self.available_nodes:
        #          self.available_nodes.remove(node_index)
        #      self.down_nodes.add(str(node_index))

        #  return newly_downed_nodes.tolist()
