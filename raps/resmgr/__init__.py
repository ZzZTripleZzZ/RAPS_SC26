"""
ResourceManager package initializer.
Exports a factory that returns the appropriate manager based on config.
"""
from .default import ExclusiveNodeResourceManager
from .multitenant import MultiTenantResourceManager
from raps.policy import AllocationStrategy


def make_resource_manager(total_nodes, down_nodes, config,
                          allocation_strategy=AllocationStrategy.CONTIGUOUS,
                          hybrid_threshold=0.5):
    """
    Factory to choose between exclusive-node and multitenant managers.

    Parameters:
    - total_nodes: Total number of nodes in the system
    - down_nodes: Set of node IDs that are down/unavailable
    - config: Configuration dictionary
    - allocation_strategy: Node allocation strategy (CONTIGUOUS, RANDOM, HYBRID)
    - hybrid_threshold: For HYBRID strategy, communication intensity threshold
    """
    if config.get("multitenant", False):
        return MultiTenantResourceManager(total_nodes, down_nodes, config)
    return ExclusiveNodeResourceManager(
        total_nodes, down_nodes, config,
        allocation_strategy=allocation_strategy,
        hybrid_threshold=hybrid_threshold
    )


# Alias for backward compatibility
ResourceManager = make_resource_manager

__all__ = [
    "make_resource_manager",
    "ResourceManager",
    "ExclusiveNodeResourceManager",
    "MultiTenantResourceManager"
]
