import pytest
from raps.network.fat_tree import (
    build_fattree,
    node_id_to_host_name,
    parse_fattree_host,
    get_host_edge_switch,
    fattree_all_shortest_paths,
    fattree_ecmp_select_path,
    fattree_adaptive_select_path,
    fattree_route,
    estimate_path_load,
    link_loads_for_job_fattree_adaptive,
)

def test_build_fattree_k4():
    """Test building a k=4 fat-tree."""
    k = 4
    G = build_fattree(k, 16)

    # Check number of nodes
    num_hosts = k * (k // 2) * (k // 2)
    num_edge_switches = k * (k // 2)
    num_agg_switches = k * (k // 2)
    num_core_switches = (k // 2) ** 2
    total_nodes = num_hosts + num_edge_switches + num_agg_switches + num_core_switches
    assert len(G.nodes) == total_nodes

    # Check number of edges
    # Host to edge switch edges
    host_edges = num_hosts
    # Edge to agg switch edges
    edge_agg_edges = k * (k // 2) * (k // 2)
    # Agg to core switch edges
    agg_core_edges = k * (k // 2) * (k // 2)
    total_edges = host_edges + edge_agg_edges + agg_core_edges
    assert len(G.edges) == total_edges

    # Check node types
    node_types = [data["type"] for _, data in G.nodes(data=True)]
    assert node_types.count("host") == num_hosts
    assert node_types.count("edge") == num_edge_switches
    assert node_types.count("agg") == num_agg_switches
    assert node_types.count("core") == num_core_switches

def test_node_id_to_host_name():
    """Test the node_id_to_host_name function."""
    k = 4
    # Test a few node IDs
    assert node_id_to_host_name(0, k) == "h_0_0_0"
    assert node_id_to_host_name(1, k) == "h_0_0_1"
    assert node_id_to_host_name(2, k) == "h_0_1_0"
    assert node_id_to_host_name(3, k) == "h_0_1_1"
    assert node_id_to_host_name(4, k) == "h_1_0_0"


# =============================================================================
# Tests for Adaptive Routing Functions
# =============================================================================

def test_parse_fattree_host():
    """Test parsing host names."""
    assert parse_fattree_host("h_0_0_0") == (0, 0, 0)
    assert parse_fattree_host("h_2_1_3") == (2, 1, 3)
    assert parse_fattree_host("h_10_5_7") == (10, 5, 7)


def test_get_host_edge_switch():
    """Test getting edge switch for a host."""
    assert get_host_edge_switch("h_0_0_0") == "e_0_0"
    assert get_host_edge_switch("h_2_1_3") == "e_2_1"
    assert get_host_edge_switch("h_3_2_1") == "e_3_2"


def test_fattree_all_shortest_paths_same_host():
    """Test paths to same host."""
    k = 4
    G = build_fattree(k, 16)
    paths = fattree_all_shortest_paths(G, "h_0_0_0", "h_0_0_0")
    assert paths == [["h_0_0_0"]]


def test_fattree_all_shortest_paths_same_edge():
    """Test paths between hosts on same edge switch."""
    k = 4
    G = build_fattree(k, 16)
    paths = fattree_all_shortest_paths(G, "h_0_0_0", "h_0_0_1")
    # Should go through edge switch only
    assert len(paths) == 1
    assert paths[0] == ["h_0_0_0", "e_0_0", "h_0_0_1"]


def test_fattree_all_shortest_paths_same_pod():
    """Test paths between hosts in same pod but different edge switches."""
    k = 4
    G = build_fattree(k, 16)
    paths = fattree_all_shortest_paths(G, "h_0_0_0", "h_0_1_0")
    # Should go through edge -> agg -> edge
    # Multiple paths possible through different agg switches
    assert len(paths) >= 1
    for path in paths:
        assert path[0] == "h_0_0_0"
        assert path[-1] == "h_0_1_0"
        assert len(path) == 5  # host -> edge -> agg -> edge -> host


def test_fattree_all_shortest_paths_different_pods():
    """Test paths between hosts in different pods."""
    k = 4
    G = build_fattree(k, 16)
    paths = fattree_all_shortest_paths(G, "h_0_0_0", "h_1_0_0")
    # Should go through edge -> agg -> core -> agg -> edge
    # Multiple paths through different agg/core switches
    assert len(paths) >= 1
    for path in paths:
        assert path[0] == "h_0_0_0"
        assert path[-1] == "h_1_0_0"
        assert len(path) == 7  # host -> edge -> agg -> core -> agg -> edge -> host


def test_fattree_ecmp_returns_valid_path():
    """Test that ECMP returns a valid shortest path."""
    k = 4
    G = build_fattree(k, 16)

    for _ in range(10):  # Run multiple times due to randomness
        path = fattree_ecmp_select_path(G, "h_0_0_0", "h_1_0_0")
        assert path[0] == "h_0_0_0"
        assert path[-1] == "h_1_0_0"
        assert len(path) == 7


def test_estimate_path_load():
    """Test path load estimation."""
    path = ["a", "b", "c", "d"]
    link_loads = {
        ("a", "b"): 100.0,
        ("b", "c"): 50.0,
        ("c", "d"): 25.0,
    }
    load = estimate_path_load(path, link_loads)
    assert load == 175.0


def test_estimate_path_load_no_loads():
    """Test path load with empty link loads."""
    path = ["a", "b", "c"]
    link_loads = {}
    load = estimate_path_load(path, link_loads)
    assert load == 0.0


def test_fattree_adaptive_selects_least_congested():
    """Test that adaptive routing selects least congested path."""
    k = 4
    G = build_fattree(k, 16)

    # Get all paths between different pods
    all_paths = fattree_all_shortest_paths(G, "h_0_0_0", "h_1_0_0")
    assert len(all_paths) >= 2  # Should have multiple paths

    # Create loads that heavily congest first path
    link_loads = {}
    for edge in G.edges():
        link_loads[tuple(sorted(edge))] = 0.0

    # Add high load to first path
    first_path = all_paths[0]
    for i in range(len(first_path) - 1):
        edge = tuple(sorted([first_path[i], first_path[i + 1]]))
        link_loads[edge] = 1000000.0

    # Adaptive should avoid the congested path
    selected_path = fattree_adaptive_select_path(G, "h_0_0_0", "h_1_0_0", link_loads)
    assert selected_path != first_path or len(all_paths) == 1


def test_fattree_route_minimal():
    """Test minimal routing dispatcher."""
    k = 4
    G = build_fattree(k, 16)

    path = fattree_route(G, "h_0_0_0", "h_1_0_0", 'minimal')
    assert path[0] == "h_0_0_0"
    assert path[-1] == "h_1_0_0"
    assert len(path) == 7


def test_fattree_route_ecmp():
    """Test ECMP routing dispatcher."""
    k = 4
    G = build_fattree(k, 16)

    path = fattree_route(G, "h_0_0_0", "h_1_0_0", 'ecmp')
    assert path[0] == "h_0_0_0"
    assert path[-1] == "h_1_0_0"
    assert len(path) == 7


def test_fattree_route_adaptive():
    """Test adaptive routing dispatcher."""
    k = 4
    G = build_fattree(k, 16)
    link_loads = {tuple(sorted(edge)): 0.0 for edge in G.edges()}

    path = fattree_route(G, "h_0_0_0", "h_1_0_0", 'adaptive', link_loads)
    assert path[0] == "h_0_0_0"
    assert path[-1] == "h_1_0_0"
    assert len(path) == 7


def test_fattree_route_same_host():
    """Test routing to same host."""
    k = 4
    G = build_fattree(k, 16)

    for algo in ['minimal', 'ecmp', 'adaptive']:
        path = fattree_route(G, "h_0_0_0", "h_0_0_0", algo)
        assert path == ["h_0_0_0"]


def test_link_loads_for_job_fattree_minimal():
    """Test link load computation with minimal routing."""
    k = 4
    G = build_fattree(k, 16)
    hosts = ["h_0_0_0", "h_0_0_1", "h_0_1_0"]

    loads = link_loads_for_job_fattree_adaptive(
        G, hosts, tx_volume_bytes=1000.0, algorithm='minimal'
    )

    # Should have some non-zero loads
    total_load = sum(loads.values())
    assert total_load > 0


def test_link_loads_for_job_fattree_adaptive():
    """Test link load computation with adaptive routing."""
    k = 4
    G = build_fattree(k, 16)
    hosts = ["h_0_0_0", "h_1_0_0"]  # Different pods

    loads = link_loads_for_job_fattree_adaptive(
        G, hosts, tx_volume_bytes=1000.0, algorithm='adaptive'
    )

    total_load = sum(loads.values())
    assert total_load > 0


def test_link_loads_single_host():
    """Test that single host produces no loads."""
    k = 4
    G = build_fattree(k, 16)
    hosts = ["h_0_0_0"]

    loads = link_loads_for_job_fattree_adaptive(
        G, hosts, tx_volume_bytes=1000.0, algorithm='minimal'
    )

    assert loads == {}
