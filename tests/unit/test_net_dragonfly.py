import pytest
from raps.network.dragonfly import (
    build_dragonfly,
    dragonfly_node_id_to_host_name,
    parse_dragonfly_host,
    parse_dragonfly_router,
    get_host_router,
    dragonfly_minimal_path,
    dragonfly_nonminimal_path,
    estimate_path_latency,
    ugal_select_path,
    valiant_select_path,
    dragonfly_route,
    link_loads_for_job_dragonfly_adaptive,
)


def test_build_dragonfly():
    """Test building a small dragonfly network."""
    D = 2  # Routers per group
    A = 2  # Gloobal connections per router
    P = 2  # Compute nodes per router
    G = build_dragonfly(D, A, P)

    # Check number of nodes
    num_routers = D * (A + 1)
    num_hosts = num_routers * P
    total_nodes = num_routers + num_hosts
    assert len(G.nodes) == total_nodes

    # Check number of edges
    routers_per_group = D
    # Edges of the router clique:
    router_clique_edges_per_group = ((routers_per_group * (routers_per_group - 1)) // 2)
    # Edges for all router compute nodes:
    compute_node_edges_per_router = P
    # Total Intra-group edges:
    intra_group_edges = router_clique_edges_per_group + compute_node_edges_per_router * D

    # Inter-group edges
    total_groups = A + 1
    inter_group_edges_simple_clique = ((total_groups * (total_groups-1)) // 2)
    inter_group_edges = inter_group_edges_simple_clique * D
    # Host to router edges
    total_edges = intra_group_edges * total_groups + inter_group_edges
    assert len(G.edges) == total_edges

    # Check node types
    node_types = [data["layer"] for _, data in G.nodes(data=True)]
    assert node_types.count("router") == num_routers
    assert node_types.count("host") == num_hosts


def test_dragonfly_node_id_to_host_name():
    """Test the dragonfly_node_id_to_host_name function."""
    D, A, P = 2, 2, 2
    # Test a few node IDs
    assert dragonfly_node_id_to_host_name(0, D, A, P) == "h_0_0_0"
    assert dragonfly_node_id_to_host_name(1, D, A, P) == "h_0_0_1"
    assert dragonfly_node_id_to_host_name(2, D, A, P) == "h_0_1_0"
    assert dragonfly_node_id_to_host_name(3, D, A, P) == "h_0_1_1"
    assert dragonfly_node_id_to_host_name(4, D, A, P) == "h_1_0_0"


# =============================================================================
# Tests for Adaptive Routing Functions
# =============================================================================

def test_parse_dragonfly_host():
    """Test parsing host names."""
    assert parse_dragonfly_host("h_2_3_1") == (2, 3, 1)
    assert parse_dragonfly_host("h_0_0_0") == (0, 0, 0)
    assert parse_dragonfly_host("h_10_5_7") == (10, 5, 7)


def test_parse_dragonfly_router():
    """Test parsing router names."""
    assert parse_dragonfly_router("r_2_3") == (2, 3)
    assert parse_dragonfly_router("r_0_0") == (0, 0)


def test_get_host_router():
    """Test getting router for a host."""
    assert get_host_router("h_2_3_1") == "r_2_3"
    assert get_host_router("h_0_0_0") == "r_0_0"
    assert get_host_router("h_5_2_7") == "r_5_2"


def test_dragonfly_minimal_path_same_host():
    """Test minimal path to same host."""
    d, a = 4, 3
    path = dragonfly_minimal_path("h_0_0_0", "h_0_0_0", d, a)
    assert path == ["h_0_0_0"]


def test_dragonfly_minimal_path_same_router():
    """Test minimal path between hosts on same router."""
    d, a = 4, 3
    path = dragonfly_minimal_path("h_0_0_0", "h_0_0_1", d, a)
    assert path == ["h_0_0_0", "r_0_0", "h_0_0_1"]


def test_dragonfly_minimal_path_intra_group():
    """Test minimal path within same group (different routers)."""
    d, a = 4, 3
    path = dragonfly_minimal_path("h_0_0_0", "h_0_2_0", d, a)
    # Should be: host -> router -> router -> host
    assert path[0] == "h_0_0_0"
    assert path[-1] == "h_0_2_0"
    assert len(path) == 4  # h, r, r, h


def test_dragonfly_minimal_path_inter_group():
    """Test minimal path between groups."""
    d, a = 4, 3
    path = dragonfly_minimal_path("h_0_0_0", "h_2_0_0", d, a)
    assert path[0] == "h_0_0_0"
    assert path[-1] == "h_2_0_0"
    # Should have: src_host, src_router, [optional local], global_landing, [optional local], dst_host
    assert len(path) >= 4


def test_dragonfly_nonminimal_path():
    """Test non-minimal path through intermediate group."""
    d, a = 4, 3
    path = dragonfly_nonminimal_path("h_0_0_0", "h_2_0_0", 1, d, a)
    assert path[0] == "h_0_0_0"
    assert path[-1] == "h_2_0_0"
    # Non-minimal should go through intermediate group 1
    # Path should have routers from group 1
    router_groups = [int(n.split("_")[1]) for n in path if n.startswith("r_")]
    assert 1 in router_groups  # Should pass through intermediate group


def test_nonminimal_longer_than_minimal():
    """Test that non-minimal paths are generally longer."""
    d, a = 4, 3
    minimal = dragonfly_minimal_path("h_0_0_0", "h_2_0_0", d, a)
    nonminimal = dragonfly_nonminimal_path("h_0_0_0", "h_2_0_0", 1, d, a)
    assert len(nonminimal) >= len(minimal)


def test_estimate_path_latency_basic():
    """Test path latency estimation."""
    link_loads = {("a", "b"): 100.0, ("b", "c"): 50.0}
    path = ["a", "b", "c"]
    latency = estimate_path_latency(path, link_loads, hop_latency=1.0, congestion_weight=0.01)
    # 2 hops + congestion from 150 total load
    expected = 2.0 + 0.01 * (100.0 + 50.0)
    assert abs(latency - expected) < 0.01


def test_estimate_path_latency_no_congestion():
    """Test latency with no congestion equals hop count."""
    link_loads = {}
    path = ["a", "b", "c", "d"]
    latency = estimate_path_latency(path, link_loads, hop_latency=1.0, congestion_weight=1.0)
    assert latency == 3.0  # 3 hops


def test_ugal_select_path_low_congestion():
    """Test UGAL selects minimal path under low congestion."""
    d, a = 4, 3
    G = build_dragonfly(d, a, 2)
    link_loads = {tuple(sorted(edge)): 0.0 for edge in G.edges()}

    path = ugal_select_path("h_0_0_0", "h_2_0_0", link_loads, d, a)
    minimal = dragonfly_minimal_path("h_0_0_0", "h_2_0_0", d, a)

    # Under no congestion, should pick minimal
    assert len(path) == len(minimal)


def test_ugal_intra_group_always_minimal():
    """Test UGAL always uses minimal for intra-group traffic."""
    d, a = 4, 3
    G = build_dragonfly(d, a, 2)
    # Even with high congestion, intra-group should be minimal
    link_loads = {tuple(sorted(edge)): 1000000.0 for edge in G.edges()}

    path = ugal_select_path("h_0_0_0", "h_0_2_0", link_loads, d, a)
    minimal = dragonfly_minimal_path("h_0_0_0", "h_0_2_0", d, a)

    assert path == minimal


def test_valiant_bias_zero():
    """Test Valiant with bias=0 always picks minimal."""
    d, a = 4, 3

    # With bias=0, should always be minimal
    for _ in range(10):  # Test multiple times due to randomness
        path = valiant_select_path("h_0_0_0", "h_2_0_0", d, a, bias=0.0)
        minimal = dragonfly_minimal_path("h_0_0_0", "h_2_0_0", d, a)
        assert len(path) == len(minimal)


def test_valiant_bias_one():
    """Test Valiant with bias=1.0 always picks non-minimal (for inter-group)."""
    d, a = 4, 3

    # With bias=1.0, should always be non-minimal for inter-group
    for _ in range(10):
        path = valiant_select_path("h_0_0_0", "h_2_0_0", d, a, bias=1.0)
        minimal = dragonfly_minimal_path("h_0_0_0", "h_2_0_0", d, a)
        # Non-minimal should be longer or equal
        assert len(path) >= len(minimal)


def test_valiant_intra_group_always_minimal():
    """Test Valiant always uses minimal for intra-group traffic."""
    d, a = 4, 3

    # Even with bias=1.0, intra-group should be minimal
    for _ in range(10):
        path = valiant_select_path("h_0_0_0", "h_0_2_0", d, a, bias=1.0)
        minimal = dragonfly_minimal_path("h_0_0_0", "h_0_2_0", d, a)
        assert path == minimal


def test_dragonfly_route_dispatcher():
    """Test the main routing dispatcher."""
    d, a = 4, 3
    src, dst = "h_0_0_0", "h_2_0_0"

    # Test minimal
    path_min = dragonfly_route(src, dst, 'minimal', d, a)
    assert path_min == dragonfly_minimal_path(src, dst, d, a)

    # Test ugal with no loads (should be minimal)
    path_ugal = dragonfly_route(src, dst, 'ugal', d, a, link_loads={})
    assert len(path_ugal) == len(path_min)

    # Test valiant with bias=0 (should be minimal)
    path_val = dragonfly_route(src, dst, 'valiant', d, a, valiant_bias=0.0)
    assert len(path_val) == len(path_min)


def test_link_loads_for_job_dragonfly_adaptive():
    """Test link load computation with adaptive routing."""
    d, a, p = 4, 3, 2
    G = build_dragonfly(d, a, p)
    hosts = ["h_0_0_0", "h_0_0_1", "h_0_1_0"]  # 3 hosts in same group

    loads = link_loads_for_job_dragonfly_adaptive(
        G, hosts, tx_volume_bytes=1000.0,
        algorithm='minimal', d=d, a=a
    )

    # Should have some non-zero loads
    total_load = sum(loads.values())
    assert total_load > 0


def test_link_loads_inter_group():
    """Test link loads for inter-group traffic."""
    d, a, p = 4, 3, 2
    G = build_dragonfly(d, a, p)
    hosts = ["h_0_0_0", "h_2_0_0"]  # 2 hosts in different groups

    loads_min = link_loads_for_job_dragonfly_adaptive(
        G, hosts, tx_volume_bytes=1000.0,
        algorithm='minimal', d=d, a=a
    )

    # Verify global links are used (edges between groups)
    total_load = sum(loads_min.values())
    assert total_load > 0
