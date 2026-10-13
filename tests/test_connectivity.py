"""Tests for the connectivity module.

Exercises:

- Baseline shortest-path connectivity on a small synthetic road graph.
- Scenario connectivity after removing a specific edge.
- Behaviour when a settlement is not reachable (already disconnected).
- Handling of an empty destination set.

The synthetic graph is defined in ``conftest.py`` as
``synthetic_road_graph``. Layout:

    A — B — C — D
            |
            E

All edges have a length of 100 m. Node coordinates are in projected
metres. This makes the graph small, deterministic, and easy to reason
about.
"""

from __future__ import annotations

import networkx as nx
import pytest

from georescue.analysis.connectivity import compute_connectivity


# ---------------------------------------------------------------------------
# Baseline connectivity
# ---------------------------------------------------------------------------

def test_baseline_all_nodes_connected(synthetic_road_graph: nx.MultiGraph):
    """Every node in the synthetic graph is reachable from any other."""
    settlements = {
        "A": (0.0, 0.0),
        "D": (300.0, 0.0),
        "E": (200.0, 100.0),
    }
    destinations = {
        "dest-D": (300.0, 0.0),
    }
    result = compute_connectivity(synthetic_road_graph, settlements, destinations)
    assert result["A"]["connected"]
    assert result["D"]["connected"]
    assert result["E"]["connected"]


def test_baseline_distance_is_correct(synthetic_road_graph: nx.MultiGraph):
    """A reaches D via three 100 m edges, so the distance should be 300 m."""
    settlements = {"A": (0.0, 0.0)}
    destinations = {"dest-D": (300.0, 0.0)}
    result = compute_connectivity(synthetic_road_graph, settlements, destinations)
    assert result["A"]["connected"]
    assert result["A"]["distance_m"] == pytest.approx(300.0, abs=0.01)


# ---------------------------------------------------------------------------
# Scenario connectivity (after removing an edge)
# ---------------------------------------------------------------------------

def test_scenario_disconnects_settlement(synthetic_road_graph: nx.MultiGraph):
    """Remove edge B-C. A can no longer reach D."""
    graph = synthetic_road_graph.copy()
    # Remove the edge between B (100, 0) and C (200, 0).
    graph.remove_edge((100.0, 0.0), (200.0, 0.0))

    settlements = {"A": (0.0, 0.0), "E": (200.0, 100.0)}
    destinations = {"dest-D": (300.0, 0.0)}
    result = compute_connectivity(graph, settlements, destinations)

    assert not result["A"]["connected"], "A should be disconnected"
    # E is on the same side as D after the removal
    assert result["E"]["connected"], "E should still reach D"


def test_scenario_settlement_already_disconnected(synthetic_road_graph: nx.MultiGraph):
    """A settlement that has no path at baseline is reported as not connected."""
    graph = synthetic_road_graph.copy()
    graph.add_node((9999.0, 9999.0))  # isolated node

    settlements = {"isolated": (9999.0, 9999.0)}
    destinations = {"dest-D": (300.0, 0.0)}
    result = compute_connectivity(graph, settlements, destinations)

    assert not result["isolated"]["connected"]
    assert result["isolated"]["distance_m"] == float("inf")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_destinations(synthetic_road_graph: nx.MultiGraph):
    """With no destinations, every settlement is reported as not connected."""
    settlements = {"A": (0.0, 0.0), "D": (300.0, 0.0)}
    destinations: dict[str, tuple[float, float]] = {}
    result = compute_connectivity(synthetic_road_graph, settlements, destinations)
    assert not result["A"]["connected"]
    assert not result["D"]["connected"]


def test_no_settlements(synthetic_road_graph: nx.MultiGraph):
    """With no settlements, the result dict is empty."""
    destinations = {"dest-D": (300.0, 0.0)}
    result = compute_connectivity(synthetic_road_graph, {}, destinations)
    assert result == {}


def test_result_has_required_fields(synthetic_road_graph: nx.MultiGraph):
    """Every result entry should have the documented keys."""
    settlements = {"A": (0.0, 0.0)}
    destinations = {"dest-D": (300.0, 0.0)}
    result = compute_connectivity(synthetic_road_graph, settlements, destinations)
    entry = result["A"]
    assert "connected" in entry
    assert "distance_m" in entry
    assert "destination_id" in entry
    assert "snapped_node" in entry