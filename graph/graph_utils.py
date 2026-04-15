"""
Graph utility functions.

Keeps graph_builder.py focused on construction logic.
This module handles anything needed to inspect or validate a graph.
"""

from __future__ import annotations

import numpy as np

from graph.graph_builder import PendulumGraph, NODE_FEAT_DIM, EDGE_FEAT_DIM


def validate_graph(graph: PendulumGraph, n_links: int) -> list[str]:
    """
    Check a PendulumGraph for structural correctness.
    Returns a list of error strings; empty list means the graph is valid.
    """
    errors = []

    expected_nodes = n_links + 1
    expected_edges = 2 * n_links

    if graph.n_nodes != expected_nodes:
        errors.append(f"Expected {expected_nodes} nodes, got {graph.n_nodes}")

    if graph.n_edges != expected_edges:
        errors.append(f"Expected {expected_edges} edges, got {graph.n_edges}")

    if graph.node_features.shape != (graph.n_nodes, NODE_FEAT_DIM):
        errors.append(f"node_features shape {graph.node_features.shape} is wrong")

    if graph.edge_index.shape != (2, graph.n_edges):
        errors.append(f"edge_index shape {graph.edge_index.shape} is wrong")

    if graph.edge_features.shape != (graph.n_edges, EDGE_FEAT_DIM):
        errors.append(f"edge_features shape {graph.edge_features.shape} is wrong")

    # Every edge node index must be in-range.
    if graph.n_edges > 0:
        if graph.edge_index.min() < 0 or graph.edge_index.max() >= graph.n_nodes:
            errors.append("edge_index contains out-of-range node indices")

    # sin²+cos² ≈ 1 for each joint node.
    for i in range(1, graph.n_nodes):
        sin_val = graph.node_features[i, 3]
        cos_val = graph.node_features[i, 4]
        norm_sq = sin_val ** 2 + cos_val ** 2
        if not np.isclose(norm_sq, 1.0, atol=1e-5):
            errors.append(f"Node {i}: sin²+cos² = {norm_sq:.6f}, expected 1.0")

    # Each rod must appear as a forward+backward pair with matching features.
    for i in range(n_links):
        fwd = 2 * i
        bwd = 2 * i + 1
        if not np.allclose(graph.edge_features[fwd], graph.edge_features[bwd]):
            errors.append(f"Rod {i}: forward and backward edge features differ")

    return errors


def graph_summary(graph: PendulumGraph) -> str:
    """Human-readable summary of a graph for debugging."""
    lines = [
        f"PendulumGraph: {graph.n_nodes} nodes, {graph.n_edges} edges",
        "Node features [is_cart, is_joint, is_end, sin_θ, cos_θ, θ̇, x, ẋ]:",
    ]
    for i, row in enumerate(graph.node_features):
        lines.append(f"  node {i}: {np.round(row, 4)}")

    lines.append("Edges [from -> to | length, mass]:")
    for e in range(graph.n_edges):
        src = graph.edge_index[0, e]
        dst = graph.edge_index[1, e]
        feat = graph.edge_features[e]
        lines.append(f"  {src} -> {dst} | L={feat[0]:.3f}m, m={feat[1]:.3f}kg")

    return "\n".join(lines)
