#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Stage 2: build the goal co-occurrence graph.

Nodes are ``(domain, intent)`` goals; an undirected edge between two goals is
weighted by the number of seed sequences in which both goals occur.
The graph is stored as JSON (``cooccurrence_graph.json``) together with summary
statistics (``cooccurrence_graph_stats.json``).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from generation.common import add_work_dir_arg, load_json, require_file, save_json

Node = Tuple[str, str]


def node_id(node: Node) -> str:
    return f"{node[0]}_{node[1]}"


def build_graph(sequences: Iterable[List[Dict[str, str]]]) -> dict:
    """Return ``{"nodes": [...], "edges": [...]}`` in first-occurrence order."""
    node_counts: Counter = Counter()
    edge_counts: Dict[Tuple[Node, Node], int] = defaultdict(int)

    for sequence in sequences:
        goals = [(goal["domain"], goal["intent"]) for goal in sequence]
        for goal in goals:
            node_counts[goal] += 1
        for i, goal_a in enumerate(goals):
            for goal_b in goals[i + 1 :]:
                if goal_a == goal_b:
                    continue
                pair = tuple(sorted([goal_a, goal_b]))
                edge_counts[pair] += 1  # type: ignore[index]

    nodes = [
        {"id": node_id(node), "domain": node[0], "intent": node[1], "frequency": count}
        for node, count in node_counts.items()
    ]
    edges = [
        {"source": node_id(a), "target": node_id(b), "weight": weight}
        for (a, b), weight in edge_counts.items()
    ]
    return {"nodes": nodes, "edges": edges}


def adjacency(graph: dict) -> Dict[str, Dict[str, int]]:
    """Weighted adjacency lists keyed by node id (neighbor order = edge insertion order)."""
    adj: Dict[str, Dict[str, int]] = {node["id"]: {} for node in graph["nodes"]}
    for edge in graph["edges"]:
        adj[edge["source"]][edge["target"]] = edge["weight"]
        adj[edge["target"]][edge["source"]] = edge["weight"]
    return adj


def graph_statistics(graph: dict) -> dict:
    adj = adjacency(graph)
    n = len(adj)
    m = len(graph["edges"])
    degrees = [len(neighbors) for neighbors in adj.values()]

    # Connected components via BFS.
    seen: set = set()
    components: List[int] = []
    for start in adj:
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adj[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(size)

    return {
        "nodes": n,
        "edges": m,
        "density": round(2 * m / (n * (n - 1)), 4) if n > 1 else 0.0,
        "avg_degree": round(sum(degrees) / n, 2) if n else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "min_degree": min(degrees) if degrees else 0,
        "connected_components": len(components),
        "largest_component": max(components) if components else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_work_dir_arg(parser)
    args = parser.parse_args()

    sequences = load_json(
        require_file(args.work_dir / "extracted_goals.json", "Run generation/extract_goals.py first.")
    )
    graph = build_graph(sequences)
    if not graph["nodes"]:
        raise SystemExit("Built an empty graph.")

    save_json(graph, args.work_dir / "cooccurrence_graph.json")
    stats = graph_statistics(graph)
    save_json(stats, args.work_dir / "cooccurrence_graph_stats.json")
    print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
