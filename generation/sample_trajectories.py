#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Stage 3: sample goal trajectories by random walks on the co-occurrence graph.

Each trajectory draws a target goal count from one of two ranges (2--8 goals,
"likely medium", or 7--12 goals, "likely complex"), then performs a weighted
random walk with an exploration factor. The complexity label itself is assigned
later (stage 4) from the annotated trajectory, so ``complexity_class`` is
``"unclassified"`` at this stage.
"""

from __future__ import annotations

import argparse
import random
import uuid
from collections import Counter
from typing import Dict, List, Optional, Tuple

from generation.build_cooccurrence_graph import adjacency
from generation.common import add_work_dir_arg, load_json, require_file, save_json

Node = Tuple[str, str]
INVALID_TOKENS = {"none", "null", "n/a", "unknown", ""}
GOAL_RANGES = {"medium": (2, 8), "complex": (7, 12)}


def is_valid_node(node: Node) -> bool:
    domain, intent = node
    if not domain or not intent:
        return False
    return (
        str(intent).strip().lower() not in INVALID_TOKENS
        and str(domain).strip().lower() not in INVALID_TOKENS
    )


class GoalTrajectorySampler:
    def __init__(self, graph: dict, rng: Optional[random.Random] = None, exploration_factor: float = 0.3):
        self.rng = rng or random.Random()
        self.exploration_factor = exploration_factor
        self.id_to_node: Dict[str, Node] = {n["id"]: (n["domain"], n["intent"]) for n in graph["nodes"]}
        self.adj = adjacency(graph)
        self.nodes: List[str] = [nid for nid, node in self.id_to_node.items() if is_valid_node(node)]
        if not self.nodes:
            raise ValueError("No valid nodes in graph")
        self.valid = set(self.nodes)

    # ------------------------------------------------------------------ #
    def sample_goal_count(self, distribution: Dict[str, float]) -> int:
        r = self.rng.random()
        cumulative = 0.0
        for range_type, prob in distribution.items():
            cumulative += prob
            if r <= cumulative:
                low, high = GOAL_RANGES[range_type]
                return self.rng.randint(low, high)
        return self.rng.randint(2, 12)

    def random_walk(self, target_length: int) -> List[Node]:
        if target_length <= 0:
            return []
        current = self.rng.choice(self.nodes)
        sequence = [current]
        for _ in range(target_length - 1):
            neighbors = [n for n in self.adj[current] if n in self.valid]
            if not neighbors:
                current = self.rng.choice(self.nodes)
            elif self.rng.random() < self.exploration_factor:
                # Exploration: random neighbor (70%) or any node (30%).
                current = self.rng.choice(neighbors) if self.rng.random() < 0.7 else self.rng.choice(self.nodes)
            else:
                # Exploitation: weighted by co-occurrence count.
                weights = [self.adj[current][n] for n in neighbors]
                total = sum(weights)
                if total > 0:
                    r = self.rng.uniform(0, total)
                    cumulative = 0
                    for neighbor, weight in zip(neighbors, weights):
                        cumulative += weight
                        if r <= cumulative:
                            current = neighbor
                            break
                else:
                    current = self.rng.choice(neighbors)
            if current != sequence[-1]:  # avoid immediate repeats
                sequence.append(current)
        return [self.id_to_node[nid] for nid in sequence]

    def sample_trajectory(self, num_goals: int) -> Optional[dict]:
        estimated_turns = num_goals * self.rng.randint(2, 6)  # rough initial estimate; refined in stage 4
        walk = self.random_walk(num_goals)
        if not walk:
            return None
        return {
            "dialogue_id": str(uuid.uuid4()),
            "complexity_class": "unclassified",
            "metadata": {"num_goals": len(walk), "estimated_turns": estimated_turns},
            "goal_list": [
                {
                    "id": f"goal_{index + 1}",
                    "domain": domain,
                    "intent": intent,
                    "slots": [],
                    "slot_values": {},
                    "dependencies": [],
                }
                for index, (domain, intent) in enumerate(walk)
            ],
        }

    def sample(self, num_trajectories: int, distribution: Dict[str, float]) -> List[dict]:
        trajectories = []
        for _ in range(num_trajectories):
            trajectory = self.sample_trajectory(self.sample_goal_count(distribution))
            if trajectory:
                trajectories.append(trajectory)
        return trajectories


def summarize(trajectories: List[dict]) -> dict:
    goals = [t["metadata"]["num_goals"] for t in trajectories]
    domains = Counter(g["domain"] for t in trajectories for g in t["goal_list"])
    intents = Counter(g["intent"] for t in trajectories for g in t["goal_list"])
    return {
        "trajectories": len(trajectories),
        "avg_goals": round(sum(goals) / len(goals), 2) if goals else 0,
        "min_goals": min(goals) if goals else 0,
        "max_goals": max(goals) if goals else 0,
        "unique_domains": len(domains),
        "unique_intents": len(intents),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--medium-ratio", type=float, default=0.65, help="Share of 2-8 goal trajectories")
    parser.add_argument("--complex-ratio", type=float, default=0.35, help="Share of 7-12 goal trajectories")
    parser.add_argument("--seed", type=int, default=42)
    add_work_dir_arg(parser)
    args = parser.parse_args()

    total = args.medium_ratio + args.complex_ratio
    distribution = {"medium": args.medium_ratio / total, "complex": args.complex_ratio / total}

    graph = load_json(
        require_file(args.work_dir / "cooccurrence_graph.json", "Run generation/build_cooccurrence_graph.py first.")
    )
    sampler = GoalTrajectorySampler(graph, random.Random(args.seed))
    trajectories = sampler.sample(args.num_trajectories, distribution)
    if not trajectories:
        raise SystemExit("No trajectories sampled.")

    output = args.work_dir / "sampled_goal_trajectories.json"
    save_json(trajectories, output)
    print(f"Saved {len(trajectories)} trajectories to {output}")
    for key, value in summarize(trajectories).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
