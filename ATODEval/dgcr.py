#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Compute Dependency-Aware Goal Completion Rate (dGCR) from ATOD dialogues.

Definitions:
  - S(g): status of goal g
  - U: goals whose all prerequisites are completed
  - E: goals in U with decided outcomes (S(g) ∈ {COMPLETED, FAILED})
  - dGCR = |{ g ∈ E | S(g) = COMPLETED }| / |E|

The implementation accepts the released ATOD schema, where dependencies refer
to goal IDs, as well as the normalized analysis schema used by older scripts.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_dgcr_for_dialogue(goals: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return ``(completed_in_E, total_E)`` for one dialogue."""
    def references(goal: Dict[str, Any]) -> List[str]:
        values = []
        for key in ("id", "goal_id", "goal_content", "content", "core_content"):
            value = goal.get(key)
            if value is not None:
                values.append(str(value))
        return values

    by_reference: Dict[str, str] = {}
    for goal in goals:
        normalized_status = str(goal.get("status", "")).upper()
        for reference in references(goal):
            by_reference[reference] = normalized_status

    def dep_completed(dependency: Any) -> bool:
        if isinstance(dependency, dict):
            candidates = references(dependency)
        else:
            candidates = [str(dependency)]
        return any(by_reference.get(candidate) == "COMPLETED" for candidate in candidates)

    completed_in_E = 0
    total_E = 0

    for g in goals:
        deps = g.get("dependencies") or []
        # U: all prerequisites completed
        if all(dep_completed(d) for d in deps):
            status = str(g.get("status", "")).upper()
            # E: decided outcomes only
            if status in {'COMPLETED', 'FAILED'}:
                total_E += 1
                if status == 'COMPLETED':
                    completed_in_E += 1

    return completed_in_E, total_E


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data  # assume it's a list


def main():
    parser = argparse.ArgumentParser(description='Compute Dependency-Aware Goal Completion Rate (dGCR) from annotated dialogues')
    parser.add_argument('--complexity', choices=['medium', 'complex', 'all'], default='all')
    parser.add_argument('--base-dir', default=None, help='Base directory containing ATOD data')
    parser.add_argument('--output', default=None, help='Optional JSON output path')
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / 'data')
    complexities = ['medium', 'complex'] if args.complexity == 'all' else [args.complexity]

    summary: Dict[str, Any] = {
        'by_complexity': {},
        'overall': {'completed_in_E': 0, 'total_E': 0, 'dGCR': 0.0},
        'base_dir': str(base_dir),
    }

    for comp in complexities:
        file_path = base_dir / comp / 'annotated_dialogues.json'
        dialogues = load_dialogues(file_path)
        comp_completed = 0
        comp_total = 0
        for dlg in dialogues:
            goals = dlg['goal_list']  # must exist
            c, t = compute_dgcr_for_dialogue(goals)
            comp_completed += c
            comp_total += t
        dGCR = (comp_completed / comp_total) if comp_total > 0 else 0.0
        summary['by_complexity'][comp] = {
            'completed_in_E': comp_completed,
            'total_E': comp_total,
            'dGCR': dGCR,
            'dialogues': len(dialogues),
        }
        summary['overall']['completed_in_E'] += comp_completed
        summary['overall']['total_E'] += comp_total

    if summary['overall']['total_E'] > 0:
        summary['overall']['dGCR'] = summary['overall']['completed_in_E'] / summary['overall']['total_E']

    print('=== Dependency-Aware Goal Completion Rate (dGCR) ===')
    for comp in complexities:
        s = summary['by_complexity'][comp]
        print(f"{comp.capitalize():8s} -> dGCR: {s['dGCR']:.3f}  (Completed/Decided: {s['completed_in_E']}/{s['total_E']}, Dialogues: {s['dialogues']})")
    print(f"Overall   -> dGCR: {summary['overall']['dGCR']:.3f}  (Completed/Decided: {summary['overall']['completed_in_E']}/{summary['overall']['total_E']})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {out_path}")


if __name__ == '__main__':
    main()
