#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Dependency-Aware Goal Completion Rate (dGCR).

Only *decided* goals enter the metric: goals whose final status is COMPLETED
or FAILED. Goals that are still OPEN/PENDING (including goals blocked by unmet
prerequisites, which the lifecycle keeps non-terminal) and ABANDONED goals are
excluded.

    dGCR = |{g decided : status(g) = COMPLETED}| / |{g decided}|

The script accepts the released ATOD schema and system-reconstructed goal
lists with the same ``status`` field.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

DECIDED = {"COMPLETED", "FAILED"}


def compute_dgcr_for_dialogue(goals: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return ``(completed, decided)`` counts for one dialogue."""
    completed = 0
    decided = 0
    for goal in goals:
        status = str(goal.get("status", "")).upper()
        if status in DECIDED:
            decided += 1
            if status == "COMPLETED":
                completed += 1
    return completed, decided


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute dGCR from annotated dialogues")
    parser.add_argument("--complexity", choices=["medium", "complex", "all"], default="all")
    parser.add_argument("--base-dir", default=None, help="Directory containing medium/ and complex/")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / "data")
    complexities = ["medium", "complex"] if args.complexity == "all" else [args.complexity]

    summary: Dict[str, Any] = {"by_complexity": {}, "overall": {"completed": 0, "decided": 0, "dGCR": 0.0}}
    for comp in complexities:
        file_path = base_dir / comp / "annotated_dialogues.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping {comp}")
            continue
        completed = decided = 0
        dialogues = load_dialogues(file_path)
        for dialogue in dialogues:
            c, d = compute_dgcr_for_dialogue(dialogue.get("goal_list", []))
            completed += c
            decided += d
        summary["by_complexity"][comp] = {
            "completed": completed,
            "decided": decided,
            "dGCR": completed / decided if decided else 0.0,
            "dialogues": len(dialogues),
        }
        summary["overall"]["completed"] += completed
        summary["overall"]["decided"] += decided
    if summary["overall"]["decided"]:
        summary["overall"]["dGCR"] = summary["overall"]["completed"] / summary["overall"]["decided"]

    print("=== Dependency-Aware Goal Completion Rate (dGCR) ===")
    for comp, s in summary["by_complexity"].items():
        print(f"{comp.capitalize():8s} -> dGCR: {s['dGCR']:.3f} ({s['completed']}/{s['decided']} decided goals, {s['dialogues']} dialogues)")
    o = summary["overall"]
    print(f"Overall   -> dGCR: {o['dGCR']:.3f} ({o['completed']}/{o['decided']})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
