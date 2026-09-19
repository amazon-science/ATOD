#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Turns to Completion (NTC).

NTC is the average number of turns between a goal's initiation
(``first_mentioned_turn``) and its decided terminal status (COMPLETED or
FAILED), computed over the same decided-goal set as dGCR. Abandoned and
unresolved goals are excluded.

The terminal turn is ``completion_turn`` for completed goals and the turn of
the FAILED entry in ``status_history`` for failed goals.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def terminal_turn(goal: Dict[str, Any]) -> Optional[int]:
    status = str(goal.get("status", "")).upper()
    if status == "COMPLETED":
        turn = goal.get("completion_turn")
        if isinstance(turn, int):
            return turn
    if status in {"COMPLETED", "FAILED"}:
        for entry in reversed(goal.get("status_history", []) or []):
            if str(entry.get("status", "")).upper() == status and isinstance(entry.get("turn"), int):
                return entry["turn"]
    return None


def compute_ntc_for_dialogue(goals: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return ``(sum_turns, decided_count)`` for one dialogue."""
    total = 0
    count = 0
    for goal in goals:
        init_turn = goal.get("first_mentioned_turn", goal.get("initiation_turn"))
        end_turn = terminal_turn(goal)
        if isinstance(init_turn, int) and isinstance(end_turn, int):
            total += max(0, end_turn - init_turn)
            count += 1
    return total, count


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute Turns to Completion (NTC) from annotated dialogues")
    parser.add_argument("--complexity", choices=["medium", "complex", "all"], default="all")
    parser.add_argument("--base-dir", default=None, help="Directory containing medium/ and complex/")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / "data")
    complexities = ["medium", "complex"] if args.complexity == "all" else [args.complexity]

    summary: Dict[str, Any] = {"by_complexity": {}, "overall": {"sum_turns": 0, "decided_goals": 0, "NTC": 0.0}}
    for comp in complexities:
        file_path = base_dir / comp / "annotated_dialogues.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping {comp}")
            continue
        total = count = 0
        dialogues = load_dialogues(file_path)
        for dialogue in dialogues:
            t, c = compute_ntc_for_dialogue(dialogue.get("goal_list", []))
            total += t
            count += c
        summary["by_complexity"][comp] = {
            "sum_turns": total,
            "decided_goals": count,
            "NTC": total / count if count else 0.0,
            "dialogues": len(dialogues),
        }
        summary["overall"]["sum_turns"] += total
        summary["overall"]["decided_goals"] += count
    if summary["overall"]["decided_goals"]:
        summary["overall"]["NTC"] = summary["overall"]["sum_turns"] / summary["overall"]["decided_goals"]

    print("=== Turns to Completion (NTC) ===")
    for comp, s in summary["by_complexity"].items():
        print(f"{comp.capitalize():8s} -> NTC: {s['NTC']:.3f} ({s['sum_turns']}/{s['decided_goals']} decided goals, {s['dialogues']} dialogues)")
    o = summary["overall"]
    print(f"Overall   -> NTC: {o['NTC']:.3f} ({o['sum_turns']}/{o['decided_goals']})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
