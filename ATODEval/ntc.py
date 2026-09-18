#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Compute Turns to Completion (NTC) from A-TOD annotated dialogues.

For each completed goal g, NTC averages the number of turns between its
initiation and completion, reflecting execution efficiency.

The released ATOD schema uses ``first_mentioned_turn`` and
``completion_turn``. The older normalized name ``initiation_turn`` is also
accepted for reproducibility.
  - NTC per set = mean over (completion_turn - initiation_turn) for completed goals
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_ntc_for_dialogue(goals: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (sum_turns, completed_count) for one dialogue."""
    s = 0
    c = 0
    for g in goals:
        init_t = g.get("first_mentioned_turn", g.get("initiation_turn"))
        comp_t = g.get("completion_turn")
        status = str(g.get("status", "")).upper()
        if isinstance(init_t, int) and isinstance(comp_t, int) and status == "COMPLETED":
            # Count only completed goals
            diff = comp_t - init_t
            if diff < 0:
                diff = 0
            s += diff
            c += 1
    return s, c


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data  # assume list


def main():
    parser = argparse.ArgumentParser(description='Compute Turns to Completion (NTC) from annotated dialogues')
    parser.add_argument('--complexity', choices=['medium', 'complex', 'all'], default='all')
    parser.add_argument('--base-dir', default=None, help='Base directory containing ATOD data')
    parser.add_argument('--output', default=None, help='Optional JSON output path')
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / 'data')
    complexities = ['medium', 'complex'] if args.complexity == 'all' else [args.complexity]

    summary: Dict[str, Any] = {
        'by_complexity': {},
        'overall': {'sum_turns': 0, 'completed_goals': 0, 'NTC': 0.0},
        'base_dir': str(base_dir),
    }

    for comp in complexities:
        file_path = base_dir / comp / 'annotated_dialogues.json'
        dialogues = load_dialogues(file_path)
        comp_sum = 0
        comp_completed = 0
        for dlg in dialogues:
            goals = dlg['goal_list']
            s, c = compute_ntc_for_dialogue(goals)
            comp_sum += s
            comp_completed += c
        NTC = (comp_sum / comp_completed) if comp_completed > 0 else 0.0
        summary['by_complexity'][comp] = {
            'sum_turns': comp_sum,
            'completed_goals': comp_completed,
            'NTC': NTC,
            'dialogues': len(dialogues),
        }
        summary['overall']['sum_turns'] += comp_sum
        summary['overall']['completed_goals'] += comp_completed

    if summary['overall']['completed_goals'] > 0:
        summary['overall']['NTC'] = summary['overall']['sum_turns'] / summary['overall']['completed_goals']

    print('=== Turns to Completion (NTC) ===')
    for comp in complexities:
        s = summary['by_complexity'][comp]
        print(f"{comp.capitalize():8s} -> NTC: {s['NTC']:.3f}  (SumTurns/Completed: {s['sum_turns']}/{s['completed_goals']}, Dialogues: {s['dialogues']})")
    print(f"Overall   -> NTC: {summary['overall']['NTC']:.3f}  (SumTurns/Completed: {summary['overall']['sum_turns']}/{summary['overall']['completed_goals']})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {out_path}")


if __name__ == '__main__':
    main()
