#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Memory Recall Accuracy (MRA) - Ground Truth Based

Computes Memory Recall Accuracy by checking if earlier goals are completed by dialogue end.
Earlier goals are defined as goals mentioned in the first half of the dialogue.

This metric validates the system's ability to maintain consistent memory of goal states
across the dialogue progression.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    """Load dialogues from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def compute_memory_recall_accuracy(dialogue: Dict[str, Any]) -> float:
    """
    Compute Memory Recall Accuracy by checking if earlier goals are completed by dialogue end.
    Earlier goals are defined as goals mentioned in the first half of the dialogue.
    """
    goals = dialogue.get('goal_list', [])
    turns = dialogue.get('turns', [])

    if not goals or not turns:
        return 0.0

    total_turns = len(turns)
    early_threshold = total_turns // 2  # First half of dialogue

    # Find goals mentioned in first half
    early_goals = []
    for goal in goals:
        first_mentioned = goal.get('first_mentioned_turn', goal.get('initiation_turn'))
        if isinstance(first_mentioned, int) and first_mentioned <= early_threshold:
            early_goals.append(goal)

    if not early_goals:
        return 1.0  # No early goals to recall

    # Check how many early goals are completed by dialogue end
    completed_early_goals = sum(1 for g in early_goals
                               if g.get('status', '').lower() == 'completed')

    return completed_early_goals / len(early_goals)


def main():
    parser = argparse.ArgumentParser(description='Compute Memory Recall Accuracy from A-TOD annotated dialogues')
    parser.add_argument('--complexity', choices=['medium', 'complex', 'all'], default='all')
    parser.add_argument('--base-dir', default=None, help='Base directory containing ATOD data')
    parser.add_argument('--sample-size', type=int, default=None, help='Limit to first N dialogues per complexity')
    parser.add_argument('--output', default=None, help='Optional JSON output path')
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / 'data')
    complexities = ['medium', 'complex'] if args.complexity == 'all' else [args.complexity]

    summary = {
        'by_complexity': {},
        'overall': {'total_dialogues': 0, 'avg_mra': 0.0},
        'base_dir': str(base_dir)
    }

    all_scores = []

    for comp in complexities:
        file_path = base_dir / comp / 'annotated_dialogues.json'
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping {comp}")
            continue

        dialogues = load_dialogues(file_path)

        # Apply sampling if specified
        if args.sample_size and args.sample_size > 0:
            dialogues = dialogues[:args.sample_size]
            print(f"Sampling first {len(dialogues)} {comp} dialogues")
        else:
            print(f"Processing all {len(dialogues)} {comp} dialogues")

        comp_scores = []
        for dialogue in dialogues:
            try:
                score = compute_memory_recall_accuracy(dialogue)
                comp_scores.append(score)
                all_scores.append(score)
            except Exception as e:
                print(f"Error processing dialogue {dialogue.get('dialogue_id', 'unknown')}: {e}")
                continue

        if comp_scores:
            summary['by_complexity'][comp] = {
                'avg_mra': np.mean(comp_scores),
                'std_mra': np.std(comp_scores),
                'n_dialogues': len(comp_scores)
            }

    # Overall summary
    if all_scores:
        summary['overall'] = {
            'avg_mra': np.mean(all_scores),
            'std_mra': np.std(all_scores),
            'total_dialogues': len(all_scores)
        }

    # Print results
    print('=== Memory Recall Accuracy (MRA) ===')
    for comp in complexities:
        if comp in summary['by_complexity']:
            s = summary['by_complexity'][comp]
            print(f"{comp.capitalize():8s} -> MRA: {s['avg_mra']:.3f} ± {s['std_mra']:.3f} ({s['n_dialogues']} dialogues)")

    if summary['overall']['total_dialogues'] > 0:
        o = summary['overall']
        print(f"Overall   -> MRA: {o['avg_mra']:.3f} ± {o['std_mra']:.3f} ({o['total_dialogues']} dialogues)")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
