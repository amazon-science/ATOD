#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Proactivity Effectiveness - LLM-Judged

Evaluates the effectiveness of proactive system actions using LLM-as-a-Judge.
Measures whether proactive turns are contextually appropriate and advance user goals.

This metric validates the system's ability to take helpful initiative in task completion.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    """Load dialogues from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def find_proactive_turns(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find proactive system turns that include goal status changes."""
    turns = dialogue.get('turns', [])
    out: List[Dict[str, Any]] = []
    for idx, t in enumerate(turns):
        if t.get('speaker') == 'SYSTEM' and t.get('goal_status_changes'):
            prev_user = ''
            if idx - 1 >= 0 and turns[idx - 1].get('speaker') == 'USER':
                prev_user = turns[idx - 1].get('utterance', '')
            out.append({
                'dialogue_id': dialogue.get('dialogue_id', 'unknown'),
                'turn_index': idx,  # 0-based
                'user_before': prev_user,
                'system_utt': t.get('utterance', ''),
                'status_changes': t.get('goal_status_changes', []),
            })
    return out


from llm_utils import evaluate_yes_no_with_llm


def compute_proactivity_effectiveness(dialogue: Dict[str, Any], model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0", verbose: bool = False) -> float:
    """Compute proactivity effectiveness using LLM-as-Judge."""
    proactive_turns = find_proactive_turns(dialogue)

    if not proactive_turns:
        return 0.0

    effective_count = 0

    for turn_info in proactive_turns:
        user_before = turn_info.get('user_before', '')
        system_utt = turn_info.get('system_utt', '')
        status_changes = turn_info.get('status_changes', [])

        # Create LLM judge prompt for proactivity effectiveness
        prompt = f"""Evaluate this system's proactive action:

User said: "{user_before}"
System responded: "{system_utt}"
Goal changes: {status_changes}

Is this system response helpful and appropriate?
- Does it advance the user's goals?
- Is it timely and relevant?

Answer with only: YES or NO"""

        # Get LLM judgment
        is_effective = evaluate_yes_no_with_llm(prompt, model_id=model_id, verbose=verbose)
        if is_effective:
            effective_count += 1

    return effective_count / len(proactive_turns)


def main():
    parser = argparse.ArgumentParser(description='Compute Proactivity Effectiveness from A-TOD annotated dialogues')
    parser.add_argument('--complexity', choices=['medium', 'complex', 'all'], default='all')
    parser.add_argument('--base-dir', default=None, help='Base directory containing ATOD data')
    parser.add_argument('--sample-size', type=int, default=None, help='Limit to first N dialogues per complexity')
    parser.add_argument('--model-id', default='us.anthropic.claude-sonnet-4-20250514-v1:0', help='LLM model ID for evaluation')
    parser.add_argument('--verbose', action='store_true', help='Show detailed debug output including LLM responses')
    parser.add_argument('--output', default=None, help='Optional JSON output path')
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / 'data')
    complexities = ['medium', 'complex'] if args.complexity == 'all' else [args.complexity]

    summary = {
        'by_complexity': {},
        'overall': {'total_dialogues': 0, 'avg_effectiveness': 0.0},
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
        total_proactive_turns = 0

        for dialogue in dialogues:
            try:
                proactive_turns = find_proactive_turns(dialogue)
                total_proactive_turns += len(proactive_turns)

                score = compute_proactivity_effectiveness(dialogue, model_id=args.model_id, verbose=args.verbose)
                comp_scores.append(score)
                all_scores.append(score)
            except Exception as e:
                print(f"Error processing dialogue {dialogue.get('dialogue_id', 'unknown')}: {e}")
                continue

        if comp_scores:
            summary['by_complexity'][comp] = {
                'avg_effectiveness': np.mean(comp_scores),
                'std_effectiveness': np.std(comp_scores),
                'n_dialogues': len(comp_scores),
                'total_proactive_turns': total_proactive_turns
            }

    # Overall summary
    if all_scores:
        summary['overall'] = {
            'avg_effectiveness': np.mean(all_scores),
            'std_effectiveness': np.std(all_scores),
            'total_dialogues': len(all_scores)
        }

    # Print results
    print('=== Proactivity Effectiveness (LLM-Judged) ===')
    for comp in complexities:
        if comp in summary['by_complexity']:
            s = summary['by_complexity'][comp]
            print(f"{comp.capitalize():8s} -> Effectiveness: {s['avg_effectiveness']:.3f} ± {s['std_effectiveness']:.3f} "
                  f"({s['n_dialogues']} dialogues, {s['total_proactive_turns']} proactive turns)")

    if summary['overall']['total_dialogues'] > 0:
        o = summary['overall']
        print(f"Overall   -> Effectiveness: {o['avg_effectiveness']:.3f} ± {o['std_effectiveness']:.3f} ({o['total_dialogues']} dialogues)")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
