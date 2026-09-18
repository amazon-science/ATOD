#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Dialogue-level Quality - LLM-Judged

Evaluates overall dialogue quality using LLM-as-a-Judge.
Measures naturalness, coherence, logical progression, and overall effectiveness.

This metric validates the quality of complete task-oriented dialogues.
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


from llm_utils import evaluate_with_llm_judge


def compute_dialogue_level_quality(dialogue: Dict[str, Any], model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0", verbose: bool = False) -> float:
    """Compute overall dialogue quality using LLM-as-Judge."""
    turns = dialogue.get('turns', [])

    if not turns:
        return 0.0

    # Create dialogue text for LLM evaluation
    dialogue_text = "\n".join([
        f"{turn.get('speaker', 'UNKNOWN')}: {turn.get('utterance', '')}"
        for turn in turns
    ])

    # Create LLM judge prompt for dialogue quality
    prompt = f"""Rate this dialogue quality (0-10):

{dialogue_text}

Consider: naturalness, coherence, goal progression, effectiveness

Score (just the number 0-10):"""

    # Get LLM judgment
    return evaluate_with_llm_judge(prompt, model_id=model_id, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description='Compute Dialogue-level Quality from A-TOD annotated dialogues')
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
        'overall': {'total_dialogues': 0, 'avg_quality': 0.0},
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
        total_turns = 0

        for dialogue in dialogues:
            try:
                total_turns += len(dialogue.get('turns', []))

                score = compute_dialogue_level_quality(dialogue, model_id=args.model_id, verbose=args.verbose)
                comp_scores.append(score)
                all_scores.append(score)
            except Exception as e:
                print(f"Error processing dialogue {dialogue.get('dialogue_id', 'unknown')}: {e}")
                continue

        if comp_scores:
            summary['by_complexity'][comp] = {
                'avg_quality': np.mean(comp_scores),
                'std_quality': np.std(comp_scores),
                'n_dialogues': len(comp_scores),
                'total_turns': total_turns
            }

    # Overall summary
    if all_scores:
        summary['overall'] = {
            'avg_quality': np.mean(all_scores),
            'std_quality': np.std(all_scores),
            'total_dialogues': len(all_scores)
        }

    # Print results
    print('=== Dialogue-level Quality (LLM-Judged) ===')
    for comp in complexities:
        if comp in summary['by_complexity']:
            s = summary['by_complexity'][comp]
            print(f"{comp.capitalize():8s} -> Quality: {s['avg_quality']:.3f} ± {s['std_quality']:.3f} "
                  f"({s['n_dialogues']} dialogues, {s['total_turns']} turns)")

    if summary['overall']['total_dialogues'] > 0:
        o = summary['overall']
        print(f"Overall   -> Quality: {o['avg_quality']:.3f} ± {o['std_quality']:.3f} ({o['total_dialogues']} dialogues)")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
