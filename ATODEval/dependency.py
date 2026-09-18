#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Dependency Handling Accuracy

Compute precision, recall, and F1 by comparing predicted vs. reference
dependency edges G = (V, E).

Dependencies may refer to goal IDs, as in the released ATOD schema, or to
goal-content strings, as in normalized evaluation outputs.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

Edge = Tuple[str, str]


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def build_edges(goals: List[Dict[str, Any]]) -> Set[Edge]:
    def content(goal: Dict[str, Any]) -> str:
        return str(
            goal.get("goal_content")
            or goal.get("content")
            or goal.get("core_content")
            or goal.get("id")
            or ""
        )

    by_reference: Dict[str, str] = {}
    for goal in goals:
        resolved_content = content(goal)
        for key in ("id", "goal_id", "goal_content", "content", "core_content"):
            value = goal.get(key)
            if value is not None:
                by_reference[str(value)] = resolved_content

    edges: Set[Edge] = set()
    for goal in goals:
        v = content(goal)
        for dependency in goal.get("dependencies") or []:
            if isinstance(dependency, dict):
                raw_reference = (
                    dependency.get("id")
                    or dependency.get("goal_id")
                    or dependency.get("goal_content")
                    or dependency.get("content")
                    or dependency.get("core_content")
                )
            else:
                raw_reference = dependency
            u = by_reference.get(str(raw_reference), str(raw_reference))
            if u != v:
                edges.add((u, v))
    return edges


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser(description='Dependency Handling Accuracy (Precision, Recall, F1)')
    parser.add_argument('--complexity', choices=['medium', 'complex', 'all'], default='all')
    parser.add_argument('--base-dir', default=None, help='Base directory containing evaluation data')
    parser.add_argument('--verbose', action='store_true', help='Show detailed debug output')
    parser.add_argument('--output', default=None, help='Optional JSON output path')
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / 'data')
    complexities = ['medium', 'complex'] if args.complexity == 'all' else [args.complexity]

    summary: Dict[str, Any] = {
        'by_complexity': {},
        'overall': {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        'base_dir': str(base_dir),
    }

    for comp in complexities:
        file_path = base_dir / comp / 'annotated_dialogues.json'
        dialogues = load_dialogues(file_path)
        tp = fp = fn = 0
        pred_edges_count = 0
        ref_edges_count = 0

        # Use tqdm for progress bar if not verbose
        if args.verbose:
            dialogue_iter = dialogues
        else:
            try:
                from tqdm import tqdm
                dialogue_iter = tqdm(dialogues, desc=f"Processing {comp}", leave=False)
            except ImportError:
                dialogue_iter = dialogues

        for dlg in dialogue_iter:
            ref_edges = build_edges(dlg['goal_list'])
            pred_edges = build_edges(dlg['predicted_goal_list'])
            tp_local = len(pred_edges & ref_edges)
            fp_local = len(pred_edges - ref_edges)
            fn_local = len(ref_edges - pred_edges)
            tp += tp_local
            fp += fp_local
            fn += fn_local
            pred_edges_count += len(pred_edges)
            ref_edges_count += len(ref_edges)
        p, r, f1 = prf(tp, fp, fn)
        summary['by_complexity'][comp] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': p,
            'recall': r,
            'f1': f1,
            'dialogues': len(dialogues),
            'pred_edges': pred_edges_count,
            'ref_edges': ref_edges_count,
        }
        summary['overall']['tp'] += tp
        summary['overall']['fp'] += fp
        summary['overall']['fn'] += fn

    o = summary['overall']
    o['precision'], o['recall'], o['f1'] = prf(o['tp'], o['fp'], o['fn'])

    print('=== Dependency Handling Accuracy (Edges) ===')
    for comp in complexities:
        s = summary['by_complexity'][comp]
        print(f"{comp.capitalize():8s} -> P: {s['precision']:.3f}  R: {s['recall']:.3f}  F1: {s['f1']:.3f}  (TP/FP/FN: {s['tp']}/{s['fp']}/{s['fn']}, Dialogues: {s['dialogues']})")
    print(f"Overall   -> P: {o['precision']:.3f}  R: {o['recall']:.3f}  F1: {o['f1']:.3f}  (TP/FP/FN: {o['tp']}/{o['fp']}/{o['fn']})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {out_path}")


if __name__ == '__main__':
    main()
