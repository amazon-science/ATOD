#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Memory Recall Accuracy (MRA) -- LLM-judged.

At every turn in which a goal status changes, a retrieval query is built from
the affected goal description and the recent context using a fixed template.
An LLM judge compares the evaluated system's retrieved memory for that turn
against the ground-truth memory snapshot derived from the annotation and
returns ``match`` in {0, 1} based on sufficiency and semantic consistency.

    MRA = (1 / |Q|) * sum_{q_t in Q} match_t

The system output is supplied as a JSON file with the same structure as the
annotated dialogues (``dialogue_id`` and ``turns[].all_goals`` holding the
system's tracked goal state after each turn), for example the per-turn states
produced by the memory evaluator.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_utils import evaluate_json_with_llm, format_dialogue_context, goal_state_json  # noqa: E402


def build_query(goal: Dict[str, Any], recent_context: str) -> str:
    """Fixed retrieval-query template."""
    description = goal.get("content") or goal.get("core_content") or goal.get("id", "")
    return f"What is the current state of the goal \"{description}\" given the recent conversation?\n{recent_context}"


def mra_prompt(dialogue_context: str, memory_query: str, gold_memory_json: str, predicted_memory: str) -> str:
    return f"""You are evaluating whether a retrieved memory output correctly matches the benchmark memory state.

Dialogue Context: {dialogue_context}
Retrieval Query: {memory_query}
Ground-Truth Memory State: {gold_memory_json}
Retrieved Memory Output: {predicted_memory}

Judging Criteria:
- Retrieved output contains information required to answer the query
- Goal statuses, slot values, and historical facts are semantically consistent with gold state
- No contradiction is introduced

Output format (JSON): {{"match": 0 or 1}}"""


def status_change_queries(dialogue: Dict[str, Any], context_window: int = 4) -> List[Dict[str, Any]]:
    """One query per (turn, changed goal) in the gold annotation."""
    goals = {g["id"]: g for g in dialogue.get("goal_list", [])}
    turns = dialogue.get("turns", [])
    queries = []
    for idx, turn in enumerate(turns):
        for change in turn.get("goal_status_changes", []) or []:
            goal = goals.get(change.get("goal_id"))
            if goal is None:
                continue
            queries.append({
                "turn_index": idx,
                "goal_id": goal["id"],
                "query": build_query(goal, format_dialogue_context(turns, upto=idx + 1, window=context_window)),
            })
    return queries


def compute_memory_recall_accuracy(
    dialogue: Dict[str, Any],
    predicted: Dict[str, Any],
    model_id: Optional[str] = None,
    verbose: bool = False,
    context_window: int = 8,
) -> Optional[float]:
    """MRA for one dialogue given the system's per-turn goal states; None without queries."""
    queries = status_change_queries(dialogue)
    if not queries:
        return None
    gold_turns = dialogue.get("turns", [])
    pred_turns = predicted.get("turns", [])
    matches = []
    for q in queries:
        idx = q["turn_index"]
        gold_state = goal_state_json(gold_turns[idx], {q["goal_id"]})
        predicted_state = goal_state_json(pred_turns[idx]) if idx < len(pred_turns) else "[]"
        prompt = mra_prompt(format_dialogue_context(gold_turns, upto=idx + 1, window=context_window), q["query"], gold_state, predicted_state)
        decision = evaluate_json_with_llm(prompt, model_id=model_id, verbose=verbose)
        matches.append(1 if str(decision.get("match", 0)).strip() in {"1", "True", "true"} else 0)
    return mean(matches)


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute Memory Recall Accuracy (MRA) with an LLM judge")
    parser.add_argument("--complexity", choices=["medium", "complex", "all"], default="all")
    parser.add_argument("--base-dir", default=None, help="Directory containing gold medium/ and complex/")
    parser.add_argument("--predictions-dir", required=True, help="Directory with the system's medium/complex per-turn goal states (same file layout)")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model-id", default=None, help="Judge model ID (defaults to ATOD_MODEL_ID)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / "data")
    pred_dir = Path(args.predictions_dir)
    complexities = ["medium", "complex"] if args.complexity == "all" else [args.complexity]

    summary: Dict[str, Any] = {"by_complexity": {}, "overall": {}}
    all_scores: List[float] = []
    for comp in complexities:
        gold_path = base_dir / comp / "annotated_dialogues.json"
        pred_path = pred_dir / comp / "annotated_dialogues.json"
        if not gold_path.exists() or not pred_path.exists():
            print(f"Warning: missing {gold_path} or {pred_path}, skipping {comp}")
            continue
        predictions = {d["dialogue_id"]: d for d in load_dialogues(pred_path)}
        dialogues = load_dialogues(gold_path)
        if args.sample_size:
            dialogues = dialogues[: args.sample_size]
        scores = []
        for dialogue in dialogues:
            predicted = predictions.get(dialogue["dialogue_id"])
            if predicted is None:
                continue
            score = compute_memory_recall_accuracy(dialogue, predicted, model_id=args.model_id, verbose=args.verbose)
            if score is not None:
                scores.append(score)
        if scores:
            summary["by_complexity"][comp] = {"MRA": mean(scores), "std": pstdev(scores), "dialogues": len(scores)}
            all_scores.extend(scores)
    if all_scores:
        summary["overall"] = {"MRA": mean(all_scores), "std": pstdev(all_scores), "dialogues": len(all_scores)}

    print("=== Memory Recall Accuracy (MRA) ===")
    for comp, s in summary["by_complexity"].items():
        print(f"{comp.capitalize():8s} -> MRA: {s['MRA']:.3f} ± {s['std']:.3f} ({s['dialogues']} dialogues)")
    if summary["overall"]:
        print(f"Overall   -> MRA: {summary['overall']['MRA']:.3f} ({summary['overall']['dialogues']} dialogues)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
