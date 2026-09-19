#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Turn-level Relevance -- LLM-judged response quality.

Every system response is scored on a 1--5 scale against the local context, the
current request and the tracked goal state (response-quality prompt, paper
appendix). Scores are averaged over system turns and reported as a fraction of
the maximum score (s / 5).
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_utils import evaluate_json_with_llm, format_dialogue_context, goal_state_json  # noqa: E402

MAX_SCORE = 5.0


def response_quality_prompt(dialogue_context: str, system_response: str, goal_state_or_full_dialogue: str) -> str:
    return f"""You are evaluating the response quality of a task-oriented dialogue system.

Dialogue Context: {dialogue_context}
Response Under Evaluation: {system_response}
Goal State / Dialogue Reference: {goal_state_or_full_dialogue}

Turn-Level Relevance: addresses current request; consistent with tracked goals; useful and natural
Dialogue-Level Coherence: globally consistent; interleaved goals progress coherently; no contradictions

Output format (JSON): {{"turn_quality": 1-5, "dialogue_quality": 1-5}}"""


def _score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(MAX_SCORE, max(1.0, score))


def compute_turn_level_quality(
    dialogue: Dict[str, Any],
    model_id: Optional[str] = None,
    verbose: bool = False,
    context_window: int = 8,
) -> Optional[float]:
    """Average turn_quality / 5 over system responses; None when no system turns."""
    turns = dialogue.get("turns", [])
    scores = []
    for idx, turn in enumerate(turns):
        if turn.get("speaker") != "SYSTEM":
            continue
        prompt = response_quality_prompt(
            format_dialogue_context(turns, upto=idx, window=context_window),
            turn.get("utterance", ""),
            goal_state_json(turn),
        )
        result = evaluate_json_with_llm(prompt, model_id=model_id, verbose=verbose)
        score = _score(result.get("turn_quality"))
        if score is not None:
            scores.append(score / MAX_SCORE)
    return mean(scores) if scores else None


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute turn-level relevance with an LLM judge")
    parser.add_argument("--complexity", choices=["medium", "complex", "all"], default="all")
    parser.add_argument("--base-dir", default=None, help="Directory containing medium/ and complex/")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model-id", default=None, help="Judge model ID (defaults to ATOD_MODEL_ID)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else (Path(__file__).resolve().parents[1] / "data")
    complexities = ["medium", "complex"] if args.complexity == "all" else [args.complexity]

    summary: Dict[str, Any] = {"by_complexity": {}, "overall": {}}
    all_scores: List[float] = []
    for comp in complexities:
        file_path = base_dir / comp / "annotated_dialogues.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping {comp}")
            continue
        dialogues = load_dialogues(file_path)
        if args.sample_size:
            dialogues = dialogues[: args.sample_size]
        scores = [s for s in (compute_turn_level_quality(d, model_id=args.model_id, verbose=args.verbose) for d in dialogues) if s is not None]
        if scores:
            summary["by_complexity"][comp] = {"turn_level_relevance": mean(scores), "std": pstdev(scores), "dialogues": len(scores)}
            all_scores.extend(scores)
    if all_scores:
        summary["overall"] = {"turn_level_relevance": mean(all_scores), "std": pstdev(all_scores), "dialogues": len(all_scores)}

    print("=== Turn-level Relevance (score / 5) ===")
    for comp, s in summary["by_complexity"].items():
        print(f"{comp.capitalize():8s} -> {s['turn_level_relevance']:.3f} ± {s['std']:.3f} ({s['dialogues']} dialogues)")
    if summary["overall"]:
        print(f"Overall   -> {summary['overall']['turn_level_relevance']:.3f} ({summary['overall']['dialogues']} dialogues)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
