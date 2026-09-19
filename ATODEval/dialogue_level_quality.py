#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Dialogue-level Coherence -- LLM-judged.

The full conversation is scored on its native 1--5 scale for global
consistency and coherent progression across interleaved goals, using the
response-quality prompt from the paper appendix (``dialogue_quality`` field).
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_utils import evaluate_json_with_llm, format_dialogue_context  # noqa: E402
from turn_level_quality import MAX_SCORE, _score, response_quality_prompt  # noqa: E402


def compute_dialogue_level_quality(
    dialogue: Dict[str, Any],
    model_id: Optional[str] = None,
    verbose: bool = False,
) -> Optional[float]:
    """dialogue_quality (1--5) for the full conversation; None when unparseable."""
    turns = dialogue.get("turns", [])
    if not turns:
        return None
    full_dialogue = format_dialogue_context(turns)
    final_system = next((t.get("utterance", "") for t in reversed(turns) if t.get("speaker") == "SYSTEM"), "")
    prompt = response_quality_prompt(full_dialogue, final_system, full_dialogue)
    result = evaluate_json_with_llm(prompt, model_id=model_id, verbose=verbose)
    return _score(result.get("dialogue_quality"))


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute dialogue-level coherence with an LLM judge")
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
        scores = [s for s in (compute_dialogue_level_quality(d, model_id=args.model_id, verbose=args.verbose) for d in dialogues) if s is not None]
        if scores:
            summary["by_complexity"][comp] = {"dialogue_level_coherence": mean(scores), "std": pstdev(scores), "dialogues": len(scores)}
            all_scores.extend(scores)
    if all_scores:
        summary["overall"] = {"dialogue_level_coherence": mean(all_scores), "std": pstdev(all_scores), "dialogues": len(all_scores)}

    print(f"=== Dialogue-level Coherence (1-{int(MAX_SCORE)}) ===")
    for comp, s in summary["by_complexity"].items():
        print(f"{comp.capitalize():8s} -> {s['dialogue_level_coherence']:.2f} ± {s['std']:.2f} ({s['dialogues']} dialogues)")
    if summary["overall"]:
        print(f"Overall   -> {summary['overall']['dialogue_level_coherence']:.2f} ({summary['overall']['dialogues']} dialogues)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
