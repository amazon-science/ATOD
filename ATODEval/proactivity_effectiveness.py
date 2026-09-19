#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Proactivity Effectiveness (PE) -- LLM-judged.

A system response is flagged as *proactive* when it acts on a goal that is
present in the tracked goal state but is not referenced in the current user
turn (e.g. a reminder about a pending booking). For each proactive response the
judge returns two binary decisions, ``grounded`` (supported by the dialogue and
memory state) and ``beneficial`` (advances an unresolved goal without
redundancy). A response receives credit only when both hold:

    b_t = grounded_t * beneficial_t,      PE = mean_t b_t
"""

import json
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_utils import evaluate_json_with_llm, format_dialogue_context, goal_state_json  # noqa: E402

STOPWORDS = {"a", "an", "the", "to", "for", "of", "in", "on", "at", "and", "or", "my", "me", "i", "you", "please", "with"}


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOPWORDS and len(w) > 2}


def find_proactive_turns(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return system turns that act on a tracked goal not referenced by the current user turn."""
    turns = dialogue.get("turns", [])
    goals = {g["id"]: g for g in dialogue.get("goal_list", [])}
    out: List[Dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        if turn.get("speaker") != "SYSTEM" or not turn.get("goal_status_changes"):
            continue
        user_turn = turns[idx - 1] if idx > 0 and turns[idx - 1].get("speaker") == "USER" else {}
        user_words = _content_words(user_turn.get("utterance", ""))
        previous_state = turns[idx - 1].get("all_goals", []) if idx > 0 else []
        tracked_before = {e.get("goal_id") for e in previous_state if e.get("status") not in (None, "not_mentioned")}

        proactive_goals = []
        for change in turn.get("goal_status_changes", []):
            gid = change.get("goal_id")
            if gid not in tracked_before:
                continue  # newly introduced goals are reactions to the user, not proactivity
            goal = goals.get(gid, {})
            goal_words = _content_words(goal.get("core_content", "")) | _content_words(goal.get("content", ""))
            if goal_words and not (goal_words & user_words):
                proactive_goals.append(gid)
        if proactive_goals:
            out.append({
                "turn_index": idx,
                "turn_id": turn.get("turn_id", idx + 1),
                "system_action": turn.get("utterance", ""),
                "goal_ids": proactive_goals,
            })
    return out


def proactivity_prompt(dialogue_context: str, system_action: str, goal_state: str) -> str:
    return f"""You are evaluating whether a system's proactive action is contextually grounded and genuinely helpful.

Dialogue Context: {dialogue_context}
Current System Action: {system_action}
Relevant Goal State: {goal_state}

Judging Criteria:
- The action is not explicitly requested by the user in the current turn
- The action is grounded in prior dialogue and memory state
- The action advances an unresolved or dependency-unlocked goal, or provides a timely reminder
- The action is not irrelevant, redundant, or distracting

Output format (JSON): {{"grounded": 0 or 1, "beneficial": 0 or 1}}"""


def _binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 1 else 0
    return 1 if str(value).strip().lower() in {"1", "true", "yes"} else 0


def compute_proactivity_effectiveness(
    dialogue: Dict[str, Any],
    model_id: Optional[str] = None,
    verbose: bool = False,
    context_window: int = 8,
) -> Optional[float]:
    """PE for one dialogue; None when the dialogue has no proactive responses."""
    proactive_turns = find_proactive_turns(dialogue)
    if not proactive_turns:
        return None
    turns = dialogue.get("turns", [])
    credits = []
    for info in proactive_turns:
        idx = info["turn_index"]
        prompt = proactivity_prompt(
            format_dialogue_context(turns, upto=idx, window=context_window),
            info["system_action"],
            goal_state_json(turns[idx - 1] if idx > 0 else {}, set(info["goal_ids"])),
        )
        decision = evaluate_json_with_llm(prompt, model_id=model_id, verbose=verbose)
        credits.append(_binary(decision.get("grounded", 0)) * _binary(decision.get("beneficial", 0)))
    return mean(credits)


def load_dialogues(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute Proactivity Effectiveness (PE) with an LLM judge")
    parser.add_argument("--complexity", choices=["medium", "complex", "all"], default="all")
    parser.add_argument("--base-dir", default=None, help="Directory containing medium/ and complex/")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit to first N dialogues per complexity")
    parser.add_argument("--model-id", default=None, help="Judge model ID (defaults to ATOD_MODEL_ID)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
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
        scores, proactive_total = [], 0
        for dialogue in dialogues:
            proactive_total += len(find_proactive_turns(dialogue))
            score = compute_proactivity_effectiveness(dialogue, model_id=args.model_id, verbose=args.verbose)
            if score is not None:
                scores.append(score)
        if scores:
            summary["by_complexity"][comp] = {
                "PE": mean(scores),
                "std": pstdev(scores),
                "dialogues_with_proactive_turns": len(scores),
                "proactive_turns": proactive_total,
            }
            all_scores.extend(scores)
    if all_scores:
        summary["overall"] = {"PE": mean(all_scores), "std": pstdev(all_scores), "dialogues": len(all_scores)}

    print("=== Proactivity Effectiveness (PE) ===")
    for comp, s in summary["by_complexity"].items():
        print(f"{comp.capitalize():8s} -> PE: {s['PE']:.3f} ± {s['std']:.3f} ({s['dialogues_with_proactive_turns']} dialogues, {s['proactive_turns']} proactive turns)")
    if summary["overall"]:
        print(f"Overall   -> PE: {summary['overall']['PE']:.3f} ({summary['overall']['dialogues']} dialogues)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
