#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Post-hoc LLM quality rating of generated dialogues (paper Appendix, Table 11).

Each dialogue is rated 1--5 on coherence, fluency, consistency, relevance and
naturalness by a single model call. Averages are reported per complexity split.
By default the released benchmark under ``data/`` is rated; pass ``--data-dir``
to rate a freshly generated ``<work-dir>/final`` directory instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List

from generation.common import (
    REPO_ROOT,
    add_llm_args,
    first_json_object,
    load_json,
    make_llm_client,
    parallel_map,
    resolve_model_id_or_exit,
    save_json,
)

METRICS = ("Coherence", "Fluency", "Consistency", "Relevance", "Naturalness")

JUDGE_PROMPT = """Rate this task-oriented dialogue on each metric (1-5 scale, where 5 is excellent):

**EVALUATION CRITERIA:**
- **Coherence** (1-5): Is the conversation well-structured with logical flow between turns?
- **Fluency** (1-5): Is the language smooth, grammatical, and natural-sounding?
- **Consistency** (1-5): Are responses consistent with previous context and stated information?
- **Relevance** (1-5): Do utterances directly address the user goals and stay on-topic?
- **Naturalness** (1-5): Does the conversation feel realistic and human-like (not robotic)?

**USER GOALS:** {goals}

**DIALOGUE:**
{dialogue}

**INSTRUCTIONS:**
- Consider this is a task-oriented dialogue where users have specific goals
- Rate each metric independently
- Be objective and consistent in your scoring

Provide scores in JSON format only:
{{"Coherence": 4, "Fluency": 5, "Consistency": 4, "Relevance": 5, "Naturalness": 4}}"""


def build_prompt(dialogue: dict) -> str:
    goals = [
        (g.get("content") or f"{g.get('domain', '')} {g.get('intent', '')}").strip()
        for g in dialogue.get("goal_list", [])
    ]
    turns = [f"{t.get('speaker', '').upper()}: {t.get('utterance', '')}" for t in dialogue.get("turns", [])]
    return JUDGE_PROMPT.format(goals=" | ".join(g for g in goals if g), dialogue=" | ".join(turns))


def rate_dialogue(client, dialogue: dict) -> dict:
    try:
        scores = first_json_object(client.call(build_prompt(dialogue), max_tokens=1024, temperature=0.1))
        scores = {m: float(scores[m]) for m in METRICS if m in scores}
    except Exception as error:  # noqa: BLE001
        print(f"[rating] {dialogue.get('dialogue_id', 'unknown')}: {error}")
        scores = {}
    return {"dialogue_id": dialogue.get("dialogue_id"), "scores": scores}


def summarize(ratings: List[dict]) -> Dict[str, float]:
    summary = {}
    for metric in METRICS:
        values = [r["scores"][metric] for r in ratings if metric in r["scores"]]
        summary[metric] = round(mean(values), 2) if values else None
    summary["rated"] = sum(1 for r in ratings if r["scores"])
    summary["total"] = len(ratings)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Directory with medium/ and complex/")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "generation" / "work" / "quality_ratings.json")
    parser.add_argument("--max-dialogues", type=int, default=None, help="Rate at most N dialogues per split")
    add_llm_args(parser)
    args = parser.parse_args()
    model_id = resolve_model_id_or_exit(parser, args.model_id)
    client = make_llm_client(model_id)

    results = {}
    for complexity in ("medium", "complex"):
        path = args.data_dir / complexity / "annotated_dialogues.json"
        if not path.exists():
            print(f"Skipping {complexity}: {path} not found")
            continue
        dialogues = load_json(path)
        if args.max_dialogues:
            dialogues = dialogues[: args.max_dialogues]
        ratings = parallel_map(dialogues, lambda d: rate_dialogue(client, d), workers=args.workers, desc=f"Rating {complexity}")
        results[complexity] = {"summary": summarize(ratings), "ratings": ratings}
        print(f"{complexity}: {results[complexity]['summary']}")

    save_json(results, args.output)
    print(f"Saved ratings to {args.output}")


if __name__ == "__main__":
    main()
