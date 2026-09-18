#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Compute unambiguous descriptive statistics for the released ATOD data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(dialogues: list[dict]) -> dict:
    utterances = [len(dialogue["turns"]) for dialogue in dialogues]
    exchanges = [
        sum(
            1
            for index in range(0, len(dialogue["turns"]) - 1, 2)
            if dialogue["turns"][index].get("speaker") == "USER"
            and dialogue["turns"][index + 1].get("speaker") == "SYSTEM"
        )
        for dialogue in dialogues
    ]
    goals = [len(dialogue["goal_list"]) for dialogue in dialogues]
    with_dependencies = sum(
        any(goal.get("dependencies") for goal in dialogue["goal_list"])
        for dialogue in dialogues
    )
    domains = {
        goal["domain"] for dialogue in dialogues for goal in dialogue["goal_list"]
    }
    intents = {
        goal["intent"] for dialogue in dialogues for goal in dialogue["goal_list"]
    }
    count = len(dialogues)
    return {
        "dialogues": count,
        "avg_utterance_records": round(sum(utterances) / count, 2),
        "avg_user_system_exchanges": round(sum(exchanges) / count, 2),
        "avg_goals": round(sum(goals) / count, 2),
        "dialogues_with_dependencies_pct": round(100 * with_dependencies / count, 2),
        "unique_domains": len(domains),
        "unique_intents": len(intents),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()

    combined = []
    output = {}
    for complexity in ("medium", "complex"):
        path = args.data_dir / complexity / "annotated_dialogues.json"
        dialogues = json.loads(path.read_text(encoding="utf-8"))
        output[complexity] = analyze(dialogues)
        combined.extend(dialogues)
    output["overall"] = analyze(combined)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
