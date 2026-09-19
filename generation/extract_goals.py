#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Stage 1: extract goal sequences from the Schema-Guided Dialogue (SGD) dataset.

A goal is a unique ``(domain, intent)`` pair observed in the user frames of one
SGD dialogue. Each dialogue yields one ordered goal sequence (first-occurrence
order). Sequences with fewer than two goals are discarded.

The SGD dataset is not redistributed with ATOD. Download it from
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue and
point ``--dialogues-dir`` at its ``train`` directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from generation.common import add_work_dir_arg, save_json

# Seed files used for the released benchmark: SGD train dialogues_044 .. dialogues_127.
DEFAULT_FILE_RANGE = (44, 127)


def extract_goals_from_dialogue(dialogue: dict) -> List[Dict[str, str]]:
    """Return the ordered list of unique (domain, intent) goals in one SGD dialogue."""
    goals: List[Dict[str, str]] = []
    seen: set = set()
    for turn in dialogue.get("turns", []):
        if turn.get("speaker") != "USER":
            continue
        for frame in turn.get("frames", []):
            service = frame.get("service")
            if not service:
                continue
            state = frame.get("state", {})
            # "Hotels_1" -> "Hotels"
            domain = service.split("_")[0]
            intent = state.get("active_intent")
            if not intent:
                if state.get("requested_slots"):
                    intent = "find"
                elif state.get("slot_values"):
                    intent = "book"
            if not intent:
                continue
            key = (domain, intent)
            if key not in seen:
                seen.add(key)
                goals.append({"domain": domain, "intent": intent})
    return goals


def extract_goal_sequences(
    dialogues_dir: Path, file_range: Tuple[int, int] = DEFAULT_FILE_RANGE
) -> List[List[Dict[str, str]]]:
    """Extract goal sequences (>= 2 goals) from ``dialogues_XXX.json`` files."""
    import json

    sequences: List[List[Dict[str, str]]] = []
    start, end = file_range
    for number in range(start, end + 1):
        path = Path(dialogues_dir) / f"dialogues_{number:03d}.json"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for dialogue in data:
            goals = extract_goals_from_dialogue(dialogue)
            if len(goals) >= 2:
                sequences.append(goals)
    return sequences


def summarize(sequences: List[List[Dict[str, str]]]) -> dict:
    lengths = [len(seq) for seq in sequences]
    domains = Counter(goal["domain"] for seq in sequences for goal in seq)
    intents = Counter(goal["intent"] for seq in sequences for goal in seq)
    multi_domain = sum(1 for seq in sequences if len({g["domain"] for g in seq}) > 1)
    return {
        "total_sequences": len(sequences),
        "avg_sequence_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "unique_domains": len(domains),
        "unique_intents": len(intents),
        "multi_domain_sequences": multi_domain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dialogues-dir", type=Path, required=True, help="SGD train directory")
    parser.add_argument("--start-file", type=int, default=DEFAULT_FILE_RANGE[0])
    parser.add_argument("--end-file", type=int, default=DEFAULT_FILE_RANGE[1])
    add_work_dir_arg(parser)
    args = parser.parse_args()

    if not args.dialogues_dir.is_dir():
        parser.error(f"{args.dialogues_dir} is not a directory")

    sequences = extract_goal_sequences(args.dialogues_dir, (args.start_file, args.end_file))
    if not sequences:
        raise SystemExit("No goal sequences extracted; check --dialogues-dir and the file range.")

    output = args.work_dir / "extracted_goals.json"
    save_json(sequences, output)
    stats = summarize(sequences)
    save_json(stats, args.work_dir / "extracted_goals_stats.json")
    print(f"Saved {len(sequences)} goal sequences to {output}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
