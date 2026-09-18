#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Validate ATOD release structure, referential integrity, and sensitive patterns."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


EXPECTED_COUNTS = {"medium": 428, "complex": 572}
REQUIRED_DIALOGUE_FIELDS = {
    "complexity_class",
    "dialogue_id",
    "dialogue_index",
    "goal_list",
    "metadata",
    "turns",
}
ALLOWED_STATUSES = {
    "open",
    "pending",
    "completed",
    "failed",
    "abandoned",
    "not_mentioned",
}
BLOCKED_PATTERNS = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "internal_domain": re.compile(
        r"\b(?:a2z\.com|aws\.dev|git\.amazon|corp\.amazon|midway-auth)\b", re.I
    ),
}
INFORMATIONAL_PATTERNS = {
    "email_like_string": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I
    ),
    "url": re.compile(r"https?://\S+", re.I),
    "phone_like_string": re.compile(
        r"(?<!\d)(?:\(\d{3}\)|\d{3})[- .]\d{3}[- .]\d{4}(?!\d)"
    ),
    "account_reference": re.compile(
        r"\baccount(?:\s+number|\s+ending\s+in|\s*#)\s*:?\s*\d{4,}\b", re.I
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(path: Path, complexity: str) -> tuple[list[str], dict]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return [f"{path}: top-level value must be an array"], {}
    if len(data) != EXPECTED_COUNTS[complexity]:
        errors.append(
            f"{path}: expected {EXPECTED_COUNTS[complexity]} dialogues, found {len(data)}"
        )

    stats = {
        "dialogues": len(data),
        "utterances": 0,
        "exchanges": 0,
        "goals": 0,
        "statuses": collections.Counter(),
        "domains": set(),
        "intents": set(),
        "informational_matches": collections.Counter(),
        "dialogue_ids": [],
    }

    for index, dialogue in enumerate(data):
        label = f"{complexity}[{index}]"
        missing = REQUIRED_DIALOGUE_FIELDS - dialogue.keys()
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
            continue
        if dialogue["complexity_class"] != complexity:
            errors.append(
                f"{label}: complexity_class={dialogue['complexity_class']!r}"
            )

        dialogue_id = dialogue["dialogue_id"]
        stats["dialogue_ids"].append(dialogue_id)
        goals = dialogue["goal_list"]
        turns = dialogue["turns"]
        stats["goals"] += len(goals)
        stats["utterances"] += len(turns)
        stats["exchanges"] += sum(
            1
            for i in range(0, len(turns) - 1, 2)
            if turns[i].get("speaker") == "USER"
            and turns[i + 1].get("speaker") == "SYSTEM"
        )

        goal_ids = [str(goal.get("id")) for goal in goals]
        goal_id_set = set(goal_ids)
        if len(goal_ids) != len(goal_id_set):
            errors.append(f"{label}: duplicate goal IDs")

        for goal in goals:
            status = str(goal.get("status"))
            stats["statuses"][status] += 1
            if status not in ALLOWED_STATUSES:
                errors.append(f"{label}: unsupported status {status!r}")
            stats["domains"].add(str(goal.get("domain", "")))
            stats["intents"].add(str(goal.get("intent", "")))
            for dependency in goal.get("dependencies") or []:
                if str(dependency) not in goal_id_set:
                    errors.append(
                        f"{label}: dependency {dependency!r} does not reference a goal"
                    )

        expected_speaker = "USER"
        for turn_index, turn in enumerate(turns):
            if turn.get("speaker") != expected_speaker:
                errors.append(
                    f"{label}: turn {turn_index} expected {expected_speaker}, "
                    f"found {turn.get('speaker')!r}"
                )
            expected_speaker = "SYSTEM" if expected_speaker == "USER" else "USER"

        serialized = json.dumps(dialogue, ensure_ascii=False)
        for name, pattern in BLOCKED_PATTERNS.items():
            if pattern.search(serialized):
                errors.append(f"{label}: blocked sensitive pattern {name}")
        for name, pattern in INFORMATIONAL_PATTERNS.items():
            stats["informational_matches"][name] += len(pattern.findall(serialized))

    return errors, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    all_ids: list[str] = []
    for complexity in EXPECTED_COUNTS:
        path = args.data_dir / complexity / "annotated_dialogues.json"
        if not path.exists():
            all_errors.append(f"Missing {path}")
            continue
        errors, stats = validate_split(path, complexity)
        all_errors.extend(errors)
        all_ids.extend(stats.get("dialogue_ids", []))
        print(
            f"{complexity}: dialogues={stats.get('dialogues', 0)}, "
            f"utterances={stats.get('utterances', 0)}, "
            f"exchanges={stats.get('exchanges', 0)}, "
            f"goals={stats.get('goals', 0)}, sha256={sha256(path)}"
        )
        info = stats.get("informational_matches", {})
        if info:
            print(f"  informational synthetic strings: {dict(info)}")

    duplicates = len(all_ids) - len(set(all_ids))
    if duplicates:
        all_errors.append(f"Found {duplicates} duplicate dialogue IDs across splits")

    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for complexity in EXPECTED_COUNTS:
            path = args.data_dir / complexity / "annotated_dialogues.json"
            expected_hash = manifest["splits"][complexity]["sha256"]
            if path.exists() and sha256(path) != expected_hash:
                all_errors.append(f"{path}: checksum differs from manifest")

    if all_errors:
        print("\nValidation failed:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"\nValidation passed: {len(all_ids)} unique dialogues.")


if __name__ == "__main__":
    main()
