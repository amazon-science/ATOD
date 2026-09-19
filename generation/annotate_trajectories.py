#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Stage 4: annotate sampled trajectories and assign complexity labels.

For every trajectory the pipeline
1. asks the model for slots, realistic slot values and a goal description
   for each ``(domain, intent)`` goal;
2. refines the turn estimate;
3. assigns the complexity label (``medium`` / ``complex``) with the rule-based
   scorer, falling back to a model-based classifier when the rules return no
   label, and sets the agentic attribute flags;
4. asks the model for inter-goal dependencies (trajectories with > 3 goals);
5. validates the annotation (no placeholder values) and, unless
   ``--disable-judge`` is given, runs the trajectory verifier prompt.

Trajectories that fail validation or the verifier are retried; the stage
succeeds when at least ``--quality-threshold`` of the input trajectories pass.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from generation.common import (
    add_llm_args,
    add_work_dir_arg,
    first_json_object,
    is_placeholder_value,
    load_json,
    make_llm_client,
    one_word_verdict,
    parallel_map,
    require_file,
    resolve_model_id_or_exit,
    save_json,
)

QUANTITATIVE_THRESHOLDS = {
    "medium": {"max_goals": 8, "max_domains": 3, "max_estimated_turns": 35, "max_dependencies": 2},
    "complex": {"min_goals": 7, "min_domains": 3, "min_estimated_turns": 30, "min_dependencies": 2},
}


# --------------------------------------------------------------------------- #
# Prompts (reproduced as used for the released benchmark)
# --------------------------------------------------------------------------- #
def slot_annotation_prompt(domain: str, intent: str) -> str:
    return f"""Annotate a goal for task-oriented dialogue.

Domain: {domain}
Intent: {intent}

Generate slots, realistic values, and descriptions. Respond in JSON:
{{
  "slots": ["slot1", "slot2"],
  "slot_values": {{"slot1": "realistic_value1", "slot2": "realistic_value2"}},
  "content": "detailed goal description with specific values",
  "core_content": "stable goal essence"
}}

Requirements:
- 2-5 relevant slots for this domain/intent
- REALISTIC values only (e.g., "Seattle downtown", "March 15, 2025", "7:30 PM")
- NO placeholders like [location] or [date]
- Content: detailed description with slot values
- Core content: stable essence (e.g., "book flight", "find restaurant")

Examples:
- Locations: "downtown Seattle", "JFK Airport", "Times Square"
- Dates: "March 15, 2025", "next Friday", "tomorrow"
- Times: "7:30 PM", "2:00 PM", "noon"
- Names: "Marriott Hotel", "Southwest Airlines", "The Cheesecake Factory"
"""


def dependency_prompt(goals: List[dict]) -> str:
    lines = []
    for index, goal in enumerate(goals):
        slots_preview = ", ".join(f"{k}={v}" for k, v in goal.get("slot_values", {}).items())
        content_preview = (goal.get("content") or "")[:140]
        lines.append(
            f"{index + 1}. intent={goal['intent']} domain={goal['domain']} "
            f'slots=[{slots_preview}] content="{content_preview}"'
        )
    return f"""Analyze these goals and determine logical dependencies:

{chr(10).join(lines)}

Rules:
- Temporal order (e.g., "find destination" -> "book hotel")
- Prerequisites (e.g., "get weather" -> "plan outdoor activity")
- Information flow (e.g., location/date from one goal used in another)
- Only include clear, necessary dependencies.

Respond in JSON format:
{{
  "dependencies": [
    {{"goal": 2, "depends_on": [1]}},
    {{"goal": 4, "depends_on": [2, 3]}}
  ]
}}

If no dependencies exist, return "dependencies": [].
"""


def complexity_prompt(trajectory: dict) -> str:
    goals = trajectory.get("goal_list", [])
    metadata = trajectory.get("metadata", {})
    analysis = []
    for index, goal in enumerate(goals, 1):
        info = f"Goal {index}: {goal.get('domain')} - {goal.get('intent')}"
        if goal.get("slot_values"):
            info += " (" + ", ".join(f"{k}: {v}" for k, v in goal["slot_values"].items()) + ")"
        if goal.get("dependencies"):
            info += f" [depends on: {', '.join(goal['dependencies'])}]"
        analysis.append(info)
    domains = sorted({g.get("domain") for g in goals if g.get("domain")})
    return f"""Classify this goal trajectory complexity based on interaction patterns and coordination needs:

TRAJECTORY ANALYSIS:
- Goals: {len(goals)} total
- Estimated turns: {metadata.get('estimated_turns', 'unknown')}
- Domains: {', '.join(domains)} ({len(domains)} unique)

GOALS:
{chr(10).join(analysis)}

CLASSIFICATION CRITERIA (2-category system):
- MEDIUM: Manageable goal coordination, moderate domain interaction, natural conversation flow with some switching
- COMPLEX: Strong interdependencies, multi-domain coordination requiring sophisticated management, proactive system behavior

Consider:
1. Goal interdependencies and coordination complexity (≤2 deps = medium, ≥3 deps = complex)
2. Domain interaction patterns (2-3 domains = medium, 4+ domains = complex)
3. Required system intelligence (proactive suggestions, failure handling needed = complex)
4. Natural conversation flow complexity (simple switching = medium, complex orchestration = complex)

Respond with exactly one word: MEDIUM or COMPLEX"""


def trajectory_verifier_prompt(trajectory: dict) -> str:
    goals = trajectory.get("goal_list", [])
    lines = []
    for index, goal in enumerate(goals, 1):
        parts = [f"Goal {index}: {goal.get('domain')} - {goal.get('intent')}", f"  Content: {goal.get('content')}"]
        if goal.get("slot_values"):
            parts.append("  Slots: " + ", ".join(f"{k}: {v}" for k, v in goal["slot_values"].items()))
        lines.append("\n".join(parts))
    return f"""You are a quality judge for annotated goal trajectories.

TRAJECTORY ({len(goals)} goals, {trajectory.get('complexity_class', 'unknown')} complexity):
{chr(10).join(lines)}

Is this trajectory ready for dialogue generation?
Check:
- Goal descriptions are clear and specific (not generic/placeholder)
- Slot values are realistic (real locations/dates/names, not [location]/[date])
- All required fields present (domain, intent, content, core_content)
- Annotations are logically consistent

Respond with exactly one word: PASS or FAIL"""


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested offline)
# --------------------------------------------------------------------------- #
def parse_slot_response(response: str) -> dict:
    data = first_json_object(response)
    slots = data.get("slots", [])
    slot_values = data.get("slot_values", {})
    if not isinstance(slots, list) or not isinstance(slot_values, dict):
        raise ValueError("Invalid slot annotation structure")
    validated = {k: v for k, v in slot_values.items() if k in slots and not is_placeholder_value(str(v))}
    return {
        "slots": [slot for slot in slots if slot in validated][:5],
        "slot_values": validated,
        "content": (data.get("content") or "").strip(),
        "core_content": (data.get("core_content") or "").strip(),
    }


def parse_dependency_response(response: str) -> List[dict]:
    """Extract ``[{"goal": i, "depends_on": [j, ...]}, ...]`` from a model response."""
    patterns = [
        r'\{[^{}]*"dependencies"[^{}]*\[[^\]]*\][^{}]*\}',
        r'\{.*?"dependencies".*?\[.*?\].*?\}',
        r"\{.*\}",
    ]
    for pattern in patterns:
        for candidate in re.findall(pattern, response, re.DOTALL):
            cleaned = re.sub(r"\s+", " ", candidate.strip()).replace("'", '"')
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and isinstance(data.get("dependencies"), list):
                return data["dependencies"]
    # Loose fallback: "goal 2 depends on 1, 3"
    dependencies = []
    for pattern in (
        r'goal["\s:]*(\d+)["\s,]*depends[_\s]*on["\s:\[]*(\d+(?:\s*,\s*\d+)*)',
        r'"goal"["\s:]*(\d+)["\s,]*"depends_on"["\s:\[]*(\d+(?:\s*,\s*\d+)*)',
    ):
        for goal_num, depends in re.findall(pattern, response, re.IGNORECASE):
            targets = [int(x) for x in re.findall(r"\d+", depends)]
            if targets:
                dependencies.append({"goal": int(goal_num), "depends_on": targets})
    return dependencies


def apply_dependencies(goals: List[dict], dependencies: List[dict]) -> None:
    """Translate 1-based goal indices into goal-id references (in place)."""
    for goal in goals:
        goal.setdefault("dependencies", [])
    for dep in dependencies:
        goal_index = int(dep.get("goal", 0)) - 1
        if not 0 <= goal_index < len(goals):
            continue
        for target in dep.get("depends_on", []):
            if 1 <= int(target) <= len(goals):
                target_id = goals[int(target) - 1]["id"]
                if target_id not in goals[goal_index]["dependencies"]:
                    goals[goal_index]["dependencies"].append(target_id)
    for goal in goals:
        goal["dependencies"] = list(dict.fromkeys(goal["dependencies"]))


def refine_turn_estimate(trajectory: dict) -> None:
    """Set ``metadata.estimated_turns`` used by the complexity scorer and the generator.

    The label is not yet assigned when this runs, so the estimate is three turns
    per goal (the value carried by the released benchmark metadata).
    """
    num_goals = len(trajectory.get("goal_list", []))
    trajectory.setdefault("metadata", {})["estimated_turns"] = int(num_goals * 3.0)


def rule_based_complexity(trajectory: dict) -> Optional[str]:
    """Score quantitative attributes; returns ``"complex"`` or ``"medium"``.

    Note: the scorer always yields a label, so the model-based classifier in
    :func:`TrajectoryAnnotator.classify` acts only as a safeguard.
    """
    goals = trajectory.get("goal_list", [])
    metadata = trajectory.get("metadata", {})
    num_goals = len(goals)
    num_domains = len({g.get("domain") for g in goals if g.get("domain")})
    estimated_turns = metadata.get("estimated_turns", 20)
    total_deps = sum(len(g.get("dependencies", [])) for g in goals)
    t_complex = QUANTITATIVE_THRESHOLDS["complex"]
    t_medium = QUANTITATIVE_THRESHOLDS["medium"]

    score = 0
    if num_goals >= 10:
        score += 3
    elif num_goals >= t_complex["min_goals"]:
        score += 2
    elif num_goals > t_medium["max_goals"]:
        score += 1
    if num_domains >= 5:
        score += 2
    elif num_domains >= t_complex["min_domains"]:
        score += 1
    if estimated_turns >= 40:
        score += 2
    elif estimated_turns >= t_complex["min_estimated_turns"]:
        score += 1
    if total_deps >= 4:
        score += 3
    elif total_deps >= t_complex["min_dependencies"]:
        score += 2
    return "complex" if score >= 2 else "medium"


def set_agentic_flags(trajectory: dict, complexity: str) -> None:
    metadata = trajectory.setdefault("metadata", {})
    if complexity == "medium":
        metadata.update({"async_execution": True, "interleaving": True, "proactivity": False})
        trajectory["dependency_label"] = False
        trajectory["defectiveness_label"] = False
    else:
        metadata.update({"async_execution": True, "interleaving": True, "proactivity": True})
        trajectory["dependency_label"] = True
        trajectory["defectiveness_label"] = True


def validate_annotation(trajectory: dict) -> bool:
    goals = trajectory.get("goal_list", [])
    if not goals:
        return False
    for goal in goals:
        if any(not goal.get(field) for field in ("domain", "intent", "content", "core_content")):
            return False
        if any(is_placeholder_value(str(v)) for v in goal.get("slot_values", {}).values()):
            return False
    return True


def sanitize_trajectory(trajectory: dict) -> Optional[dict]:
    clean = dict(trajectory)
    goals = [g for g in clean.get("goal_list", []) or [] if g.get("domain") and g.get("intent")]
    if not goals:
        return None
    for index, goal in enumerate(goals, 1):
        goal.setdefault("id", f"goal_{index}")
        goal.setdefault("dependencies", [])
    clean["goal_list"] = goals
    clean["metadata"] = {**clean.get("metadata", {}), "num_goals": len(goals)}
    return clean


# --------------------------------------------------------------------------- #
# LLM-backed annotator
# --------------------------------------------------------------------------- #
class TrajectoryAnnotator:
    def __init__(self, client, enable_judge: bool = True, retries_per_item: int = 2):
        self.client = client
        self.enable_judge = enable_judge
        self.retries_per_item = retries_per_item

    # -- model calls --------------------------------------------------------
    def annotate_goal(self, goal: dict) -> None:
        domain, intent = goal["domain"], goal["intent"]
        try:
            data = parse_slot_response(self.client.call(slot_annotation_prompt(domain, intent), max_tokens=3000, temperature=0.3))
            goal["slots"] = data["slots"]
            goal["slot_values"] = data["slot_values"]
            goal["content"] = data["content"] or _content_from_slots(goal)
            goal["core_content"] = data["core_content"] or f"{intent.lower()} {domain.lower().rstrip('s')}"
        except Exception as error:  # noqa: BLE001 - keep the pipeline running
            print(f"[slots] {domain}.{intent}: {error}")
            goal.setdefault("slots", [])
            goal.setdefault("slot_values", {})
            goal["core_content"] = goal.get("core_content") or f"{intent.lower()} {domain.lower().rstrip('s')}"
            goal["content"] = goal.get("content") or _content_from_slots(goal)

    def generate_dependencies(self, trajectory: dict) -> None:
        goals = trajectory.get("goal_list", [])
        for goal in goals:
            goal.setdefault("dependencies", [])
        if len(goals) <= 3:
            return
        try:
            response = self.client.call(dependency_prompt(goals), max_tokens=1500, temperature=0.1)
            apply_dependencies(goals, parse_dependency_response(response))
        except Exception as error:  # noqa: BLE001
            print(f"[dependencies] {trajectory.get('dialogue_id', 'unknown')}: {error}")

    def model_based_complexity(self, trajectory: dict) -> str:
        try:
            response = self.client.call(complexity_prompt(trajectory), max_tokens=64, temperature=0.1)
            return "complex" if "COMPLEX" in response.upper() else "medium"
        except Exception as error:  # noqa: BLE001
            print(f"[complexity] {trajectory.get('dialogue_id', 'unknown')}: {error}")
            return "medium"

    def classify(self, trajectory: dict) -> str:
        label = rule_based_complexity(trajectory)
        method = "pre_defined"
        if not label:
            label = self.model_based_complexity(trajectory)
            method = "model_based"
        trajectory["complexity_class"] = label
        trajectory["classification_method"] = method
        set_agentic_flags(trajectory, label)
        return label

    def verify(self, trajectory: dict) -> bool:
        try:
            return one_word_verdict(self.client.call(trajectory_verifier_prompt(trajectory), max_tokens=16, temperature=0.1))
        except Exception as error:  # noqa: BLE001
            print(f"[verifier] {trajectory.get('dialogue_id', 'unknown')}: {error}")
            return False

    # -- per-trajectory flow -------------------------------------------------
    def annotate(self, trajectory: dict) -> dict:
        annotated = dict(trajectory)
        annotated["goal_list"] = [dict(goal) for goal in annotated["goal_list"]]
        for goal in annotated["goal_list"]:
            self.annotate_goal(goal)
        refine_turn_estimate(annotated)
        self.classify(annotated)
        self.generate_dependencies(annotated)
        return annotated

    def process(self, trajectory: dict) -> Tuple[Optional[dict], Optional[dict]]:
        """Return ``(annotated, None)`` on success or ``(None, original)`` on failure."""
        sanitized = sanitize_trajectory(trajectory)
        if sanitized is None:
            return None, trajectory
        for _ in range(self.retries_per_item + 1):
            try:
                annotated = self.annotate(sanitized)
            except Exception as error:  # noqa: BLE001
                print(f"[annotate] {trajectory.get('dialogue_id', 'unknown')}: {error}")
                continue
            if not validate_annotation(annotated):
                continue
            if self.enable_judge and not self.verify(annotated):
                continue
            return annotated, None
        return None, trajectory

    def run(self, trajectories: List[dict], workers: int, max_rounds: int, quality_threshold: float) -> List[dict]:
        required = max(1, int(len(trajectories) * quality_threshold))
        accepted: List[dict] = []
        pending = list(trajectories)
        for round_index in range(max_rounds):
            print(f"Annotation round {round_index + 1}/{max_rounds}: {len(pending)} trajectories")
            results = parallel_map(pending, self.process, workers=workers, desc="Annotating trajectories")
            accepted.extend(annotated for annotated, _ in results if annotated is not None)
            pending = [failed for _, failed in results if failed is not None]
            print(f"  accepted so far: {len(accepted)}/{len(trajectories)} (required {required})")
            if len(accepted) >= required or not pending:
                return accepted
        if len(accepted) < required:
            sys.exit(f"Quality threshold not met: {len(accepted)} < {required}")
        return accepted


def _content_from_slots(goal: dict) -> str:
    core = goal.get("core_content", "")
    slot_values = goal.get("slot_values", {})
    if not slot_values:
        return core
    return f"{core} with " + ", ".join(f"{k}: {v}" for k, v in slot_values.items())


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_work_dir_arg(parser)
    add_llm_args(parser)
    parser.add_argument("--disable-judge", action="store_true", help="Skip the trajectory verifier prompt")
    parser.add_argument("--max-retries", type=int, default=3, help="Annotation rounds over failed trajectories")
    parser.add_argument("--quality-threshold", type=float, default=0.7, help="Required pass ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_id = resolve_model_id_or_exit(parser, args.model_id)
    random.seed(args.seed)

    trajectories = load_json(
        require_file(args.work_dir / "sampled_goal_trajectories.json", "Run generation/sample_trajectories.py first.")
    )
    annotator = TrajectoryAnnotator(make_llm_client(model_id), enable_judge=not args.disable_judge)
    annotated = annotator.run(trajectories, args.workers, args.max_retries, args.quality_threshold)

    output = args.work_dir / "annotated_goal_trajectories.json"
    save_json(annotated, output)
    counts: Dict[str, int] = {}
    for trajectory in annotated:
        counts[trajectory["complexity_class"]] = counts.get(trajectory["complexity_class"], 0) + 1
    print(f"Saved {len(annotated)} annotated trajectories to {output}")
    print(f"  complexity distribution: {counts}")
    print(f"  verifier: {'disabled' if args.disable_judge else 'enabled'}")


if __name__ == "__main__":
    main()
