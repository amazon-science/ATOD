#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Stage 6: turn-level goal status annotation.

Every dialogue turn is processed sequentially. Given the dialogue so far and the
current status of each goal, the model returns the updated status of every goal
for the current turn. Transitions are validated against the lifecycle
``not_mentioned -> open -> pending -> {completed, failed, abandoned}``; invalid
or backwards transitions are rejected and the previous status is kept.

The output has the schema of the released benchmark (``data/schema.json``):
per-turn ``all_goals`` snapshots, ``goal_status_changes``, and per-goal
``status``, ``status_history``, ``first_mentioned_turn`` and ``completion_turn``.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional

from generation.common import (
    STATUSES,
    VALID_TRANSITIONS,
    add_llm_args,
    add_work_dir_arg,
    first_json_object,
    load_json,
    make_llm_client,
    parallel_map,
    require_file,
    resolve_model_id_or_exit,
    save_json,
)


# --------------------------------------------------------------------------- #
# Prompts (reproduced as used for the released benchmark)
# --------------------------------------------------------------------------- #
def status_prompt(last_turn: str, goals: List[dict], current: Dict[str, str]) -> str:
    descriptions = []
    template = {}
    for goal in goals:
        content = goal.get("content") or _fallback_content(goal)
        descriptions.append(f"Goal {goal['id']}: {content} [Current: {current[goal['id']]}]")
        template[goal["id"]] = {"status": current[goal["id"]]}
    terminal = [
        f"- {goal['id']}: {current[goal['id']].upper()}"
        for goal in goals
        if current[goal["id"]] in ("completed", "failed", "abandoned")
    ]
    return f"""You are tracking goal status in a task-oriented dialogue. Analyze ONLY the current turn and update statuses based on what actually happens.

CURRENT TURN TO ANALYZE: {last_turn}

GOALS AND CURRENT STATUS:
{chr(10).join(descriptions)}

STATUS MEANINGS:
- NOT_MENTIONED: Goal exists but hasn't been mentioned in dialogue yet
- OPEN: Goal mentioned by user, no action started yet
- PENDING: System actively working on goal
- COMPLETED: Goal successfully finished
- FAILED: Goal failed due to system issues (no availability, errors)
- ABANDONED: User cancelled/changed mind about goal

CRITICAL RULES:
1. ONLY change status if something definitive happens in the current turn
2. Once COMPLETED/FAILED/ABANDONED, goals NEVER change again
3. PENDING goals can only become COMPLETED/FAILED/ABANDONED
4. If nothing clear happens to a goal, keep its current status

WHAT TO LOOK FOR IN CURRENT TURN:
- User first mentions goal: "I need a hotel" → OPEN
- System says "Let me search..." or "I'm checking..." → PENDING
- System says "I've booked..." or "Here's your..." → COMPLETED
- System says "Sorry, no availability" or "System error" → FAILED
- User says "forget it" or "cancel that" → ABANDONED
- Goal not mentioned yet → NOT_MENTIONED

IMPORTANT: If a goal is already COMPLETED/FAILED/ABANDONED, it stays that way forever. Never change these back to PENDING or OPEN.

TERMINAL STATES - DO NOT CHANGE:
{chr(10).join(terminal)}

Current statuses as JSON template:
{json.dumps(template, indent=2)}

Respond with ONLY the JSON above, updating ONLY goals that clearly change in the current turn:"""


def goal_content_prompt(domain: str, intent: str, slot_values: dict) -> str:
    slot_items = [f"{k}: {v}" for k, v in slot_values.items() if v and str(v).lower() not in ("none", "unknown", "")]
    slot_context = f" with details: {', '.join(slot_items)}" if slot_items else ""
    return f"""Generate realistic goal descriptions for a task-oriented dialogue goal:

                            Domain: {domain}
                            Intent: {intent}{slot_context}

                            Generate TWO descriptions:
                            1. content: Detailed, specific goal description that clearly expresses what the user wants to accomplish
                            2. core_content: Simple, stable identifier for this type of goal (2-3 words, actionable)

                            CONSISTENT EXAMPLES (use these patterns):
                            - Domain: hotel, Intent: book → content: "Book hotel room in Seattle for 2 nights", core_content: "book hotel"
                            - Domain: flight, Intent: search → content: "Search for flights from NYC to LA on March 15", core_content: "find flight"
                            - Domain: restaurant, Intent: reserve → content: "Reserve table at Italian restaurant for dinner", core_content: "book restaurant"
                            - Domain: taxi, Intent: book → content: "Book taxi from downtown to airport", core_content: "book taxi"
                            - Domain: attraction, Intent: find → content: "Find tourist attractions in San Francisco", core_content: "find attraction"
                            - Domain: weather, Intent: get → content: "Get weather forecast for tomorrow", core_content: "get weather"

                            CORE_CONTENT PATTERNS:
                            - Booking/Reservations: "book [domain]" (book hotel, book flight, book restaurant)
                            - Information/Search: "find [domain]" or "get [domain]" (find restaurant, get weather)
                            - Other actions: "[action] [domain]" (cancel booking, modify reservation)

                            Respond with JSON:
                            {{"content": "detailed actionable description", "core_content": "action domain"}}"""


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested offline)
# --------------------------------------------------------------------------- #
def is_valid_transition(current: str, new: str) -> bool:
    current, new = current.lower(), new.lower()
    return current == new or new in VALID_TRANSITIONS.get(current, [])


def validate_transition(current: str, new: str, goal_id: str = "") -> str:
    """Return ``new`` when the transition is allowed, otherwise keep ``current``."""
    if is_valid_transition(current, new):
        return new.lower()
    print(f"Warning: invalid transition for {goal_id}: {current} -> {new}; keeping {current}")
    return current.lower()


def clean_status_history(history: List[dict]) -> List[dict]:
    if not history:
        return [{"turn": 0, "status": "not_mentioned"}]
    cleaned = [history[0]]
    for entry in history[1:]:
        if is_valid_transition(cleaned[-1]["status"], entry["status"]):
            cleaned.append(entry)
    return cleaned


def parse_status_response(response: str, goals: List[dict], current: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Map every goal id to its (validated) new status. None when nothing parseable."""
    # Strategy 1: JSON object {goal_id: {"status": ...}} or {goal_id: "status"}
    try:
        data = first_json_object(response)
        if isinstance(data, dict):
            result = {}
            for goal in goals:
                gid = goal["id"]
                value = data.get(gid)
                if isinstance(value, dict) and "status" in value:
                    result[gid] = validate_transition(current[gid], str(value["status"]), gid)
                elif isinstance(value, str):
                    result[gid] = validate_transition(current[gid], value, gid)
                else:
                    result[gid] = current[gid]
            return result
    except (ValueError, json.JSONDecodeError):
        pass

    # Strategy 2: per-goal regular expressions
    result = {}
    found_any = False
    for goal in goals:
        gid = goal["id"]
        status = None
        for pattern in (
            rf'"{re.escape(gid)}":\s*{{\s*"status":\s*"([^"]+)"',
            rf'"{re.escape(gid)}":\s*"([^"]+)"',
            rf"{re.escape(gid)}.*?status.*?([a-z_]+)",
        ):
            match = re.search(pattern, response, re.IGNORECASE)
            if match and match.group(1).lower() in STATUSES:
                status = match.group(1).lower()
                break
        if status:
            found_any = True
            result[gid] = validate_transition(current[gid], status, gid)
        else:
            result[gid] = current[gid]
    return result if found_any else None


def _fallback_content(goal: dict) -> str:
    text = f"{goal.get('intent', 'Unknown')} in {goal.get('domain', 'Unknown')}"
    if goal.get("slot_values"):
        text += " (" + ", ".join(f"{k}: {v}" for k, v in goal["slot_values"].items()) + ")"
    return text


# --------------------------------------------------------------------------- #
class StatusAnnotator:
    def __init__(self, client):
        self.client = client

    def ensure_goal_content(self, goal: dict) -> dict:
        goal = dict(goal)
        if goal.get("content") and goal.get("core_content"):
            return goal
        domain, intent = goal.get("domain", "unknown"), goal.get("intent", "unknown")
        try:
            data = first_json_object(
                self.client.call(goal_content_prompt(domain, intent, goal.get("slot_values", {})), max_tokens=3000, temperature=0.3)
            )
            goal["content"] = data.get("content") or goal.get("content")
            goal["core_content"] = data.get("core_content") or goal.get("core_content")
        except Exception as error:  # noqa: BLE001
            print(f"[goal content] {goal.get('id', 'unknown')}: {error}")
        goal.setdefault("content", f"{intent} in {domain}")
        goal.setdefault("core_content", f"{intent} {domain}")
        return goal

    def annotate_turn(self, dialogue_context: str, goals: List[dict], current: Dict[str, str], turn_num: int) -> Dict[str, str]:
        last_turn = dialogue_context.strip().split("\n")[-1] if dialogue_context.strip() else ""
        try:
            response = self.client.call(status_prompt(last_turn, goals, current), max_tokens=3000, temperature=0.3)
            parsed = parse_status_response(response, goals, current)
            if parsed:
                return parsed
            print(f"Could not parse status response for turn {turn_num}; keeping statuses")
        except Exception as error:  # noqa: BLE001
            print(f"[status] turn {turn_num}: {error}")
        return dict(current)

    def annotate_dialogue(self, dialogue: dict) -> dict:
        annotated = dict(dialogue)
        goals = [self.ensure_goal_content(goal) for goal in annotated.get("goal_list", [])]
        turns = annotated.get("turns", [])
        if not goals or not turns:
            return annotated
        annotated["goal_list"] = goals

        current: Dict[str, str] = {g["id"]: "not_mentioned" for g in goals}
        history: Dict[str, List[dict]] = {g["id"]: [{"turn": 0, "status": "not_mentioned"}] for g in goals}
        first_mentioned: Dict[str, Optional[int]] = {g["id"]: None for g in goals}
        completion: Dict[str, Optional[int]] = {g["id"]: None for g in goals}

        context_lines: List[str] = []
        annotated_turns = []
        for index, turn in enumerate(turns):
            turn_num = index + 1
            speaker = "User" if turn.get("speaker") == "USER" else "System"
            context_lines.append(f"{speaker}: {turn.get('utterance', '')}")
            new_statuses = self.annotate_turn("\n".join(context_lines) + "\n", goals, current, turn_num)

            snapshot, changes = [], []
            for goal in goals:
                gid = goal["id"]
                old, new = current[gid], new_statuses.get(gid, current[gid])
                changed = new != old
                if changed:
                    current[gid] = new
                    history[gid].append({"turn": turn_num, "status": new})
                    changes.append({"goal_id": gid, "new_status": new, "turn": turn_num})
                    if old == "not_mentioned" and new in ("open", "pending", "completed", "abandoned"):
                        first_mentioned[gid] = turn_num
                    if new == "completed":
                        completion[gid] = turn_num
                snapshot.append(
                    {
                        "goal_id": gid,
                        "goal_content": goal.get("content", _fallback_content(goal)),
                        "status": current[gid],
                        "status_changed": changed,
                    }
                )
            annotated_turns.append({**turn, "turn_id": turn_num, "all_goals": snapshot, "goal_status_changes": changes})

        annotated["turns"] = annotated_turns
        for goal in annotated["goal_list"]:
            gid = goal["id"]
            cleaned = clean_status_history(history[gid])
            goal.update(
                {
                    "status": cleaned[-1]["status"],
                    "status_history": cleaned,
                    "first_mentioned_turn": first_mentioned[gid],
                    "completion_turn": completion[gid],
                }
            )
        return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_work_dir_arg(parser)
    add_llm_args(parser)
    args = parser.parse_args()
    model_id = resolve_model_id_or_exit(parser, args.model_id)

    dialogues = load_json(
        require_file(args.work_dir / "synthetic_dialogues.json", "Run generation/generate_dialogues.py first.")
    )
    annotator = StatusAnnotator(make_llm_client(model_id))
    annotated = parallel_map(dialogues, annotator.annotate_dialogue, workers=args.workers, desc="Annotating status")
    annotated = [d for d in annotated if d.get("turns")]

    save_json(annotated, args.work_dir / "annotated_dialogues.json")
    for complexity in ("medium", "complex"):
        subset = [d for d in annotated if d.get("complexity_class") == complexity]
        for index, dialogue in enumerate(subset):
            dialogue.setdefault("dialogue_index", index)
        save_json(subset, args.work_dir / "final" / complexity / "annotated_dialogues.json")
        print(f"{complexity}: {len(subset)} dialogues -> {args.work_dir / 'final' / complexity}")


if __name__ == "__main__":
    main()
