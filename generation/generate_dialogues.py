#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Stage 5: generate synthetic dialogues from annotated trajectories.

Each annotated trajectory (goals, slot values, dependencies, complexity label
and agentic attribute flags) is turned into a structured prompt; the model
writes an alternating USER/SYSTEM dialogue. Unless ``--disable-judge`` is given,
a verifier prompt screens every dialogue and rejected dialogues are dropped.
Generation is retried until ``--quality-threshold`` of the trajectories yield
an accepted dialogue with at least four turns.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional

from generation.common import (
    add_llm_args,
    add_work_dir_arg,
    format_dialogue,
    load_json,
    make_llm_client,
    one_word_verdict,
    parallel_map,
    require_file,
    resolve_model_id_or_exit,
    save_json,
)

TURN_PATTERN = re.compile(r"(USER|SYSTEM):\s*(.+?)(?=(?:USER|SYSTEM):|$)", re.DOTALL | re.IGNORECASE)
MIN_TURNS = 4


# --------------------------------------------------------------------------- #
# Prompts (reproduced as used for the released benchmark)
# --------------------------------------------------------------------------- #
def dialogue_generation_prompt(trajectory: dict) -> str:
    complexity = trajectory.get("complexity_class", "medium")
    metadata = trajectory.get("metadata", {})
    goals = trajectory.get("goal_list", [])

    goal_descriptions = []
    for index, goal in enumerate(goals, 1):
        description = f"Goal {index}: {goal.get('domain', 'Unknown')} - {goal.get('intent', 'Unknown')}"
        if goal.get("slot_values"):
            description += " (Required info: " + ", ".join(f"{k}: {v}" for k, v in goal["slot_values"].items()) + ")"
        if goal.get("dependencies"):
            description += f" (Depends on: {', '.join(goal['dependencies'])})"
        goal_descriptions.append(description)

    agentic_attrs, agentic_guidance = [], []
    if metadata.get("async_execution"):
        agentic_attrs.append("asynchronous execution (goals can be worked on simultaneously)")
        agentic_guidance.append("ASYNC: System can handle multiple goals at once, switching between them naturally")
    if metadata.get("interleaving"):
        agentic_attrs.append("interleaving (user can switch between goals mid-conversation)")
        agentic_guidance.append("INTERLEAVING: User may jump between goals mid-conversation, system adapts smoothly")
    if trajectory.get("dependency_label"):
        agentic_attrs.append("dependencies (some goals depend on completion of others)")
        agentic_guidance.append("DEPENDENCIES: Some goals may depend on others, but not all goals need dependencies - use when logical")
    if metadata.get("proactivity"):
        agentic_attrs.append("proactive system behavior (system suggests related actions)")
        agentic_guidance.append("PROACTIVITY: System suggests helpful related services, asks clarifying questions to prevent issues, but doesn't overwhelm or force")
    if trajectory.get("defectiveness_label"):
        agentic_attrs.append("defectiveness (include realistic failures and error scenarios)")
        agentic_guidance.append("DEFECTIVENESS: Include realistic failures (no availability, booking errors, system issues)")

    combined_guidance = ""
    if agentic_guidance:
        combined_guidance = f"""
                AGENTIC BEHAVIOR GUIDELINES:
                {chr(10).join(f"- {guidance}" for guidance in agentic_guidance)}"""

    try:
        estimated_turns = int(metadata.get("estimated_turns", 20))
    except (TypeError, ValueError):
        estimated_turns = 20

    outcome_guidance = """
                GOAL PROGRESSION GUIDANCE:
                Create clear, forward-moving goal progressions that are easy to track:

                1. INTRODUCE goals clearly: "I need to book a hotel"
                2. SYSTEM STARTS WORK: "Let me search for hotels" (goal becomes active)
                3. CLEAR OUTCOMES: Choose one path per goal:
                   - FIRST MENTION: "I need a hotel" (goal becomes open)
                   - SYSTEM STARTS: "Let me search" (goal becomes pending)
                   - SUCCESS: "I've booked your hotel room" (then goal stays completed)
                   - FAILURE: "No hotels available" (then goal stays failed)
                   - USER CANCELS: "Actually, forget the hotel" (then goal stays abandoned)
                   - STILL WORKING: "I'm still checking availability" (goal remains pending)

                IMPORTANT: Once a goal reaches SUCCESS/FAILURE/ABANDONED, don't revisit it.
                This creates clear status tracking without confusing back-and-forth."""

    return f"""Generate a realistic task-oriented dialogue between USER and SYSTEM.

                REQUIREMENTS:
                - Complexity: {complexity.upper()}
                - Length: ~{estimated_turns} turns
                - Goals: {chr(10).join(goal_descriptions)}
                - Attributes: {chr(10).join(f"- {attr}" for attr in agentic_attrs) if agentic_attrs else "- Standard processing"}
                {combined_guidance}
                {outcome_guidance}

                DIALOGUE STRUCTURE:
                1. User introduces goals naturally throughout the conversation
                2. System works on goals with realistic constraints and limitations
                3. Natural obstacles, delays, and user preference changes may occur
                4. Conversation ends when it reaches a natural stopping point
                5. Goal completion varies based on circumstances - this reflects reality

                NATURAL CONVERSATION PATTERNS:
                - User expresses needs and preferences as they arise
                - System responds helpfully while working within realistic constraints
                - Users may add, modify, or abandon goals based on information received
                - System may encounter availability issues, pricing concerns, or technical limitations
                - Conversations conclude when users are satisfied with progress or need time to decide

                FORMAT: Alternating USER/SYSTEM turns, start with USER.

                EXAMPLE (clear goal progressions):
                USER: I need to book a flight to Seattle and find a restaurant there.
                SYSTEM: I'll help with both. What date for your flight?
                USER: This Friday. Also, I'll need a hotel.
                SYSTEM: Let me search for Friday flights to Seattle... I found some options around $300.
                USER: That works. Book the cheapest one.
                SYSTEM: Done! I've booked your Friday flight to Seattle. Now searching for hotels...
                USER: Actually, forget about the restaurant - I'll just eat at the hotel.
                SYSTEM: No problem, I've cancelled the restaurant search. Still looking for hotels - this may take a moment.
                USER: Thanks, I'll call back later about the hotel.
                SYSTEM: Perfect! Your flight is confirmed. I'll keep the hotel search ready for when you call back.
                [Clear progression: flight NOT_MENTIONED→OPEN→PENDING→COMPLETED, restaurant NOT_MENTIONED→OPEN→ABANDONED, hotel NOT_MENTIONED→OPEN→PENDING]

                Generate the dialogue:"""


def dialogue_verifier_prompt(dialogue: dict) -> str:
    turns = dialogue.get("turns", [])
    goals_text = [f"Goal: {g.get('content', g.get('core_content', 'N/A'))}" for g in dialogue.get("goal_list", [])]
    return f"""You are a quality judge for synthetic task-oriented dialogues.

                DIALOGUE ({len(turns)} turns, {dialogue.get('complexity_class', 'unknown')} complexity):
                {format_dialogue(turns)}

                GOALS:
                {chr(10).join(goals_text)}

                Is this dialogue suitable for evaluation?
                Check:
                - Realistic task-oriented conversation
                - Goals match dialogue content
                - No critical errors (empty dialogue, nonsensical content, wrong goals)

                Minor issues are acceptable, but reject dialogues with critical errors.

                Respond with exactly one word: PASS or FAIL"""


# --------------------------------------------------------------------------- #
def parse_generated_dialogue(response: str, trajectory: dict) -> Optional[dict]:
    """Split a model response into USER/SYSTEM turns; None when fewer than MIN_TURNS."""
    turns = []
    for speaker, utterance in TURN_PATTERN.findall(response):
        utterance = utterance.strip()
        if utterance:
            turns.append({"turn_id": len(turns) + 1, "speaker": speaker.upper(), "utterance": utterance})
    if len(turns) < MIN_TURNS:
        return None
    return {
        "dialogue_id": trajectory["dialogue_id"],
        "complexity_class": trajectory["complexity_class"],
        "goal_list": trajectory["goal_list"],
        "metadata": trajectory.get("metadata", {}),
        "turns": turns,
    }


class DialogueGenerator:
    def __init__(self, client, enable_judge: bool = True):
        self.client = client
        self.enable_judge = enable_judge

    def generate_one(self, trajectory: dict) -> Optional[dict]:
        if not trajectory.get("goal_list"):
            return None
        try:
            response = self.client.call(dialogue_generation_prompt(trajectory), max_tokens=5000, temperature=0.8)
        except Exception as error:  # noqa: BLE001
            print(f"[generate] {trajectory.get('dialogue_id', 'unknown')}: {error}")
            return None
        dialogue = parse_generated_dialogue(response, trajectory)
        if dialogue is None:
            return None
        if self.enable_judge and not self.verify(dialogue):
            return None
        return dialogue

    def verify(self, dialogue: dict) -> bool:
        try:
            return one_word_verdict(self.client.call(dialogue_verifier_prompt(dialogue), max_tokens=16, temperature=0.1))
        except Exception as error:  # noqa: BLE001
            print(f"[verifier] {dialogue.get('dialogue_id', 'unknown')}: {error}")
            return False

    def run(self, trajectories: List[dict], workers: int, max_rounds: int, quality_threshold: float) -> List[dict]:
        required = max(1, int(len(trajectories) * quality_threshold))
        accepted: List[dict] = []
        pending = list(trajectories)
        for round_index in range(max_rounds):
            print(f"Generation round {round_index + 1}/{max_rounds}: {len(pending)} trajectories")
            results = parallel_map(pending, self.generate_one, workers=workers, desc="Generating dialogues")
            accepted.extend(d for d in results if d is not None)
            pending = [t for t, d in zip(pending, results) if d is None]
            print(f"  accepted so far: {len(accepted)}/{len(trajectories)} (required {required})")
            if len(accepted) >= required or not pending:
                return accepted
        if len(accepted) < required:
            sys.exit(f"Quality threshold not met: {len(accepted)} < {required}")
        return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_work_dir_arg(parser)
    add_llm_args(parser)
    parser.add_argument("--disable-judge", action="store_true", help="Skip the dialogue verifier prompt")
    parser.add_argument("--max-retries", type=int, default=3, help="Generation rounds over failed trajectories")
    parser.add_argument("--quality-threshold", type=float, default=0.8, help="Required accepted ratio")
    args = parser.parse_args()
    model_id = resolve_model_id_or_exit(parser, args.model_id)

    trajectories = load_json(
        require_file(args.work_dir / "annotated_goal_trajectories.json", "Run generation/annotate_trajectories.py first.")
    )
    generator = DialogueGenerator(make_llm_client(model_id), enable_judge=not args.disable_judge)
    dialogues = generator.run(trajectories, args.workers, args.max_retries, args.quality_threshold)

    output = args.work_dir / "synthetic_dialogues.json"
    save_json(dialogues, output)
    total_turns = sum(len(d["turns"]) for d in dialogues)
    print(f"Saved {len(dialogues)} dialogues to {output}")
    print(f"  avg utterances per dialogue: {total_turns / len(dialogues):.1f}")
    print(f"  verifier: {'disabled' if args.disable_judge else 'enabled'}")


if __name__ == "__main__":
    main()
