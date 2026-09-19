#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
LLM utilities for the ATOD-Eval metric scripts.

The judge calls reuse the shared Bedrock client in ``MemSys.utils.llm_controller``
(model ID from ``--model-id`` or ``ATOD_MODEL_ID``).
"""

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate_json_with_llm(
    prompt: str,
    model_id: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Evaluate with an LLM judge that must answer with a JSON object.

    Args:
        prompt: Evaluation prompt whose output format is a JSON object
        model_id: LLM model identifier

    Returns:
        Parsed JSON object (empty dict when parsing fails)
    """
    # Imported lazily so that the pure helpers below stay importable offline.
    from MemSys.utils.llm_controller import LLMController, extract_json_from_llm_response

    controller = LLMController(model_id=model_id)
    response = controller.get_completion(prompt, temperature=0.0)
    if verbose:
        print(f"LLM JSON Response: {response[:200]}...")
    try:
        data = json.loads(extract_json_from_llm_response(response))
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def format_dialogue_context(turns, upto: Optional[int] = None, window: Optional[int] = None) -> str:
    """Render ``turns`` (optionally only the first ``upto`` and/or the last ``window``) as text."""
    selected = list(turns[:upto] if upto is not None else turns)
    if window is not None:
        selected = selected[-window:]
    return "\n".join(f"{t.get('speaker', 'UNKNOWN')}: {t.get('utterance', '')}" for t in selected)


def goal_state_json(turn: dict, goal_ids: Optional[set] = None) -> str:
    """Serialize the ``all_goals`` snapshot of a turn (optionally restricted to ``goal_ids``)."""
    snapshot = []
    for entry in turn.get("all_goals", []) or []:
        if goal_ids is None or entry.get("goal_id") in goal_ids:
            snapshot.append({
                "goal_id": entry.get("goal_id"),
                "goal_content": entry.get("goal_content"),
                "status": entry.get("status"),
            })
    return json.dumps(snapshot, ensure_ascii=False)
