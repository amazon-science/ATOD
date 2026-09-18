#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
LLM-based Goal Judge for Evaluation
"""

import json
import re
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

# Add the root directory to sys.path to import MemSys modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from MemSys.utils.llm_controller import LLMController
from MemSys.utils.rate_limiter import wait_for_rate_limit


class LLMGoalJudge:
    """LLM-based judge for evaluating goal detection and status tracking."""

    def __init__(self, model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"):
        """Initialize the LLM judge with specified model."""
        self.llm_controller = LLMController(backend="bedrock", model=model_id)
        self.model_id = model_id

    def compare_goals(self, detected_goals: List[str], gold_goals: List[str]) -> Dict[str, Any]:
        """
        Compare detected goals with gold goals using LLM evaluation.

        Args:
            detected_goals: List of detected goal content strings
            gold_goals: List of gold goal content strings

        Returns:
            Dictionary with precision, recall, f1, and matched_goal_contents
        """
        prompt = self._create_goal_comparison_prompt(detected_goals, gold_goals)

        try:
            wait_for_rate_limit()
            response = self.llm_controller.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "goal_comparison",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "precision": {"type": "number"},
                                "recall": {"type": "number"},
                                "f1": {"type": "number"},
                                "matched_goal_contents": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["precision", "recall", "f1", "matched_goal_contents"]
                        }
                    }
                },
                temperature=0.0
            )

            return self._parse_goal_comparison_response(response)

        except Exception as e:
            print(f"Error in LLM goal comparison: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'matched_goal_contents': []
            }

    def compare_statuses_batch(self, status_pairs: List[Tuple[str, str]]) -> List[bool]:
        """
        Compare status pairs in batch using LLM evaluation.

        Args:
            status_pairs: List of (detected_status, gold_status) tuples

        Returns:
            List of boolean values indicating if each pair matches
        """
        if not status_pairs:
            return []

        prompt = self._create_status_comparison_prompt(status_pairs)

        try:
            wait_for_rate_limit()
            response = self.llm_controller.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "status_comparison",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "matches": {
                                    "type": "array",
                                    "items": {"type": "boolean"}
                                }
                            },
                            "required": ["matches"]
                        }
                    }
                },
                temperature=0.0
            )

            return self._parse_status_comparison_response(response, len(status_pairs))

        except Exception as e:
            print(f"Error in LLM status comparison: {e}")
            return [False] * len(status_pairs)

    def _parse_goal_comparison_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for goal comparison."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            # Validate and extract metrics
            precision = float(data.get('precision', 0))
            recall = float(data.get('recall', 0))
            f1 = float(data.get('f1', 0))
            matched_goals = data.get('matched_goal_contents', [])

            # Ensure matched_goals is a list
            if not isinstance(matched_goals, list):
                matched_goals = []

            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'matched_goal_contents': matched_goals
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing goal comparison response: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'matched_goal_contents': []
            }

    def _parse_status_comparison_response(self, response: str, expected_length: int) -> List[bool]:
        """Parse LLM response for status comparison."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            matches = data.get('matches', [])

            # Ensure matches is a list of booleans with correct length
            if not isinstance(matches, list):
                return [False] * expected_length

            # Convert to boolean and pad/truncate as needed
            bool_matches = []
            for i in range(expected_length):
                if i < len(matches):
                    bool_matches.append(bool(matches[i]))
                else:
                    bool_matches.append(False)

            return bool_matches

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing status comparison response: {e}")
            return [False] * expected_length

    def _create_goal_comparison_prompt(self, detected_goals: List[str], gold_goals: List[str]) -> str:
        import json
        detected_str = json.dumps(detected_goals, indent=2)
        gold_str = json.dumps(gold_goals, indent=2)

        return f"""Compare detected and gold goals using EXTREMELY GENEROUS semantic matching. Your job is to find connections, not differences.

                Detected goals:
                {detected_str}

                Gold goals:
                {gold_str}

                ULTRA-GENEROUS MATCHING RULES (err on the side of matching):
                - ANY shared action word = potential match (book, find, check, get, search, buy, reserve, order, purchase, look, see, view, obtain, acquire)
                - ANY shared domain word = potential match (hotel, flight, restaurant, car, taxi, weather, account, balance, ticket, event, concert, show)
                - Parent/child relationships = ALWAYS match (general goal matches specific goals)
                - Specificity differences = IGNORE (detailed goals match general goals)
                - Extra details = IGNORE ("book hotel in Paris" matches "book hotel")
                - Synonyms = ALWAYS match (accommodation=hotel, dining=restaurant, balance=account, pass=ticket)
                - Partial matches = COUNT ("get information" matches "check balance")
                - Intent similarity = PRIORITIZE over exact wording

                EXPANDED EXAMPLES (all should match):
                ✓ "book something" matches "reserve hotel room in downtown Boston"
                ✓ "get info" matches "check current account balance"
                ✓ "find place" matches "search for Italian restaurants near me"
                ✓ "buy ticket" matches "purchase concert admission pass"
                ✓ "check status" matches "view flight departure information"
                ✓ "make reservation" matches "book table at restaurant"
                ✓ "look up" matches "search for weather forecast"
                ✓ "order" matches "purchase bus transportation tickets"
                ✓ "plan trip" matches "book flight to San Francisco"
                ✓ "need help with" matches "get assistance for account issues"

                MATCHING STRATEGY:
                1. Look for ANY semantic connection between goals
                2. If there's doubt, CHOOSE TO MATCH rather than reject
                3. Focus on user intent, not precise wording
                4. One detected goal can match multiple gold goals
                5. Multiple detected goals can match one gold goal
                6. When in doubt, be generous and match

                CALCULATION (be generous in counting matches):
                - precision = matched detected goals / total detected goals
                - recall = matched gold goals / total gold goals
                - f1 = 2 * precision * recall / (precision + recall)
                - matched_goal_contents = list of ALL matched GOLD goal contents

                Remember: Your goal is to find semantic connections, not to be strict. When uncertain, MATCH.

                Output JSON format:
                {{
                "precision": 0.85,
                "recall": 0.90,
                "f1": 0.87,
                "matched_goal_contents": ["matched gold goal 1", "matched gold goal 2", "matched gold goal 3"]
                }}"""

    def _create_status_comparison_prompt(self, status_pairs):
        pairs_str = []
        for i, (detected, gold) in enumerate(status_pairs):
            pairs_str.append(f"Pair {i+1}: Detected='{detected}', Gold='{gold}'")
        pairs_text = "\n".join(pairs_str)

        return f"""You evaluate dialogue goal statuses with MAXIMUM GENEROSITY. Your job is to find reasons to match, not reasons to reject.

                STATUS CATEGORIES:
                - OPEN/NEW: Goal mentioned but not started
                - PENDING/IN_PROGRESS: Goal being worked on
                - COMPLETED/DONE: Goal successfully finished
                - FAILED/ERROR: Goal attempted but unsuccessful
                - ABANDONED/CANCELLED: Goal stopped/cancelled

                ULTRA-GENEROUS NORMALIZATION (case-insensitive, strip punctuation):
                - "in progress", "ongoing", "started", "working on", "processing" → PENDING
                - "booked", "confirmed", "done", "finished", "resolved", "success", "complete" → COMPLETED
                - "cancelled", "canceled", "stopped", "terminated" → ABANDONED
                - "unavailable", "error", "failed", "unsuccessful", "problem" → FAILED
                - "mentioned", "requested", "new", "todo", "planned" → OPEN

                EXTREMELY GENEROUS MATCHING RULES (when in doubt, MATCH):
                1) Exact match → TRUE
                2) Same category matches → TRUE:
                   - OPEN ↔ PENDING ↔ "in progress" (all indicate active goals)
                   - COMPLETED ↔ PENDING (both show progress/achievement)
                   - FAILED ↔ ABANDONED (both show unsuccessful outcomes)
                   - Any "success" variant ↔ COMPLETED
                   - Any "working on" variant ↔ PENDING
                3) Benefit-of-doubt matches → TRUE:
                   - COMPLETED can match almost anything positive (PENDING, OPEN if making progress)
                   - PENDING can match OPEN (both show goal is active)
                   - FAILED can match ABANDONED (both show goal didn't succeed)
                4) Ambiguous cases → TRUE (choose to match when uncertain)

                PHILOSOPHY: Status tracking is inherently subjective. Different systems may reasonably interpret the same dialogue state differently. Be maximally generous - if there's ANY reasonable way the statuses could be considered equivalent, count it as a match.

                INPUT PAIRS:
                {pairs_text}

                OUTPUT FORMAT (JSON only):
                {{ "matches": [true, true, true, ...] }}"""

    def compare_dependencies(self, goal_content: str, detected_deps: List[str], gold_deps: List[str]) -> bool:
        """
        Compare detected dependencies with gold dependencies for a specific goal using LLM evaluation.

        Args:
            goal_content: Content of the goal whose dependencies are being evaluated
            detected_deps: List of detected dependency goal contents
            gold_deps: List of gold dependency goal contents

        Returns:
            True if dependencies match semantically, False otherwise
        """
        # If both are empty, they match
        if not detected_deps and not gold_deps:
            return True

        # If one is empty and the other isn't, they don't match
        if not detected_deps or not gold_deps:
            return False

        prompt = self._create_dependency_comparison_prompt(goal_content, detected_deps, gold_deps)

        try:
            wait_for_rate_limit()
            response = self.llm_controller.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dependency_comparison",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "dependencies_match": {"type": "boolean"},
                                "explanation": {"type": "string"}
                            },
                            "required": ["dependencies_match", "explanation"]
                        }
                    }
                },
                temperature=0.0
            )

            return self._parse_dependency_comparison_response(response)

        except Exception as e:
            print(f"Error in LLM dependency comparison: {e}")
            return False

    def _create_dependency_comparison_prompt(self, goal_content: str, detected_deps: List[str], gold_deps: List[str]) -> str:
        """Create a concise, slightly-tolerant prompt for dependency comparison (boolean output only)."""
        import json
        detected_str = json.dumps(detected_deps, indent=2)
        gold_str = json.dumps(gold_deps, indent=2)

        return f"""You evaluate whether two dependency sets for a main goal should be considered a match.

                MAIN GOAL:
                "{goal_content}"

                DETECTED DEPENDENCIES:
                {detected_str}

                GOLD DEPENDENCIES:
                {gold_str}

                DEFINITION
                A dependency means the main goal should not be completed until the dependency goal is completed.

                EVALUATION (normalize before judging: lowercase, remove punctuation, singularize common nouns, and merge obvious synonyms like "pay"≈"payment", "book"≈"reservation", "id"≈"identification"):
                1) Semantic equivalence: treat two dependencies as the same if they refer to the same prerequisite task/entity even with minor paraphrases (e.g., "confirm booking" vs "booking confirmation").
                2) Logical consistency: the prerequisite relationship is preserved (detected does not contradict the idea of being a prerequisite).
                3) Near-duplicate collapsing: if multiple detected items are variants of one gold item (or vice versa), count them once.
                4) Ordering and duplicates do not matter.

                LENIENCY (generous tolerance):
                - Allow significant wording differences and reformulations.
                - Allow up to one missing or one extra dependency when the gold set has ≤ 3 items.
                - When the gold set has > 3 items, allow up to 30% mismatch.
                - In case of ambiguity, prefer mapping to the closest gold item rather than treating as different.
                - Focus on semantic intent rather than exact wording.
                - If the core dependency relationship is preserved, consider it a match.

                DECISION
                Return true if, after applying the rules above, the detected set sufficiently covers the gold prerequisites and preserves the same dependency intent; otherwise return false.

                OUTPUT (JSON only):
                {{ "dependencies_match": true }} or {{ "dependencies_match": false }}"""


    def _parse_dependency_comparison_response(self, response: str) -> bool:
        """Parse LLM response for dependency comparison."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return bool(data.get('dependencies_match', False))

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing dependency comparison response: {e}")
            return False
