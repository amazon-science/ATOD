# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Offline tests for the generation pipeline (no model calls)."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from generation.annotate_dialogue_status import (
    clean_status_history,
    parse_status_response,
    validate_transition,
)
from generation.annotate_trajectories import (
    apply_dependencies,
    complexity_score,
    parse_dependency_response,
    parse_slot_response,
    refine_turn_estimate,
    rule_based_complexity,
    sanitize_trajectory,
    set_agentic_flags,
    validate_annotation,
)
from generation.build_cooccurrence_graph import build_graph, graph_statistics
from generation.common import is_placeholder_value
from generation.extract_goals import extract_goal_sequences, extract_goals_from_dialogue
from generation.generate_dialogues import parse_generated_dialogue
from generation.sample_trajectories import GoalTrajectorySampler


def sgd_dialogue(*frames):
    """Build a minimal SGD-style dialogue with one user turn per (service, intent)."""
    turns = []
    for service, intent in frames:
        turns.append(
            {
                "speaker": "USER",
                "utterance": "...",
                "frames": [{"service": service, "state": {"active_intent": intent}}],
            }
        )
        turns.append({"speaker": "SYSTEM", "utterance": "...", "frames": []})
    return {"dialogue_id": "x", "turns": turns}


class ExtractGoalsTests(unittest.TestCase):
    def test_unique_domain_intent_pairs_in_order(self):
        dialogue = sgd_dialogue(("Hotels_1", "ReserveHotel"), ("Hotels_1", "ReserveHotel"), ("Flights_2", "SearchFlights"))
        self.assertEqual(
            extract_goals_from_dialogue(dialogue),
            [{"domain": "Hotels", "intent": "ReserveHotel"}, {"domain": "Flights", "intent": "SearchFlights"}],
        )

    def test_intent_fallbacks_and_min_length(self):
        dialogue = {
            "turns": [
                {"speaker": "USER", "frames": [{"service": "Banks_1", "state": {"requested_slots": ["balance"]}}]},
                {"speaker": "USER", "frames": [{"service": "Buses_1", "state": {"slot_values": {"to": "Denver"}}}]},
            ]
        }
        self.assertEqual([g["intent"] for g in extract_goals_from_dialogue(dialogue)], ["find", "book"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dialogues_001.json").write_text(
                json.dumps([sgd_dialogue(("Hotels_1", "ReserveHotel")), sgd_dialogue(("Hotels_1", "ReserveHotel"), ("Flights_2", "SearchFlights"))])
            )
            sequences = extract_goal_sequences(root, (1, 2))  # file 002 is missing and skipped
        self.assertEqual(len(sequences), 1)


class GraphAndSamplingTests(unittest.TestCase):
    SEQUENCES = [
        [{"domain": "Hotels", "intent": "Reserve"}, {"domain": "Flights", "intent": "Search"}],
        [{"domain": "Hotels", "intent": "Reserve"}, {"domain": "Flights", "intent": "Search"}, {"domain": "Events", "intent": "Buy"}],
        [{"domain": "Weather", "intent": "Get"}, {"domain": "Events", "intent": "Buy"}],
    ]

    def test_graph_weights_and_statistics(self):
        graph = build_graph(self.SEQUENCES)
        weights = {(e["source"], e["target"]): e["weight"] for e in graph["edges"]}
        self.assertEqual(weights[("Flights_Search", "Hotels_Reserve")], 2)
        stats = graph_statistics(graph)
        self.assertEqual(stats["nodes"], 4)
        self.assertEqual(stats["edges"], 4)
        self.assertEqual(stats["density"], round(2 * 4 / (4 * 3), 4))
        self.assertEqual(stats["connected_components"], 1)

    def test_sampling_respects_goal_ranges_and_seed(self):
        graph = build_graph(self.SEQUENCES)
        first = GoalTrajectorySampler(graph, random.Random(7)).sample(30, {"medium": 0.65, "complex": 0.35})
        second = GoalTrajectorySampler(graph, random.Random(7)).sample(30, {"medium": 0.65, "complex": 0.35})
        self.assertEqual(len(first), 30)
        self.assertEqual(
            [[(g["domain"], g["intent"]) for g in t["goal_list"]] for t in first],
            [[(g["domain"], g["intent"]) for g in t["goal_list"]] for t in second],
        )
        for trajectory in first:
            n = trajectory["metadata"]["num_goals"]
            self.assertTrue(1 <= n <= 12)
            self.assertEqual(trajectory["complexity_class"], "unclassified")
            self.assertEqual([g["id"] for g in trajectory["goal_list"]], [f"goal_{i + 1}" for i in range(n)])
            for a, b in zip(trajectory["goal_list"], trajectory["goal_list"][1:]):
                self.assertNotEqual((a["domain"], a["intent"]), (b["domain"], b["intent"]))
        self.assertEqual(len({t["dialogue_id"] for t in first}), 30)


class TrajectoryAnnotationTests(unittest.TestCase):
    def test_placeholder_detection(self):
        for value in ("[location]", "{date}", "TBD", "n/a", "", "sample value", "...", "unknown"):
            self.assertTrue(is_placeholder_value(value), value)
        for value in ("downtown Seattle", "March 15, 2025", "2", "$", "7:30 PM"):
            self.assertFalse(is_placeholder_value(value), value)

    def test_slot_response_filters_placeholders(self):
        response = 'Sure:\n{"slots": ["city", "date", "extra"], "slot_values": {"city": "Boston", "date": "[date]"}, "content": "Book hotel in Boston", "core_content": "book hotel"}'
        parsed = parse_slot_response(response)
        self.assertEqual(parsed["slots"], ["city"])
        self.assertEqual(parsed["slot_values"], {"city": "Boston"})
        self.assertEqual(parsed["core_content"], "book hotel")

    def test_dependency_parsing_and_application(self):
        goals = [{"id": f"goal_{i}", "domain": "D", "intent": "I"} for i in range(1, 5)]
        deps = parse_dependency_response('{"dependencies": [{"goal": 2, "depends_on": [1]}, {"goal": 4, "depends_on": [2, 3, 9]}]}')
        apply_dependencies(goals, deps)
        self.assertEqual(goals[1]["dependencies"], ["goal_1"])
        self.assertEqual(goals[3]["dependencies"], ["goal_2", "goal_3"])
        self.assertEqual(goals[0]["dependencies"], [])
        loose = parse_dependency_response("goal 3 depends on 1, 2")
        self.assertEqual(loose, [{"goal": 3, "depends_on": [1, 2]}])

    def test_rule_based_complexity_and_flags(self):
        small = {"goal_list": [{"domain": "A", "intent": "x"}, {"domain": "A", "intent": "y"}], "metadata": {}}
        refine_turn_estimate(small)
        self.assertEqual(small["metadata"]["estimated_turns"], 6)
        self.assertEqual(rule_based_complexity(small), "medium")

        large = {"goal_list": [{"domain": f"D{i}", "intent": "x", "dependencies": []} for i in range(9)], "metadata": {}}
        refine_turn_estimate(large)
        self.assertGreaterEqual(complexity_score(large), 3)
        self.assertEqual(rule_based_complexity(large), "complex")

        # overlapping range (7 goals, two domains, no dependencies) -> model-based branch
        overlap = {"goal_list": [{"domain": "A" if i % 2 else "B", "intent": f"x{i}", "dependencies": []} for i in range(7)], "metadata": {}}
        refine_turn_estimate(overlap)
        self.assertEqual(complexity_score(overlap), 2)
        self.assertIsNone(rule_based_complexity(overlap))

        set_agentic_flags(small, "medium")
        self.assertFalse(small["metadata"]["proactivity"])
        self.assertFalse(small["dependency_label"])
        set_agentic_flags(large, "complex")
        self.assertTrue(large["metadata"]["proactivity"])
        self.assertTrue(large["defectiveness_label"])

    def test_sanitize_and_validate(self):
        trajectory = {"dialogue_id": "d", "metadata": {"num_goals": 3}, "goal_list": [
            {"domain": "Hotels", "intent": "Reserve"},
            {"domain": "", "intent": "Broken"},
        ]}
        clean = sanitize_trajectory(trajectory)
        self.assertEqual(len(clean["goal_list"]), 1)
        self.assertEqual(clean["metadata"]["num_goals"], 1)
        self.assertEqual(clean["goal_list"][0]["id"], "goal_1")
        self.assertFalse(validate_annotation(clean))  # no content yet
        clean["goal_list"][0].update({"content": "Reserve a hotel", "core_content": "book hotel", "slot_values": {"city": "Boston"}})
        self.assertTrue(validate_annotation(clean))
        clean["goal_list"][0]["slot_values"]["date"] = "[date]"
        self.assertFalse(validate_annotation(clean))


class DialogueGenerationTests(unittest.TestCase):
    TRAJECTORY = {"dialogue_id": "d1", "complexity_class": "medium", "goal_list": [{"id": "goal_1"}], "metadata": {}}

    def test_turn_parsing(self):
        response = "USER: Hi, I need a hotel.\nSYSTEM: Sure, where?\nUSER: Boston.\nSYSTEM: Searching now.\n"
        dialogue = parse_generated_dialogue(response, self.TRAJECTORY)
        self.assertEqual([t["speaker"] for t in dialogue["turns"]], ["USER", "SYSTEM", "USER", "SYSTEM"])
        self.assertEqual([t["turn_id"] for t in dialogue["turns"]], [1, 2, 3, 4])
        self.assertEqual(dialogue["turns"][2]["utterance"], "Boston.")

    def test_short_dialogue_rejected(self):
        self.assertIsNone(parse_generated_dialogue("USER: Hi\nSYSTEM: Hello", self.TRAJECTORY))


class StatusAnnotationTests(unittest.TestCase):
    GOALS = [{"id": "g1"}, {"id": "g2"}]

    def test_transition_rules(self):
        self.assertEqual(validate_transition("open", "pending", "g"), "pending")
        self.assertEqual(validate_transition("pending", "open", "g"), "pending")
        self.assertEqual(validate_transition("completed", "pending", "g"), "completed")
        self.assertEqual(validate_transition("not_mentioned", "completed", "g"), "completed")

    def test_history_cleaning(self):
        history = [
            {"turn": 0, "status": "not_mentioned"},
            {"turn": 2, "status": "pending"},
            {"turn": 3, "status": "open"},  # invalid backwards transition
            {"turn": 5, "status": "completed"},
        ]
        self.assertEqual([h["status"] for h in clean_status_history(history)], ["not_mentioned", "pending", "completed"])

    def test_status_response_parsing(self):
        current = {"g1": "open", "g2": "not_mentioned"}
        parsed = parse_status_response('```json\n{"g1": {"status": "PENDING"}, "g2": "not_mentioned"}\n```', self.GOALS, current)
        self.assertEqual(parsed, {"g1": "pending", "g2": "not_mentioned"})
        # invalid transition is rejected, missing goal keeps status
        parsed = parse_status_response('{"g1": {"status": "not_mentioned"}}', self.GOALS, current)
        self.assertEqual(parsed, {"g1": "open", "g2": "not_mentioned"})
        # regex fallback
        parsed = parse_status_response('g1: status completed; g2 unchanged', self.GOALS, current)
        self.assertEqual(parsed["g1"], "completed")
        self.assertIsNone(parse_status_response("no useful content", self.GOALS, current))


class _StubClient:
    """Deterministic stand-in for the Bedrock client (no network)."""

    def __init__(self):
        self.calls = 0
        self.model_classifications = 0

    def call(self, prompt, max_tokens=None, temperature=None, **_kwargs):
        self.calls += 1
        if prompt.startswith("Annotate a goal for task-oriented dialogue."):
            return json.dumps({"slots": ["city", "date"], "slot_values": {"city": "Boston", "date": "March 15, 2025"},
                               "content": "Book something in Boston on March 15, 2025", "core_content": "book item"})
        if prompt.startswith("Analyze these goals and determine logical dependencies"):
            return json.dumps({"dependencies": [{"goal": 2, "depends_on": [1]}]})
        if prompt.startswith("Classify this goal trajectory complexity"):
            self.model_classifications += 1
            return "COMPLEX"
        if "Respond with exactly one word: PASS or FAIL" in prompt:
            return "PASS"
        if prompt.startswith("Generate a realistic task-oriented dialogue"):
            return ("USER: I need to book a hotel in Boston.\nSYSTEM: Let me search for hotels.\n"
                    "USER: Also a flight please.\nSYSTEM: Your hotel is booked. Searching flights now.\n"
                    "USER: Never mind the flight.\nSYSTEM: Understood, I cancelled the flight search.")
        if prompt.startswith("You are tracking goal status in a task-oriented dialogue."):
            turn = prompt.split("CURRENT TURN TO ANALYZE:")[1].split("\n")[0].lower()
            updates = {}
            if "need to book a hotel" in turn:
                updates["goal_1"] = {"status": "open"}
            elif "search for hotels" in turn:
                updates["goal_1"] = {"status": "pending"}
            elif "also a flight" in turn:
                updates["goal_2"] = {"status": "open"}
            elif "hotel is booked" in turn:
                updates.update({"goal_1": {"status": "completed"}, "goal_2": {"status": "pending"}})
            elif "never mind" in turn:
                updates["goal_2"] = {"status": "abandoned"}
                updates["goal_1"] = {"status": "open"}  # invalid: must be ignored
            return json.dumps(updates)
        raise AssertionError("unexpected prompt: " + prompt[:60])


class EndToEndStubTests(unittest.TestCase):
    def test_model_backed_stages_with_stub_client(self):
        from generation.annotate_dialogue_status import StatusAnnotator
        from generation.annotate_trajectories import TrajectoryAnnotator
        from generation.generate_dialogues import DialogueGenerator

        graph = build_graph(GraphAndSamplingTests.SEQUENCES)
        trajectories = GoalTrajectorySampler(graph, random.Random(1)).sample(4, {"medium": 1.0, "complex": 0.0})
        client = _StubClient()

        annotated = TrajectoryAnnotator(client, enable_judge=True).run(trajectories, workers=2, max_rounds=2, quality_threshold=0.7)
        self.assertEqual(len(annotated), 4)
        for trajectory in annotated:
            self.assertIn(trajectory["complexity_class"], {"medium", "complex"})
            self.assertIn(trajectory["classification_method"], {"pre_defined", "model_based"})
            if trajectory["classification_method"] == "model_based":
                self.assertEqual(trajectory["complexity_class"], "complex")
            self.assertEqual(trajectory["metadata"]["estimated_turns"], 3 * len(trajectory["goal_list"]))
            for goal in trajectory["goal_list"]:
                self.assertEqual(goal["slot_values"]["city"], "Boston")
                self.assertTrue(goal["content"] and goal["core_content"])
            if len(trajectory["goal_list"]) > 3:
                self.assertEqual(trajectory["goal_list"][1]["dependencies"], ["goal_1"])
        self.assertEqual(client.model_classifications, sum(t["classification_method"] == "model_based" for t in annotated))

        dialogues = DialogueGenerator(client, enable_judge=True).run(annotated, workers=1, max_rounds=1, quality_threshold=0.8)
        self.assertEqual(len(dialogues), 4)
        self.assertEqual(len(dialogues[0]["turns"]), 6)

        final = StatusAnnotator(client).annotate_dialogue(dialogues[0])
        by_id = {g["id"]: g for g in final["goal_list"]}
        self.assertEqual(by_id["goal_1"]["status"], "completed")
        self.assertEqual(by_id["goal_1"]["first_mentioned_turn"], 1)
        self.assertEqual(by_id["goal_1"]["completion_turn"], 4)
        self.assertEqual([h["status"] for h in by_id["goal_1"]["status_history"]], ["not_mentioned", "open", "pending", "completed"])
        if "goal_2" in by_id:
            self.assertEqual(by_id["goal_2"]["status"], "abandoned")
        self.assertEqual(len(final["turns"]), 6)
        self.assertTrue(all("all_goals" in t and "goal_status_changes" in t for t in final["turns"]))
        self.assertEqual(final["turns"][3]["goal_status_changes"][0]["new_status"], "completed")


if __name__ == "__main__":
    unittest.main()
