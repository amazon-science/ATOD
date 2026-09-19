# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import unittest

from ATODEval.dependency import build_edges, prf
from ATODEval.dgcr import compute_dgcr_for_dialogue
from ATODEval.ntc import compute_ntc_for_dialogue


class MetricTests(unittest.TestCase):
    def test_dependency_edges_and_prf(self):
        goals = [
            {"id": "g1", "content": "book flight", "dependencies": []},
            {"id": "g2", "content": "book hotel", "dependencies": ["g1"]},
        ]
        self.assertEqual(build_edges(goals), {("book flight", "book hotel")})
        self.assertEqual(prf(1, 0, 0), (1.0, 1.0, 1.0))

    def test_dgcr_and_ntc_accept_toy_goals(self):
        goals = [
            {
                "id": "g1",
                "content": "book flight",
                "status": "completed",
                "dependencies": [],
                "completion_turn": 2,
                "first_mentioned_turn": 1,
            },
            {
                "id": "g2",
                "content": "book hotel",
                "status": "completed",
                "dependencies": ["g1"],
                "completion_turn": 4,
                "first_mentioned_turn": 1,
            },
        ]
        self.assertEqual(compute_dgcr_for_dialogue(goals), (2, 2))
        sum_turns, decided_goals = compute_ntc_for_dialogue(goals)
        self.assertEqual(decided_goals, 2)
        self.assertEqual(sum_turns, (2 - 1) + (4 - 1))

    def test_decided_goal_set_matches_paper_definition(self):
        goals = [
            {"id": "g1", "status": "completed", "dependencies": [], "first_mentioned_turn": 1, "completion_turn": 3},
            {"id": "g2", "status": "failed", "dependencies": [], "first_mentioned_turn": 2,
             "status_history": [{"turn": 0, "status": "not_mentioned"}, {"turn": 2, "status": "open"}, {"turn": 6, "status": "failed"}]},
            {"id": "g3", "status": "pending", "dependencies": ["g2"], "first_mentioned_turn": 4},
            {"id": "g4", "status": "abandoned", "dependencies": [], "first_mentioned_turn": 5},
        ]
        # g3 (blocked, non-terminal) and g4 (abandoned) are excluded: dGCR = 1/2
        self.assertEqual(compute_dgcr_for_dialogue(goals), (1, 2))
        # NTC covers completed and failed goals: (3-1) + (6-2) over 2 decided goals
        self.assertEqual(compute_ntc_for_dialogue(goals), (6, 2))

    def test_proactive_turn_detection_and_query_construction(self):
        from ATODEval.memory_recall_accuracy import status_change_queries
        from ATODEval.proactivity_effectiveness import find_proactive_turns

        goals = [
            {"id": "g1", "content": "Book hotel in Boston", "core_content": "book hotel"},
            {"id": "g2", "content": "Book flight to Boston", "core_content": "book flight"},
        ]
        state = lambda s1, s2: [{"goal_id": "g1", "goal_content": "book hotel", "status": s1},
                                {"goal_id": "g2", "goal_content": "book flight", "status": s2}]
        turns = [
            {"turn_id": 1, "speaker": "USER", "utterance": "I need a hotel in Boston.",
             "goal_status_changes": [{"goal_id": "g1", "new_status": "open"}], "all_goals": state("open", "not_mentioned")},
            {"turn_id": 2, "speaker": "SYSTEM", "utterance": "Searching hotels now.",
             "goal_status_changes": [{"goal_id": "g1", "new_status": "pending"}], "all_goals": state("pending", "not_mentioned")},
            {"turn_id": 3, "speaker": "USER", "utterance": "Also a flight please.",
             "goal_status_changes": [{"goal_id": "g2", "new_status": "open"}], "all_goals": state("pending", "open")},
            {"turn_id": 4, "speaker": "SYSTEM", "utterance": "Your hotel is confirmed. Looking at flights.",
             "goal_status_changes": [{"goal_id": "g1", "new_status": "completed"}, {"goal_id": "g2", "new_status": "pending"}],
             "all_goals": state("completed", "pending")},
        ]
        dialogue = {"dialogue_id": "d", "goal_list": goals, "turns": turns}
        proactive = find_proactive_turns(dialogue)
        # turn 2 reacts to the user's hotel request; turn 4 completes the hotel goal unprompted
        self.assertEqual([p["turn_id"] for p in proactive], [4])
        self.assertEqual(proactive[0]["goal_ids"], ["g1"])
        queries = status_change_queries(dialogue)
        self.assertEqual([(q["turn_index"], q["goal_id"]) for q in queries], [(0, "g1"), (1, "g1"), (2, "g2"), (3, "g1"), (3, "g2")])
        self.assertIn("Book hotel in Boston", queries[0]["query"])

    def test_quality_score_bounds(self):
        from ATODEval.turn_level_quality import _score

        self.assertEqual(_score("4"), 4.0)
        self.assertEqual(_score(9), 5.0)
        self.assertEqual(_score(0), 1.0)
        self.assertIsNone(_score("n/a"))


if __name__ == "__main__":
    unittest.main()
