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
        completed_turns, completed_goals = compute_ntc_for_dialogue(goals)
        self.assertEqual(completed_goals, 2)
        self.assertGreaterEqual(completed_turns, 0)


if __name__ == "__main__":
    unittest.main()
