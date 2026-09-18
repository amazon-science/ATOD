# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseDataTests(unittest.TestCase):
    def test_release_split_counts_and_ids(self):
        counts = {"medium": 428, "complex": 572}
        ids = []
        for complexity, expected in counts.items():
            path = ROOT / "data" / complexity / "annotated_dialogues.json"
            dialogues = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(dialogues), expected)
            self.assertTrue(
                all(d["complexity_class"] == complexity for d in dialogues)
            )
            ids.extend(d["dialogue_id"] for d in dialogues)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 1000)


if __name__ == "__main__":
    unittest.main()
