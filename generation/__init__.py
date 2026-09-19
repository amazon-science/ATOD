# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""ATOD synthetic dialogue generation pipeline.

Stages (see README.md in this directory):

1. ``extract_goals``            -- goal sequences from Schema-Guided Dialogue
2. ``build_cooccurrence_graph`` -- weighted goal co-occurrence graph
3. ``sample_trajectories``      -- random-walk trajectory sampling
4. ``annotate_trajectories``    -- LLM slot/dependency annotation + complexity label
5. ``generate_dialogues``       -- LLM dialogue generation (+ verifier)
6. ``annotate_dialogue_status`` -- turn-level goal status annotation
"""
