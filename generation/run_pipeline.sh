#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Run the six ATOD generation stages end to end.
#
# Required:
#   SGD_DIR        path to the Schema-Guided Dialogue "train" directory
#   ATOD_MODEL_ID  Bedrock model ID used for stages 4-6
#
# Optional (defaults in parentheses):
#   ATOD_WORK_DIR        output directory (generation/work)
#   NUM_TRAJECTORIES     trajectories to sample (1000)
#   MEDIUM_RATIO         share of 2-8 goal trajectories (0.65)
#   COMPLEX_RATIO        share of 7-12 goal trajectories (0.35)
#   ENABLE_JUDGES        run the trajectory and dialogue verifiers (true)
#   MAX_RETRIES          retry rounds for stages 4 and 5 (3)
#   ATOD_WORKERS         concurrent model calls (1)
#   SEED                 random seed for sampling (42)
#
# Example:
#   SGD_DIR=/path/to/dstc8-schema-guided-dialogue/train ATOD_MODEL_ID=<model-id> \
#     NUM_TRAJECTORIES=20 generation/run_pipeline.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${SGD_DIR:?Set SGD_DIR to the SGD train directory}"
: "${ATOD_MODEL_ID:?Set ATOD_MODEL_ID to a Bedrock model ID}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-1000}"
MEDIUM_RATIO="${MEDIUM_RATIO:-0.65}"
COMPLEX_RATIO="${COMPLEX_RATIO:-0.35}"
ENABLE_JUDGES="${ENABLE_JUDGES:-true}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SEED="${SEED:-42}"
export ATOD_WORK_DIR="${ATOD_WORK_DIR:-generation/work}"

JUDGE_FLAG=""
if [[ "$ENABLE_JUDGES" != "true" ]]; then
  JUDGE_FLAG="--disable-judge"
fi

echo "ATOD generation | trajectories=$NUM_TRAJECTORIES | verifiers=$ENABLE_JUDGES | work dir=$ATOD_WORK_DIR"

echo "[1/6] Extract goal sequences from SGD"
python -m generation.extract_goals --dialogues-dir "$SGD_DIR"

echo "[2/6] Build co-occurrence graph"
python -m generation.build_cooccurrence_graph

echo "[3/6] Sample goal trajectories"
python -m generation.sample_trajectories --num-trajectories "$NUM_TRAJECTORIES" \
  --medium-ratio "$MEDIUM_RATIO" --complex-ratio "$COMPLEX_RATIO" --seed "$SEED"

echo "[4/6] Annotate trajectories and assign complexity"
python -m generation.annotate_trajectories --max-retries "$MAX_RETRIES" --seed "$SEED" $JUDGE_FLAG

echo "[5/6] Generate dialogues"
python -m generation.generate_dialogues --max-retries "$MAX_RETRIES" $JUDGE_FLAG

echo "[6/6] Annotate turn-level goal status"
python -m generation.annotate_dialogue_status

echo "Done. Final dialogues: $ATOD_WORK_DIR/final/{medium,complex}/annotated_dialogues.json"
