# ATOD generation pipeline

This directory contains the pipeline that produced the ATOD benchmark. It is
released so that the construction procedure described in the paper can be
inspected and re-run. The released benchmark under `data/` is the fixed
artifact used in the paper; re-running the pipeline produces a *new* synthetic
dataset because stages 4–6 depend on model sampling.

## Stages

| Stage | Module | Model calls | Input → output (inside the work directory) |
|---|---|---|---|
| 1 | `extract_goals` | no | SGD `train/` → `extracted_goals.json` |
| 2 | `build_cooccurrence_graph` | no | → `cooccurrence_graph.json`, `cooccurrence_graph_stats.json` |
| 3 | `sample_trajectories` | no | → `sampled_goal_trajectories.json` |
| 4 | `annotate_trajectories` | yes | → `annotated_goal_trajectories.json` |
| 5 | `generate_dialogues` | yes | → `synthetic_dialogues.json` |
| 6 | `annotate_dialogue_status` | yes | → `annotated_dialogues.json`, `final/{medium,complex}/annotated_dialogues.json` |
| – | `quality_rating` | yes | post-hoc 1–5 quality ratings of a dialogue set |

Stage details:

1. **Goal extraction.** Each SGD dialogue yields the ordered list of unique
   `(domain, intent)` pairs found in its user frames (`Hotels_1 → Hotels`).
   Sequences with fewer than two goals are dropped. The released benchmark used
   SGD train files `dialogues_044` … `dialogues_127`.
2. **Co-occurrence graph.** Nodes are goals; edge weights count the seed
   sequences in which two goals co-occur.
3. **Trajectory sampling.** A goal count is drawn from the 2–8 range (65 %) or
   the 7–12 range (35 %); a weighted random walk with exploration factor 0.3
   produces the goal sequence. Complexity is *not* assigned here.
4. **Trajectory annotation.** For each goal the model proposes slots, realistic
   slot values, `content` and `core_content`. The turn estimate is set to three
   turns per goal, the complexity label is assigned by the rule-based scorer
   (goal count, domain count, turn estimate, dependency count; a model-based
   classifier is the fallback when the scorer yields no label), agentic flags
   are set from the label, and inter-goal dependencies are proposed by the model
   for trajectories with more than three goals. Annotations with placeholder
   values are rejected; with verifiers enabled a PASS/FAIL trajectory judge is
   applied as well.
5. **Dialogue generation.** The annotated trajectory is rendered into the
   structured generation prompt (paper Appendix). The response is split into
   alternating `USER`/`SYSTEM` turns; dialogues with fewer than four turns are
   discarded. With verifiers enabled a PASS/FAIL dialogue judge screens each
   dialogue.
6. **Status annotation.** Turns are processed sequentially; the model returns
   the status of every goal after the current turn. Transitions are validated
   against `not_mentioned → open → pending → {completed, failed, abandoned}`
   (terminal states never change; backwards transitions are rejected).

Prompts are reproduced verbatim from the pipeline runs that produced the
released data. Retry handling was simplified: failed items are retried on their
own instead of regenerating the full batch.

## Requirements

* Python 3.10+, packages from `requirements.txt` (only `boto3`/`tqdm` are used
  here; stages 1–3 need no third-party packages).
* A local copy of the Schema-Guided Dialogue dataset
  (<https://github.com/google-research-datasets/dstc8-schema-guided-dialogue>,
  CC BY-SA 4.0). It is **not** redistributed in this repository.
* Model access through the Bedrock runtime for stages 4–6. Configure AWS
  credentials and a region through the standard SDK settings and pass a model
  ID with `ATOD_MODEL_ID` or `--model-id`. The model used for the released data
  is stated in the paper.

## Running

```bash
export ATOD_MODEL_ID="<model-id>"
SGD_DIR=/path/to/dstc8-schema-guided-dialogue/train NUM_TRAJECTORIES=20 \
  generation/run_pipeline.sh
```

or stage by stage from the repository root:

```bash
python -m generation.extract_goals --dialogues-dir /path/to/sgd/train
python -m generation.build_cooccurrence_graph
python -m generation.sample_trajectories --num-trajectories 1000 --seed 42
python -m generation.annotate_trajectories            # add --disable-judge to skip the verifier
python -m generation.generate_dialogues               # add --disable-judge to skip the verifier
python -m generation.annotate_dialogue_status
python -m generation.quality_rating --data-dir generation/work/final --max-dialogues 20
```

All stages accept `--work-dir` (default `generation/work`, ignored by Git) and
the model-backed stages accept `--workers N` for concurrent calls. Start with a
small `NUM_TRAJECTORIES`: stages 4–6 make several model calls per goal and per
turn.

## Output format

Stage 6 writes dialogues in the schema of the released benchmark
(`data/schema.json`). Validate a generated set with

```bash
python scripts/validate_data.py --data-dir generation/work/final --skip-counts
```

## Offline tests

```bash
python -m unittest tests.test_generation
```

The tests exercise stages 1–3 on a tiny synthetic SGD-like fixture and the
response parsers / lifecycle validation used by stages 4–6, without model calls.
