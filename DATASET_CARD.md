# Dataset Card for ATOD

## Dataset summary

ATOD is a synthetic benchmark for evaluating task-oriented dialogue systems under multi-goal, long-horizon, asynchronous, interleaved, dependency-aware, and proactive interactions. It contains 1,000 dialogues divided into 428 medium-complexity and 572 complex dialogues.

## Supported tasks

- Goal detection
- Goal lifecycle and status tracking
- Dependency-aware goal completion
- Memory recall
- Turn efficiency
- Proactivity and dialogue-quality evaluation

## Data composition

Each dialogue contains:

- `dialogue_id`: stable UUID
- `complexity_class`: `medium` or `complex`
- `dialogue_index`: generation-time index
- `goal_list`: goal content, domain, intent, slots, status, history, and dependencies
- `turns`: alternating user/system utterances with turn-level goal annotations
- `metadata`: complexity attributes used during generation

The complete structural specification is in `data/schema.json`.

## Creation process

The internal generation process extracts goal structures from the publicly available Schema-Guided Dialogue dataset, constructs goal co-occurrence graphs, samples goal trajectories, and uses an LLM to generate and annotate synthetic dialogues. Neither the generation pipeline nor the upstream dataset is redistributed here.

The released files are the fixed benchmark artifacts used for evaluation, not regenerated outputs. Random regeneration may differ because it depends on model availability and stochastic model behavior.

## Validation

Automated release validation checks:

- Expected split sizes and globally unique dialogue IDs
- Required fields and supported status values
- Goal-ID and dependency consistency
- Alternating user/system turns
- Absence of common credential patterns and Amazon-internal domains
- Release-file checksums

The paper additionally reports a single-author manual audit of evaluator decisions.

## Personal and sensitive information

The dialogues are synthetic and are not intended to describe real individuals. They may contain fictional names, businesses, locations, contact details, reservation details, or financial-task language as part of simulated task-oriented interactions. Users should not treat these strings as verified real-world information.

No credentials or Amazon-internal identifiers are intentionally included. Run `python scripts/validate_data.py` to repeat the automated scan.

## Recommended uses

ATOD is intended for academic research on task-oriented dialogue evaluation, memory, planning, goal tracking, and agentic behavior.

## Out-of-scope uses

- Production decision-making
- Evaluation involving real users without additional review
- Inferring personal attributes
- Treating synthetic entities or facts as real
- Safety-critical or financial decisions

## Limitations

- The benchmark is synthetic and may not capture the full linguistic and behavioral diversity of real dialogue.
- Generation and annotation can inherit biases or errors from the source ontology and LLM backbone.
- LLM-judged metrics depend on prompting, model family, and model version.
- Domain coverage is bounded by the generation ontology.
- The benchmark is a fixed research artifact and is not designed as a continuously updated dataset.

## Licensing

The ATOD release is covered by the repository license unless a file states otherwise. Upstream materials remain covered by their original licenses; see `THIRD_PARTY_LICENSES.md`.
