# ATOD

ATOD is a benchmark and evaluation framework for agentic task-oriented dialogue systems. It accompanies the AACL 2026 paper:

> **ATOD: An Evaluation Framework and Benchmark for Agentic Task-Oriented Dialogue Systems**

The repository contains the fixed 1,000-dialogue ATOD benchmark and the code required to run ATOD-Eval.

## Repository contents

```text
data/                 Fixed ATOD benchmark and dataset documentation
MemSys/               Agentic symbolic-vector memory evaluator
evaluation/           End-of-dialogue ATOD evaluation
ATODEval/             Dependency, completion, memory, and quality metrics
scripts/              Dataset validation and statistics utilities
tests/                Offline smoke tests
```

The repository intentionally excludes the data-generation pipeline, baseline experiment harnesses, model weights, AWS credentials, generated result caches, the upstream Schema-Guided Dialogue dataset, and paper/review materials.

## Dataset

The release contains:

| Split | Dialogues |
|---|---:|
| Medium | 428 |
| Complex | 572 |
| Total | 1,000 |

Each file is a compact JSON array. Every dialogue contains a stable ID, a complexity label, a goal graph, dialogue turns, goal-status transitions, and complexity metadata. See [data/README.md](data/README.md) and [DATASET_CARD.md](DATASET_CARD.md).

Validate the release before use:

```bash
python scripts/validate_data.py
python scripts/compute_dataset_stats.py
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The evaluator and memory components invoke models through Amazon Bedrock using the standard `boto3` credential chain. No credentials are included in this repository.

The paper uses Claude-Sonnet-4 as the memory backbone. You may supply another supported Bedrock model ID through the command line or environment variables.

## Run a smoke evaluation

Configure AWS credentials and a region, then run a small sample:

```bash
export AWS_REGION="us-east-2"
python evaluation/evaluation_atod.py \
  --complexity medium \
  --max-samples 5 \
  --model-id "us.anthropic.claude-sonnet-4-20250514-v1:0" \
  --output-dir results/smoke
```

The full evaluation makes multiple model calls per dialogue and may incur substantial cost. Start with `--max-samples`.

## Reproducibility status

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## License

Unless otherwise noted, the code and ATOD data are released under the Creative Commons Attribution-NonCommercial 4.0 International license. Third-party materials retain their original licenses; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The proceedings entry should be updated when the final Anthology record becomes available.
