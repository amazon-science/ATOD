# ATOD

ATOD is a benchmark and evaluation framework for agentic task-oriented dialogue systems. It accompanies the AACL 2026 paper:

> [**ATOD: An Evaluation Framework and Benchmark for Agentic Task-Oriented Dialogue Systems**](https://arxiv.org/abs/2601.11854)

The repository contains the fixed 1,000-dialogue ATOD benchmark, the pipeline that generated it, and the code required to run ATOD-Eval.

## Repository contents

```text
data/                 Fixed ATOD benchmark and dataset documentation
generation/           Six-stage synthetic dialogue generation pipeline
MemSys/               Agentic symbolic-vector memory evaluator
evaluation/           End-of-dialogue ATOD evaluation
ATODEval/             Dependency, completion, memory, and quality metrics
scripts/              Dataset validation and statistics utilities
tests/                Offline unit tests
```

The repository excludes baseline experiment harnesses, model weights, credentials, generated result caches, the upstream Schema-Guided Dialogue dataset, and paper/review materials.

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

The released evaluator uses the Bedrock runtime. Configure credentials and a
region through the standard AWS SDK settings, then provide a compatible model
ID either through `ATOD_MODEL_ID` or the `--model-id` option:

```bash
export ATOD_MODEL_ID="<model-id>"
```

The exact model configuration used for the reported experiments is described
in the paper.

## Regenerate dialogues (optional)

`generation/` contains the pipeline that produced the benchmark: goal extraction from the Schema-Guided Dialogue dataset, co-occurrence graph construction, random-walk trajectory sampling, model-based trajectory annotation and complexity labelling, dialogue generation with verifiers, and turn-level goal status annotation. Re-running it produces a new synthetic set, not the released data. See [generation/README.md](generation/README.md).

```bash
SGD_DIR=/path/to/dstc8-schema-guided-dialogue/train NUM_TRAJECTORIES=20 \
  generation/run_pipeline.sh
```

## Run a smoke evaluation

After configuring model access, run a small sample:

```bash
python evaluation/evaluation_atod.py \
  --max-samples 5
```

The full evaluation makes multiple model calls per dialogue and may incur substantial cost. Start with `--max-samples`.

Offline tests (no model access needed):

```bash
python -m unittest discover -s tests
```

## Reproducibility status

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## License

Unless otherwise noted, the code and ATOD data are released under the Creative Commons Attribution-NonCommercial 4.0 International license. Third-party materials retain their original licenses; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Citation

If you use ATOD or ATOD-Eval, please cite:

```bibtex
@article{zhang2026atod,
  title={ATOD: An Evaluation Framework and Benchmark for Agentic Task-Oriented Dialogue Systems},
  author={Zhang, Yifei and Nayyeri, Hooshang and Khaziev, Rinat and Yilmaz, Emine and Tur, Gokhan and Hakkani-T{\"u}r, Dilek and Thadakamalla, Hari P},
  journal={arXiv preprint arXiv:2601.11854},
  year={2026}
}
```

The paper is available on [arXiv](https://arxiv.org/abs/2601.11854). Citation metadata is also provided in [CITATION.cff](CITATION.cff) and will be updated when the final Anthology record becomes available.
