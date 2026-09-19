# Third-Party Materials

## Schema-Guided Dialogue Dataset

- Project: Schema-Guided Dialogue Dataset
- Authors: Abhinav Rastogi, Xiaoxue Zang, Srinivas Sunkara, Raghav Gupta, and Pranav Khaitan
- Paper: “Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset”
- Source: `https://github.com/google-research-datasets/dstc8-schema-guided-dialogue`
- License: Creative Commons Attribution-ShareAlike 4.0 International
- Use in ATOD: The generation pipeline (`generation/extract_goals.py`) reads the upstream dataset to extract goal sequences and build the co-occurrence graph.
- Distribution: The upstream dataset and its Git history are not included in this repository. Users who re-run the generation pipeline must obtain it from the source above under its own license.

The released evaluator and benchmark data do not require the upstream dataset.

## Python dependencies

The Python packages listed in `requirements.txt` are referenced as dependencies and are not redistributed in this repository. Each package remains governed by its own license.
