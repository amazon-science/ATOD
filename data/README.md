# ATOD Data

## Files

```text
medium/annotated_dialogues.json   428 dialogues
complex/annotated_dialogues.json  572 dialogues
schema.json                       Structural schema
manifest.json                     Counts, sizes, and SHA-256 checksums
```

The dialogue files are compact JSON arrays. They preserve the object order, fields, and values of the fixed evaluation data.

```python
import json
from pathlib import Path

dialogues = json.loads(
    Path("data/medium/annotated_dialogues.json").read_text(encoding="utf-8")
)
```

Run the validator after cloning or transferring the repository:

```bash
python scripts/validate_data.py
```

`turns` contains individual speaker utterances, not user/system exchange pairs. Statistics scripts report both utterance records and complete user/system exchanges to avoid ambiguity.

The upstream Schema-Guided Dialogue dataset is not included. See `THIRD_PARTY_LICENSES.md`.
