# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Shared helpers for the ATOD generation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = REPO_ROOT / "generation" / "work"

STATUSES = ("not_mentioned", "open", "pending", "completed", "failed", "abandoned")

# Valid lifecycle transitions. Terminal states never change.
VALID_TRANSITIONS = {
    "not_mentioned": ["open", "pending", "completed", "failed", "abandoned"],
    "open": ["pending", "completed", "failed", "abandoned"],
    "pending": ["completed", "failed", "abandoned"],
    "completed": [],
    "failed": [],
    "abandoned": [],
}

T = TypeVar("T")
R = TypeVar("R")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def require_file(path: Path, hint: str) -> Path:
    path = Path(path)
    if not path.exists():
        sys.exit(f"Error: {path} not found. {hint}")
    return path


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def add_work_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("ATOD_WORK_DIR", DEFAULT_WORK_DIR)),
        help="Directory holding intermediate and final pipeline outputs "
        "(default: generation/work or $ATOD_WORK_DIR).",
    )


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ATOD_MODEL_ID"),
        help="Bedrock model ID. Defaults to the ATOD_MODEL_ID environment variable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("ATOD_WORKERS", "1")),
        help="Number of concurrent model calls (default: 1).",
    )


def resolve_model_id_or_exit(parser: argparse.ArgumentParser, model_id: Optional[str]) -> str:
    if not model_id:
        parser.error("provide --model-id or set ATOD_MODEL_ID")
    return model_id


def make_llm_client(model_id: str):
    """Create the shared Bedrock client used by the evaluator as well."""
    # Imported lazily so that the offline stages do not require boto3.
    from MemSys.utils.llm_controller import LLMClient

    return LLMClient(model_id)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def parallel_map(
    items: Sequence[T],
    fn: Callable[[T], R],
    workers: int = 1,
    desc: str = "",
) -> List[R]:
    """Apply ``fn`` to every item, preserving input order.

    ``workers <= 1`` runs sequentially. Progress is shown with tqdm when available.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is in requirements.txt
        def tqdm(iterable, **_kwargs):  # type: ignore
            return iterable

    if workers <= 1:
        return [fn(item) for item in tqdm(items, desc=desc)]

    results: List[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        for future in tqdm(futures, desc=desc, total=len(futures)):
            results[futures[future]] = future.result()
    return results  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Text / JSON parsing
# --------------------------------------------------------------------------- #
_PLACEHOLDER_PATTERNS = [
    r"^\[.*\]$",
    r"^\{.*\}$",
    r"^placeholder",
    r"^example",
    r"^sample",
    r"^xxx+$",
    r"^\.\.\.+$",
    r"^tbd$",
    r"^n/?a$",
    r"^unknown$",
    r"^null$",
    r"^none$",
]
_PLACEHOLDER_PHRASES = {
    "[location]",
    "[date]",
    "[time]",
    "[name]",
    "[value]",
    "your location",
    "your name",
    "your email",
    "enter here",
}


def is_placeholder_value(value) -> bool:
    """Return True when a slot value looks like an unfilled template value."""
    if not isinstance(value, str) or not value.strip():
        return True
    stripped = value.strip()
    lowered = stripped.lower()
    if stripped.isdigit():
        return False
    if len(stripped) == 1:
        return not (stripped.isalnum() or stripped in {"$", "%", "#", "*", "+", "-"})
    if any(re.match(pattern, lowered) for pattern in _PLACEHOLDER_PATTERNS):
        return True
    return lowered in _PLACEHOLDER_PHRASES


def first_json_object(text: str):
    """Parse the first ``{...}`` block in a model response, or raise ValueError."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group())


def one_word_verdict(response: str, positive: str = "PASS") -> bool:
    """Interpret a PASS/FAIL style verdict."""
    return positive.upper() in response.upper()


def format_dialogue(turns: Iterable[dict]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['utterance']}" for turn in turns)
