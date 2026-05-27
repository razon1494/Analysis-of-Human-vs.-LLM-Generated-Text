"""I/O helpers — single source of truth for jsonl loading and label coding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


LABEL_LLM = 1
LABEL_HUMAN = 0


def label_to_int(label: str) -> int:
    return LABEL_LLM if label == "llm" else LABEL_HUMAN


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_xy(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    X = [r["text"] for r in rows]
    y = np.array([label_to_int(r["label"]) for r in rows], dtype=int)
    return X, y


def rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def load_p0(path: str | Path) -> list[dict]:
    return load_jsonl(path)


def load_test_ids(path: str | Path) -> set[str]:
    return set(Path(path).read_text(encoding="utf-8").splitlines())
