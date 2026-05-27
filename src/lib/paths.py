"""Centralized paths so every script can import them rather than re-deriving."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SPLITS = DATA / "splits"

P0_PATH = DATA / "p0" / "p0.jsonl"
TEST_IDS = SPLITS / "test_ids.txt"
TRAIN_IDS = SPLITS / "train_ids.txt"
VAL_IDS = SPLITS / "val_ids.txt"

PARAPHRASE_PATHS = {
    "P0_test":            P0_PATH,                                       # special — use test_ids subset
    "P1_test_standard":   DATA / "p1" / "p1_test.jsonl",
    "P2_test_standard":   DATA / "p2" / "p2_test.jsonl",
    "P1_test_simplified": DATA / "p1" / "p1_test_simplified.jsonl",
    "P2_test_simplified": DATA / "p2" / "p2_test_simplified.jsonl",
    # P6 — paraphraser diversity tracks
    "P1_test_nllb":       DATA / "p1" / "p1_test_nllb.jsonl",
    "P2_test_nllb":       DATA / "p2" / "p2_test_nllb.jsonl",
    "P1_test_mistral":              DATA / "p1" / "p1_test_mistral.jsonl",
    "P2_test_mistral":              DATA / "p2" / "p2_test_mistral.jsonl",
    "P1_test_mistral_simplified":   DATA / "p1" / "p1_test_mistral_simplified.jsonl",
    "P2_test_mistral_simplified":   DATA / "p2" / "p2_test_mistral_simplified.jsonl",
}

# Conditioning for plots and tables
STANDARD_TRACK           = ["P0_test", "P1_test_standard",            "P2_test_standard"]
SIMPLIFIED_TRACK         = ["P0_test", "P1_test_simplified",          "P2_test_simplified"]
NLLB_TRACK               = ["P0_test", "P1_test_nllb",                "P2_test_nllb"]
MISTRAL_STANDARD_TRACK   = ["P0_test", "P1_test_mistral",             "P2_test_mistral"]
MISTRAL_SIMPLIFIED_TRACK = ["P0_test", "P1_test_mistral_simplified",  "P2_test_mistral_simplified"]

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
