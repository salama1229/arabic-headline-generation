"""With the real weights in artifacts/, generation must reproduce the headlines recorded in the notebook."""
import json
from pathlib import Path

import pytest

from src.inference import HeadlineGenerator, artifacts_available

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not artifacts_available(), reason="artifacts/ not present")

SAMPLES = json.loads((ROOT / "data" / "sample_predictions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("cell,key", [("rnn", "simplernn"), ("lstm", "lstm")])
def test_matches_notebook_outputs(cell, key):
    gen = HeadlineGenerator.from_artifacts(cell)
    for s in SAMPLES:
        assert gen.generate(s["article"]) == s[key], f"article {s['id']}"
