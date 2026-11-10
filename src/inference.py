"""Pure-NumPy inference for the two trained encoder-decoder models.

Why NumPy instead of TensorFlow?  The notebook's models are tiny recurrent nets
(one SimpleRNN / LSTM layer per side) whose forward pass is a few matrix products.
Re-implementing it removes the ~600 MB TensorFlow dependency, so the app starts fast
and fits comfortably in Streamlit Community Cloud's memory limits.
`tests/test_numpy_matches_keras.py` checks that this code reproduces Keras exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import (
    EOS_TOKEN, MAX_ARTICLE_LEN, MAX_TITLE_LEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN,
    clean_arabic, tokenize_text,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CELLS = ("rnn", "lstm")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Vocab:
    id_to_word: list[str]

    def __post_init__(self) -> None:
        self.word_to_id = {w: i for i, w in enumerate(self.id_to_word)}
        self.pad = self.word_to_id[PAD_TOKEN]
        self.unk = self.word_to_id[UNK_TOKEN]
        self.sos = self.word_to_id[SOS_TOKEN]
        self.eos = self.word_to_id[EOS_TOKEN]

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["id_to_word"])

    def encode_article(self, text: str, max_len: int = MAX_ARTICLE_LEN) -> list[int]:
        tokens = tokenize_text(clean_arabic(text))
        return [self.word_to_id.get(t, self.unk) for t in tokens][:max_len]

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            w = self.id_to_word[int(i)] if 0 <= int(i) < len(self.id_to_word) else UNK_TOKEN
            if w in (PAD_TOKEN, SOS_TOKEN):
                continue
            if w == EOS_TOKEN:
                break
            words.append(w)
        return " ".join(words)


class HeadlineGenerator:
    """Greedy-decoding headline generator for one model ('rnn' or 'lstm')."""

    def __init__(self, vocab: Vocab, weights: dict[str, np.ndarray], cell: str):
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}")
        self.vocab, self.cell = vocab, cell
        w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.enc_emb, self.dec_emb = w["enc_emb"], w["dec_emb"]
        self.enc_k, self.enc_r, self.enc_b = w["enc_kernel"], w["enc_recurrent"], w["enc_bias"]
        self.dec_k, self.dec_r, self.dec_b = w["dec_kernel"], w["dec_recurrent"], w["dec_bias"]
        self.out_k, self.out_b = w["out_kernel"], w["out_bias"]
        self.units = self.enc_r.shape[0]

    @classmethod
    def from_artifacts(cls, cell: str, directory: Path = ARTIFACT_DIR) -> "HeadlineGenerator":
        vocab = Vocab.load(directory / "vocab.json")
        with np.load(directory / f"{cell}_weights.npz") as z:
            weights = {k: z[k] for k in z.files}
        return cls(vocab, weights, cell)

    # -- recurrent cells ----------------------------------------------------------
    def _step(self, x, h, c, k, r, b):
        z = x @ k + h @ r + b
        if self.cell == "rnn":
            return np.tanh(z), None
        u = self.units
        i, f, g, o = (_sigmoid(z[:u]), _sigmoid(z[u:2 * u]), np.tanh(z[2 * u:3 * u]), _sigmoid(z[3 * u:]))
        c = f * c + i * g
        return o * np.tanh(c), c

    def encode(self, article: str):
        """Return the encoder's final state (h, c). Padding is masked like Keras mask_zero."""
        ids = self.vocab.encode_article(article)
        h = np.zeros(self.units, dtype=np.float32)
        c = np.zeros(self.units, dtype=np.float32)
        for t in ids:
            h, c_new = self._step(self.enc_emb[t], h, c, self.enc_k, self.enc_r, self.enc_b)
            if c_new is not None:
                c = c_new
        return h, c, ids

    def generate(self, article: str, max_len: int = MAX_TITLE_LEN - 1, avoid_repeats: bool = False) -> str:
        """Greedy decoding exactly as in the notebook (optionally blocking repeated words)."""
        h, c, _ = self.encode(article)
        token, out = self.vocab.sos, []
        for _ in range(max_len):
            h, c_new = self._step(self.dec_emb[token], h, c, self.dec_k, self.dec_r, self.dec_b)
            if c_new is not None:
                c = c_new
            logits = h @ self.out_k + self.out_b
            if avoid_repeats and out:
                logits[out] = -np.inf
            token = int(np.argmax(logits))
            if token in (self.vocab.eos, self.vocab.pad):
                break
            out.append(token)
        return self.vocab.decode(out)


def artifacts_available(directory: Path = ARTIFACT_DIR) -> bool:
    return (directory / "vocab.json").exists() and all(
        (directory / f"{c}_weights.npz").exists() for c in CELLS
    )
