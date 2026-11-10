"""Verifies the NumPy inference engine reproduces the notebook's Keras inference models.

Builds miniature models with the *same architecture code as the notebook* (small vocab/units),
exports them with scripts/export_artifacts.py and compares encoder states and greedy outputs.
Requires tensorflow (dev-only):  pip install tensorflow-cpu pytest
"""
import sys, tempfile
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, SimpleRNN  # noqa: E402
from tensorflow.keras.models import Model  # noqa: E402

from export_artifacts import export_artifacts  # noqa: E402
from src.inference import HeadlineGenerator  # noqa: E402

V, E, U, MA, MT = 60, 8, 8, 80, 20
WORDS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"] + [f"w{i}" for i in range(V - 4)]


def _randomise(model, seed):
    rng = np.random.default_rng(seed)
    model.set_weights([rng.normal(0, 0.6, w.shape).astype("float32") for w in model.get_weights()])


def build(cell):
    cls = SimpleRNN if cell == "rnn" else LSTM
    enc_in = Input(shape=(MA,)); enc_emb = Embedding(V, E, mask_zero=True)
    enc_rnn = cls(U, return_state=True, dropout=0.1)
    enc_out = enc_rnn(enc_emb(enc_in))
    state = list(enc_out[1:])
    dec_in = Input(shape=(MT - 1,)); dec_emb = Embedding(V, E, mask_zero=True)
    dec_rnn = cls(U, return_sequences=True, return_state=True, dropout=0.1)
    dense = Dense(V, activation="softmax")
    dec_out = dec_rnn(dec_emb(dec_in), initial_state=state if cell == "lstm" else state[0])
    full = Model([enc_in, dec_in], dense(dec_out[0]))
    _randomise(full, 7 if cell == "rnn" else 8)
    enc_model = Model(enc_in, state if cell == "lstm" else state[0])
    tok = Input(shape=(1,))
    if cell == "rnn":
        s_in = Input(shape=(U,)); o = dec_rnn(dec_emb(tok), initial_state=s_in)
        dec_model = Model([tok, s_in], [dense(o[0]), o[1]])
    else:
        h_in, c_in = Input(shape=(U,)), Input(shape=(U,)); o = dec_rnn(dec_emb(tok), initial_state=[h_in, c_in])
        dec_model = Model([tok, h_in, c_in], [dense(o[0]), o[1], o[2]])
    layers = dict(enc_emb=enc_emb, enc_rnn=enc_rnn, dec_emb=dec_emb, dec_rnn=dec_rnn, dense=dense)
    return enc_model, dec_model, layers


def keras_generate(cell, enc_model, dec_model, ids):
    x = np.zeros((1, MA), dtype="int32"); x[0, :len(ids)] = ids          # post-padding
    st = enc_model.predict(x, verbose=0)
    st = [st] if cell == "rnn" else list(st)
    tok, out, logits_seq = np.array([[2]]), [], []
    for _ in range(MT - 1):
        res = dec_model.predict([tok] + st, verbose=0)
        probs, st = res[0], list(res[1:])
        nxt = int(np.argmax(probs[0, -1]))
        logits_seq.append(probs[0, -1])
        if nxt in (3, 0):
            break
        out.append(nxt); tok = np.array([[nxt]])
    return out, np.array(logits_seq)


@pytest.mark.parametrize("cell", ["rnn", "lstm"])
@pytest.mark.parametrize("n_tokens", [3, 25, 80, 120])
def test_numpy_equals_keras(cell, n_tokens, tmp_path):
    enc_model, dec_model, layers = build(cell)
    export_artifacts(tmp_path, WORDS, rnn=layers if cell == "rnn" else _dummy(layers),
                     lstm=layers if cell == "lstm" else _dummy(layers), dtype="float32")
    gen = HeadlineGenerator.from_artifacts(cell, tmp_path)
    rng = np.random.default_rng(n_tokens)
    text = " ".join(WORDS[i] for i in rng.integers(4, V, n_tokens))
    ids = gen.vocab.encode_article(text)
    assert len(ids) == min(n_tokens, MA)
    ref_ids, _ = keras_generate(cell, enc_model, dec_model, ids)
    assert gen.generate(text) == gen.vocab.decode(ref_ids)


def _dummy(layers):  # the other model is not under test, reuse layers so export succeeds
    return layers
