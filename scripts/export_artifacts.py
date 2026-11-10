"""Export the trained Keras layers from the notebook into the files the Streamlit app needs.

The notebook itself is NOT modified. After running NLP2.ipynb in Colab, open a *new,
unsaved* cell (or a second notebook attached to the same runtime) and run the contents of
`scripts/colab_export_snippet.py`, which calls `export_artifacts` with the notebook's layer objects.

Outputs (float16 on disk, so each model stays below GitHub's 100 MB file limit):
    artifacts/vocab.json        id -> word list
    artifacts/rnn_weights.npz   SimpleRNN encoder-decoder weights
    artifacts/lstm_weights.npz  LSTM encoder-decoder weights
"""
import json
import os

import numpy as np


def _layer_weights(enc_emb, enc_rnn, dec_emb, dec_rnn, dense):
    ek, er, eb = enc_rnn.get_weights()
    dk, dr, db = dec_rnn.get_weights()
    ok, ob = dense.get_weights()
    return dict(
        enc_emb=enc_emb.get_weights()[0], enc_kernel=ek, enc_recurrent=er, enc_bias=eb,
        dec_emb=dec_emb.get_weights()[0], dec_kernel=dk, dec_recurrent=dr, dec_bias=db,
        out_kernel=ok, out_bias=ob,
    )


def export_artifacts(out_dir, id_to_word, rnn, lstm, dtype="float16"):
    """rnn / lstm: dicts with keys enc_emb, enc_rnn, dec_emb, dec_rnn, dense (Keras layer objects)."""
    os.makedirs(out_dir, exist_ok=True)
    words = [id_to_word[i] for i in range(len(id_to_word))]
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"id_to_word": words}, f, ensure_ascii=False)
    for name, layers in (("rnn", rnn), ("lstm", lstm)):
        w = {k: v.astype(dtype) for k, v in _layer_weights(**layers).items()}
        path = os.path.join(out_dir, f"{name}_weights.npz")
        np.savez_compressed(path, **w)
        print(f"saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print("done - commit the artifacts/ folder to GitHub and redeploy the app")
