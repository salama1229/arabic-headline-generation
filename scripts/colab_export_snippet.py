# ======================================================================================
# Paste this WHOLE cell into a NEW cell at the bottom of the notebook runtime where you
# just ran NLP2.ipynb (all 55 cells). Do not save it into the notebook itself.
# It creates the files the Streamlit app needs and downloads them as artifacts.zip.
# ======================================================================================
import json, os
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
    os.makedirs(out_dir, exist_ok=True)
    words = [id_to_word[i] for i in range(len(id_to_word))]
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"id_to_word": words}, f, ensure_ascii=False)
    for name, layers in (("rnn", rnn), ("lstm", lstm)):
        w = {k: v.astype(dtype) for k, v in _layer_weights(**layers).items()}
        path = os.path.join(out_dir, f"{name}_weights.npz")
        np.savez_compressed(path, **w)
        print(f"saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


# These names all come from the notebook cells (25, 26, 36, 38, 39, 10)
export_artifacts(
    "artifacts", id_to_word,
    rnn=dict(enc_emb=encoder_embedding, enc_rnn=encoder_rnn,
             dec_emb=decoder_embedding, dec_rnn=decoder_rnn, dense=decoder_dense),
    lstm=dict(enc_emb=encoder_embedding_lstm, enc_rnn=encoder_lstm,
              dec_emb=decoder_embedding_lstm, dec_rnn=decoder_lstm, dense=decoder_dense_lstm),
)

# Zip and download (Colab). Elsewhere, just copy the artifacts/ folder.
try:
    import shutil
    from google.colab import files
    shutil.make_archive("artifacts", "zip", ".", "artifacts")
    files.download("artifacts.zip")
except ImportError:
    print("Not on Colab: copy the artifacts/ folder manually.")
