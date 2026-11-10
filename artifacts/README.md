# artifacts/

Put the exported model files here so the Streamlit app can generate headlines live:

```
artifacts/
├── vocab.json          id -> word list (92,121 entries)
├── rnn_weights.npz     SimpleRNN encoder-decoder (float16)
└── lstm_weights.npz    LSTM encoder-decoder (float16)
```

Create them with `scripts/colab_export_snippet.py` after running `notebooks/NLP2.ipynb`. See `docs/DEPLOYMENT.md`.
Each `.npz` is about 70 MB, below GitHub's 100 MB per-file limit (no Git LFS needed). Avoid Git LFS: it is not reliably supported on Streamlit Community Cloud.
