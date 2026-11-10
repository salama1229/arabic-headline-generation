# artifacts/

Trained model files used by the Streamlit app:

```
artifacts/
├── vocab.json          id -> word list (92,121 entries)
├── rnn_weights.npz     SimpleRNN encoder-decoder (float16, ~60 MB)
└── lstm_weights.npz    LSTM encoder-decoder (float16, ~60 MB)
```

They come from the trained Keras weights of the 20-epoch, dropout-0.1 notebook runs (BLEU-1 0.0587 / 0.0736 on the test set). The weights were converted to NumPy arrays and stored as float16; generation from these files reproduces all 20 recorded notebook headlines for the ten test articles (`tests/test_recorded_outputs.py`).

To replace them after retraining, use `scripts/colab_export_snippet.py` (see `docs/DEPLOYMENT.md`). Each file must stay below GitHub's 100 MB limit. Avoid Git LFS: it is not reliably supported on Streamlit Community Cloud.
