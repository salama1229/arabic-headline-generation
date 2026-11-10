# Deployment guide

The repository already contains the trained weights in `artifacts/`, so the app can be deployed as is (section 2). Section 1 is only needed if you retrain the models.

## 1. Re-export the models (only after retraining)

`notebooks/NLP2.ipynb` trains the models in memory and does not save them. The notebook is left unchanged; use a separate cell.

1. Open the notebook in the runtime you trained it in (for example Colab) and run it to the end (the recorded run took about 24 minutes per model).
2. Add a **new cell at the bottom** (do not save it into the notebook) and paste the contents of [`scripts/colab_export_snippet.py`](../scripts/colab_export_snippet.py). The snippet is self-contained: nothing to download or edit.
3. Run it. You get:

   ```
   artifacts/vocab.json
   artifacts/rnn_weights.npz     (~70 MB)
   artifacts/lstm_weights.npz    (~70 MB)
   ```
4. On Colab the snippet also zips the folder and downloads `artifacts.zip` automatically.
5. Unzip into the `artifacts/` folder of this repository and push:

   ```bash
   git add artifacts
   git commit -m "Add trained model weights"
   git push
   ```

Each weight file stays below GitHub's 100 MB limit because weights are stored as float16 (the app converts them back to float32; the effect on the output is negligible). Do not use Git LFS.

### Why the app does not need TensorFlow

`src/inference.py` re-implements the two networks (embedding, SimpleRNN/LSTM step, dense output, greedy decoding) in NumPy. `tests/test_numpy_matches_keras.py` builds miniature Keras models with the same architecture code as the notebook and checks that both implementations generate identical headlines, for articles shorter and longer than the 80-token window and for both cell types:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 2. Deploy on Streamlit Community Cloud

Prerequisites: the repository is on GitHub and public, and your GitHub account is connected to [share.streamlit.io](https://share.streamlit.io).

1. Open share.streamlit.io and select **Create app** → **Deploy a public app from GitHub**.
2. Fill in:
   - Repository: `<your-user>/arabic-headline-generation`
   - Branch: `main`
   - Main file path: `app.py`
3. Open **Advanced settings** and choose Python 3.11 or newer.
4. Select **Deploy**. The first start installs `requirements.txt` (about a minute).
5. Optional: choose a custom sub-domain in the app settings, then add the URL to the top of `README.md`.

Pre-filled form link (replace the user name if needed):
`https://share.streamlit.io/deploy?repository=salama1229/arabic-headline-generation&branch=main&mainModule=app.py`

### Behaviour

| State | What the app does |
|---|---|
| `artifacts/` missing | The Generate tab fails when it loads the models; the other tabs still work. Restore the files in `artifacts/`. |
| `artifacts/` present (default) | Runs in **live mode**: paste any Arabic article and both models generate a headline. Models are loaded once and cached. |

### Troubleshooting

- *App sleeps after inactivity:* normal on the free tier; the first visit wakes it up.
- *Out of memory:* both float16 files expand to ~290 MB of float32 in memory, which fits the free tier. If it ever does not, load only one model in `load_generators()`.
- *Arabic text shows as boxes:* the app loads IBM Plex Sans Arabic from Google Fonts; the browser must be able to reach it. Fallback fonts are used otherwise.
