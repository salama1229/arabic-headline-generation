"""Arabic Headline Generator - Streamlit app.

Compares a SimpleRNN and an LSTM encoder-decoder trained on AraFinNews (see notebooks/NLP2.ipynb).
Team: Ali Omar, Shaza Abdulnasser, Zizi Mostafa - Badr University in Assiut.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import HeadlineGenerator, artifacts_available
from src.ui import CSS, INK, LSTM_COLOR, MUTED, REF_COLOR, RNN_COLOR, RULE, esc, headline_row, tag

ROOT = Path(__file__).parent
st.set_page_config(page_title="Arabic Headline Generator", page_icon="📰", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- data ----------
@st.cache_data
def load_json(name: str):
    with open(ROOT / "data" / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading models...")
def load_generators():
    return {c: HeadlineGenerator.from_artifacts(c) for c in ("rnn", "lstm")}


SAMPLES = load_json("sample_predictions.json")
METRICS = load_json("metrics.json")
HISTORY = load_json("training_history.json")
LIVE = artifacts_available()
MODEL_COLORS = {"SimpleRNN": RNN_COLOR, "LSTM": LSTM_COLOR}

# ---------------------------------------------------------------- hero ----------
hero_sample = SAMPLES[4]
st.markdown(
    f"""
<div class="hero">
  <div>
    <h1>Arabic headline generator</h1>
    <p>Give the model an Arabic financial news article and it writes a headline.
       Two sequence-to-sequence networks, a SimpleRNN and an LSTM, were trained on
       5,000 articles from AraFinNews and are compared side by side.</p>
    <div class="meta">Badr University in Assiut · School of Artificial Intelligence &amp; Data Management</div>
  </div>
  <div class="wire" aria-label="Example from the test set">
    {headline_row("Reference headline", REF_COLOR, hero_sample["reference"])}
    {headline_row("SimpleRNN", RNN_COLOR, hero_sample["simplernn"], small=True)}
    {headline_row("LSTM", LSTM_COLOR, hero_sample["lstm"], small=True)}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_gen, tab_res, tab_ex, tab_how, tab_lim = st.tabs(
    ["Generate", "Results", "Examples", "How it works", "Limits"]
)

# ------------------------------------------------------------ generate ----------
with tab_gen:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        if LIVE:
            st.markdown("#### Article")
            options = ["Write my own"] + [f"Test article {s['id'] + 1}" for s in SAMPLES]
            choice = st.selectbox("Start from", options, index=1, label_visibility="collapsed")
            sample = None if choice == options[0] else SAMPLES[options.index(choice) - 1]
            text = st.text_area(
                "Arabic article", value=sample["article"] if sample else "", height=260,
                placeholder="الصق نص الخبر هنا...", label_visibility="collapsed",
                key=f"article_{choice}",
            )
            avoid = st.checkbox(
                "Block repeated words",
                help="Not part of the notebook. Greedy decoding often loops (e.g. 'ارتفاع في ارتفاع'); "
                     "this option forbids using a word twice in one headline.",
            )
            go = st.button("Generate headline", type="primary", disabled=not text.strip()) or (sample is not None and bool(text.strip()))
        else:
            st.markdown("#### Pick a test article")
            idx = st.selectbox(
                "Test article", range(len(SAMPLES)), format_func=lambda i: f"Test article {i + 1}",
                label_visibility="collapsed",
            )
            sample = SAMPLES[idx]
            text, go, avoid = sample["article"], True, False
            st.markdown(f'<div class="articlebox ar">{esc(text)}</div>', unsafe_allow_html=True)

    with right:
        st.markdown("#### Headlines")
        if LIVE and not go:
            st.markdown('<div class="note">Paste an article or choose a test article, then select '
                        '<b>Generate headline</b>.</div>', unsafe_allow_html=True)
        else:
            if LIVE:
                gens = load_generators()
                rnn_out = gens["rnn"].generate(text, avoid_repeats=avoid)
                lstm_out = gens["lstm"].generate(text, avoid_repeats=avoid)
            else:
                rnn_out, lstm_out = sample["simplernn"], sample["lstm"]
            ref = sample["reference"] if sample else None
            rows = (headline_row("Reference headline", REF_COLOR, ref) if ref else "")
            rows += headline_row("SimpleRNN", RNN_COLOR, rnn_out) + headline_row("LSTM", LSTM_COLOR, lstm_out)
            st.markdown(f'<div class="wire">{rows}</div>', unsafe_allow_html=True)
            if not LIVE:
                st.markdown(
                    '<div class="note" style="margin-top:.9rem">Live generation is switched off because the '
                    'trained weights are not in <code>artifacts/</code>. These headlines are the outputs '
                    'recorded in the notebook for the same test articles. '
                    'See <b>docs/DEPLOYMENT.md</b> to enable the live mode.</div>',
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------- results ----------
with tab_res:
    st.markdown("#### Test-set scores")
    st.caption("1,000 held-out articles, greedy decoding. Higher is better; a perfect score is 1.0.")
    c = st.columns(4)
    kpis = [("SimpleRNN", "BLEU-1", "bleu1"), ("LSTM", "BLEU-1", "bleu1"),
            ("SimpleRNN", "ROUGE-L", "rouge_l"), ("LSTM", "ROUGE-L", "rouge_l")]
    for col, (model, label, key) in zip(c, kpis):
        cls = "accent-rnn" if model == "SimpleRNN" else "accent-lstm"
        col.markdown(
            f'<div class="card {cls}">{tag(model, MODEL_COLORS[model])}'
            f'<div class="kpi">{METRICS[model][key]:.4f}<small>{label}</small></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
    score_df = pd.DataFrame(
        [{"Model": m, "Metric": lbl, "Score": METRICS[m][k]}
         for m in ("SimpleRNN", "LSTM") for lbl, k in (("BLEU-1", "bleu1"), ("ROUGE-L", "rouge_l"))]
    )
    color_scale = alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))
    base = alt.Chart(score_df).encode(
        x=alt.X("Metric:N", axis=alt.Axis(labelAngle=0, title=None)),
        xOffset="Model:N",
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 0.1]), axis=alt.Axis(title=None, format=".2f")),
        color=alt.Color("Model:N", scale=color_scale, legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")],
    )
    bars = base.mark_bar(size=46, cornerRadiusEnd=4)
    labels = base.mark_text(dy=-8, fontSize=12, color=INK).encode(text=alt.Text("Score:Q", format=".3f"))
    st.altair_chart((bars + labels).properties(height=260, title="Overlap with the reference headline"),
                    width="stretch")

    lstm_gain_b = METRICS["LSTM"]["bleu1"] / METRICS["SimpleRNN"]["bleu1"] - 1
    lstm_gain_r = METRICS["LSTM"]["rouge_l"] / METRICS["SimpleRNN"]["rouge_l"] - 1
    st.markdown(
        f'<div class="note">LSTM scores {lstm_gain_b:.0%} higher on BLEU-1 and {lstm_gain_r:.0%} higher on '
        f"ROUGE-L, but both models stay below 0.08. It is a single training run on 1,000 test articles, "
        f"so treat the gap as indicative, not proven.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Training curves")
    metric_choice = st.radio("Show", ["Loss", "Token accuracy"], horizontal=True, label_visibility="collapsed")
    key_train, key_val, ytitle = (("loss", "val_loss", "Loss") if metric_choice == "Loss"
                                  else ("acc", "val_acc", "Accuracy"))
    hist_df = pd.DataFrame(
        [{"Epoch": r["epoch"], "Model": m, "Split": split, "Value": r[k]}
         for m, rows in HISTORY.items() for r in rows
         for split, k in (("Training", key_train), ("Validation", key_val))]
    )
    line = alt.Chart(hist_df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1, title="Epoch")),
        y=alt.Y("Value:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title=ytitle)),
        color=alt.Color("Model:N", scale=color_scale, legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Split:N", scale=alt.Scale(domain=["Training", "Validation"],
                                                             range=[[1, 0], [5, 4]]),
                                  legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", "Split", "Epoch", alt.Tooltip("Value:Q", format=".4f")],
    ).properties(height=300)
    st.altair_chart(line, width="stretch")
    st.caption("Solid = training, dashed = validation. SimpleRNN validation loss bottoms out at epoch 10 and "
               "then rises while training loss keeps falling, a sign of overfitting.")

    with st.expander("Per-epoch numbers"):
        for m, rows in HISTORY.items():
            st.markdown(f"**{m}**")
            st.dataframe(pd.DataFrame(rows).rename(columns={
                "epoch": "Epoch", "seconds": "Seconds", "loss": "Loss", "acc": "Accuracy",
                "val_loss": "Val loss", "val_acc": "Val accuracy"}), hide_index=True)

# ------------------------------------------------------------ examples ----------
with tab_ex:
    st.markdown("#### First 10 test articles")
    st.caption("Recorded from the notebook (greedy decoding). The text is normalised Arabic: no diacritics, "
               "alef forms unified.")
    for s in SAMPLES:
        with st.expander(f"Article {s['id'] + 1}: {s['reference']}"):
            st.markdown(f'<div class="articlebox ar">{esc(s["article"])}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="wire" style="margin-top:.7rem">'
                f'{headline_row("Reference headline", REF_COLOR, s["reference"])}'
                f'{headline_row("SimpleRNN", RNN_COLOR, s["simplernn"], small=True)}'
                f'{headline_row("LSTM", LSTM_COLOR, s["lstm"], small=True)}</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------- how it works ----------
with tab_how:
    st.markdown("#### Pipeline")
    st.markdown(
        """
<div class="pipe">
  <div><b>Load</b><span>AraFinNews article and headline pairs from Hugging Face</span></div>
  <div><b>Clean</b><span>Strip HTML and URLs, remove diacritics, unify alef forms</span></div>
  <div><b>Sample</b><span>5,000 train, 1,000 validation, 1,000 test; no article shared between splits</span></div>
  <div><b>Encode</b><span>Whitespace tokens to ids; article cut to 80, headline to 20</span></div>
  <div><b>Train</b><span>Encoder-decoder with teacher forcing, 20 epochs</span></div>
  <div><b>Generate</b><span>Greedy decoding until [EOS] or 19 words</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Architecture")
        st.markdown(
            "- Encoder: embedding (128) then one SimpleRNN or LSTM layer (128 units)\n"
            "- The encoder's final state initialises the decoder\n"
            "- Decoder: embedding (128), same recurrent layer type, softmax over the vocabulary\n"
            "- Vocabulary: 92,121 words (4 special tokens + every word in the training articles and headlines)\n"
            "- About 35.5 M parameters (SimpleRNN) and 35.7 M (LSTM); 99% of them sit in the embeddings and "
            "the output layer"
        )
    with b:
        st.markdown("#### Training setup")
        st.table(pd.DataFrame(
            {"Setting": ["Optimizer", "Learning rate", "Gradient clipping", "Batch size", "Epochs", "Dropout",
                         "Loss", "Padding", "Seed"],
             "Value": ["Adam", "0.0005", "clipnorm 1.0", "32", "20", "0.1", "Sparse categorical cross-entropy",
                       "Masked with sample weights", "42"]}
        ).set_index("Setting"))
    st.markdown("#### Metrics")
    st.markdown(
        "**BLEU-1** counts how many generated words also appear in the reference. **ROUGE-L** uses the longest "
        "common subsequence of words and reports an F-score. Both are computed on whitespace tokens of the "
        "normalised headline, so a correct headline worded differently still scores low."
    )

# --------------------------------------------------------------- limits ----------
with tab_lim:
    st.markdown("#### What the results tell us")
    st.markdown(
        '<div class="note">The headlines are fluent-looking Arabic, but they usually do not describe the article. '
        "The same phrases (“US stocks rise”, “China intends to…”) come back for unrelated inputs, so the models "
        "learned which headline words are common, not how to summarise. On the ten recorded test articles every LSTM "
        "headline opens with either “ارتفاع” or “وكالة: الصين تعتزم”.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
**Why it happens**

- **Too little data for the vocabulary.** 5,000 training pairs against a 92,121-word vocabulary and ~35 M parameters. Most words are seen once.
- **No attention.** The whole article must pass through one 128-number state, and only its first 80 words are used (articles average 180 to 190 words).
- **Word-level tokens.** Punctuation sticks to words (`اسمه.` and `اسمه` are different tokens) and there is no frequency cut-off, which inflates the vocabulary.
- **Overfitting.** SimpleRNN training accuracy reaches 23% while validation stays near 16%; validation loss rises after epoch 10.
- **Greedy decoding** repeats itself (“ارتفاع في ارتفاع”).
- **Limited compute.** About 70 seconds per epoch limited how much data and tuning was feasible.

**Where to go next**

1. Train on the full ~170,000 pairs, with a minimum word frequency or subword tokens (SentencePiece/BPE).
2. Add attention, or fine-tune a pretrained Arabic model such as AraT5, AraBART or mT5.
3. Use beam search with a repetition penalty.
4. Report ROUGE-1/2/L and BERTScore, add a “first sentence” baseline, and run a small human evaluation.
"""
    )

st.markdown(
    '<div class="foot"><b>Team</b> · Ali Omar · Shaza Abdulnasser · Zizi Mostafa<br>'
    "Dataset: AraFinNews (drelhaj/AraFinNews on Hugging Face). Source code, report and notebook are in the GitHub repository.</div>",
    unsafe_allow_html=True,
)
