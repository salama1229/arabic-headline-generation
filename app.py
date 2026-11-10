"""Arabic Headline Generator - Streamlit app.

Compares a SimpleRNN and an LSTM encoder-decoder trained on AraFinNews (see notebooks/NLP2.ipynb).
Team: Ali Omar, Shaza Abdulnasser, Zizi Mostafa - Badr University in Assiut.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import HeadlineGenerator

ROOT = Path(__file__).parent
RNN_COLOR, LSTM_COLOR = "#EB6834", "#2A78D6"
MODEL_COLORS = {"SimpleRNN": RNN_COLOR, "LSTM": LSTM_COLOR}

st.set_page_config(page_title="Arabic Headline Generator", page_icon="📰", layout="centered")
st.markdown(
    "<style>.ar{direction:rtl;text-align:right;font-size:1.25rem;line-height:1.8;}</style>",
    unsafe_allow_html=True,
)


@st.cache_data
def load_json(name: str):
    with open(ROOT / "data" / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading models...")
def load_generators():
    return {c: HeadlineGenerator.from_artifacts(c) for c in ("rnn", "lstm")}


METRICS = load_json("metrics.json")
HISTORY = load_json("training_history.json")

st.title("Arabic headline generator")
st.write("Paste an Arabic news article and two models, a SimpleRNN and an LSTM, each write a headline.")

tab_gen, tab_res, tab_how, tab_lim = st.tabs(["Generate", "Results", "How it works", "Limits"])

# ------------------------------------------------------------ generate ----------
with tab_gen:
    text = st.text_area("Article", height=260, placeholder="الصق نص الخبر هنا...")
    avoid = st.checkbox(
        "Block repeated words",
        help="Not part of the notebook. Greedy decoding often loops; this forbids using a word twice in one headline.",
    )
    if st.button("Generate headline", type="primary", disabled=not text.strip()):
        gens = load_generators()
        for name, key in (("SimpleRNN", "rnn"), ("LSTM", "lstm")):
            st.subheader(name)
            out = gens[key].generate(text, avoid_repeats=avoid)
            st.markdown(f'<div class="ar">{html.escape(out) if out else "—"}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------- results ----------
with tab_res:
    st.markdown("#### Test-set scores")
    st.caption("1,000 held-out articles, greedy decoding. Higher is better; a perfect score is 1.0.")
    c = st.columns(4)
    for col, (model, label, key) in zip(c, [("SimpleRNN", "BLEU-1", "bleu1"), ("LSTM", "BLEU-1", "bleu1"),
                                           ("SimpleRNN", "ROUGE-L", "rouge_l"), ("LSTM", "ROUGE-L", "rouge_l")]):
        col.metric(f"{model} {label}", f"{METRICS[model][key]:.4f}")

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
    labels = base.mark_text(dy=-8, fontSize=12, color="#31333F").encode(text=alt.Text("Score:Q", format=".3f"))
    st.altair_chart((bars + labels).properties(height=260, title="Overlap with the reference headline"),
                    width="stretch")

    lstm_gain_b = METRICS["LSTM"]["bleu1"] / METRICS["SimpleRNN"]["bleu1"] - 1
    lstm_gain_r = METRICS["LSTM"]["rouge_l"] / METRICS["SimpleRNN"]["rouge_l"] - 1
    st.info(
        f"LSTM scores {lstm_gain_b:.0%} higher on BLEU-1 and {lstm_gain_r:.0%} higher on "
        f"ROUGE-L, but both models stay below 0.08. It is a single training run on 1,000 test articles, "
        f"so treat the gap as indicative, not proven."
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

# ---------------------------------------------------------- how it works ----------
with tab_how:
    st.markdown("#### Pipeline")
    st.markdown(
        "1. **Load**: AraFinNews article and headline pairs from Hugging Face\n"
        "2. **Clean**: strip HTML and URLs, remove diacritics, unify alef forms\n"
        "3. **Sample**: 5,000 train, 1,000 validation, 1,000 test; no article shared between splits\n"
        "4. **Encode**: whitespace tokens to ids; article cut to 80, headline to 20\n"
        "5. **Train**: encoder-decoder with teacher forcing, 20 epochs\n"
        "6. **Generate**: greedy decoding until [EOS] or 19 words"
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
    st.info(
        "The headlines are fluent-looking Arabic, but they usually do not describe the article. "
        "The same phrases (“US stocks rise”, “China intends to…”) come back for unrelated inputs, so the models "
        "learned which headline words are common, not how to summarise. On the ten recorded test articles every LSTM "
        "headline opens with either “ارتفاع” or “وكالة: الصين تعتزم”."
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

st.caption("Team: Ali Omar, Shaza Abdulnasser, Zizi Mostafa · Badr University in Assiut · Dataset: AraFinNews")
