# Notebook guide: `notebooks/NLP2.ipynb`

The notebook is kept exactly as submitted. This guide explains each part, the key numbers it prints, and the decisions behind it. Cell numbers are 0-based positions in the notebook.

**Goal:** generate a news headline from an Arabic article with an encoder-decoder network, and compare a SimpleRNN with an LSTM.
**Runtime:** TensorFlow/Keras, about 65 to 80 s per epoch in the recorded run.

## 1. Load the data (cell 0)

Reads the parquet version of `drelhaj/AraFinNews` from Hugging Face.

| Split | Rows loaded |
|---|---|
| Train | 340,024 |
| Validation | 42,502 |
| Test | 42,504 |

## 2. Validate and clean (cells 1 to 2)

Cell 1 counts missing values and duplicates, then `remove_invalid` drops missing/blank rows and duplicate pairs.

| Split | Missing (article = title) | Duplicate pairs |
|---|---|---|
| Train | 170,012 | 23 |
| Validation | 21,251 | 1 |
| Test | 21,252 | 0 |

Exactly half of every split is empty, so about 170,000 usable training pairs remain.

Cell 2 defines `clean_arabic`:

1. `html.unescape`: `&quot;` becomes `"`
2. remove URLs and HTML tags
3. remove Arabic diacritics (U+0610–061A, U+064B–065F, U+0670, U+06D6–06ED) and tatweel (U+0640)
4. unify alef forms أ إ آ ٱ to ا
5. collapse whitespace

> Not normalised: ي/ى and ة/ه, and punctuation attached to words. Both increase vocabulary size (see the report).

## 3. Select records (cells 3 to 6)

`select_records` shuffles with `SEED = 42`, cleans each pair, skips empty results and any article already used by another split, and keeps **5,000 train / 1,000 validation / 1,000 test**. It checks that there are no missing values or duplicates left. Cells 4 to 6 show five raw examples, average lengths and before/after cleaning:

| Split | Avg. article words | Avg. headline words |
|---|---|---|
| Train | 181.57 | 10.32 |
| Validation | 182.45 | 10.25 |
| Test | 193.70 | 10.29 |

## 4. Tokens, vocabulary, sequences (cells 7 to 22)

| Item | Value |
|---|---|
| Special tokens | `[PAD]`=0, `[UNK]`=1, `[SOS]`=2, `[EOS]`=3 |
| Tokeniser | `text.split()` |
| Vocabulary | **92,121** words (all training articles + headlines, no frequency cut-off) |
| Article length | cut/padded to **80** (post-padding) |
| Headline length | up to 18 words + `[SOS]`/`[EOS]` = **20** |
| Decoder input / target | first 19 tokens / the same sequence shifted left by one |
| Shapes | encoder `(N, 80)`, decoder input and target `(N, 19)` for N = 5000, 1000, 1000 |

`text_to_sequence`, `encode_article`, `encode_title` and `sequence_to_text` convert in both directions (cell 22 verifies the round trip).

## 4b. Sample weights (cell 28)

`sample_weight = (target != PAD)` so padding positions do not count in the loss.

## 5. SimpleRNN model (cells 23 to 34)

Hyper-parameters (cell 24): embedding 128, RNN units 128, 20 epochs, batch 32, dropout 0.1, seed 42.

- Encoder: `Embedding(92121, 128, mask_zero=True)` → `SimpleRNN(128, return_state=True)`
- Decoder: `Embedding(92121, 128, mask_zero=True)` → `SimpleRNN(128, return_sequences=True, return_state=True)` initialised with the encoder state → `Dense(92121, softmax)`
- Compile: Adam (lr 0.0005, clipnorm 1.0), sparse categorical cross-entropy, weighted accuracy
- About 35.5 M parameters; trained in about 23 minutes.

| Epoch | Train loss | Val. loss | Train acc. | Val. acc. |
|---|---|---|---|---|
| 1 | 8.2312 | 7.4691 | 8.4% | 8.9% |
| 10 | 5.5749 | **6.9801** | 15.7% | 13.6% |
| 20 | 4.4528 | 7.1101 | 23.2% | 16.4% |

Validation loss is lowest at epoch 10 and then rises while training loss keeps falling: overfitting.

Cells 31 to 34 build separate **inference** models (encoder → state; decoder one step at a time) and `generate_rnn_headline`, which decodes greedily from `[SOS]` until `[EOS]`, `[PAD]` or 19 words.

## 6. LSTM model (cells 35 to 47)

Same data and settings. The encoder returns both hidden and cell state and passes them to the decoder. About 35.7 M parameters; about 24 minutes.

| Epoch | Train loss | Val. loss | Train acc. | Val. acc. |
|---|---|---|---|---|
| 1 | 8.3796 | 7.4565 | 8.7% | 8.9% |
| 3 | 6.7817 | 7.2633 | 9.7% | 10.6% |
| 19 | 5.2178 | **7.1695** | 18.8% | 12.9% |
| 20 | 5.1348 | 7.1715 | 19.4% | 13.1% |

## 7. Evaluation (cells 48 to 54)

- **BLEU-1:** NLTK `sentence_bleu`, weights (1,0,0,0), smoothing method 1, whitespace tokens.
- **ROUGE-L:** own implementation: LCS of reference and prediction tokens, then the F-score of LCS precision and recall. (`rouge-score` is installed but not used, because it does not tokenise Arabic.)

| Model | BLEU-1 | ROUGE-L |
|---|---|---|
| SimpleRNN | 0.0587 | 0.0615 |
| LSTM | 0.0736 | 0.0779 |

Cell 54 prints a table of ten test articles with both models' headlines (stored in `data/sample_predictions.json`).

## 8. Reading the results

- The LSTM wins on BLEU-1 and ROUGE-L; the SimpleRNN wins on validation loss and accuracy. Accuracy rewards frequent words, so it is the weaker signal.
- Predicted headlines are generic ("US stocks rise", "China intends to…") and repeat words ("ارتفاع في ارتفاع"). Greedy decoding plus a small, sparse training set pushes the models to the most common openings.
- With one run and 1,000 test articles, the LSTM–SimpleRNN gap is indicative, not proven.

## 9. Known limitations and ideas

| Limitation | Idea |
|---|---|
| 5,000 pairs vs 92,121-word vocabulary | Use all ~170 k pairs; minimum word frequency or SentencePiece/BPE |
| Punctuation glued to words; ي/ى, ة/ه not unified | Split punctuation; extend normalisation |
| 80-word window, no attention | Attention, or fine-tune AraT5 / AraBART / mT5 |
| Greedy decoding | Beam search with repetition penalty |
| Metrics: BLEU-1 and ROUGE-L only, one run | ROUGE-1/2/L, BERTScore, first-sentence baseline, several seeds, human evaluation |
| Models are not saved by the notebook | Export script in `scripts/` (see `docs/DEPLOYMENT.md`) |
