"""Presentation helpers: theme CSS and small HTML snippets for the Streamlit app."""
from __future__ import annotations

import html

INK, PAPER, RULE, MUTED = "#10243B", "#F4F6F9", "#D5DCE5", "#5A6B7D"
RNN_COLOR, LSTM_COLOR, REF_COLOR = "#EB6834", "#2A78D6", "#10243B"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&display=swap');

.hero, .wire, .card, .note, .pipe, .foot, .articlebox {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
.block-container, .stMainBlockContainer {{ max-width: 1080px; padding-top: 5rem !important; padding-bottom: 4rem; }}
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', 'IBM Plex Sans', sans-serif; letter-spacing: -0.01em; color: {INK}; }}

/* hero ------------------------------------------------------------------ */
.hero {{ display:grid; grid-template-columns: 1.05fr 1fr; gap: 2.2rem; align-items:start;
        padding: 0 0 1.6rem; border-bottom: 2px solid {INK}; margin-bottom: 1.2rem; }}
.hero h1 {{ font-size: 2.6rem; line-height: 1.05; margin: 0 0 .7rem; }}
.hero p {{ color: {MUTED}; font-size: 1.02rem; line-height: 1.55; margin: 0; max-width: 34rem; }}
.hero .meta {{ margin-top: .9rem; font-size: .82rem; color: {MUTED}; }}
.wire {{ background: #fff; border: 1px solid {RULE}; border-radius: 6px; padding: 1.05rem 1.2rem; }}
.wire .row {{ padding: .55rem 0; border-top: 1px solid {RULE}; }}
.wire .row:first-child {{ border-top: 0; padding-top: 0; }}
.tag {{ display:inline-flex; align-items:center; gap:.4rem; font-size:.78rem; font-weight:600; color:{MUTED}; margin-bottom:.15rem; }}
.dot {{ width:.62rem; height:.62rem; border-radius:50%; display:inline-block; }}
.ar {{ direction: rtl; text-align: right; font-family: 'IBM Plex Sans Arabic', 'Noto Naskh Arabic', sans-serif; line-height: 1.75; }}
.headline {{ font-family: 'IBM Plex Sans Arabic', 'Noto Naskh Arabic', sans-serif; font-weight: 500; font-size: 1.3rem; line-height: 1.5; color:{INK}; }}
.stat {{ font-size: 1rem; color:{INK}; font-weight:500; }}
.stat.sm {{ font-size:.82rem; color:{MUTED}; font-weight:400; }}
.articlebox {{ background:#fff; border:1px solid {RULE}; border-radius:6px; padding: .9rem 1.1rem; font-size: 1.02rem; max-height: 16rem; overflow:auto; }}
.card {{ background:#fff; border:1px solid {RULE}; border-radius:6px; padding: 1rem 1.15rem; height:100%; margin-bottom:1.1rem; }}
.card.accent-rnn {{ border-top: 4px solid {RNN_COLOR}; }}
.card.accent-lstm {{ border-top: 4px solid {LSTM_COLOR}; }}
.card.accent-ref {{ border-top: 4px solid {INK}; }}
.kpi {{ font-family:'Bricolage Grotesque', sans-serif; font-size: 2rem; font-weight:700; line-height:1.1; color:{INK}; }}
.kpi small {{ font-family:'IBM Plex Sans', system-ui, sans-serif; font-size:.8rem; font-weight:500; color:{MUTED}; margin-left:.3rem; }}
.note {{ background:#fff; border-left: 4px solid {INK}; padding:.8rem 1rem; border-radius:0 6px 6px 0; color:{INK}; font-size:.95rem; }}
.pipe {{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:stretch; margin:.4rem 0 1rem; }}
.pipe div {{ flex:1 1 130px; background:#fff; border:1px solid {RULE}; border-radius:6px; padding:.65rem .75rem; font-size:.86rem; }}
.pipe b {{ display:block; font-size:.9rem; color:{INK}; margin-bottom:.15rem; }}
.pipe span {{ color:{MUTED}; }}
.foot {{ margin-top:2.5rem; padding-top:1rem; border-top:1px solid {RULE}; font-size:.85rem; color:{MUTED}; }}
button:focus-visible, a:focus-visible, textarea:focus-visible {{ outline: 3px solid {LSTM_COLOR} !important; outline-offset: 2px; }}
@media (max-width: 760px) {{ .hero {{ grid-template-columns: 1fr; }} .hero h1 {{ font-size: 2rem; }} }}
</style>
"""


def esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def tag(label: str, color: str) -> str:
    return f'<div class="tag"><span class="dot" style="background:{color}"></span>{esc(label)}</div>'


def headline_row(label: str, color: str, text: str, small: bool = False) -> str:
    cls = "headline sm ar" if small else "headline ar"
    body = esc(text) if text else "—"
    return f'<div class="row">{tag(label, color)}<div class="{cls}">{body}</div></div>'
