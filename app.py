import streamlit as st
from transformers import pipeline
import PyPDF2
import spacy
import random
import re
import pandas as pd
from datetime import datetime
import requests
import html
from openai import OpenAI
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Personal Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

  :root {
    --accent:   #7c6ef7;
    --accent2:  #5eead4;
    --accent3:  #f472b6;
    --bg:       #0d0f18;
    --surface:  #13162a;
    --card:     #181c30;
    --border:   rgba(255,255,255,0.08);
    --text:     #e8eaf6;
    --muted:    #6b7280;
    --danger:   #f87171;
    --success:  #4ade80;
    --warn:     #fbbf24;
  }

  html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
  }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: rgba(124,110,247,0.4); border-radius: 99px; }

  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 12px;
    padding: 4px;
    gap: 3px;
    border: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    border-radius: 9px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: .85rem;
    padding: 8px 14px;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(124,110,247,0.18) !important;
    color: #a89cf7 !important;
    border: 1px solid rgba(124,110,247,0.3) !important;
  }

  .stButton > button {
    background: rgba(124,110,247,0.1) !important;
    border: 1px solid rgba(124,110,247,0.3) !important;
    color: #a89cf7 !important;
    border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: .88rem !important;
    transition: all .18s !important;
  }
  .stButton > button:hover {
    background: rgba(124,110,247,0.2) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c6ef7, #5eead4) !important;
    color: #0d0f18 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: .9rem !important;
  }

  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: rgba(124,110,247,0.5) !important;
    box-shadow: 0 0 0 3px rgba(124,110,247,0.1) !important;
  }

  .stSelectbox > div > div,
  .stRadio label {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
  }

  .stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-size: .8rem !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stDownloadButton > button:hover {
    border-color: rgba(124,110,247,0.4) !important;
    color: #a89cf7 !important;
  }

  .hero {
    text-align: center;
    padding: 48px 0 20px;
  }
  .hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a89cf7 0%, #5eead4 60%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 12px;
    line-height: 1.1;
  }
  .hero p {
    color: var(--muted);
    font-size: 1rem;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.7;
  }

  .page-header {
    padding: 28px 0 18px;
  }
  .page-header h2 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.6rem, 3vw, 2.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a89cf7 0%, #5eead4 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
  }
  .page-header p {
    color: var(--muted);
    font-size: .9rem;
  }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
  }

  .sec-label {
    font-family: 'Manrope', sans-serif;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 12px;
  }

  .input-choice {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 28px 28px 22px;
    text-align: center;
    transition: border-color .2s, transform .15s;
    cursor: pointer;
    height: 100%;
  }
  .input-choice:hover {
    border-color: rgba(124,110,247,0.4);
    transform: translateY(-2px);
  }
  .input-choice-icon {
    font-size: 2.4rem;
    margin-bottom: 12px;
  }
  .input-choice-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--text);
    margin-bottom: 6px;
  }
  .input-choice-desc {
    color: var(--muted);
    font-size: .83rem;
    line-height: 1.6;
  }

  .summary-text {
    color: var(--text);
    font-size: .95rem;
    line-height: 1.85;
    border-left: 3px solid rgba(124,110,247,0.5);
    padding-left: 16px;
  }

  .note-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 11px 14px;
    background: rgba(124,110,247,0.06);
    border-left: 2px solid rgba(124,110,247,0.4);
    border-radius: 0 10px 10px 0;
    margin: 7px 0;
  }
  .note-num {
    font-family: 'Manrope', sans-serif;
    font-size: .72rem;
    font-weight: 700;
    color: #a89cf7;
    min-width: 22px;
    margin-top: 3px;
  }
  .note-text {
    color: var(--text);
    font-size: .9rem;
    line-height: 1.7;
  }

  .def-card {
    background: rgba(94,234,212,0.05);
    border: 1px solid rgba(94,234,212,0.18);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
  }
  .def-term {
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: .95rem;
    color: var(--accent2);
    margin-bottom: 5px;
  }
  .def-meaning {
    color: var(--text);
    font-size: .88rem;
    line-height: 1.7;
  }

  .fc-front {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 28px;
    min-height: 170px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 1rem;
    color: var(--text);
    line-height: 1.65;
  }
  .fc-back {
    background: rgba(94,234,212,0.06);
    border: 1px solid rgba(94,234,212,0.25);
    border-radius: 16px;
    padding: 36px 28px;
    min-height: 170px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: .95rem;
    color: var(--text);
    line-height: 1.7;
  }
  .prog-bg  { background: rgba(255,255,255,0.07); border-radius: 99px; height: 5px; overflow: hidden; margin: 8px 0; }
  .prog-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #7c6ef7, #5eead4); }

  .chat-wrap { overflow: hidden; margin: 5px 0; }
  .chat-u {
    float: right; clear: both;
    background: rgba(124,110,247,0.15);
    border: 1px solid rgba(124,110,247,0.25);
    border-radius: 16px 16px 4px 16px;
    padding: 10px 15px;
    max-width: 78%;
    color: var(--text);
    font-size: .88rem;
    line-height: 1.65;
  }
  .chat-b {
    float: left; clear: both;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 10px 15px;
    max-width: 82%;
    color: var(--text);
    font-size: .88rem;
    line-height: 1.75;
  }
  .chat-lbl {
    font-size: .65rem; font-weight: 600;
    letter-spacing: .07em; text-transform: uppercase;
    margin-bottom: 2px;
  }
  .chat-lbl-u { color: #a89cf7; text-align: right; }
  .chat-lbl-b { color: var(--muted); }

  .q-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 12px 0;
  }
  .q-badge {
    display: inline-block;
    font-size: .68rem; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px; margin-bottom: 9px;
  }
  .qb-def  { background: rgba(124,110,247,0.12); color: #a89cf7; }
  .qb-fill { background: rgba(94,234,212,0.12);  color: var(--accent2); }
  .qb-id   { background: rgba(244,114,182,0.12); color: var(--accent3); }

  .dp-easy   { background:rgba(74,222,128,0.12);  color:var(--success); border:1px solid rgba(74,222,128,0.3);  border-radius:20px; padding:2px 12px; font-size:.74rem; font-weight:600; display:inline-block; margin-left:8px; }
  .dp-medium { background:rgba(251,191,36,0.12);  color:var(--warn);    border:1px solid rgba(251,191,36,0.3);  border-radius:20px; padding:2px 12px; font-size:.74rem; font-weight:600; display:inline-block; margin-left:8px; }
  .dp-hard   { background:rgba(248,113,113,0.12); color:var(--danger);  border:1px solid rgba(248,113,113,0.3); border-radius:20px; padding:2px 12px; font-size:.74rem; font-weight:600; display:inline-block; margin-left:8px; }

  .ok-box {
    background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.25);
    border-radius: 9px; padding: 10px 15px; color: var(--success);
    margin: 5px 0; font-size: .88rem;
  }
  .err-box {
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.25);
    border-radius: 9px; padding: 10px 15px; color: var(--danger);
    margin: 5px 0; font-size: .88rem;
  }

  .score-big {
    text-align: center; padding: 24px 0 10px;
  }
  .score-num {
    font-family: 'Manrope', sans-serif; font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #7c6ef7, #5eead4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .score-sub { color: var(--muted); font-size: .9rem; margin-top: 6px; }
  .streak-tag {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(244,114,182,0.12); border: 1px solid rgba(244,114,182,0.25);
    border-radius: 20px; padding: 5px 14px;
    color: var(--accent3); font-weight: 600; font-size: .85rem;
  }

  .metric-g { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 18px; }
  .metric-c {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 13px; padding: 16px; text-align: center;
  }
  .metric-v {
    font-family: 'Manrope', sans-serif; font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(135deg, #7c6ef7, #5eead4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .metric-l { color: var(--muted); font-size: .73rem; margin-top: 3px; }

  .weak-row {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 9px;
    background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.15);
    margin: 6px 0;
  }
  .weak-name { color: var(--text); font-size: .87rem; font-weight: 500; flex: 1; }
  .weak-pct  { color: var(--danger); font-size: .8rem; font-weight: 600; }

  .divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,110,247,0.2), transparent);
    margin: 18px 0; border: none;
  }

  .stepper-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 0 auto 28px;
    max-width: 720px;
    overflow-x: auto;
    padding: 0 8px;
  }
  .step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 68px;
  }
  .step-dot {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .72rem; font-weight: 700;
    font-family: 'Manrope', sans-serif;
    transition: all .2s;
  }
  .step-dot-active {
    background: linear-gradient(135deg, #7c6ef7, #5eead4);
    color: #0d0f18;
    box-shadow: 0 0 14px rgba(124,110,247,0.5);
  }
  .step-dot-done {
    background: rgba(94,234,212,0.2);
    border: 1px solid rgba(94,234,212,0.4);
    color: var(--accent2);
  }
  .step-dot-todo {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: var(--muted);
  }
  .step-label {
    font-size: .6rem; font-weight: 600;
    letter-spacing: .04em; text-transform: uppercase;
    color: var(--muted);
    margin-top: 5px;
    text-align: center;
  }
  .step-label-active { color: #a89cf7; }
  .step-connector {
    width: 28px; height: 2px;
    background: rgba(255,255,255,0.07);
    margin-bottom: 22px;
    flex-shrink: 0;
  }
  .step-connector-done {
    background: linear-gradient(90deg, rgba(94,234,212,0.4), rgba(124,110,247,0.4));
  }

  .nav-bar {
    position: sticky;
    bottom: 0;
    left: 0; right: 0;
    background: rgba(13,15,24,0.9);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-top: 1px solid var(--border);
    padding: 14px 0 16px;
    margin-top: 32px;
    z-index: 100;
  }

  .block-container {
    padding-bottom: 0 !important;
  }

  @media (max-width: 900px) {
    .metric-g {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
_defaults = dict(
    page=0,
    history=[],
    chat_history=[],
    summary=None,
    notes=[],
    defs=[],
    quiz=[],
    flashcards=[],
    fc_idx=0,
    fc_flipped=False,
    streak=0,
    difficulty="Medium",
    generated=False,
    q_stats={},
    groq_api_key="",
    openai_api_key="",
    input_mode=None,
    raw_text="",
    live_chart_x=[],
    live_chart_y=[],
)

for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    st.markdown("**🤖 OpenAI ChatGPT (Recommended)**")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        placeholder="sk-..."
    )

    if openai_key:
        st.session_state.openai_api_key = openai_key
        st.success("✅ ChatGPT enabled (Best AI)")

    st.markdown("---")
    st.markdown("**🤖 Groq AI Chat (optional)**")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        placeholder="gsk_… (free at console.groq.com)",
    )

    if groq_key:
        st.session_state.groq_api_key = groq_key
        st.success("✅ Real AI chat enabled")
    else:
        if not st.session_state.openai_api_key:
            st.caption("Without key: smart keyword-based chat")

    st.markdown("---")
    st.markdown("**🎯 Quiz Difficulty**")
    diff = st.radio(
        "",
        ["Easy", "Medium", "Hard"],
        index=["Easy", "Medium", "Hard"].index(st.session_state.difficulty),
        label_visibility="collapsed"
    )
    st.session_state.difficulty = diff

    if st.session_state.page > 0:
        st.markdown("---")
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = 0
            st.rerun()

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    nlp = spacy.load("en_core_web_sm")
    return summarizer, nlp

with st.spinner("⚙️ Loading AI models…"):
    summarizer, nlp = load_models()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def clean_text(raw: str) -> str:
    raw = re.sub(r'\[[0-9]+\]', '', raw)
    raw = re.sub(r'\s+', ' ', raw)
    return raw.strip()

def safe_html(text: str) -> str:
    return html.escape(str(text)).replace("\n", "<br>")

def is_question_from_document(question: str, context: str) -> bool:
    q_words = set(question.lower().split())
    ctx_words = set(context.lower().split())
    common = q_words.intersection(ctx_words)
    return len(common) > 3

def create_live_ai_chart_figure(x_vals, y_vals):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        name="AI Data",
        line=dict(color="#7c6ef7", width=3),
        marker=dict(size=8, color="#5eead4")
    ))

    fig.update_layout(
        title="Live AI Data Stream",
        xaxis_title="Time",
        yaxis_title="AI Value",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#13162a",
        font=dict(color="#e8eaf6", family="Inter"),
        xaxis=dict(showgrid=False, color="#9ca3af"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#9ca3af"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380
    )
    return fig

def get_ai_value():
    if st.session_state.openai_api_key:
        try:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Give a random number between 10 and 100. Return only the number."}],
                temperature=0.8,
                max_tokens=10
            )
            raw = response.choices[0].message.content.strip()
            num = int(re.findall(r"\d+", raw)[0])
            return max(10, min(100, num))
        except Exception:
            return random.randint(10, 100)
    return random.randint(10, 100)

# ─────────────────────────────────────────
# STEPPER COMPONENT
# ─────────────────────────────────────────
PAGES = [
    ("🏠", "Home"),
    ("📌", "Summary"),
    ("📖", "Definitions"),
    ("🃏", "Flashcards"),
    ("🤖", "AI Chat"),
    ("🧠", "Quiz"),
    ("📊", "Dashboard"),
    ("📥", "Download"),
]

def render_stepper(current_page: int):
    if current_page == 0:
        return
    items = ""
    for i, (icon, label) in enumerate(PAGES):
        if i == 0:
            continue
        idx = i
        if idx == current_page:
            dot_cls = "step-dot-active"
            lbl_cls = "step-label-active"
        elif idx < current_page:
            dot_cls = "step-dot-done"
            lbl_cls = "step-label"
        else:
            dot_cls = "step-dot-todo"
            lbl_cls = "step-label"

        if i > 1:
            conn_cls = "step-connector-done" if (i - 1) < current_page else "step-connector"
            items += f'<div class="step-connector {conn_cls}"></div>'

        items += f'''
        <div class="step-item">
          <div class="step-dot {dot_cls}">{icon}</div>
          <div class="step-label {lbl_cls}">{label}</div>
        </div>'''

    st.markdown(f'<div class="stepper-wrap">{items}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# NAV BUTTONS
# ─────────────────────────────────────────
def render_nav(current_page: int, can_go_next: bool = True, next_label: str = "Next ➡️"):
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    cols = st.columns([1, 3, 1])

    with cols[0]:
        if current_page == 1:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = 0
                st.rerun()
        elif current_page > 1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.page = current_page - 1
                st.rerun()

    with cols[2]:
        if current_page < len(PAGES) - 1 and can_go_next:
            if st.button(next_label, use_container_width=True, type="primary"):
                st.session_state.page = current_page + 1
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# NLP FUNCTIONS
# ─────────────────────────────────────────
def generate_summary(txt: str) -> str:
    cut = txt[:6000]
    chunks = [cut[i:i+800] for i in range(0, len(cut), 800)]
    parts = []
    for ch in chunks[:8]:
        if len(ch.split()) < 25:
            continue
        try:
            r = summarizer(ch, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
            parts.append(r.strip())
        except Exception:
            pass

    full = " ".join(parts)
    seen, clean = set(), []
    for s in re.split(r'(?<=[.!?])\s+', full):
        s = s.strip()
        if s and s not in seen and len(s) > 15:
            clean.append(s)
            seen.add(s)
    return " ".join(clean).strip() or "Summary could not be generated. Try a longer text."

def generate_notes(txt: str):
    doc = nlp(txt)
    out, seen = [], set()
    skip_starts = ("the ", "a ", "an ", "this ", "that ", "these ", "it ", "in ", "on ", "at ", "by ")
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s.split()) < 8 or len(s) < 50:
            continue
        if s.lower() in seen:
            continue
        has_ent = len(sent.ents) > 0
        has_noun_vb = any(t.pos_ in ("NOUN", "PROPN") for t in sent) and any(t.pos_ == "VERB" for t in sent)
        if not (has_ent or has_noun_vb):
            continue
        if s.lower().startswith(skip_starts) and not has_ent:
            continue
        out.append(s)
        seen.add(s.lower())
        if len(out) >= 12:
            break
    return out

def extract_definitions(txt: str):
    defs = {}
    SKIP = {"it","this","that","they","which","who","these","those","there","one","all","both","he","she","we","you","i"}

    pattern1 = re.compile(
        r'([A-Z][a-zA-Z\s\-]{2,40}?)\s+(?:is|are|was|were)\s+(a\s|an\s|the\s)?(.{20,200}?)(?:\.|;|,\s+and\b)',
        re.MULTILINE
    )
    for m in pattern1.finditer(txt):
        term = m.group(1).strip().rstrip(',')
        defn = ((m.group(2) or "") + m.group(3)).strip()
        term_clean = term.lower().split()[-1] if term else ""
        if (term_clean in SKIP or len(term.split()) > 6 or len(term) < 3 or len(defn) < 15 or term.capitalize() in defs):
            continue
        defs[term.strip()] = defn
        if len(defs) >= 10:
            break

    pattern2 = re.compile(
        r'([A-Z][a-zA-Z\s\-]{2,40}?)\s+(?:refers to|defined as|known as|called)\s+(.{20,180}?)(?:\.|;)',
        re.MULTILINE
    )
    for m in pattern2.finditer(txt):
        term = m.group(1).strip().rstrip(',')
        defn = m.group(2).strip()
        if term.strip() not in defs and len(defn) > 15 and len(term.split()) <= 6:
            defs[term.strip()] = defn
        if len(defs) >= 10:
            break

    if len(defs) < 3:
        doc = nlp(txt[:5000])
        for sent in doc.sents:
            s = sent.text.strip()
            if " is " not in s and " are " not in s:
                continue
            for tok in sent:
                if tok.lemma_ != "be":
                    continue
                subj = None
                for chunk in sent.noun_chunks:
                    if chunk.end <= tok.i:
                        subj = chunk
                if subj is None:
                    break
                term = subj.text.strip(" ,.()'\"")
                term = re.split(r',\s*(which|that|who)', term, flags=re.I)[0].strip()
                if (term.lower() in SKIP or len(term) < 3 or len(term.split()) > 5 or term in defs):
                    break
                def_start = tok.idx + len(tok.text)
                defn = s[def_start:].split(".")[0].strip(" ,")
                if len(defn) > 20:
                    defs[term] = defn
                break
            if len(defs) >= 10:
                break

    return list(defs.items())

# ─────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────
def chat_with_openai(question: str, context: str, history: list) -> str:
    try:
        client = OpenAI(api_key=st.session_state.openai_api_key)

        use_doc = is_question_from_document(question, context)

        if use_doc:
            system_prompt = f"""
You are an AI tutor.

Answer using the study material below.
If needed, you can also use general knowledge.

STUDY MATERIAL:
{context[:4000]}
"""
        else:
            system_prompt = """
You are a smart AI assistant like ChatGPT.

Answer clearly, simply, and helpfully.
"""

        messages = [{"role": "system", "content": system_prompt}]

        for q, a in history[-5:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=600
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"

def chat_with_groq(question: str, context: str, history: list) -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert AI tutor. Answer the student's question using the study material below. Be clear, educational, and concise.\n\nSTUDY MATERIAL:\n" + context[:4000]
            }
        ]

        for q, a in history[-4:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": question})

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.session_state.groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.4
            },
            timeout=15
        )
        data = resp.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()

        return f"API error: {data.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"Connection error: {e}"

def chat_keyword_improved(question: str, txt: str) -> str:
    q_doc = nlp(question.lower())
    keywords = {
        tok.lemma_ for tok in q_doc
        if tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not tok.is_stop and len(tok.text) > 2
    }
    if not keywords:
        keywords = {tok.lemma_ for tok in q_doc if not tok.is_stop}

    doc = nlp(txt)
    scored = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) < 20:
            continue
        s_lower = s.lower()
        s_doc = nlp(s_lower)
        s_lemmas = {tok.lemma_ for tok in s_doc if not tok.is_stop}
        s_ents = {ent.text.lower() for ent in sent.ents}
        score = len(keywords & s_lemmas)
        for kw in keywords:
            if kw in s_ents:
                score += 3
            if s_lower.startswith(kw):
                score += 2
            if re.search(rf'\b{re.escape(kw)}\b\s+(is|are|was|were)\b', s_lower):
                score += 3
        if score > 0:
            scored.append((score, s))

    if not scored:
        return "I couldn't find a direct answer in the material. Try rephrasing your question or ask about a specific term mentioned in the text."

    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:2]]
    return " ".join(top)

def chat_answer(question: str, txt: str) -> str:
    if st.session_state.openai_api_key:
        return chat_with_openai(question, txt, st.session_state.chat_history)
    elif st.session_state.groq_api_key:
        return chat_with_groq(question, txt, st.session_state.chat_history)
    else:
        return chat_keyword_improved(question, txt)

# ─────────────────────────────────────────
# FLASHCARDS
# ─────────────────────────────────────────
def generate_flashcards(notes: list, defs: list) -> list:
    cards = []
    seen_fronts = set()

    for term, defn in defs:
        front = f"What is {term}?"
        if front in seen_fronts:
            continue
        back = defn.strip().rstrip('.,;')
        if len(back) > 20:
            cards.append({"front": front, "back": back})
            seen_fronts.add(front)

    for note in notes:
        if len(cards) >= 12:
            break
        if len(note.split()) < 8:
            continue
        doc = nlp(note)
        candidates = [
            tok for tok in doc
            if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop
            and not tok.is_punct and len(tok.text) > 3
            and 2 <= tok.i <= len(doc) - 3
        ]
        if candidates:
            chosen = random.choice(candidates)
            blanked = note[:chosen.idx] + "_______" + note[chosen.idx + len(chosen.text):]
            front = blanked
            back = f"Answer: {chosen.text}\n\nFull: {note}"
            if front not in seen_fronts:
                cards.append({"front": front, "back": back})
                seen_fronts.add(front)
        elif " is " in note:
            parts = note.split(" is ", 1)
            front = f"What is {parts[0].strip()[:80]}?"
            back = parts[1].split(".")[0].strip()
            if front not in seen_fronts and len(back) > 15:
                cards.append({"front": front, "back": back})
                seen_fronts.add(front)

    return cards

# ─────────────────────────────────────────
# QUIZ
# ─────────────────────────────────────────
_QUIZ_SKIP = {"it","this","that","which","who","these","those","there","one","all","both","blood","time","way","part","type","kind","form"}

def _extract_pairs(txt: str):
    doc = nlp(txt)
    pairs, seen = [], set()
    for sent in doc.sents:
        s = sent.text.strip()
        if not re.search(r'\b(is|are|was|were)\b', s):
            continue
        be_token = None
        for tok in sent:
            if tok.lemma_ == "be":
                be_token = tok
                break
        if be_token is None:
            continue

        subject_chunk = None
        for chunk in sent.noun_chunks:
            if chunk.end <= be_token.i:
                subject_chunk = chunk

        if subject_chunk is None:
            cap_match = re.match(r'^([A-Z][a-zA-Z\s\-]{2,40}?)\s+(?:is|are|was|were)\s+', s)
            if cap_match:
                subject = cap_match.group(1).strip()
                def_start = cap_match.end()
                predicate = s[def_start:].split(".")[0].strip(" ,")
            else:
                continue
        else:
            subject = subject_chunk.text.strip(" ,.()'\"")
            subject = re.split(r',\s*(which|that|who)', subject, flags=re.I)[0].strip()
            def_start = be_token.idx + len(be_token.text)
            predicate = s[def_start:].split(".")[0].strip(" ,")

        if (subject.lower() in _QUIZ_SKIP or len(subject) < 3 or len(subject.split()) > 6 or
                subject.lower() in seen or len(predicate) < 12):
            continue

        seen.add(subject.lower())
        pairs.append({"subject": subject, "predicate": predicate, "sentence": s})
    return pairs

def generate_quiz(txt: str, difficulty: str = "Medium"):
    pairs = _extract_pairs(txt)

    if len(pairs) < 3:
        pairs = []
        seen = set()
        for m in re.finditer(r'([A-Z][a-zA-Z\s\-]{2,35}?)\s+(is|are|was|were)\s+(.{15,200}?)(?:\.|;)', txt):
            subject = m.group(1).strip()
            predicate = m.group(3).strip()
            if subject.lower() not in seen and len(predicate) > 12:
                seen.add(subject.lower())
                pairs.append({"subject": subject, "predicate": predicate, "sentence": m.group(0)})
            if len(pairs) >= 10:
                break

    if len(pairs) < 2:
        return []

    cfg = {
        "Easy": {"max_q": 3, "hint": True},
        "Medium": {"max_q": 5, "hint": False},
        "Hard": {"max_q": 7, "hint": False}
    }[difficulty]

    all_preds = [p["predicate"] for p in pairs]
    q_templates = [
        ("definition", lambda s: f"What is {s}?"),
        ("fill", lambda s: f"Complete: '{s} is ___'"),
        ("identify", lambda s: f"Which statement best describes {s}?"),
    ]

    random.shuffle(pairs)
    quiz = []

    for i, pair in enumerate(pairs):
        subj = pair["subject"]
        correct = pair["predicate"]
        sentence = pair["sentence"]

        wrong_pool = [p for p in all_preds if p != correct]
        if len(wrong_pool) < 3:
            wrong_pool += ["a secondary process", "an external factor", "a structural component"]

        wrong = random.sample(wrong_pool[:max(3, len(wrong_pool))], 3)
        options = wrong + [correct]
        random.shuffle(options)

        labeled = {chr(65+j): opt for j, opt in enumerate(options)}
        correct_label = next(k for k, v in labeled.items() if v == correct)
        q_type, q_fn = q_templates[i % len(q_templates)]
        hint = f"💡 Starts with: '{correct[:25]}…'" if cfg["hint"] else ""

        quiz.append({
            "q": q_fn(subj),
            "q_type": q_type,
            "labeled": labeled,
            "answer": correct,
            "correct_label": correct_label,
            "exp": sentence,
            "hint": hint,
            "subject": subj,
        })

        if len(quiz) >= cfg["max_q"]:
            break

    return quiz

# ─────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────
def build_txt(summary, notes, defs) -> str:
    lines = [
        "="*55,
        "AI PERSONAL TUTOR — STUDY NOTES",
        "="*55,
        "",
        "SUMMARY",
        "-"*55,
        summary or "N/A",
        "",
        "KEY NOTES",
        "-"*55,
    ]
    for i, n in enumerate(notes, 1):
        lines.append(f"{i}. {n}")
    lines += ["", "DEFINITIONS", "-"*55]
    for i, (term, meaning) in enumerate(defs, 1):
        lines.append(f"{i}. {term}: {meaning}")
    return "\n".join(lines)

# ─────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return clean_text(text)

def process_content(raw_text: str):
    txt = clean_text(raw_text)
    st.session_state.raw_text = txt
    st.session_state.summary = generate_summary(txt)
    st.session_state.notes = generate_notes(txt)
    st.session_state.defs = extract_definitions(txt)
    st.session_state.flashcards = generate_flashcards(st.session_state.notes, st.session_state.defs)
    st.session_state.quiz = generate_quiz(txt, st.session_state.difficulty)
    st.session_state.fc_idx = 0
    st.session_state.fc_flipped = False
    st.session_state.generated = True
    st.session_state.page = 1

# ─────────────────────────────────────────
# PAGE ROUTER
# ─────────────────────────────────────────
page = st.session_state.page

if page == 0:
    st.markdown("""
    <div class="hero">
      <h1>AI Personal Tutor</h1>
      <p>Paste text or upload a PDF to generate a summary, notes, flashcards, quiz, AI chat, dashboard, and downloads.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="input-choice">
          <div class="input-choice-icon">📄</div>
          <div class="input-choice-title">Paste Text</div>
          <div class="input-choice-desc">Paste study material directly and generate learning tools instantly.</div>
        </div>
        """, unsafe_allow_html=True)
        txt = st.text_area("Paste your content", height=280, placeholder="Paste your notes, article, or study material here...")
        if st.button("Generate from Text", type="primary", use_container_width=True):
            if txt.strip():
                process_content(txt)
                st.rerun()
            else:
                st.error("Please paste some text first.")

    with c2:
        st.markdown("""
        <div class="input-choice">
          <div class="input-choice-icon">📚</div>
          <div class="input-choice-title">Upload PDF</div>
          <div class="input-choice-desc">Upload a PDF file and extract content automatically for learning.</div>
        </div>
        """, unsafe_allow_html=True)
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if st.button("Generate from PDF", use_container_width=True):
            if pdf is not None:
                try:
                    raw = extract_text_from_pdf(pdf)
                    if raw.strip():
                        process_content(raw)
                        st.rerun()
                    else:
                        st.error("Could not extract readable text from the PDF.")
                except Exception as e:
                    st.error(f"PDF processing failed: {e}")
            else:
                st.error("Please upload a PDF first.")

elif page == 1:
    render_stepper(page)
    st.markdown("""
    <div class="page-header">
      <h2>Summary</h2>
      <p>Quick understanding of your material.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="card"><div class="summary-text">{safe_html(st.session_state.summary)}</div></div>', unsafe_allow_html=True)
    render_nav(page)

elif page == 2:
    render_stepper(page)
    st.markdown("""
    <div class="page-header">
      <h2>Definitions & Notes</h2>
      <p>Core concepts extracted from your material.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card"><div class="sec-label">Definitions</div>', unsafe_allow_html=True)
        if st.session_state.defs:
            for term, meaning in st.session_state.defs:
                st.markdown(f'''
                <div class="def-card">
                  <div class="def-term">{safe_html(term)}</div>
                  <div class="def-meaning">{safe_html(meaning)}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No strong definitions found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="sec-label">Key Notes</div>', unsafe_allow_html=True)
        if st.session_state.notes:
            for i, note in enumerate(st.session_state.notes, 1):
                st.markdown(f'''
                <div class="note-row">
                  <div class="note-num">{i:02d}</div>
                  <div class="note-text">{safe_html(note)}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No key notes generated.")
        st.markdown('</div>', unsafe_allow_html=True)

    render_nav(page)

elif page == 3:
    render_stepper(page)
    st.markdown("""
    <div class="page-header">
      <h2>Flashcards</h2>
      <p>Review important concepts interactively.</p>
    </div>
    """, unsafe_allow_html=True)

    cards = st.session_state.flashcards

    if not cards:
        st.warning("No flashcards available.")
    else:
        idx = st.session_state.fc_idx
        card = cards[idx]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.progress((idx + 1) / len(cards))
        st.caption(f"Card {idx + 1} of {len(cards)}")

        if st.session_state.fc_flipped:
            st.markdown(f'<div class="fc-back">{safe_html(card["back"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fc-front">{safe_html(card["front"])}</div>', unsafe_allow_html=True)

        a, b, c = st.columns(3)
        with a:
            if st.button("⬅ Prev", use_container_width=True, disabled=(idx == 0)):
                st.session_state.fc_idx -= 1
                st.session_state.fc_flipped = False
                st.rerun()
        with b:
            if st.button("🔄 Flip", use_container_width=True):
                st.session_state.fc_flipped = not st.session_state.fc_flipped
                st.rerun()
        with c:
            if st.button("Next ➡", use_container_width=True, disabled=(idx == len(cards) - 1)):
                st.session_state.fc_idx += 1
                st.session_state.fc_flipped = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    render_nav(page)

elif page == 4:
    render_stepper(page)
    st.markdown("""
    <div class="page-header">
      <h2>AI Chat</h2>
      <p>Ask questions from your material or general doubts.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    for q, a in st.session_state.chat_history:
        st.markdown(f'''
        <div class="chat-wrap">
          <div class="chat-lbl chat-lbl-u">You</div>
          <div class="chat-u">{safe_html(q)}</div>
        </div>
        <div class="chat-wrap">
          <div class="chat-lbl chat-lbl-b">AI</div>
          <div class="chat-b">{safe_html(a)}</div>
        </div>
        ''', unsafe_allow_html=True)

    question = st.text_input("Ask a question", placeholder="Ask from your document or any related concept...")
    if st.button("Ask AI", type="primary"):
        if question.strip():
            answer = chat_answer(question, st.session_state.raw_text)
            st.session_state.chat_history.append((question, answer))
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    render_nav(page)

elif page == 5:
    render_stepper(page)
    st.markdown(f"""
    <div class="page-header">
      <h2>Quiz <span class="dp-{st.session_state.difficulty.lower()}">{st.session_state.difficulty}</span></h2>
      <p>Test your understanding of the material.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.quiz:
        st.warning("No quiz could be generated from this content.")
        if st.button("Regenerate Quiz", use_container_width=True):
            st.session_state.quiz = generate_quiz(st.session_state.raw_text, st.session_state.difficulty)
            st.rerun()
    else:
        with st.form("quiz_form"):
            user_answers = {}
            for i, q in enumerate(st.session_state.quiz):
                badge = "qb-def" if q["q_type"] == "definition" else ("qb-fill" if q["q_type"] == "fill" else "qb-id")
                st.markdown(f'''
                <div class="q-block">
                  <div class="q-badge {badge}">{q["q_type"]}</div>
                  <div style="font-weight:600; margin-bottom:10px;">Q{i+1}. {safe_html(q["q"])}</div>
                </div>
                ''', unsafe_allow_html=True)

                opts = [f"{k}. {v}" for k, v in q["labeled"].items()]
                user_answers[i] = st.radio(
                    f"Select answer for question {i+1}",
                    opts,
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                if q["hint"]:
                    st.caption(q["hint"])

            submitted = st.form_submit_button("Submit Quiz", type="primary")

        if submitted:
            score = 0
            total = len(st.session_state.quiz)

            for i, q in enumerate(st.session_state.quiz):
                chosen_label = user_answers[i].split(".")[0].strip()
                is_correct = chosen_label == q["correct_label"]
                if is_correct:
                    score += 1

                subj = q["subject"]
                if subj not in st.session_state.q_stats:
                    st.session_state.q_stats[subj] = {"correct": 0, "total": 0}
                st.session_state.q_stats[subj]["total"] += 1
                if is_correct:
                    st.session_state.q_stats[subj]["correct"] += 1

            pct = round((score / total) * 100, 1) if total else 0

            if pct >= 70:
                st.session_state.streak += 1
            else:
                st.session_state.streak = 0

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "score": pct,
                "correct": score,
                "total": total
            })

            if len(st.session_state.live_chart_x) == 0:
                st.session_state.live_chart_x = [1]
                st.session_state.live_chart_y = [pct]
            else:
                st.session_state.live_chart_x.append(len(st.session_state.live_chart_x) + 1)
                st.session_state.live_chart_y.append(pct)

            st.markdown(f'''
            <div class="card">
              <div class="score-big">
                <div class="score-num">{pct}%</div>
                <div class="score-sub">{score}/{total} correct</div>
              </div>
            </div>
            ''', unsafe_allow_html=True)

            for i, q in enumerate(st.session_state.quiz):
                chosen_label = user_answers[i].split(".")[0].strip()
                is_correct = chosen_label == q["correct_label"]
                box_cls = "ok-box" if is_correct else "err-box"
                label = "✅ Correct" if is_correct else f"❌ Wrong — Correct: {q['correct_label']}"
                st.markdown(f'''
                <div class="{box_cls}">
                  <strong>Q{i+1}:</strong> {label}<br>
                  <span>{safe_html(q["exp"])}</span>
                </div>
                ''', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔁 Regenerate Quiz", use_container_width=True):
                    st.session_state.quiz = generate_quiz(st.session_state.raw_text, st.session_state.difficulty)
                    st.rerun()
            with c2:
                if st.button("📊 Go to Dashboard", use_container_width=True):
                    st.session_state.page = 6
                    st.rerun()

    render_nav(page)

elif page == 6:
    render_stepper(page)
    st.markdown("""
    <div class="page-header">
      <h2>Dashboard</h2>
      <p>Track your quiz performance and view a free live chart.</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.history
    q_stats = st.session_state.q_stats

    total_attempts = len(history)
    avg_score = round(sum(h["score"] for h in history) / total_attempts, 1) if total_attempts else 0
    best_score = round(max((h["score"] for h in history), default=0), 1)
    streak = st.session_state.streak

    st.markdown(f"""
    <div class="metric-g">
      <div class="metric-c"><div class="metric-v">{total_attempts}</div><div class="metric-l">Attempts</div></div>
      <div class="metric-c"><div class="metric-v">{avg_score}%</div><div class="metric-l">Average Score</div></div>
      <div class="metric-c"><div class="metric-v">{best_score}%</div><div class="metric-l">Best Score</div></div>
      <div class="metric-c"><div class="metric-v">{streak}</div><div class="metric-l">Current Streak</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])

    with c1:
        st.markdown('<div class="card"><div class="sec-label">Quiz Score Trend</div>', unsafe_allow_html=True)
        if history:
            df = pd.DataFrame(history)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df["score"],
                mode="lines+markers",
                line=dict(color="#7c6ef7", width=3),
                marker=dict(size=8, color="#5eead4"),
                name="Score %"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#13162a",
                font=dict(color="#e8eaf6", family="Inter"),
                xaxis=dict(title="Attempt", color="#9ca3af", showgrid=False),
                yaxis=dict(title="Score %", color="#9ca3af", gridcolor="rgba(255,255,255,0.08)", range=[0, 100]),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Take a quiz to see your performance chart.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="sec-label">Free Live Chart</div>', unsafe_allow_html=True)

        live_col1, live_col2, live_col3 = st.columns([1, 1, 1])

        with live_col1:
            if st.button("➕ Add Live Point", use_container_width=True):
                next_x = len(st.session_state.live_chart_x) + 1
                st.session_state.live_chart_x.append(next_x)
                st.session_state.live_chart_y.append(get_ai_value())
                st.rerun()

        with live_col2:
            if st.button("🧹 Reset Live Chart", use_container_width=True):
                st.session_state.live_chart_x = []
                st.session_state.live_chart_y = []
                st.rerun()

        with live_col3:
            auto_live = st.checkbox("Auto refresh", value=False)

        if len(st.session_state.live_chart_x) == 0:
            st.session_state.live_chart_x = [1]
            st.session_state.live_chart_y = [get_ai_value()]

        live_fig = create_live_ai_chart_figure(
            st.session_state.live_chart_x,
            st.session_state.live_chart_y
        )
        st.plotly_chart(live_fig, use_container_width=True)

        if auto_live:
            time.sleep(2)
            next_x = len(st.session_state.live_chart_x) + 1
            st.session_state.live_chart_x.append(next_x)
            st.session_state.live_chart_y.append(get_ai_value())
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="sec-label">Weak Areas</div>', unsafe_allow_html=True)
        weak_list = []
        for topic, stats in q_stats.items():
            pct = round((stats["correct"] / stats["total"]) * 100, 1) if stats["total"] else 0
            weak_list.append((topic, pct))

        weak_list.sort(key=lambda x: x[1])

        if weak_list:
            for topic, pct in weak_list[:6]:
                st.markdown(f'''
                <div class="weak-row">
                  <div class="weak-name">{safe_html(topic)}</div>
                  <div class="weak-pct">{pct}%</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No topic stats yet.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="sec-label">Recent Attempts</div>', unsafe_allow_html=True)
        if history:
            for h in history[-5:][::-1]:
                st.markdown(f"""
                <div class="note-row">
                  <div class="note-num">⏱</div>
                  <div class="note-text">
                    {h['time']}<br>
                    Score: <strong>{h['score']}%</strong> ({h['correct']}/{h['total']})
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No attempts yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    render_nav(page)

elif page == 7:
    render_stepper(page)
    st.markdown("""
    <div class="page-header">
      <h2>Download</h2>
      <p>Export your summary, notes, and definitions.</p>
    </div>
    """, unsafe_allow_html=True)

    export_text = build_txt(
        st.session_state.summary,
        st.session_state.notes,
        st.session_state.defs
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.download_button(
        "📥 Download TXT Notes",
        data=export_text,
        file_name="ai_personal_tutor_notes.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    render_nav(page, can_go_next=False)