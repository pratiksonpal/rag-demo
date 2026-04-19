import html
import math
import numpy as np
import pandas as pd
import ollama
import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.chunker import get_chunks, STRATEGIES, count_tokens
from src.embedder import get_model, embed_texts, build_faiss_index
from src.retriever import retrieve_chunks
from src.generator import check_ollama, stream_response
from src.logger import get_logger, separator

log = get_logger("app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Playground",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Corporate CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp { background: #EEF2F7; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #032D60 0%, #083178 55%, #0D3E8F 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label span { color: #B8D4F0 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #90CAF9 !important;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(144,202,249,0.25);
}
[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255,255,255,0.08) !important;
    color: #D0E8FF !important;
    border: 1px solid rgba(144,202,249,0.3) !important;
    border-radius: 6px !important;
}
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 6px 6px 0 !important;
    gap: 6px !important;
    border-bottom: 3px solid #0070D2 !important;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 11px 26px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #5A6872 !important;
    border: none !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    background: #0070D2 !important;
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #FFFFFF !important;
    border-radius: 0 0 12px 12px !important;
    padding: 28px 28px 32px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.08) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0070D2 0%, #032D60 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(0,112,210,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 5px 18px rgba(0,112,210,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    border: 1.5px solid #0070D2 !important;
    color: #0070D2 !important;
    background: transparent !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #D8DDE6 !important;
    border-top: 3px solid #0070D2 !important;
    border-radius: 9px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] > div {
    color: #5A6872 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}
[data-testid="stMetricValue"] > div {
    color: #032D60 !important;
    font-weight: 800 !important;
    font-size: 26px !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1.5px solid #D8DDE6 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    background: #FAFBFD !important;
    color: #1A202C !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #0070D2 !important;
    box-shadow: 0 0 0 3px rgba(0,112,210,0.12) !important;
}

/* ── Widget labels — explicit dark color so they're visible on white tab panels ── */
label {
    color: #1A202C !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
/* Sidebar labels stay light */
[data-testid="stSidebar"] label { color: #B8D4F0 !important; font-weight: 500 !important; }

/* ── Headings ── */
h1 { color: #032D60 !important; font-weight: 800 !important; letter-spacing:-0.5px !important; }
h2 { color: #0070D2 !important; font-weight: 700 !important; }
h3 { color: #0070D2 !important; font-weight: 600 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #EBF5FB !important;
    border-radius: 7px !important;
    color: #0070D2 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 1px solid #D0E8F8 !important;
}
.streamlit-expanderContent {
    border: 1px solid #D8DDE6 !important;
    border-top: none !important;
    border-radius: 0 0 7px 7px !important;
    background: #FAFBFD !important;
}

/* ── Divider / Caption ── */
hr { border-color: #D8DDE6 !important; margin: 20px 0 !important; }
.stCaption p { color: #5A6872 !important; font-size: 13px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #EEF2F7; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #B0BEC5; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0070D2; }

/* ── Markdown text — ensure dark on white backgrounds ── */
.stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] p,
.stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] li,
.stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] span {
    color: #1A202C !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div { background: #0070D2 !important; }

/* ── Success / Warning / Error ── */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Chunk highlight colors (Tailwind-200 palette, vivid but light) ────────────
COLORS = [
    "#BFDBFE", "#A7F3D0", "#FDE68A", "#FBCFE8", "#DDD6FE",
    "#A5F3FC", "#FECACA", "#BBF7D0", "#FED7AA", "#BAE6FD",
    "#E9D5FF", "#99F6E4", "#F5D0FE", "#C7D2FE", "#FCA5A5",
    "#6EE7B7", "#FCD34D", "#93C5FD", "#F9A8D4", "#A5B4FC",
]

# Matching darker border/text accent per color for readability
COLOR_BORDERS = [
    "#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6",
    "#06B6D4", "#EF4444", "#22C55E", "#F97316", "#0EA5E9",
    "#A855F7", "#14B8A6", "#D946EF", "#6366F1", "#F87171",
    "#34D399", "#EAB308", "#60A5FA", "#F472B6", "#818CF8",
]


def color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def border_color(i: int) -> str:
    return COLOR_BORDERS[i % len(COLOR_BORDERS)]


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentence-transformer model (all-MiniLM-L6-v2)…")
def load_embedding_model():
    return get_model()


@st.cache_data
def load_sample_doc() -> str:
    with open("data/sample_docs.txt", "r", encoding="utf-8") as f:
        return f.read()


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    for k, v in {
        "source_text": load_sample_doc(),
        "chunks": [],
        "chunk_meta": {},
        "embeddings": None,
        "faiss_index": None,
        "last_chunk_params": None,
        "_applied_just_now": False,
        "response_text": "",       # persists LLM response across rerenders
        "response_for_query": "",  # which query produced the stored response
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Slider + number-input helper (bidirectional sync) ─────────────────────────
def slider_input(label: str, min_val, max_val, default, step, key: str, _wsuffix: str = ""):
    """
    Renders a labelled slider beside a compact number input, both in sync.
    _wsuffix differentiates widget keys when two sliders share the same canonical key.
    """
    is_float = isinstance(step, float)
    coerce = float if is_float else int

    sl_key = f"_sl_{key}{_wsuffix}"
    ni_key = f"_ni_{key}{_wsuffix}"
    ls_key = f"_ls_{key}{_wsuffix}"  # last-synced canonical value for this widget pair

    if key not in st.session_state:
        st.session_state[key] = coerce(default)
    canonical = coerce(st.session_state[key])

    if sl_key not in st.session_state:
        st.session_state[sl_key] = canonical
    if ni_key not in st.session_state:
        st.session_state[ni_key] = canonical
    if ls_key not in st.session_state:
        st.session_state[ls_key] = canonical

    last_sync = coerce(st.session_state[ls_key])
    sl_val = coerce(st.session_state[sl_key])
    ni_val = coerce(st.session_state[ni_key])

    if canonical != last_sync:
        # Canonical was changed by another widget (e.g., shared Top-K from a different tab)
        sl_val = canonical
        ni_val = canonical
        st.session_state[sl_key] = canonical
        st.session_state[ni_key] = canonical
        st.session_state[ls_key] = canonical
    elif sl_val != canonical:
        # This slider was moved
        canonical = sl_val
        st.session_state[key] = canonical
        st.session_state[ni_key] = canonical
        st.session_state[ls_key] = canonical
    elif ni_val != canonical:
        # This number input was edited
        canonical = ni_val
        st.session_state[key] = canonical
        st.session_state[sl_key] = canonical
        st.session_state[ls_key] = canonical

    col_s, col_n = st.columns([5, 1])
    with col_s:
        st.slider(label, coerce(min_val), coerce(max_val), step=step, key=sl_key)
    with col_n:
        st.number_input(
            label,
            min_value=coerce(min_val), max_value=coerce(max_val),
            step=step, key=ni_key,
            label_visibility="hidden",
            format="%.2f" if is_float else "%d",
        )
    return canonical

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 RAG Play")
    st.markdown("---")

    st.markdown("### 📄 Document")
    src_option = st.radio("Source", ["Sample Document", "Upload .txt file"], label_visibility="collapsed")
    if src_option == "Upload .txt file":
        uploaded = st.file_uploader("Upload", type=["txt"], label_visibility="collapsed")
        if uploaded:
            new_src = uploaded.read().decode("utf-8")
            if new_src != st.session_state.source_text:
                st.session_state.source_text = new_src
                st.session_state.chunks = []
                st.session_state.chunk_meta = {}
                st.session_state.embeddings = None
                st.session_state.faiss_index = None
                if "source_editor" in st.session_state:
                    del st.session_state["source_editor"]
    else:
        sample = load_sample_doc()
        if st.session_state.source_text != sample:
            st.session_state.source_text = sample
            st.session_state.chunks = []
            st.session_state.chunk_meta = {}
            st.session_state.embeddings = None
            st.session_state.faiss_index = None
            if "source_editor" in st.session_state:
                del st.session_state["source_editor"]

    st.markdown("---")
    st.markdown("### 🤖 Ollama Model")
    try:
        _installed = [m.model for m in ollama.list().models]
    except Exception:
        _installed = []
    _model_options = _installed if _installed else ["tinyllama:latest", "llama3.2:latest"]
    _default_idx = next((i for i, m in enumerate(_model_options) if "tinyllama" in m), 0)
    model_name = st.selectbox("LLM", _model_options, index=_default_idx, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ System Prompt")
    system_message = st.text_area(
        "System",
        value=(
            "You are a helpful AI assistant. Answer the user's question using only "
            "the provided document excerpts.\n"
            "Be direct and concise. Use bullet points or numbered lists where appropriate.\n"
            "Do NOT repeat or reference excerpt labels (e.g. 'Excerpt 1') in your answer.\n"
            "If the answer is not in the excerpts, say 'I don't know' — do not make up information."
        ),
        height=160,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#7EB3E8;text-align:center;line-height:2">'
        '🔒 Fully local &nbsp;·&nbsp; No API keys<br>'
        '📦 FAISS &nbsp;·&nbsp; sentence-transformers<br>'
        '⚡ Ollama local inference</div>',
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _log_chunk_table(
    chunks: list, strategy: str, meta: dict, source_text: str, chunk_size: int, overlap: int
) -> None:
    """Log a math breakdown + strategy-specific ASCII table for every Apply click."""
    if not chunks:
        return

    PREVIEW = 55

    # ── Generic table renderer ────────────────────────────────────────────────
    def _render(headers: list, rows: list) -> None:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        sep = "  +" + "+".join("-" * (w + 2) for w in widths) + "+"
        def _row(cells):
            return "  |" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(cells, widths)) + "|"
        log.info(sep)
        log.info(_row(headers))
        log.info(sep)
        for row in rows:
            log.info(_row(row))
        log.info(sep)

    def _preview(s: str) -> str:
        p = s.replace("\n", " ").strip()
        return (p[:PREVIEW - 3] + "...") if len(p) > PREVIEW else p

    def _pct(n: int, total: int) -> str:
        return f"{100 * n / total:.1f}%" if total else "?"

    # ── Math breakdown (strategy-specific) ───────────────────────────────────
    doc_len = len(source_text)
    n       = len(chunks)
    sizes   = [len(c) for c in chunks]

    log.info("  ── Math Breakdown ─────────────────────────────────────────────")

    if strategy == "Fixed Character":
        step   = max(chunk_size - overlap, 1)
        approx = math.ceil(doc_len / step)
        log.info("  Formula : ⌈ doc_chars ÷ (chunk_size − overlap) ⌉")
        log.info("  Values  : ⌈ %d ÷ (%d − %d) ⌉ = ⌈ %d ÷ %d ⌉ = ⌈ %.2f ⌉ ≈ %d  →  actual: %d",
                 doc_len, chunk_size, overlap, doc_len, step, doc_len / step, approx, n)
        log.info("  Note    : effective step = chunk_size − overlap = %d − %d = %d chars", chunk_size, overlap, step)

    elif strategy == "Recursive Character":
        avg = int(sum(sizes) / n) if n else 0
        log.info("  Formula : hierarchical split — try \\\\n\\\\n → \\\\n → '. ' → ' ' → '' until chunk ≤ %d chars", chunk_size)
        log.info("  Values  : %d chars  →  %d chunks  (avg %d | min %d | max %d)",
                 doc_len, n, avg, min(sizes), max(sizes))
        ov_pairs = [
            next((sz for sz in range(min(len(chunks[i]), len(chunks[i+1]), overlap+10), 0, -1)
                  if chunks[i].endswith(chunks[i+1][:sz])), 0)
            for i in range(n - 1)
        ]
        nonzero = sum(1 for v in ov_pairs if v > 0)
        log.info("  Overlap : setting=%d chars | only applied when merging pieces, not at separator cuts", overlap)
        log.info("  Result  : %d of %d boundaries show actual shared text (rest split at natural separators)",
                 nonzero, n - 1)

    elif strategy == "Parent-Child":
        pc        = meta.get("parent_count", "?")
        psize     = meta.get("parent_size", chunk_size * 2)
        cov       = meta.get("child_overlap", max(0, overlap // 2))
        p_step    = max(psize - overlap, 1)
        c_step    = max(chunk_size - cov, 1)
        p_approx  = math.ceil(doc_len / p_step)
        c_per_p   = f"{n / pc:.1f}" if isinstance(pc, int) and pc else "?"
        log.info("  Pass 1 — Parent split")
        log.info("    Formula : ⌈ doc_chars ÷ (parent_size − overlap) ⌉")
        log.info("    Values  : ⌈ %d ÷ (%d − %d) ⌉ = ⌈ %d ÷ %d ⌉ = ⌈ %.2f ⌉ ≈ %d  →  actual: %s",
                 doc_len, psize, overlap, doc_len, p_step, doc_len / p_step, p_approx, pc)
        log.info("  Pass 2 — Child split (per parent)")
        log.info("    Formula : ⌈ parent_chars ÷ (child_size − child_overlap) ⌉  for each parent")
        log.info("    Values  : child_size=%d | child_overlap=%d | step=%d", chunk_size, cov, c_step)
        log.info("    Result  : %s parents  →  %d child chunks  (avg %s children/parent)", pc, n, c_per_p)

    elif strategy == "Token (Sentence-Transformers)":
        tpc      = meta.get("tokens_per_chunk", min(chunk_size, 256))
        tok_ov   = meta.get("token_overlap", overlap)
        t_step   = max(tpc - tok_ov, 1)
        total_t  = count_tokens(source_text)
        approx   = math.ceil(total_t / t_step)
        log.info("  Formula : ⌈ total_tokens ÷ (tokens_per_chunk − token_overlap) ⌉")
        log.info("  Values  : ⌈ %d ÷ (%d − %d) ⌉ = ⌈ %d ÷ %d ⌉ = ⌈ %.2f ⌉ ≈ %d  →  actual: %d",
                 total_t, tpc, tok_ov, total_t, t_step, total_t / t_step, approx, n)
        log.info("  Note    : total_tokens=%d | tokens_per_chunk=%d | token_overlap=%d | model=all-MiniLM-L6-v2",
                 total_t, tpc, tok_ov)

    elif strategy == "Markdown Header":
        h1 = meta.get("h1_count", 0)
        h2 = meta.get("h2_count", 0)
        h3 = meta.get("h3_count", 0)
        log.info("  Formula : 1 section per heading boundary  (chunk_size / overlap not applied)")
        log.info("  Values  : %d H1  +  %d H2  +  %d H3  =  %d heading boundaries  →  %d sections",
                 h1, h2, h3, h1 + h2 + h3, n)

    elif strategy == "Python Code":
        avg = int(sum(sizes) / n) if n else 0
        log.info("  Formula : language-aware recursive split  (class → def → \\\\n\\\\n → \\\\n → statements ≤ %d chars)", chunk_size)
        log.info("  Values  : %d chars  →  %d chunks  (avg %d | min %d | max %d)",
                 doc_len, n, avg, min(sizes), max(sizes))

    log.info("  ── Chunk Table ─────────────────────────────────────────────────")

    # ── Strategy-specific table ───────────────────────────────────────────────
    if strategy == "Fixed Character":
        headers = ["#", "Start", "End", "Chars", "Overlap→Next", "Preview"]
        rows    = []
        pos     = 0
        for i, chunk in enumerate(chunks):
            start = pos
            end   = start + len(chunk)
            if i < n - 1:
                # actual shared suffix/prefix length between this and next chunk
                nxt     = chunks[i + 1]
                actual  = next(
                    (sz for sz in range(min(len(chunk), len(nxt), overlap + 10), 0, -1)
                     if chunk.endswith(nxt[:sz])),
                    0,
                )
                ov_str = f"{actual} chars"
            else:
                ov_str = "—"
            rows.append([i + 1, start, end, len(chunk), ov_str, _preview(chunk)])
            pos = end - overlap
        _render(headers, rows)

    elif strategy in ("Recursive Character", "Python Code"):
        headers = ["#", "Chars", "% of Doc", "Cumul. Chars", "Overlap→Next (chars)", "Preview"]
        cumul   = 0
        rows    = []
        for i, chunk in enumerate(chunks):
            cumul += len(chunk)
            if i < n - 1:
                nxt        = chunks[i + 1]
                ov_actual  = next(
                    (sz for sz in range(min(len(chunk), len(nxt), overlap + 10), 0, -1)
                     if chunk.endswith(nxt[:sz])),
                    0,
                )
                ov_str = str(ov_actual)
            else:
                ov_str = "—"
            rows.append([i + 1, len(chunk), _pct(len(chunk), doc_len), cumul, ov_str, _preview(chunk)])
        _render(headers, rows)

    elif strategy == "Parent-Child":
        parent_sizes    = meta.get("parent_sizes", [])
        child_to_parent = meta.get("child_to_parent", [])

        if parent_sizes and child_to_parent:
            from collections import Counter
            child_counts = Counter(child_to_parent)

            log.info("  [Table 1 / 2]  Parent Chunks")
            p_headers = ["Parent #", "Chars", "Children", "Child # Range"]
            p_rows    = []
            child_idx = 1
            for pi, psize in enumerate(parent_sizes, 1):
                cnt       = child_counts.get(pi, 0)
                rng       = f"{child_idx}–{child_idx + cnt - 1}" if cnt else "—"
                p_rows.append([pi, psize, cnt, rng])
                child_idx += cnt
            _render(p_headers, p_rows)

            log.info("  [Table 2 / 2]  Child Chunks  (these are indexed for retrieval)")
            c_headers = ["Child #", "Parent #", "Chars", "Preview"]
            c_rows    = [
                [i + 1,
                 child_to_parent[i] if i < len(child_to_parent) else "?",
                 len(c),
                 _preview(c)]
                for i, c in enumerate(chunks)
            ]
            _render(c_headers, c_rows)
        else:
            _render(["Child #", "Chars", "Preview"],
                    [[i + 1, len(c), _preview(c)] for i, c in enumerate(chunks)])

    elif strategy == "Token (Sentence-Transformers)":
        headers = ["#", "Tokens", "Chars", "Chars / Token", "Preview"]
        rows    = []
        for i, chunk in enumerate(chunks):
            tok = count_tokens(chunk)
            cpt = f"{len(chunk) / tok:.1f}" if tok else "?"
            rows.append([i + 1, tok, len(chunk), cpt, _preview(chunk)])
        _render(headers, rows)

    elif strategy == "Markdown Header":
        section_hdrs = meta.get("section_headers", [])
        if section_hdrs:
            headers = ["#", "H-Level", "Header Text", "Chars", "Preview"]
            rows    = [
                [i + 1,
                 sh["level"],
                 sh["text"][:35],
                 len(chunk),
                 _preview(chunk)]
                for i, (chunk, sh) in enumerate(zip(chunks, section_hdrs))
            ]
            _render(headers, rows)
        else:
            _render(["#", "Chars", "Preview"],
                    [[i + 1, len(c), _preview(c)] for i, c in enumerate(chunks)])


def _render_ui_chunk_breakdown(
    chunks: list, strategy: str, meta: dict,
    source_text: str, chunk_size: int, overlap: int,
) -> None:
    """Render math breakdown + strategy-specific dataframe(s) inside an expander."""
    doc_len = len(source_text)
    n       = len(chunks)
    sizes   = [len(c) for c in chunks]

    # ── Math summary block ────────────────────────────────────────────────────
    if strategy == "Fixed Character":
        step   = max(chunk_size - overlap, 1)
        approx = math.ceil(doc_len / step)
        math_md = (
            f"**Formula:** `⌈ doc_chars ÷ (chunk_size − overlap) ⌉`  \n"
            f"**Applied:** `⌈ {doc_len} ÷ ({chunk_size} − {overlap}) ⌉ = ⌈ {doc_len} ÷ {step} ⌉"
            f" = ⌈ {doc_len/step:.2f} ⌉ ≈ {approx}` → **actual: {n}**  \n"
            f"**Note:** effective step = chunk_size − overlap = {chunk_size} − {overlap} = **{step} chars**"
        )
    elif strategy == "Recursive Character":
        avg      = int(sum(sizes) / n) if n else 0
        ov_pairs = [
            next((sz for sz in range(min(len(chunks[i]), len(chunks[i+1]), overlap+10), 0, -1)
                  if chunks[i].endswith(chunks[i+1][:sz])), 0)
            for i in range(n - 1)
        ]
        nonzero  = sum(1 for v in ov_pairs if v > 0)
        math_md  = (
            f"**Formula:** hierarchical — `\\n\\n → \\n → '. ' → ' ' → ''` until chunk ≤ {chunk_size} chars  \n"
            f"**Applied:** {doc_len} chars → **{n} chunks** (avg {avg} | min {min(sizes)} | max {max(sizes)} chars)  \n"
            f"**Overlap setting:** {overlap} chars — overlap is added only when the splitter **merges** "
            f"split pieces; natural separator cuts produce 0 overlap. "
            f"**{nonzero} of {n-1} boundary/ies** show actual shared text in this split."
        )
    elif strategy == "Parent-Child":
        pc     = meta.get("parent_count", "?")
        psize  = meta.get("parent_size", chunk_size * 2)
        cov    = meta.get("child_overlap", max(0, overlap // 2))
        p_step = max(psize - overlap, 1)
        c_step = max(chunk_size - cov, 1)
        p_apx  = math.ceil(doc_len / p_step)
        avg_c  = f"{n/pc:.1f}" if isinstance(pc, int) and pc else "?"
        math_md = (
            f"**Pass 1 — Parent split**  \n"
            f"Formula: `⌈ doc_chars ÷ (parent_size − overlap) ⌉`  \n"
            f"Applied: `⌈ {doc_len} ÷ ({psize} − {overlap}) ⌉ = ⌈ {doc_len/p_step:.2f} ⌉ ≈ {p_apx}` → **actual: {pc} parents**  \n\n"
            f"**Pass 2 — Child split** (per parent)  \n"
            f"Formula: `⌈ parent_chars ÷ (child_size − child_overlap) ⌉`  \n"
            f"Applied: child_size={chunk_size} | child_overlap={cov} | step={c_step}  \n"
            f"Result: **{pc} parents → {n} child chunks** (avg {avg_c} children/parent)"
        )
    elif strategy == "Token (Sentence-Transformers)":
        tpc     = meta.get("tokens_per_chunk", min(chunk_size, 256))
        tok_ov  = meta.get("token_overlap", overlap)
        t_step  = max(tpc - tok_ov, 1)
        total_t = count_tokens(source_text)
        approx  = math.ceil(total_t / t_step)
        math_md = (
            f"**Formula:** `⌈ total_tokens ÷ (tokens_per_chunk − token_overlap) ⌉`  \n"
            f"**Applied:** `⌈ {total_t} ÷ ({tpc} − {tok_ov}) ⌉ = ⌈ {total_t} ÷ {t_step} ⌉"
            f" = ⌈ {total_t/t_step:.2f} ⌉ ≈ {approx}` → **actual: {n}**  \n"
            f"**Note:** total_tokens={total_t} | tokens_per_chunk={tpc} | token_overlap={tok_ov} | model=all-MiniLM-L6-v2"
        )
    elif strategy == "Markdown Header":
        h1 = meta.get("h1_count", 0)
        h2 = meta.get("h2_count", 0)
        h3 = meta.get("h3_count", 0)
        math_md = (
            f"**Formula:** 1 section per heading boundary (chunk_size / overlap not applied)  \n"
            f"**Applied:** {h1} H1 + {h2} H2 + {h3} H3 = **{h1+h2+h3} heading boundaries** → **{n} sections**"
        )
    elif strategy == "Python Code":
        avg = int(sum(sizes) / n) if n else 0
        math_md = (
            f"**Formula:** `class → def → \\n\\n → \\n → statements` until chunk ≤ {chunk_size} chars  \n"
            f"**Applied:** {doc_len} chars → **{n} chunks** (avg {avg} | min {min(sizes)} | max {max(sizes)} chars)"
        )
    else:
        math_md = ""

    st.markdown(
        '<div style="background:#EBF5FB;border:1px solid #90CAF9;border-left:4px solid #0070D2;'
        'border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:14px">'
        '<div style="font-size:11px;font-weight:700;color:#0070D2;text-transform:uppercase;'
        'letter-spacing:0.6px;margin-bottom:8px">📐 Math Breakdown</div>'
        f'<div style="font-size:13px;color:#1A202C;line-height:1.9">{math_md.replace(chr(10), "<br>")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Strategy-specific dataframe(s) ────────────────────────────────────────
    def _prev(s: str, limit: int = 90) -> str:
        p = s.replace("\n", " ").strip()
        return (p[:limit - 3] + "…") if len(p) > limit else p

    def _pct(v: int, total: int) -> str:
        return f"{100 * v / total:.1f}%" if total else "?"

    if strategy == "Fixed Character":
        rows, pos = [], 0
        for i, chunk in enumerate(chunks):
            start, end = pos, pos + len(chunk)
            if i < n - 1:
                nxt = chunks[i + 1]
                ov_actual = next(
                    (sz for sz in range(min(len(chunk), len(nxt), overlap + 10), 0, -1)
                     if chunk.endswith(nxt[:sz])), 0)
                ov_str = str(ov_actual)
            else:
                ov_str = "—"
            rows.append({"#": i + 1, "Start": start, "End": end,
                         "Chars": len(chunk), "Overlap→Next (chars)": ov_str, "Preview": _prev(chunk)})
            pos = end - overlap
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif strategy in ("Recursive Character", "Python Code"):
        cumul, rows = 0, []
        for i, chunk in enumerate(chunks):
            cumul += len(chunk)
            if i < n - 1:
                nxt       = chunks[i + 1]
                ov_actual = next(
                    (sz for sz in range(min(len(chunk), len(nxt), overlap + 10), 0, -1)
                     if chunk.endswith(nxt[:sz])),
                    0,
                )
                ov_str = str(ov_actual)
            else:
                ov_str = "—"
            rows.append({"#": i + 1, "Chars": len(chunk),
                         "% of Doc": _pct(len(chunk), doc_len),
                         "Cumul. Chars": cumul,
                         "Overlap→Next (chars)": ov_str,
                         "Preview": _prev(chunk)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"ℹ️ **Overlap setting: {overlap} chars.** "
            "RecursiveCharacterTextSplitter only adds overlap when merging split pieces together. "
            "Boundaries at natural separators (paragraph / sentence) show **0** — the cut landed "
            "exactly on a separator so no overlap was needed. Non-zero values appear where pieces "
            "were merged and the overlap window was applied."
        )

    elif strategy == "Parent-Child":
        parent_sizes    = meta.get("parent_sizes", [])
        child_to_parent = meta.get("child_to_parent", [])
        from collections import Counter
        child_counts = Counter(child_to_parent)

        st.markdown("**Table 1 / 2 — Parent Chunks**")
        p_rows, child_idx = [], 1
        for pi, psize in enumerate(parent_sizes, 1):
            cnt = child_counts.get(pi, 0)
            p_rows.append({"Parent #": pi, "Chars": psize, "Children": cnt,
                           "Child # Range": f"{child_idx}–{child_idx + cnt - 1}" if cnt else "—"})
            child_idx += cnt
        st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

        st.markdown("**Table 2 / 2 — Child Chunks** *(indexed for retrieval)*")
        c_rows = [{"Child #": i + 1,
                   "Parent #": child_to_parent[i] if i < len(child_to_parent) else "?",
                   "Chars": len(c), "Preview": _prev(c)}
                  for i, c in enumerate(chunks)]
        st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)

    elif strategy == "Token (Sentence-Transformers)":
        rows = []
        for i, chunk in enumerate(chunks):
            tok = count_tokens(chunk)
            rows.append({"#": i + 1, "Tokens": tok, "Chars": len(chunk),
                         "Chars / Token": f"{len(chunk)/tok:.1f}" if tok else "?",
                         "Preview": _prev(chunk)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif strategy == "Markdown Header":
        section_hdrs = meta.get("section_headers", [])
        rows = []
        for i, chunk in enumerate(chunks):
            sh = section_hdrs[i] if i < len(section_hdrs) else {"level": "—", "text": "—"}
            rows.append({"#": i + 1, "H-Level": sh["level"],
                         "Header Text": sh["text"], "Chars": len(chunk), "Preview": _prev(chunk)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def highlight_source_html(source: str, chunks: list) -> str:
    char_color = [""] * len(source)
    char_border = [""] * len(source)
    pos = 0
    for i, chunk in enumerate(chunks):
        idx = source.find(chunk, max(0, pos - len(chunk)))
        if idx == -1:
            idx = source.find(chunk)
        if idx != -1:
            for j in range(idx, min(idx + len(chunk), len(source))):
                char_color[j] = color(i)
                char_border[j] = border_color(i)
            pos = idx + len(chunk)
    parts = []
    i = 0
    while i < len(source):
        c = char_color[i]
        j = i + 1
        while j < len(source) and char_color[j] == c:
            j += 1
        segment = html.escape(source[i:j]).replace("\n", "<br>")
        if c:
            parts.append(
                f'<span style="background:{c};color:#1A202C;padding:1px 0;'
                f'border-radius:2px;outline:1px solid {char_border[i]}22">{segment}</span>'
            )
        else:
            parts.append(f'<span style="color:#374151">{segment}</span>')
        i = j
    return "".join(parts)


def scrollable(content_html: str, height: int = 520) -> None:
    st.markdown(
        f'<div style="height:{height}px;overflow-y:auto;border:1.5px solid #D8DDE6;'
        f'padding:14px 16px;border-radius:9px;font-size:12.5px;line-height:1.75;'
        f'background:#FFFFFF;color:#1A202C;font-family:"SFMono-Regular",Menlo,monospace;'
        f'box-shadow:inset 0 2px 6px rgba(0,0,0,0.04)">{content_html}</div>',
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = f'<div style="font-size:13px;color:#5A6872;margin-top:3px">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div style="border-left:4px solid #0070D2;padding:6px 0 6px 14px;margin-bottom:18px">'
        f'<div style="font-size:19px;font-weight:700;color:#032D60">{icon} {title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def info_cards(tab_title: str, cards: list) -> None:
    """Styled knowledge panel at the bottom of each tab."""
    n = len(cards)
    cols = 3 if n >= 3 else n
    cards_html = "".join(
        f'<div style="background:#F8FAFD;border:1px solid #D8DDE6;border-left:4px solid {card.get("accent","#0070D2")};'
        f'border-radius:0 9px 9px 0;padding:16px 18px">'
        f'<div style="font-size:13px;font-weight:700;color:#032D60;margin-bottom:8px">{card["icon"]} {card["title"]}</div>'
        f'<div style="font-size:12px;color:#374151;line-height:1.8">{card["body"]}</div>'
        f'</div>'
        for card in cards
    )
    st.markdown(
        f'<div style="margin-top:40px;padding-top:28px;border-top:2px solid #D8DDE6">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">'
        f'<div style="background:linear-gradient(135deg,#0070D2,#032D60);border-radius:6px;'
        f'padding:5px 12px;font-size:11px;font-weight:700;color:#FFF;letter-spacing:0.6px;text-transform:uppercase">📚 How it works</div>'
        f'<div style="font-size:17px;font-weight:700;color:#032D60">{tab_title}</div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);gap:14px">'
        f'{cards_html}</div></div>',
        unsafe_allow_html=True,
    )


# ── App Header ────────────────────────────────────────────────────────────────
_indexed_badge = (
    '<span style="background:#D1FAE5;color:#065F46;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:700">✅ Indexed</span>'
    if st.session_state.faiss_index is not None else
    '<span style="background:#FEF3C7;color:#78350F;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:700">⬜ Not indexed</span>'
)
_chunk_count = len(st.session_state.chunks)

st.markdown(
    f'<div style="background:linear-gradient(135deg,#032D60 0%,#0F52A0 60%,#1565C0 100%);'
    f'padding:22px 30px;border-radius:14px;margin-bottom:22px;'
    f'box-shadow:0 6px 28px rgba(3,45,96,0.32);display:flex;align-items:center;gap:22px">'
    f'<div style="background:rgba(255,255,255,0.12);border-radius:10px;padding:10px 18px;'
    f'font-size:24px;font-weight:900;color:white;letter-spacing:3px;font-family:monospace">RAG</div>'
    f'<div style="flex:1">'
    f'<div style="font-size:24px;font-weight:800;color:white;letter-spacing:-0.4px">RAG Playground</div>'
    f'<div style="font-size:13px;color:rgba(255,255,255,0.65);margin-top:4px">'
    f'Interactive step-by-step demonstration of <b style="color:#90CAF9">Retrieval · Augmentation · Generation</b></div>'
    f'</div>'
    f'<div style="display:flex;gap:30px;flex-shrink:0">'
    + "".join(
        f'<div style="text-align:center">'
        f'<div style="font-size:10px;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:5px">{label}</div>'
        f'<div style="font-size:12px;color:#90CAF9;font-weight:700">{value}</div>'
        f'</div>'
        for label, value in [
            ("Embedding Model", "all-MiniLM-L6-v2"),
            ("Vector Store", "FAISS · IndexFlatIP"),
            ("LLM Backend", model_name),
            ("Pipeline Status", f"{_indexed_badge} · {_chunk_count} chunks"),
        ]
    )
    + f'</div></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["① Text Splitting", "⊕ Vector Embedding", "💬 Response Generation"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT SPLITTING
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    section_header("✂️", "Text Splitting",
                   "Configure how your document is divided into chunks before embedding.")

    with st.expander("📖 Chunking strategy guide"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:14px">'
                '<div style="font-weight:700;color:#1D4ED8;margin-bottom:6px">📏 Fixed Character</div>'
                '<div style="font-size:12px;color:#374151">Splits every N characters with no NLP awareness. '
                'Fast, predictable, O(n). Best for structured/tabular data or quick prototyping.</div>'
                '</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(
                '<div style="background:#F0FDF4;border:1px solid #A7F3D0;border-radius:8px;padding:14px">'
                '<div style="font-weight:700;color:#15803D;margin-bottom:6px">🔄 Recursive Character</div>'
                '<div style="font-size:12px;color:#374151">Tries separators in order: paragraph → sentence → word → character. '
                'Preserves semantic boundaries. <b>Recommended for natural language.</b></div>'
                '</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(
                '<div style="background:#FAF5FF;border:1px solid #DDD6FE;border-radius:8px;padding:14px">'
                '<div style="font-weight:700;color:#6D28D9;margin-bottom:6px">👪 Parent-Child</div>'
                '<div style="font-size:12px;color:#374151">Two-pass split: large parent chunks (2× size) for context, '
                'small child chunks (1× size) for precise retrieval. Used in advanced RAG pipelines.</div>'
                '</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(
                '<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;padding:14px">'
                '<div style="font-weight:700;color:#C2410C;margin-bottom:6px">🔤 Token (Sentence-Transformers)</div>'
                '<div style="font-size:12px;color:#374151">Splits by token count of all-MiniLM-L6-v2 (max 256 tokens). '
                'Guarantees chunks never exceed the embedding model context window.</div>'
                '</div>', unsafe_allow_html=True)
        with c5:
            st.markdown(
                '<div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:8px;padding:14px">'
                '<div style="font-weight:700;color:#0369A1;margin-bottom:6px"># Markdown Header</div>'
                '<div style="font-size:12px;color:#374151">Splits at # / ## / ### heading boundaries. '
                'Preserves document section structure. Best for markdown-formatted content.</div>'
                '</div>', unsafe_allow_html=True)
        with c6:
            st.markdown(
                '<div style="background:#F7FEE7;border:1px solid #D9F99D;border-radius:8px;padding:14px">'
                '<div style="font-weight:700;color:#4D7C0F;margin-bottom:6px">🐍 Python Code</div>'
                '<div style="font-size:12px;color:#374151">RecursiveCharacterTextSplitter tuned for Python: '
                'splits at class/def boundaries first. Best for Python source files.</div>'
                '</div>', unsafe_allow_html=True)

    # Controls
    col1, col2, col3, col4 = st.columns([2, 3, 3, 1])
    with col1:
        strategy = st.selectbox("Split Strategy", STRATEGIES)
    with col2:
        chunk_size = slider_input("Chunk Size (chars)", 100, 1200, 500, 50, "chunk_size")
    with col3:
        overlap = slider_input("Overlap (chars)", 0, 300, 50, 10, "overlap_size")
    with col4:
        st.markdown("<br><br>", unsafe_allow_html=True)
        apply_btn = st.button("Apply ▶", type="primary", use_container_width=True)

    current_source = st.session_state.get("source_editor", st.session_state.source_text)

    if apply_btn:
        with st.spinner("Splitting document…"):
            st.session_state.source_text = current_source
            separator(log)
            log.info("TAB1 apply | strategy=%s  chunk_size=%d  overlap=%d  doc_len=%d",
                     strategy, chunk_size, overlap, len(current_source))
            _chunks, _meta = get_chunks(current_source, strategy, chunk_size, overlap)
            st.session_state.chunks = _chunks
            st.session_state.chunk_meta = _meta
            st.session_state.last_chunk_params = (strategy, chunk_size, overlap, current_source[:64])
            st.session_state.embeddings = None
            st.session_state.faiss_index = None
            log.info("TAB1 result | chunks=%d", len(st.session_state.chunks))
            _log_chunk_table(_chunks, strategy, _meta, current_source, chunk_size, overlap)
            st.session_state._applied_just_now = True

    chunks = st.session_state.chunks
    chunk_meta = st.session_state.chunk_meta

    # ── Dynamic metrics — built per-strategy ─────────────────────────────────
    def _metrics(strat, src, cks, meta, cs, ov):
        """Return [(label, value)] relevant to the active strategy."""
        items = [("Total Chars", f"{len(src):,}")]

        # Total Tokens: real-time for Token strategy; dash otherwise
        if strat == "Token (Sentence-Transformers)":
            items.append(("Total Tokens", f"{count_tokens(src):,}"))
        else:
            items.append(("Total Tokens", "—"))

        if not cks:
            return items

        sizes = [len(c) for c in cks]
        avg = int(np.mean(sizes))

        if strat == "Parent-Child":
            items += [
                ("Parent Chunks",  str(meta.get("parent_count", "—"))),
                ("Child Chunks",   str(len(cks))),
                ("Avg Child Size", f"{avg} chars"),
                ("Smallest Child", f"{min(sizes)} chars"),
                ("Largest Child",  f"{max(sizes)} chars"),
                ("Child Overlap",  f"{meta.get('child_overlap', '—')} chars"),
            ]
        elif strat == "Token (Sentence-Transformers)":
            items += [
                ("Total Chunks",   str(len(cks))),
                ("Tokens / Chunk", str(meta.get("tokens_per_chunk", min(cs, 256)))),
                ("Token Overlap",  str(meta.get("token_overlap", "—"))),
                ("Avg Chunk Size", f"{avg} chars"),
                ("Smallest",       f"{min(sizes)} chars"),
                ("Largest",        f"{max(sizes)} chars"),
            ]
        elif strat == "Markdown Header":
            items += [
                ("Sections Found",   str(len(cks))),
                ("H1 Sections",      str(meta.get("h1_count", "—"))),
                ("H2 Sections",      str(meta.get("h2_count", "—"))),
                ("H3 Sections",      str(meta.get("h3_count", "—"))),
                ("Avg Section Size", f"{avg} chars"),
            ]
        else:  # Fixed Character / Recursive Character / Python Code
            items += [
                ("Total Chunks", str(len(cks))),
                ("Avg Size",     f"{avg} chars"),
                ("Smallest",     f"{min(sizes)} chars"),
                ("Largest",      f"{max(sizes)} chars"),
                ("Overlap",      f"{ov} chars"),
            ]
        return items

    metric_items = _metrics(strategy, current_source, chunks, chunk_meta, chunk_size, overlap)
    cols = st.columns(len(metric_items))
    for col, (label, value) in zip(cols, metric_items):
        col.metric(label, value)

    # ── Chunk Breakdown Table (auto-opens on Apply) ───────────────────────────
    if chunks and st.session_state.last_chunk_params:
        _app_strategy, _app_cs, _app_ov, _ = st.session_state.last_chunk_params
        _expand_now = st.session_state.get("_applied_just_now", False)
        st.session_state._applied_just_now = False   # reset for subsequent renders
        with st.expander("📊 Chunk Breakdown Table", expanded=_expand_now):
            _render_ui_chunk_breakdown(
                chunks, _app_strategy, chunk_meta,
                st.session_state.source_text, _app_cs, _app_ov,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_src, col_chunks = st.columns([1, 1])

    with col_src:
        st.markdown(
            '<div style="font-size:14px;font-weight:600;color:#032D60;margin-bottom:8px">'
            '📝 Source Document <span style="font-size:11px;font-weight:400;color:#5A6872">'
            '— paste or type your text</span></div>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "source_doc",
            value=st.session_state.source_text,
            height=540,
            key="source_editor",
            label_visibility="collapsed",
            placeholder="Paste your document text here…",
        )

    with col_chunks:
        st.markdown(
            '<div style="font-size:14px;font-weight:600;color:#032D60;margin-bottom:8px">'
            f'🧩 Generated Chunks '
            f'<span style="background:#0070D2;color:white;border-radius:20px;padding:2px 10px;font-size:11px">'
            f'{len(chunks)}</span></div>',
            unsafe_allow_html=True,
        )
        if chunks:
            cards = ""
            for i, chunk in enumerate(chunks):
                preview = html.escape(chunk[:200].replace("\n", " "))
                if len(chunk) > 200:
                    preview += "…"
                cards += (
                    f'<div style="margin-bottom:10px;padding:10px 13px;background:{color(i)};'
                    f'border-left:4px solid {border_color(i)};border-radius:0 8px 8px 0;'
                    f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
                    f'<div style="font-size:11px;font-weight:700;color:#032D60;margin-bottom:4px">'
                    f'Chunk {i + 1} &nbsp;·&nbsp; {len(chunk)} chars</div>'
                    f'<div style="font-size:11px;color:#1A202C">{preview}</div>'
                    f'</div>'
                )
            scrollable(cards, height=540)
        else:
            st.info("Click **Apply ▶** to generate chunks.")

    if chunks:
        with st.expander("🎨 Chunk highlight map — see how chunks map onto the source"):
            highlighted = highlight_source_html(st.session_state.source_text, chunks)
            st.markdown(
                f'<div style="max-height:400px;overflow-y:auto;border:1.5px solid #D8DDE6;'
                f'padding:14px 16px;border-radius:9px;font-size:12.5px;line-height:1.75;'
                f'background:#FFFFFF;color:#1A202C;font-family:monospace">{highlighted}</div>',
                unsafe_allow_html=True,
            )

    # ── Details section ───────────────────────────────────────────────────────
    info_cards("Text Splitting — Technical Reference", [
        {
            "icon": "❓", "title": "Why chunk at all?", "accent": "#0070D2",
            "body": (
                "LLMs and embedding models have a fixed <b>context window</b> (typically 256–512 tokens). "
                "A full document rarely fits. Chunking divides it into pieces that:<br>"
                "① Fit within model limits &nbsp; ② Carry focused meaning &nbsp; ③ Enable precise retrieval"
            ),
        },
        {
            "icon": "📏", "title": "Fixed Character Strategy", "accent": "#3B82F6",
            "body": (
                "Splits at every <code>N</code> characters regardless of words or sentences. "
                "Simple O(n) algorithm, zero NLP overhead. Produces uniform-size chunks but may cut "
                "mid-sentence. Best for: logs, CSVs, structured records."
            ),
        },
        {
            "icon": "🔄", "title": "Recursive Character Strategy", "accent": "#10B981",
            "body": (
                "Hierarchically tries separators: <code>[\\n\\n, \\n, '. ', ' ', '']</code>. "
                "Respects paragraph → sentence → word boundaries. Produces semantically coherent chunks. "
                "<b>Recommended default</b> for prose, documentation, and Q&A corpora."
            ),
        },
        {
            "icon": "👪", "title": "Parent-Child Strategy", "accent": "#8B5CF6",
            "body": (
                "Two-pass split: <b>parent</b> chunks (2× size) preserve broad context; "
                "<b>child</b> chunks (1× size) are indexed for retrieval. "
                "At query time, child matches are retrieved, then expanded to parent for richer context."
            ),
        },
        {
            "icon": "🔤", "title": "Token (Sentence-Transformers)", "accent": "#C2410C",
            "body": (
                "Splits by <b>token count</b> rather than character count, using the same tokeniser as all-MiniLM-L6-v2. "
                "Chunk size is clamped to the model's 256-token limit. "
                "Ensures every chunk can be embedded without truncation."
            ),
        },
        {
            "icon": "#", "title": "Markdown Header Strategy", "accent": "#0369A1",
            "body": (
                "Splits text at <code>#</code>, <code>##</code>, and <code>###</code> heading boundaries. "
                "Each section becomes one chunk, preserving document hierarchy. "
                "If no headers exist the full text is returned as one chunk."
            ),
        },
        {
            "icon": "🐍", "title": "Python Code Strategy", "accent": "#4D7C0F",
            "body": (
                "Uses <code>RecursiveCharacterTextSplitter.from_language(PYTHON)</code>. "
                "Separators prioritise <code>class</code>/<code>def</code> boundaries, then blank lines, then statements. "
                "Keeps logical code units intact for code-search RAG."
            ),
        },
        {
            "icon": "⚙️", "title": "Chunk Size & Overlap", "accent": "#F59E0B",
            "body": (
                "<b>Too small</b> → loses context, retrieval noise. "
                "<b>Too large</b> → dilutes relevance, may exceed context window.<br>"
                "<b>Overlap</b> prevents boundary information loss — typically 10–15% of chunk size. "
                "Rule of thumb: <b>300–600 chars</b>, overlap <b>30–60 chars</b>."
            ),
        },
        {
            "icon": "📊", "title": "Impact on RAG Quality", "accent": "#EF4444",
            "body": (
                "Chunking strategy is often the <b>highest-impact</b> RAG parameter. "
                "Poor chunking → retrieved context is irrelevant or incomplete → LLM hallucinates. "
                "Good chunking → retrieved chunks are precise, self-contained, and meaningful."
            ),
        },
    ])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — VECTOR EMBEDDING
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    section_header("⊕", "Vector Embedding & Semantic Search",
                   "Transform text chunks into numerical vectors, index them in FAISS, and visualize semantic similarity.")

    model = load_embedding_model()

    col_build, col_status = st.columns([2, 5])
    with col_build:
        build_btn = st.button("⚡ Build FAISS Index", type="primary")
    with col_status:
        if st.session_state.embeddings is not None:
            emb_shape = st.session_state.embeddings.shape
            st.markdown(
                f'<div style="margin-top:8px;padding:8px 14px;background:#D1FAE5;border-radius:7px;'
                f'font-size:13px;color:#065F46;font-weight:600">'
                f'✅ Index ready — {emb_shape[0]} chunks × {emb_shape[1]} dimensions (FAISS IndexFlatIP)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin-top:8px;padding:8px 14px;background:#FEF3C7;border-radius:7px;'
                'font-size:13px;color:#78350F">'
                '⚠️ No index yet — apply chunking in Tab 1, then build the index here.</div>',
                unsafe_allow_html=True,
            )

    if build_btn:
        if not st.session_state.chunks:
            st.warning("Go to the Text Splitting tab and apply chunking first.")
        else:
            with st.spinner(f"Encoding {len(st.session_state.chunks)} chunks with all-MiniLM-L6-v2…"):
                separator(log)
                log.info("TAB2 build index | chunks=%d", len(st.session_state.chunks))
                embs, idx = build_faiss_index(model, st.session_state.chunks)
                st.session_state.embeddings = embs
                st.session_state.faiss_index = idx
                log.info("TAB2 index ready | shape=%s", embs.shape)
            st.success(
                f"✅ {len(st.session_state.chunks)} chunks encoded → "
                f"{embs.shape[1]}-dimensional vectors → FAISS IndexFlatIP"
            )

    if st.session_state.embeddings is not None:
        embs = st.session_state.embeddings
        chunks = st.session_state.chunks

        n_components = min(2, embs.shape[0], embs.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(embs)

        col_q, col_k = st.columns([5, 2])
        with col_q:
            emb_query = st.text_input(
                "🔍 Enter a query to find semantically similar chunks",
                placeholder="e.g.  What is the annual leave policy?",
                key="emb_query",
            )
        with col_k:
            top_k = slider_input("Top K", 1, min(10, len(chunks)), 3, 1, "shared_top_k", _wsuffix="_t2")

        query_2d = None
        retrieved_idxs: list = []
        retrieved_scores: list = []

        if emb_query:
            q_emb = embed_texts(model, [emb_query])
            retrieved_idxs, retrieved_scores = retrieve_chunks(st.session_state.faiss_index, q_emb, top_k)
            query_2d = pca.transform(q_emb)[0]

        # ── Build scatter plot ───────────────────────────────────────────────
        fig = go.Figure()

        # Draw dashed retrieval lines from query to retrieved chunks
        if query_2d is not None:
            for ri in retrieved_idxs:
                fig.add_shape(
                    type="line",
                    x0=query_2d[0], y0=query_2d[1],
                    x1=float(coords[ri][0]), y1=float(coords[ri][1]),
                    line=dict(color="#EF4444", width=1.5, dash="dot"),
                    layer="below",
                )

        # Non-retrieved chunks
        non_retrieved = [i for i in range(len(chunks)) if i not in retrieved_idxs]
        if non_retrieved:
            fig.add_trace(go.Scatter(
                x=[coords[i][0] for i in non_retrieved],
                y=[coords[i][1] for i in non_retrieved],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=[color(i) for i in non_retrieved],
                    line=dict(width=1.5, color=[border_color(i) for i in non_retrieved]),
                    symbol="circle",
                ),
                text=[f"C{i + 1}" for i in non_retrieved],
                textposition="top center",
                textfont=dict(size=9, color="#374151"),
                hovertext=[
                    f"<b>Chunk {i + 1}</b><br>"
                    f"Size: {len(chunks[i])} chars<br><br>"
                    f"<i>{html.escape(chunks[i][:150])}…</i>"
                    for i in non_retrieved
                ],
                hoverinfo="text",
                name="📄 Chunk",
                showlegend=True,
            ))

        # Retrieved chunks
        for rank_idx, (ri, score) in enumerate(zip(retrieved_idxs, retrieved_scores)):
            fig.add_trace(go.Scatter(
                x=[coords[ri][0]],
                y=[coords[ri][1]],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=color(ri),
                    line=dict(width=3, color="#EF4444"),
                    symbol="circle",
                ),
                text=[f"#{rank_idx + 1}"],
                textposition="middle center",
                textfont=dict(size=10, color="#7F1D1D", family="Inter Bold"),
                hovertext=(
                    f"<b>Rank #{rank_idx + 1} — Chunk {ri + 1}</b><br>"
                    f"Cosine Similarity: <b>{score:.4f}</b><br>"
                    f"Size: {len(chunks[ri])} chars<br><br>"
                    f"<i>{html.escape(chunks[ri][:160])}…</i>"
                ),
                hoverinfo="text",
                name=f"🎯 Rank #{rank_idx + 1} — Chunk {ri + 1} (sim: {score:.3f})",
                showlegend=True,
            ))

        # Query point
        if query_2d is not None:
            fig.add_trace(go.Scatter(
                x=[query_2d[0]],
                y=[query_2d[1]],
                mode="markers+text",
                marker=dict(size=22, color="#EF4444", symbol="star", line=dict(width=2, color="#7F1D1D")),
                text=["Query"],
                textposition="top center",
                textfont=dict(size=11, color="#7F1D1D", family="Inter Bold"),
                hovertext=f"<b>Your Query</b><br><i>{html.escape(emb_query)}</i>",
                hoverinfo="text",
                name="⭐ Your Query",
                showlegend=True,
            ))

        var1 = pca.explained_variance_ratio_[0] if n_components >= 1 else 0
        var2 = pca.explained_variance_ratio_[1] if n_components >= 2 else 0

        fig.update_layout(
            title=dict(
                text="Semantic Embedding Space — 2D PCA Projection of Chunk Vectors",
                font=dict(size=15, color="#032D60", family="Inter"),
                x=0,
            ),
            xaxis=dict(
                title=f"Semantic Axis 1 — PC1 (captures {var1:.1%} of total variance)",
                title_font=dict(size=12, color="#5A6872"),
                gridcolor="#EEF2F7",
                zerolinecolor="#CBD5E0",
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title=f"Semantic Axis 2 — PC2 (captures {var2:.1%} of total variance)",
                title_font=dict(size=12, color="#5A6872"),
                gridcolor="#EEF2F7",
                zerolinecolor="#CBD5E0",
                tickfont=dict(size=10),
            ),
            legend=dict(
                title=dict(text="Legend", font=dict(size=12, color="#032D60")),
                font=dict(size=11),
                bordercolor="#D8DDE6",
                borderwidth=1,
                bgcolor="#FFFFFF",
                x=1.01,
                y=1,
            ),
            height=520,
            plot_bgcolor="#FAFBFD",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=60, b=60, l=60, r=200),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#D8DDE6",
                font_size=12,
                font_family="Inter",
            ),
            annotations=[
                dict(
                    text=(
                        "💡 <b>How to read this chart:</b> Each dot is a text chunk projected "
                        "into 2D.<br>Dots that are <b>closer together</b> have more similar meaning. "
                        "The ⭐ query point<br>and 🎯 retrieved chunks are connected by dashed lines."
                    )
                    if query_2d is not None else
                    "💡 <b>Tip:</b> Enter a query above to see which chunks are semantically closest.",
                    xref="paper", yref="paper",
                    x=0, y=-0.18,
                    xanchor="left",
                    showarrow=False,
                    font=dict(size=11, color="#5A6872"),
                    align="left",
                )
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Retrieved chunks displayed in a grid
        if emb_query and retrieved_idxs:
            st.markdown(
                '<div style="font-size:15px;font-weight:700;color:#032D60;'
                'margin-bottom:14px;margin-top:4px">🎯 Top-K Retrieved Chunks</div>',
                unsafe_allow_html=True,
            )
            cols_ret = st.columns(min(len(retrieved_idxs), 3))
            for rank, (ri, score) in enumerate(zip(retrieved_idxs, retrieved_scores)):
                with cols_ret[rank % 3]:
                    bar_pct = int(score * 100)
                    st.markdown(
                        f'<div style="padding:14px;background:{color(ri)};border-left:4px solid {border_color(ri)};'
                        f'border-radius:0 9px 9px 0;height:100%">'
                        f'<div style="font-size:12px;font-weight:700;color:#032D60;margin-bottom:6px">'
                        f'Rank #{rank + 1} &nbsp;·&nbsp; Chunk {ri + 1}</div>'
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                        f'<div style="font-size:11px;color:#374151;font-weight:600">Cosine Sim:</div>'
                        f'<code style="font-size:12px;color:#032D60;font-weight:700">{score:.4f}</code>'
                        f'</div>'
                        f'<div style="background:rgba(255,255,255,0.6);border-radius:4px;height:6px;margin-bottom:10px">'
                        f'<div style="background:{border_color(ri)};width:{bar_pct}%;height:100%;border-radius:4px"></div>'
                        f'</div>'
                        f'<div style="font-size:11px;color:#1A202C;font-family:monospace;line-height:1.6">'
                        f'{html.escape(chunks[ri][:280])}…</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            '<div style="text-align:center;padding:48px;background:#F8FAFD;border:2px dashed #D8DDE6;'
            'border-radius:12px;color:#5A6872">'
            '<div style="font-size:40px;margin-bottom:12px">⊕</div>'
            '<div style="font-size:16px;font-weight:600;color:#032D60;margin-bottom:8px">No Embeddings Yet</div>'
            '<div style="font-size:13px">Apply chunking in <b>Tab 1</b>, then click <b>⚡ Build FAISS Index</b> above.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Details section ───────────────────────────────────────────────────────
    info_cards("Vector Embedding & Semantic Search — Technical Reference", [
        {
            "icon": "🔢", "title": "What are Embeddings?", "accent": "#0070D2",
            "body": (
                "An embedding is a <b>dense numerical vector</b> that captures the meaning of text. "
                "The model maps semantically similar sentences to <b>nearby points</b> in high-dimensional space. "
                "Example: 'employee vacation days' and 'annual leave policy' will have high cosine similarity."
            ),
        },
        {
            "icon": "🤖", "title": "all-MiniLM-L6-v2 Model", "accent": "#3B82F6",
            "body": (
                "A 6-layer <b>MiniLM</b> model distilled from a larger BERT. Produces <b>384-dimensional</b> vectors. "
                "22M parameters, ~50ms per batch on CPU. Trained on 1B+ sentence pairs. "
                "Outputs are L2-normalized — enabling cosine similarity via dot product."
            ),
        },
        {
            "icon": "⚡", "title": "FAISS — IndexFlatIP", "accent": "#10B981",
            "body": (
                "<b>F</b>acebook <b>AI</b> <b>S</b>imilarity <b>S</b>earch. "
                "<code>IndexFlatIP</code> = exact brute-force inner product search. "
                "Since embeddings are L2-normalized, inner product = cosine similarity. "
                "Time complexity: <code>O(n × d)</code> per query. For large corpora, use <code>IndexIVFFlat</code>."
            ),
        },
        {
            "icon": "📉", "title": "PCA — Why 2D?", "accent": "#8B5CF6",
            "body": (
                "Embeddings live in <b>384 dimensions</b> — impossible to visualize directly. "
                "<b>Principal Component Analysis</b> finds the axes of maximum variance and projects "
                "onto 2D. The % on each axis shows how much original structure is preserved. "
                "Proximity in 2D approximates (but doesn't guarantee) similarity."
            ),
        },
        {
            "icon": "📐", "title": "Cosine Similarity", "accent": "#F59E0B",
            "body": (
                "Score ∈ [−1, 1]. <b>1.0</b> = identical meaning · <b>0</b> = unrelated · <b>−1</b> = opposite. "
                "Computed as the dot product of L2-normalized vectors: <code>sim(A,B) = A·B / (|A||B|)</code>. "
                "FAISS returns scores closest to 1.0 as the top matches."
            ),
        },
        {
            "icon": "🔍", "title": "Retrieval (the R in RAG)", "accent": "#EF4444",
            "body": (
                "At query time: ① query is embedded into the same 384D space ② FAISS searches "
                "for the <b>Top-K nearest chunk vectors</b> by cosine similarity ③ those chunks "
                "are returned as context. This is the <b>Retrieval</b> step — no LLM involved yet."
            ),
        },
    ])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESPONSE GENERATION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    section_header("💬", "Response Generation",
                   "Retrieved chunks are injected into the prompt. The LLM generates a grounded, context-aware answer.")

    col_t, col_m, col_k, col_btn = st.columns([3, 3, 2, 2])
    with col_t:
        temperature = slider_input("Temperature", 0.0, 1.0, 0.3, 0.05, "gen_temperature")
    with col_m:
        max_tokens = slider_input("Max Tokens", 200, 2000, 1000, 100, "gen_max_tokens")
    with col_k:
        max_k = max(1, len(st.session_state.chunks)) if st.session_state.chunks else 10
        gen_top_k = slider_input("Top K", 1, min(10, max_k), 3, 1, "shared_top_k", _wsuffix="_t3")
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        gen_btn = st.button("⚡ Generate Response", type="primary", use_container_width=True)

    user_query = st.text_input(
        "💬 Your Question",
        placeholder="Ask something about the document… e.g. What is the leave encashment policy?",
        key="gen_query",
    )

    st.markdown("---")

    retrieved_texts: list = []
    retrieved_idxs_g: list = []

    if user_query and st.session_state.faiss_index is not None:
        model = load_embedding_model()
        q_emb = embed_texts(model, [user_query])
        retrieved_idxs_g, _ = retrieve_chunks(st.session_state.faiss_index, q_emb, gen_top_k)
        retrieved_texts = [st.session_state.chunks[i] for i in retrieved_idxs_g]

    col_prompt, col_response = st.columns(2)

    # ── Prompt Preview ────────────────────────────────────────────────────────
    with col_prompt:
        st.markdown(
            '<div style="font-size:14px;font-weight:700;color:#032D60;margin-bottom:10px">'
            '📋 Prompt Preview <span style="font-size:11px;font-weight:400;color:#5A6872">'
            '— what the LLM receives</span></div>',
            unsafe_allow_html=True,
        )
        sys_html = html.escape(system_message).replace("\n", "<br>")
        prompt_html = (
            f'<div style="background:#EBF5FB;border:1px solid #90CAF9;border-left:4px solid #0070D2;'
            f'border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:10px;color:#1A202C">'
            f'<div style="font-size:11px;font-weight:700;color:#0070D2;text-transform:uppercase;'
            f'letter-spacing:0.6px;margin-bottom:6px">⚙️ System Message</div>'
            f'<div style="font-size:12px">{sys_html}</div></div>'
        )
        if retrieved_texts:
            for i, (txt, ri) in enumerate(zip(retrieved_texts, retrieved_idxs_g)):
                body = html.escape(txt).replace("\n", "<br>")
                prompt_html += (
                    f'<div style="background:{color(ri)};border-left:4px solid {border_color(ri)};'
                    f'border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:8px;color:#1A202C">'
                    f'<div style="font-size:11px;font-weight:700;color:#032D60;text-transform:uppercase;'
                    f'letter-spacing:0.6px;margin-bottom:6px">📄 Context {i + 1} — Chunk {ri + 1}</div>'
                    f'<div style="font-size:11.5px;font-family:monospace;line-height:1.7">{body}</div></div>'
                )
        else:
            prompt_html += (
                '<div style="background:#FEF3C7;border:1px solid #FDE68A;border-radius:8px;'
                'padding:12px 14px;color:#78350F;font-size:12px">'
                '⚠️ No context chunks retrieved yet. Enter a question and ensure the FAISS index is built.</div>'
            )
        if user_query:
            prompt_html += (
                f'<div style="background:#D1FAE5;border:1px solid #A7F3D0;border-left:4px solid #10B981;'
                f'border-radius:0 8px 8px 0;padding:12px 14px;margin-top:8px;color:#1A202C">'
                f'<div style="font-size:11px;font-weight:700;color:#065F46;text-transform:uppercase;'
                f'letter-spacing:0.6px;margin-bottom:6px">💬 User Question</div>'
                f'<div style="font-size:12.5px">{html.escape(user_query)}</div></div>'
            )
        scrollable(prompt_html, height=560)

    # ── Model Response ────────────────────────────────────────────────────────
    with col_response:
        st.markdown(
            '<div style="font-size:14px;font-weight:700;color:#032D60;margin-bottom:10px">'
            '🤖 Model Response <span style="font-size:11px;font-weight:400;color:#5A6872">'
            f'— {model_name}</span></div>',
            unsafe_allow_html=True,
        )

        if gen_btn:
            # ── Validate pre-conditions ──────────────────────────────────────
            if not user_query:
                st.warning("Enter a question in the **Your Question** field above.")
            elif not retrieved_texts:
                st.error(
                    "No context available.\n\n"
                    "① Apply chunking in **Tab 1**\n"
                    "② Build the FAISS index in **Tab 2**\n"
                    "③ Then try again."
                )
            else:
                ok, err = check_ollama(model_name)
                if not ok:
                    st.error(
                        f"**Ollama error:** {err}\n\n"
                        f"```\nollama serve\nollama pull {model_name}\n```"
                    )
                else:
                    # ── Stream tokens ────────────────────────────────────────
                    try:
                        full_text = ""
                        stream_placeholder = st.empty()
                        separator(log)
                        log.info("TAB3 generate | model=%s  top_k=%d  query='%.80s'",
                                 model_name, gen_top_k, user_query)
                        with st.spinner(f"⚡ Generating response from **{model_name}**…"):
                            for token in stream_response(
                                model_name=model_name,
                                system_message=system_message,
                                context_chunks=retrieved_texts,
                                query=user_query,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            ):
                                full_text += token
                                stream_placeholder.markdown(full_text)
                        st.session_state.response_text = full_text
                        st.session_state.response_for_query = user_query
                        log.info("TAB3 done | response_len=%d chars", len(full_text))
                        st.rerun()
                    except Exception as e:
                        err_msg = str(e)
                        if "memory" in err_msg.lower():
                            st.error(
                                f"**Not enough memory to run `{model_name}`.**\n\n"
                                f"{err_msg}\n\n"
                                f"👉 Select a smaller model from the sidebar (e.g. `tinyllama:latest`)."
                            )
                        else:
                            st.error(f"**Generation error:** {err_msg}")

        # ── Show stored response (all non-generating renders) ────────────────
        elif st.session_state.response_text:
            st.markdown(
                f'<div style="background:#F0FDF4;border:1px solid #A7F3D0;border-radius:7px;'
                f'padding:8px 14px;font-size:11px;color:#065F46;margin-bottom:10px">'
                f'✅ Response for: <i>{html.escape(st.session_state.response_for_query)}</i></div>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown(st.session_state.response_text)

        else:
            st.markdown(
                '<div style="height:480px;border:2px dashed #D8DDE6;padding:20px;border-radius:9px;'
                'display:flex;flex-direction:column;align-items:center;justify-content:center;'
                'gap:12px;background:#FAFBFD">'
                '<div style="font-size:40px">🤖</div>'
                '<div style="font-size:15px;font-weight:600;color:#032D60">Awaiting your query</div>'
                '<div style="font-size:13px;color:#5A6872;text-align:center">'
                'Enter a question above and click<br><b>⚡ Generate Response</b></div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── RAG pipeline visual ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,#032D60,#0F52A0);border-radius:12px;'
        'padding:20px 28px;display:flex;align-items:center;justify-content:space-around;gap:10px">'
        + "".join(
            f'<div style="text-align:center;color:white">'
            f'<div style="font-size:28px;margin-bottom:6px">{icon}</div>'
            f'<div style="font-size:13px;font-weight:700;letter-spacing:0.3px">{step}</div>'
            f'<div style="font-size:11px;color:rgba(255,255,255,0.65);margin-top:3px">{desc}</div>'
            f'</div>'
            + (f'<div style="color:rgba(255,255,255,0.35);font-size:24px">→</div>' if step != "Generate" else "")
            for icon, step, desc in [
                ("📝", "Chunk", "Document → chunks"),
                ("🔢", "Embed", "Chunks → 384D vectors"),
                ("⚡", "Index", "FAISS IndexFlatIP"),
                ("🔍", "Retrieve", "Top-K cosine search"),
                ("📋", "Augment", "Inject context into prompt"),
                ("🤖", "Generate", "LLM → grounded answer"),
            ]
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Details section ───────────────────────────────────────────────────────
    info_cards("Response Generation — Technical Reference", [
        {
            "icon": "🔄", "title": "The Full RAG Pipeline", "accent": "#0070D2",
            "body": (
                "<b>R — Retrieve:</b> Query → embed → FAISS top-K search → relevant chunks<br>"
                "<b>A — Augment:</b> System prompt + retrieved chunks + user question = full prompt<br>"
                "<b>G — Generate:</b> LLM reads the augmented prompt → produces a grounded answer"
            ),
        },
        {
            "icon": "📋", "title": "Prompt Augmentation", "accent": "#10B981",
            "body": (
                "The retrieved chunks are <b>injected verbatim</b> into the prompt as context. "
                "The LLM sees your document content without any retraining. "
                "The system prompt instructs it to only answer from the provided context."
            ),
        },
        {
            "icon": "🌡️", "title": "Temperature", "accent": "#F59E0B",
            "body": (
                "Controls output randomness. <b>0.0</b> = fully deterministic (best for factual QA). "
                "<b>1.0</b> = highly creative/varied. For RAG, keep between <b>0.1–0.4</b> "
                "to stay accurate while remaining fluent."
            ),
        },
        {
            "icon": "📝", "title": "Max Tokens", "accent": "#8B5CF6",
            "body": (
                "Hard limit on response length. 1 token ≈ 0.75 English words. "
                "<b>1000 tokens ≈ 750 words</b>. Hitting the limit mid-sentence means "
                "the response is truncated — increase if answers are cut off."
            ),
        },
        {
            "icon": "🛡️", "title": "Grounding vs Hallucination", "accent": "#EF4444",
            "body": (
                "Without RAG: LLM answers from training data only → may <b>hallucinate</b> facts. "
                "With RAG: LLM has access to your exact documents → answers are <b>verifiable and grounded</b>. "
                "If the answer isn't in the context, it should say 'I don't know'."
            ),
        },
        {
            "icon": "⚡", "title": "Ollama — Local Inference", "accent": "#3B82F6",
            "body": (
                "Ollama runs quantized LLMs (GGUF format) locally on CPU or GPU. "
                "No internet required after model download. "
                "llama3.2:latest ≈ 2GB, tinyllama:latest ≈ 637MB. "
                "Streams tokens in real-time via the Ollama API on <code>localhost:11434</code>."
            ),
        },
        {
            "icon": "🔑", "title": "Top-K Retrieval & Context Window", "accent": "#05A8AA",
            "body": (
                "Higher K → more context → better coverage but larger prompt. "
                "Each LLM has a <b>context window limit</b> (e.g. 4096–128K tokens). "
                "If K × chunk_size exceeds the limit, the prompt is truncated. "
                "Tune K based on your chunk size and model's context window."
            ),
        },
    ])
