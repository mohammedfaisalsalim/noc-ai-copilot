import os
import json
import sqlite3
import pandas as pd
import streamlit as st
import google.generativeai as genai
import chromadb
import requests
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DB_PATH = "noc_inventory.db"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "noc_sops_v2"

OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

st.set_page_config(
    page_title="NOC.AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# UI — NOC mission-control theme
# ==============================================================================
st.markdown(
    """
<style>
/* ===== Base ===== */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    color: #E8EAED;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(0,212,255,0.06) 0%, transparent 40%),
        radial-gradient(circle at 80% 100%, rgba(16,240,160,0.04) 0%, transparent 40%),
        linear-gradient(180deg, #0A0E14 0%, #0B1018 100%);
    background-attachment: fixed;
}
/* Subtle grid pattern overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1280px;
    position: relative;
    z-index: 1;
}

/* ===== Header / brand ===== */
.brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 1.1rem;
    background: linear-gradient(135deg, rgba(19,25,34,0.85), rgba(13,18,25,0.85));
    border: 1px solid #1F2A38;
    border-radius: 14px;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(8px);
}
.brand-left { display: flex; align-items: center; gap: 0.75rem; }
.brand-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #10F0A0;
    box-shadow: 0 0 8px #10F0A0, 0 0 16px rgba(16,240,160,0.4);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}
.brand-name {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #E8EAED;
}
.brand-name .accent { color: #00D4FF; }
.brand-tag {
    font-size: 0.78rem;
    color: #6B7785;
    margin-left: 0.5rem;
    border-left: 1px solid #2A3445;
    padding-left: 0.75rem;
    letter-spacing: 0.02em;
}
.status-strip { display: flex; gap: 1.25rem; align-items: center; }
.status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    color: #9AA5B1;
    display: flex; align-items: center; gap: 0.4rem;
}
.status-pill .dot { width: 6px; height: 6px; border-radius: 50%; }
.status-pill.ok .dot { background: #10F0A0; box-shadow: 0 0 6px #10F0A0; }
.status-pill.warn .dot { background: #FFB020; box-shadow: 0 0 6px #FFB020; }
.status-pill.off .dot { background: #4A5563; }

/* ===== Section labels ===== */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #00D4FF;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::before {
    content: "";
    width: 18px; height: 1px;
    background: #00D4FF;
    display: inline-block;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #E8EAED;
    letter-spacing: -0.01em;
    margin: 0.1rem 0 0.3rem;
}
.section-sub {
    color: #6B7785;
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #1F2A38;
    margin-bottom: 1.3rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none !important;
    color: #6B7785;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.6rem 1.2rem;
    position: relative;
    border-radius: 0 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #9AA5B1; }
.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    background: transparent !important;
    box-shadow: none !important;
}
.stTabs [aria-selected="true"]::after {
    content: "";
    position: absolute;
    bottom: -1px; left: 1rem; right: 1rem;
    height: 2px;
    background: #00D4FF;
    box-shadow: 0 0 8px #00D4FF;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ===== Cards & surfaces ===== */
.card {
    background: rgba(19,25,34,0.6);
    border: 1px solid #1F2A38;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    backdrop-filter: blur(4px);
}

/* ===== Buttons ===== */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #2A3445;
    background: rgba(19,25,34,0.6);
    color: #C5CCD4;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 0.4rem 0.9rem;
    white-space: nowrap !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #00D4FF;
    color: #00D4FF;
    background: rgba(0,212,255,0.05);
    box-shadow: 0 0 12px rgba(0,212,255,0.15);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00D4FF, #00A8CC);
    color: #0A0E14;
    border-color: #00D4FF;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #5BE5FF, #00D4FF);
    box-shadow: 0 0 20px rgba(0,212,255,0.4);
    color: #0A0E14;
}

/* ===== Inputs ===== */
.stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: rgba(10,14,20,0.6) !important;
    border: 1px solid #2A3445 !important;
    border-radius: 8px !important;
    color: #E8EAED !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 1px #00D4FF, 0 0 12px rgba(0,212,255,0.2) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #4A5563; }
label, .stRadio label, .stCheckbox label {
    color: #9AA5B1 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}

/* ===== Chat ===== */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 0.5rem 0;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    background: rgba(19,25,34,0.7);
    border: 1px solid #1F2A38;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    line-height: 1.65;
    color: #DDE2E8;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: rgba(0,212,255,0.06);
    border-color: rgba(0,212,255,0.25);
}
[data-testid="stChatMessage"] code {
    background: rgba(0,212,255,0.1);
    color: #00D4FF;
    border-radius: 4px;
    padding: 0.1rem 0.35rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
}
[data-testid="stChatInput"] textarea {
    background: rgba(19,25,34,0.8) !important;
    border: 1px solid #2A3445 !important;
    border-radius: 12px !important;
    color: #E8EAED !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 1px #00D4FF, 0 0 16px rgba(0,212,255,0.25) !important;
}

/* ===== Suggestion chips ===== */
.suggestion-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.8rem; }

/* ===== Dataframes / data editor ===== */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1F2A38;
}

/* ===== File uploader ===== */
[data-testid="stFileUploader"] section {
    background: rgba(19,25,34,0.5);
    border: 1.5px dashed #2A3445;
    border-radius: 12px;
    padding: 1.4rem;
    transition: all 0.2s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #00D4FF;
    background: rgba(0,212,255,0.04);
}
[data-testid="stFileUploaderDropzone"] { color: #9AA5B1; }

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E141C 0%, #0A0F16 100%);
    border-right: 1px solid #1F2A38;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] hr { border-color: #1F2A38; }

/* ===== KB registry items ===== */
.kb-item {
    background: rgba(19,25,34,0.6);
    border: 1px solid #1F2A38;
    border-left: 3px solid #00D4FF;
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.4rem;
    font-size: 0.86rem;
    color: #DDE2E8;
}
.kb-item.green { border-left-color: #10F0A0; }
.kb-meta {
    font-family: 'JetBrains Mono', monospace;
    color: #6B7785;
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    margin-top: 0.15rem;
}

/* ===== Icon button (small, single-glyph) ===== */
.stButton.icon-btn > button {
    min-width: 38px;
    padding: 0.4rem 0.6rem;
    font-size: 0.95rem;
}

/* ===== Misc ===== */
hr { border-color: #1F2A38; margin: 1.5rem 0; }
.stAlert { border-radius: 10px; }
[data-testid="stToast"] {
    background: #131922 !important;
    border: 1px solid #00D4FF !important;
    border-radius: 10px !important;
    color: #E8EAED !important;
}
code {
    background: rgba(0,212,255,0.08);
    color: #00D4FF;
    border-radius: 4px;
    padding: 0.1rem 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
footer, #MainMenu, header { visibility: hidden; }

/* Hide the Streamlit running spinner during stream */
[data-testid="stStatusWidget"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# VECTOR DB + SQL
# ==============================================================================
@st.cache_resource
def get_vector_db():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)


def get_sql_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


chroma_collection = get_vector_db()

# ==============================================================================
# OLLAMA BACKEND (zero-config local LLM)
# ==============================================================================
@st.cache_data(ttl=15)
def ollama_status():
    """Returns dict: {available: bool, models: [str]}."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"available": True, "models": models}
    except Exception:
        pass
    return {"available": False, "models": []}


def ollama_stream(prompt: str, model: str):
    """Yields text deltas from Ollama's /api/generate stream."""
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if "response" in data:
                yield data["response"]
            if data.get("done"):
                break


def gemini_stream(prompt: str, model_name: str):
    model = genai.GenerativeModel(model_name)
    payload = [{"role": "user", "parts": [prompt]}]
    response = model.generate_content(payload, stream=True)
    for chunk in response:
        try:
            piece = chunk.text or ""
        except Exception:
            piece = ""
        if piece:
            yield piece


# ==============================================================================
# SESSION STATE
# ==============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_table" not in st.session_state:
    st.session_state.selected_table = None
if "processed_sop_sigs" not in st.session_state:
    st.session_state.processed_sop_sigs = set()
if "processed_excel_sigs" not in st.session_state:
    st.session_state.processed_excel_sigs = set()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "backend" not in st.session_state:
    st.session_state.backend = None

# ==============================================================================
# SIDEBAR — backend selection
# ==============================================================================
ollama_info = ollama_status()

with st.sidebar:
    st.markdown(
        """
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem;">
  <div style="width:8px;height:8px;border-radius:50%;background:#10F0A0;
              box-shadow:0 0 8px #10F0A0;animation:pulse 2s ease-in-out infinite;"></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;letter-spacing:0.08em;">
    NOC<span style="color:#00D4FF;">.AI</span>
  </div>
</div>
<div style="color:#6B7785;font-size:0.78rem;margin-bottom:1.4rem;letter-spacing:0.02em;">
  Mission-control assistant
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-label'>Backend</div>", unsafe_allow_html=True)

    backend_options = ["Local (Ollama)", "Cloud (Gemini)"]
    default_backend_idx = 0 if ollama_info["available"] else 1
    if st.session_state.backend is None:
        st.session_state.backend = backend_options[default_backend_idx]

    backend_choice = st.radio(
        "Backend",
        backend_options,
        index=backend_options.index(st.session_state.backend),
        label_visibility="collapsed",
    )
    st.session_state.backend = backend_choice

    api_key = ""
    selected_model = DEFAULT_OLLAMA_MODEL

    if backend_choice == "Local (Ollama)":
        if ollama_info["available"]:
            st.markdown(
                "<div class='status-pill ok'><span class='dot'></span>"
                "OLLAMA · ONLINE</div>",
                unsafe_allow_html=True,
            )
            if ollama_info["models"]:
                # Prefer the default model if installed, else first
                default_idx = (
                    ollama_info["models"].index(DEFAULT_OLLAMA_MODEL)
                    if DEFAULT_OLLAMA_MODEL in ollama_info["models"]
                    else 0
                )
                selected_model = st.selectbox(
                    "Model", ollama_info["models"], index=default_idx
                )
            else:
                st.warning(
                    f"No models found. In a terminal, run:\n\n"
                    f"`ollama pull {DEFAULT_OLLAMA_MODEL}`"
                )
                selected_model = None
        else:
            st.markdown(
                "<div class='status-pill off'><span class='dot'></span>"
                "OLLAMA · OFFLINE</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Free, no API key needed. Install once and the chat runs entirely "
                "on this machine."
            )
            st.markdown(
                """
**Setup (one-time):**
1. Download from [ollama.com/download](https://ollama.com/download)
2. In a terminal, run: `ollama pull llama3.1:8b`
3. Refresh this page.
""",
            )
            selected_model = None
    else:
        st.markdown(
            "<div class='status-pill warn'><span class='dot'></span>"
            "GEMINI · CLOUD</div>",
            unsafe_allow_html=True,
        )
        api_key = st.text_input(
            "API key",
            type="password",
            placeholder="Paste your Gemini key…",
            help="Free key at aistudio.google.com/apikey",
        )
        if api_key:
            genai.configure(api_key=api_key)
        selected_model = DEFAULT_GEMINI_MODEL
        st.caption(f"Model: `{DEFAULT_GEMINI_MODEL}`")

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ==============================================================================
# HEADER BAR — brand + status indicators
# ==============================================================================
# Pull counts for header status pills
try:
    kb_count = chroma_collection.count()
except Exception:
    kb_count = 0
try:
    with get_sql_connection() as conn:
        _tables = [
            t[0]
            for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]
        inv_rows = 0
        for _t in _tables:
            inv_rows += conn.execute(f"SELECT COUNT(*) FROM {_t}").fetchone()[0]
        inv_tables = len(_tables)
except Exception:
    inv_rows, inv_tables = 0, 0

if st.session_state.backend == "Local (Ollama)" and ollama_info["available"]:
    backend_pill = "<div class='status-pill ok'><span class='dot'></span>OLLAMA</div>"
elif st.session_state.backend == "Cloud (Gemini)" and api_key:
    backend_pill = "<div class='status-pill warn'><span class='dot'></span>GEMINI</div>"
else:
    backend_pill = "<div class='status-pill off'><span class='dot'></span>OFFLINE</div>"

st.markdown(
    f"""
<div class="brand-bar">
  <div class="brand-left">
    <div class="brand-dot"></div>
    <div class="brand-name">NOC<span class="accent">.AI</span></div>
    <div class="brand-tag">network operations copilot</div>
  </div>
  <div class="status-strip">
    <div class="status-pill ok"><span class="dot"></span>KB · {kb_count} chunks</div>
    <div class="status-pill ok"><span class="dot"></span>INV · {inv_rows} rows / {inv_tables} tables</div>
    {backend_pill}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_chat, tab_inventory, tab_kb = st.tabs(["Chat", "Inventory", "Knowledge base"])

# ==============================================================================
# TAB: CHAT
# ==============================================================================
with tab_chat:
    backend_ready = (
        st.session_state.backend == "Local (Ollama)"
        and ollama_info["available"]
        and selected_model
    ) or (st.session_state.backend == "Cloud (Gemini)" and api_key)

    if not st.session_state.chat_history:
        st.markdown(
            "<div class='section-label'>Console</div>"
            "<div class='section-title'>How can I help with the network today?</div>"
            "<div class='section-sub'>"
            "Ask about SOPs, troubleshooting, asset status, or QC logs — answers are "
            "grounded in your indexed knowledge base."
            "</div>",
            unsafe_allow_html=True,
        )

        suggestions = [
            "Which doors are flagged 'not working' in the on-campus QC form?",
            "Walk me through the off-campus mesh router pairing procedure.",
            "A CCTV camera shows no signal. What's the standard troubleshooting flow?",
            "Show me all access points currently in Maintenance status.",
        ]
        sc1, sc2 = st.columns(2)
        for i, s in enumerate(suggestions):
            col = sc1 if i % 2 == 0 else sc2
            if col.button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_prompt = s
                st.rerun()

    # Render history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    typed = st.chat_input("Ask about SOPs, devices, or QC logs…")
    if typed:
        prompt = typed

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if not backend_ready:
            with st.chat_message("assistant"):
                if st.session_state.backend == "Local (Ollama)":
                    st.warning(
                        "Ollama isn't reachable. Install it from "
                        "[ollama.com/download](https://ollama.com/download), then run "
                        f"`ollama pull {DEFAULT_OLLAMA_MODEL}` and refresh."
                    )
                else:
                    st.warning("Add your Gemini API key in the sidebar to continue.")
        else:
            # RAG: top-5 SOP chunks with source metadata
            rag_context = ""
            try:
                qr = chroma_collection.query(
                    query_texts=[prompt],
                    n_results=5,
                    include=["documents", "metadatas"],
                )
                docs = (qr.get("documents") or [[]])[0]
                metas = (qr.get("metadatas") or [[]])[0]
                if docs:
                    blocks = []
                    for d, m in zip(docs, metas):
                        src = (m or {}).get("source", "unknown SOP")
                        blocks.append(f"[Source PDF: {src}]\n{d}")
                    rag_context = "\n\n---\n\n".join(blocks)
                else:
                    rag_context = "No matching SOP passages found."
            except Exception as e:
                rag_context = f"SOP vector query failed: {type(e).__name__}: {e}"

            # Full table dumps (capped)
            db_context = ""
            MAX_ROWS_PER_TABLE = 1000
            try:
                with get_sql_connection() as conn:
                    chat_tables = [
                        t[0]
                        for t in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                        ).fetchall()
                    ]
                    if chat_tables:
                        for table in chat_tables:
                            total_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            df_snap = pd.read_sql_query(
                                f"SELECT * FROM {table} LIMIT {MAX_ROWS_PER_TABLE}", conn
                            )
                            db_context += (
                                f"\nTABLE: {table} (showing {len(df_snap)} of {total_rows} rows)\n"
                            )
                            db_context += df_snap.to_string(index=False) + "\n---"
                    else:
                        db_context = "No relational tables present."
            except Exception as e:
                db_context = f"DB read failure: {e}"

            system_prompt = f"""You are a Principal NOC Systems Engineer assistant for a campus network.
Help operators with Wi-Fi uptime, switch configs, door access controllers, and Hikvision IP surveillance.

Ground every answer in the two sources below — do not invent procedures or device names.

Tables you may see:
- 'wifi_qc': Access point signal mapping and parameters.
- 'cctv_qc': Hikvision NVR streams, channel logs, camera status.
- 'door_access_qc': Door panels, card readers, controllers, weekly QC logs.
- 'general_inventory': Master hardware and asset registry.

1. SOP CONTEXT (each block tagged with its source PDF):
---
{rag_context}
---

2. RELATIONAL TABLES SNAPSHOT:
---
{db_context}
---

RULES:
- Procedure questions → quote/paraphrase the SOP and cite the source PDF, e.g. "(per APU Off-Campus mesh pairing SOP.pdf)".
- Inventory questions → read the table and cite the row's `source_file` (and `sheet_name` if present).
- If neither source contains the answer, say so plainly. Do not guess.
- Tone: senior network engineer — concise, technical, no filler. Use bullets where it helps.
"""

            full_prompt = f"{system_prompt}\n\nUser question: {prompt}"

            try:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""

                    if st.session_state.backend == "Local (Ollama)":
                        stream_iter = ollama_stream(full_prompt, selected_model)
                    else:
                        stream_iter = gemini_stream(full_prompt, selected_model)

                    for piece in stream_iter:
                        if piece:
                            full_response += piece
                            placeholder.markdown(full_response + "▌")

                    if not full_response:
                        full_response = "_(Model returned no text — possibly a safety filter or empty response.)_"
                    placeholder.markdown(full_response)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )
            except Exception as e:
                import traceback

                st.error(f"Model error: {type(e).__name__}: {e}")
                st.code(traceback.format_exc(), language="text")

# ==============================================================================
# TAB: INVENTORY
# ==============================================================================
with tab_inventory:
    st.markdown(
        "<div class='section-label'>Asset matrix</div>"
        "<div class='section-title'>Network inventory</div>"
        "<div class='section-sub'>Browse, search, and edit data ingested from your spreadsheets.</div>",
        unsafe_allow_html=True,
    )

    with get_sql_connection() as conn:
        tables = [
            t[0]
            for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]

    if not tables:
        st.info("No data ingested yet. Head to **Knowledge base** to upload spreadsheets.")
    else:
        if st.session_state.selected_table not in tables:
            st.session_state.selected_table = tables[0]

        st.session_state.selected_table = st.selectbox(
            "Table",
            tables,
            index=tables.index(st.session_state.selected_table),
        )

        with get_sql_connection() as conn:
            df_current = pd.read_sql_query(
                f"SELECT * FROM {st.session_state.selected_table}", conn
            )

        st.caption(f"{len(df_current)} rows in `{st.session_state.selected_table}`")

        f1, f2 = st.columns([1, 2])
        with f1:
            source_options = []
            if "source_file" in df_current.columns:
                source_options = sorted(df_current["source_file"].dropna().unique().tolist())
            selected_sources = st.multiselect(
                "Filter by source file",
                source_options,
                default=[],
                key=f"src_filter_{st.session_state.selected_table}",
            )
        with f2:
            search_term = st.text_input(
                "Search",
                value="",
                key=f"search_{st.session_state.selected_table}",
                placeholder="e.g. 'not working', '10.113.168', 'GYM'",
            )

        df_view = df_current.copy()
        if selected_sources:
            df_view = df_view[df_view["source_file"].isin(selected_sources)]
        if search_term.strip():
            term = search_term.strip().lower()
            mask = df_view.apply(
                lambda row: row.astype(str).str.lower().str.contains(term, na=False).any(),
                axis=1,
            )
            df_view = df_view[mask]

        filtering = bool(selected_sources) or bool(search_term.strip())

        if filtering:
            st.info(
                f"Showing **{len(df_view)} of {len(df_current)}** rows. "
                "Editing is locked while filtering — clear filters to edit."
            )
            st.dataframe(df_view, use_container_width=True, hide_index=True)
        else:
            edited_df = st.data_editor(
                df_current,
                num_rows="dynamic",
                use_container_width=True,
                key=f"editor_{st.session_state.selected_table}_{len(df_current)}",
            )

            if st.button("Save changes", type="primary"):
                try:
                    with get_sql_connection() as conn:
                        conn.execute(f"DROP TABLE IF EXISTS {st.session_state.selected_table};")
                        edited_df.to_sql(
                            st.session_state.selected_table,
                            conn,
                            if_exists="replace",
                            index=False,
                        )
                        conn.commit()
                    st.success(f"Saved to `{st.session_state.selected_table}`.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

# ==============================================================================
# TAB: KNOWLEDGE BASE
# ==============================================================================
with tab_kb:
    st.markdown(
        "<div class='section-label'>Ingestion</div>"
        "<div class='section-title'>Knowledge base</div>"
        "<div class='section-sub'>Upload SOPs and inventory spreadsheets. Everything stays on this machine.</div>",
        unsafe_allow_html=True,
    )

    upload_col1, upload_col2 = st.columns(2)

    with upload_col1:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            "color:#00D4FF;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;'>"
            "SOP manuals</div>"
            "<div class='kb-meta'>PDF, TXT, MD · indexed for semantic search</div>",
            unsafe_allow_html=True,
        )
        sop_files = st.file_uploader(
            "Upload SOPs",
            accept_multiple_files=True,
            type=["txt", "md", "pdf"],
            key="kb_tab_sop_uploader",
            label_visibility="collapsed",
        )

        if sop_files:
            new_sop_indexed = False
            for file in sop_files:
                if file.size == 0:
                    continue
                file_id = f"{file.name}_{file.size}"
                if file_id in st.session_state.processed_sop_sigs:
                    continue
                existing = chroma_collection.get(ids=[f"{file_id}_0"])
                if existing and existing.get("ids"):
                    st.session_state.processed_sop_sigs.add(file_id)
                    continue

                content = ""
                if file.name.lower().endswith(".pdf"):
                    try:
                        reader = PdfReader(file)
                        for page in reader.pages:
                            t = page.extract_text()
                            if t:
                                content += t + "\n"
                    except Exception as e:
                        st.error(f"Error parsing {file.name}: {e}")
                        continue
                else:
                    content = file.read().decode("utf-8", errors="ignore")

                if content.strip():
                    chunks = []
                    size, overlap = 600, 120
                    i = 0
                    while i < len(content):
                        chunks.append(content[i : i + size])
                        i += size - overlap

                    ids = [f"{file_id}_{idx}" for idx in range(len(chunks))]
                    metadatas = [{"source": file.name} for _ in chunks]
                    chroma_collection.add(documents=chunks, metadatas=metadatas, ids=ids)
                    st.toast(f"Indexed {len(chunks)} chunks from {file.name}")
                    st.session_state.processed_sop_sigs.add(file_id)
                    new_sop_indexed = True

            if new_sop_indexed:
                st.rerun()

    with upload_col2:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            "color:#10F0A0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;'>"
            "Inventory spreadsheets</div>"
            "<div class='kb-meta'>CSV, XLSX · multi-sheet workbooks fully ingested</div>",
            unsafe_allow_html=True,
        )
        excel_files = st.file_uploader(
            "Upload spreadsheets",
            accept_multiple_files=True,
            type=["csv", "xlsx"],
            key="kb_tab_excel_uploader",
            label_visibility="collapsed",
        )

        if excel_files:
            new_excel_ingested = False
            for file in excel_files:
                if file.size == 0:
                    continue
                file_id = f"{file.name}_{file.size}"
                if file_id in st.session_state.processed_excel_sigs:
                    continue

                try:
                    if file.name.endswith(".csv"):
                        df_upload = pd.read_csv(file)
                    else:
                        def _detect_header_row(df_raw, max_scan=10):
                            best_idx, best_score = 0, -1
                            for i in range(min(max_scan, len(df_raw))):
                                score = 0
                                for val in df_raw.iloc[i].dropna():
                                    s = str(val).strip()
                                    if 0 < len(s) <= 40 and any(c.isalpha() for c in s):
                                        score += 1
                                if score > best_score:
                                    best_idx, best_score = i, score
                            return best_idx

                        def _normalize_col(c):
                            return str(c).strip().replace(" ", "_").lower()

                        all_sheets = pd.read_excel(file, sheet_name=None, header=None)
                        sheet_frames = []
                        for sheet_name, df_raw in all_sheets.items():
                            if df_raw.empty:
                                continue
                            h_idx = _detect_header_row(df_raw)
                            df_s = df_raw.iloc[h_idx:].reset_index(drop=True)
                            df_s.columns = [_normalize_col(c) for c in df_s.iloc[0]]
                            df_s = df_s.iloc[1:].reset_index(drop=True)
                            min_filled = max(1, int(0.3 * len(df_s.columns)))
                            df_s = df_s[df_s.notna().sum(axis=1) >= min_filled].reset_index(
                                drop=True
                            )
                            df_s = df_s.loc[:, ~df_s.columns.str.contains("^unnamed|^nan$|^$")]
                            df_s = df_s.loc[:, ~df_s.columns.duplicated()]
                            if df_s.empty:
                                continue
                            df_s["sheet_name"] = sheet_name
                            sheet_frames.append(df_s)

                        if not sheet_frames:
                            st.error(f"No usable data rows detected in {file.name}.")
                            continue
                        df_upload = pd.concat(sheet_frames, ignore_index=True, sort=False)

                    df_upload = df_upload.dropna(how="all")
                    df_upload.columns = [
                        str(c).strip().replace(" ", "_").lower() for c in df_upload.columns
                    ]
                    df_upload = df_upload.loc[:, ~df_upload.columns.str.contains("^unnamed|^nan$")]
                    df_upload = df_upload.loc[:, ~df_upload.columns.duplicated()]

                    if len(df_upload) == 0:
                        st.error(f"Skipping empty or malformed file: {file.name}")
                        continue

                    df_upload["source_file"] = file.name

                    name_norm = file.name.lower().replace(" ", "_")
                    if (
                        "wifi" in name_norm
                        or "bssid" in df_upload.columns
                        or "ssid" in df_upload.columns
                    ):
                        target_table = "wifi_qc"
                    elif "cctv" in name_norm or "nvr" in name_norm or "camera" in name_norm:
                        target_table = "cctv_qc"
                    elif (
                        "on_campus" in name_norm
                        or "door" in name_norm
                        or "access_qc" in name_norm
                        or "qc_form" in name_norm
                    ):
                        target_table = "door_access_qc"
                    elif "inventory" in name_norm or "materials" in name_norm:
                        target_table = "general_inventory"
                    else:
                        clean_name = "".join(
                            e for e in name_norm.split(".")[0] if e.isalnum() or e == "_"
                        ).strip("_")
                        target_table = f"table_{clean_name}"

                    df_upload = df_upload.drop_duplicates(keep="first")

                    with get_sql_connection() as conn:
                        try:
                            cursor = conn.cursor()
                            cursor.execute(f"PRAGMA table_info({target_table});")
                            existing_columns = {row[1] for row in cursor.fetchall()}
                            for col in df_upload.columns:
                                if col not in existing_columns:
                                    conn.execute(
                                        f"ALTER TABLE {target_table} ADD COLUMN {col} TEXT;"
                                    )
                                    existing_columns.add(col)
                        except Exception:
                            pass

                        try:
                            df_existing = pd.read_sql_query(
                                f"SELECT * FROM {target_table} WHERE source_file != ?",
                                conn,
                                params=(file.name,),
                            )
                            df_final = pd.concat(
                                [df_existing, df_upload], ignore_index=True
                            )
                            df_final.to_sql(
                                target_table, conn, if_exists="replace", index=False
                            )
                        except Exception:
                            df_upload.to_sql(
                                target_table, conn, if_exists="append", index=False
                            )
                        conn.commit()

                    st.toast(
                        f"Merged {len(df_upload)} rows into `{target_table}` from {file.name}"
                    )
                    st.session_state.processed_excel_sigs.add(file_id)
                    new_excel_ingested = True
                except Exception as e:
                    import traceback

                    st.error(f"Error processing {file.name}: {type(e).__name__}: {e}")
                    st.code(traceback.format_exc(), language="text")

            if new_excel_ingested:
                st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-label'>Registry</div>"
        "<div class='section-title'>Indexed sources</div>"
        "<div class='section-sub'>Remove a file to drop it from the knowledge base.</div>",
        unsafe_allow_html=True,
    )

    reg_col1, reg_col2 = st.columns(2)

    with reg_col1:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            "color:#00D4FF;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;'>"
            "SOP manuals</div>",
            unsafe_allow_html=True,
        )
        try:
            all_vectors = chroma_collection.get(include=["metadatas"])
            if all_vectors and all_vectors["metadatas"]:
                unique_sources = {}
                for doc_id, meta in zip(all_vectors["ids"], all_vectors["metadatas"]):
                    src = meta.get("source", "Unknown")
                    unique_sources.setdefault(src, []).append(doc_id)

                for s_name, id_list in unique_sources.items():
                    c_doc, c_doc_del = st.columns([8, 1])
                    c_doc.markdown(
                        f"<div class='kb-item'>{s_name}"
                        f"<div class='kb-meta'>{len(id_list)} chunks</div></div>",
                        unsafe_allow_html=True,
                    )
                    with c_doc_del:
                        st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
                        if st.button("✕", key=f"del_chroma_{s_name}", help=f"Remove {s_name}"):
                            chroma_collection.delete(ids=id_list)
                            st.toast(f"Removed {s_name}")
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption("No SOPs indexed yet.")
        except Exception:
            st.caption("No SOPs indexed yet.")

    with reg_col2:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            "color:#10F0A0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;'>"
            "Inventory spreadsheets</div>",
            unsafe_allow_html=True,
        )
        try:
            with get_sql_connection() as conn:
                check_tables = [
                    t[0]
                    for t in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                    ).fetchall()
                ]

                if not check_tables:
                    st.caption("No spreadsheets ingested yet.")
                else:
                    for t in check_tables:
                        sources = pd.read_sql_query(
                            f"SELECT DISTINCT source_file FROM {t}", conn
                        )
                        st.markdown(
                            f"<div class='kb-meta' style='margin-top:0.6rem;'>"
                            f"Table · <code>{t}</code></div>",
                            unsafe_allow_html=True,
                        )
                        for _, row in sources.iterrows():
                            fname = row["source_file"]
                            count_res = conn.execute(
                                f"SELECT COUNT(*) FROM {t} WHERE source_file = ?", (fname,)
                            ).fetchone()[0]
                            c_file, c_del = st.columns([8, 1])
                            c_file.markdown(
                                f"<div class='kb-item green'>{fname}"
                                f"<div class='kb-meta'>{count_res} records</div></div>",
                                unsafe_allow_html=True,
                            )
                            with c_del:
                                st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
                                if st.button(
                                    "✕",
                                    key=f"del_sql_{t}_{fname}",
                                    help=f"Remove {fname}",
                                ):
                                    with get_sql_connection() as rm_conn:
                                        rm_conn.execute(
                                            f"DELETE FROM {t} WHERE source_file = ?", (fname,)
                                        )
                                        rm_conn.commit()
                                        remaining = rm_conn.execute(
                                            f"SELECT COUNT(*) FROM {t}"
                                        ).fetchone()[0]
                                        if remaining == 0:
                                            rm_conn.execute(f"DROP TABLE IF EXISTS {t};")
                                            rm_conn.commit()
                                    st.toast(f"Removed {fname}")
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to query SQL registry: {e}")
