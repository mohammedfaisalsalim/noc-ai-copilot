import os
import json
import re
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
# Models below are aligned with the llm-gateway-platform fleet so both
# projects share a single source of truth on what's installed.
#   - llama3.1:8b   → gateway Tier 2 (heavy local) + chatbot direct-Ollama default
#   - gemini-2.5-flash → gateway Tier 1 (balanced cloud) + chatbot direct-Gemini default
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# LLM Gateway (companion project: llm-gateway-platform)
DEFAULT_GATEWAY_URL = "http://localhost:8000"
DEFAULT_TEAM_ID = "noc-team"

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

/* ===== Routing telemetry footer (shown under gateway answers) ===== */
.route-footer {
    margin-top: 0.6rem;
    padding: 0.45rem 0.8rem;
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    color: #9AA5B1;
    display: flex;
    flex-wrap: wrap;
    gap: 1.1rem;
}
.route-footer .key { color: #00D4FF; }
.route-footer .val { color: #E8EAED; }

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
    """Yields text deltas from Ollama's /api/generate stream.

    Sends `num_ctx=32768` so the full SOP + table-dump context fits.
    Ollama defaults to a 2048-token window which is far too small once
    door_access_qc + wifi_qc + other tables are dumped into the prompt.
    llama3.1:8b supports up to 128K — 32K is a safe middle ground that
    won't OOM most desktops.
    """
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": 32768,
                "temperature": 0.2,
            },
        },
        stream=True,
        timeout=600,
    ) as r:
        if r.status_code >= 400:
            # Surface Ollama's actual error body for diagnosability.
            try:
                err_body = r.text or ""
            except Exception:
                err_body = "(no body)"
            raise requests.HTTPError(
                f"Ollama returned {r.status_code}. "
                f"Likely causes: prompt exceeds context window, model not loaded, "
                f"or out of memory. Body: {err_body[:400]}",
                response=r,
            )
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
# LLM GATEWAY BACKEND
# Talks to the companion `llm-gateway-platform` (FastAPI service on :8000).
# The gateway internally classifies prompts (tier 0/1/2), routes to
# Llama 3.2 / Gemini Flash / Mistral, applies rate limits + budget guardrails,
# and returns a single non-streamed response with rich routing telemetry.
# ==============================================================================
@st.cache_data(ttl=15)
def gateway_status(url: str):
    """Pings gateway root. Returns {available: bool}."""
    try:
        r = requests.get(f"{url}/", timeout=1.5)
        if r.status_code == 200 and r.json().get("status") == "online":
            return {"available": True}
    except Exception:
        pass
    return {"available": False}


class GatewayError(RuntimeError):
    """Raised on 429 / 402 / 503 from the gateway with a user-friendly message."""


def gateway_call(prompt: str, url: str, team_id: str) -> dict:
    """POST to /v1/chat/completions and return the full StandardResponse payload.
    Raises GatewayError with a friendly message on 429/402/503."""
    try:
        r = requests.post(
            f"{url}/v1/chat/completions",
            headers={"X-Team-Id": team_id, "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        raise GatewayError(
            f"Cannot reach the gateway at {url}. "
            "Is `docker-compose up` running in the llm-gateway-platform project?"
        )
    except requests.exceptions.Timeout:
        raise GatewayError("Gateway timed out after 5 minutes. The upstream model may be stuck.")

    if r.status_code == 429:
        retry = r.headers.get("Retry-After", "?")
        raise GatewayError(
            f"Rate limit reached for team `{team_id}` (60 requests/min). "
            f"Retry after {retry}s."
        )
    if r.status_code == 402:
        raise GatewayError(
            f"Daily budget exceeded for team `{team_id}` ($5.00 cap). "
            "Resets at midnight UTC."
        )
    if r.status_code == 503:
        raise GatewayError(
            "All upstream LLM providers are down (circuit breakers open). "
            "Wait ~15s for the health-check loop to retry, then try again."
        )
    r.raise_for_status()
    return r.json()


# ==============================================================================
# DOOR INSPECTION HELPERS (FR-04 / FR-05 / FR-06 / FR-07)
# Resolves on-disk paths under door_images/{unit_id}/{filename}.
# Image binaries never enter SQLite — only the filename string is stored.
# ==============================================================================
DOOR_IMAGES_DIR = "door_images"
# Matches G1EE, G2EE, F1EE, F2EE, D1EE, D2EE and future units that follow
# the {letter}{digit}{letter}{letter} convention. Case-insensitive at call time.
DOOR_UNIT_RE = re.compile(r"\b([A-Za-z]\d[A-Za-z]{2})\b")
# Phrases that imply "show all doors" — bulk render mode for FR-06.
BROAD_DOOR_RE = re.compile(
    r"\b(all\s+doors?|every\s+door|list\s+doors?|door\s+status(?:es)?|status\s+of\s+(?:all\s+)?doors?)\b",
    re.IGNORECASE,
)


def door_image_path(unit_id, filename) -> str:
    """Resolve the on-disk path for a door inspection photo."""
    if not unit_id or not filename:
        return ""
    if pd.isna(unit_id) or pd.isna(filename):
        return ""
    uid = str(unit_id).strip()
    fn = str(filename).strip()
    if not uid or not fn:
        return ""
    return os.path.join(DOOR_IMAGES_DIR, uid, fn)


def door_image_exists(unit_id, filename) -> bool:
    p = door_image_path(unit_id, filename)
    return bool(p) and os.path.isfile(p)


def extract_unit_ids(text: str) -> list:
    """Pull candidate door unit IDs out of free text, uppercased + deduped."""
    if not text:
        return []
    seen, out = set(), []
    for match in DOOR_UNIT_RE.findall(text):
        uid = match.upper()
        if uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def fetch_all_door_units_with_photos() -> list:
    """Return all distinct unit_ids in door_access_qc that have a photo reference."""
    try:
        with get_sql_connection() as conn:
            tables = {
                t[0]
                for t in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='door_access_qc';"
                ).fetchall()
            }
            if "door_access_qc" not in tables:
                return []
            cols = {row[1] for row in conn.execute("PRAGMA table_info(door_access_qc);").fetchall()}
            if "unit_id" not in cols or "image_filename" not in cols:
                return []
            rows = conn.execute(
                "SELECT DISTINCT UPPER(TRIM(unit_id)) FROM door_access_qc "
                "WHERE image_filename IS NOT NULL AND TRIM(image_filename) != ''"
            ).fetchall()
            return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def fetch_door_photos_for_units(unit_ids: list) -> list:
    """For each unit_id, return its photo records sorted newest-first.

    Falls back to row order (SQLite rowid) when inspection_date is missing
    or unparseable, so admins who forget the date never lose photo display.

    Returns: [{unit_id, photos: [{filename, date, pic, path, exists}, ...]}, ...]
    """
    if not unit_ids:
        return []
    results = []
    try:
        with get_sql_connection() as conn:
            tables = {
                t[0]
                for t in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='door_access_qc';"
                ).fetchall()
            }
            if "door_access_qc" not in tables:
                return []
            cols = {row[1] for row in conn.execute("PRAGMA table_info(door_access_qc);").fetchall()}
            if "unit_id" not in cols or "image_filename" not in cols:
                return []
            has_date = "inspection_date" in cols
            has_pic = "pic" in cols
            date_select = "inspection_date" if has_date else "NULL AS inspection_date"
            pic_select = "pic" if has_pic else "NULL AS pic"
            order_clause = (
                "ORDER BY inspection_date DESC, rowid DESC"
                if has_date
                else "ORDER BY rowid DESC"
            )
            for uid in unit_ids:
                q = (
                    f"SELECT image_filename, {date_select}, {pic_select} "
                    f"FROM door_access_qc "
                    f"WHERE UPPER(TRIM(unit_id)) = ? "
                    f"AND image_filename IS NOT NULL AND TRIM(image_filename) != '' "
                    f"{order_clause}"
                )
                rows = conn.execute(q, (uid,)).fetchall()
                if not rows:
                    continue
                photos = []
                for filename, date_val, pic_val in rows:
                    p = door_image_path(uid, filename)
                    photos.append(
                        {
                            "filename": str(filename),
                            "date": str(date_val) if date_val else "—",
                            "pic": str(pic_val) if pic_val else "—",
                            "path": p,
                            "exists": bool(p) and os.path.isfile(p),
                        }
                    )
                results.append({"unit_id": uid, "photos": photos})
    except Exception:
        return []
    return results


def render_door_photos(photo_groups: list, bulk: bool = False):
    """Render inspection photos under a chat answer or inventory expander.

    photo_groups is the output of fetch_door_photos_for_units.
    bulk=True compacts thumbnail width for broad "show all doors" queries.
    """
    if not photo_groups:
        return
    primary_width = 320 if bulk else 500
    secondary_width = 260 if bulk else 380
    for group in photo_groups:
        uid = group.get("unit_id", "")
        photos = group.get("photos") or []
        if not photos:
            continue
        newest = photos[0]
        caption = f"{uid} · {newest['date']} · Submitted by: {newest['pic']}"
        if newest["exists"]:
            st.image(newest["path"], caption=caption, width=primary_width)
        else:
            st.caption(f"📷 {caption} — Photo not available — file missing.")
        if len(photos) > 1:
            with st.expander(f"Show all photos for {uid} ({len(photos)})"):
                for p in photos[1:]:
                    cap = f"{uid} · {p['date']} · Submitted by: {p['pic']}"
                    if p["exists"]:
                        st.image(p["path"], caption=cap, width=secondary_width)
                    else:
                        st.caption(f"📷 {cap} — file missing")


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
if "gateway_url" not in st.session_state:
    st.session_state.gateway_url = DEFAULT_GATEWAY_URL
if "team_id" not in st.session_state:
    st.session_state.team_id = DEFAULT_TEAM_ID

# ==============================================================================
# SIDEBAR — backend selection
# ==============================================================================
ollama_info = ollama_status()
gateway_info = gateway_status(st.session_state.gateway_url)

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

    backend_options = ["LLM Gateway", "Local (Ollama)", "Cloud (Gemini)"]
    # Default: Gateway if online, else Ollama if online, else Gemini
    if st.session_state.backend is None:
        if gateway_info["available"]:
            st.session_state.backend = "LLM Gateway"
        elif ollama_info["available"]:
            st.session_state.backend = "Local (Ollama)"
        else:
            st.session_state.backend = "Cloud (Gemini)"

    backend_choice = st.radio(
        "Backend",
        backend_options,
        index=backend_options.index(st.session_state.backend),
        label_visibility="collapsed",
    )
    st.session_state.backend = backend_choice

    api_key = ""
    selected_model = None

    if backend_choice == "LLM Gateway":
        if gateway_info["available"]:
            st.markdown(
                "<div class='status-pill ok'><span class='dot'></span>"
                "GATEWAY · ONLINE</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Smart routing across Llama 3.2 · Gemini 2.5 Flash · Llama 3.1 "
                "with rate limits, budget caps, and circuit breakers."
            )
        else:
            st.markdown(
                "<div class='status-pill off'><span class='dot'></span>"
                "GATEWAY · OFFLINE</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
**Setup:** In your `llm-gateway-platform` folder run:
```
docker-compose up -d
```
Then refresh this page.
"""
            )

        st.session_state.gateway_url = st.text_input(
            "Gateway URL",
            value=st.session_state.gateway_url,
            help="The companion llm-gateway-platform service. Default is localhost:8000.",
        )
        st.session_state.team_id = st.text_input(
            "Team ID",
            value=st.session_state.team_id,
            help="Sent as the X-Team-Id header. Drives rate limits and budget tracking.",
        )
        st.caption(
            f"Dashboards: "
            f"[Grafana](http://localhost:3000) · "
            f"[Prometheus](http://localhost:9090) · "
            f"[Metrics]({st.session_state.gateway_url}/metrics)"
        )

    elif backend_choice == "Local (Ollama)":
        if ollama_info["available"]:
            st.markdown(
                "<div class='status-pill ok'><span class='dot'></span>"
                "OLLAMA · ONLINE</div>",
                unsafe_allow_html=True,
            )
            if ollama_info["models"]:
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
        else:
            st.markdown(
                "<div class='status-pill off'><span class='dot'></span>"
                "OLLAMA · OFFLINE</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
**Setup (one-time):**
1. Download from [ollama.com/download](https://ollama.com/download)
2. In a terminal, run: `ollama pull llama3.1:8b`
3. Refresh this page.
"""
            )

    else:  # Cloud (Gemini)
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

if st.session_state.backend == "LLM Gateway" and gateway_info["available"]:
    backend_pill = (
        "<div class='status-pill ok'><span class='dot'></span>"
        f"GATEWAY · {st.session_state.team_id}</div>"
    )
elif st.session_state.backend == "Local (Ollama)" and ollama_info["available"]:
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
    <div class="status-pill ok"><span class="dot"></span>QC · {inv_rows} rows / {inv_tables} tables</div>
    {backend_pill}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_chat, tab_inventory, tab_kb = st.tabs(["Chat", "QC Records", "Knowledge base"])

# ==============================================================================
# TAB: CHAT
# ==============================================================================
def _render_route_footer(meta: dict):
    """Render a small mono-font telemetry strip under a gateway answer."""
    if not meta:
        return
    model = meta.get("model_used", "?")
    latency_ms = meta.get("latency_ms", 0)
    total_tokens = meta.get("total_tokens", 0)
    cost = meta.get("cost_usd", 0)
    st.markdown(
        f"""
<div class="route-footer">
  <span><span class="key">ROUTED</span> · <span class="val">{model}</span></span>
  <span><span class="key">LATENCY</span> · <span class="val">{latency_ms/1000:.2f}s</span></span>
  <span><span class="key">TOKENS</span> · <span class="val">{total_tokens}</span></span>
  <span><span class="key">COST</span> · <span class="val">${cost:.6f}</span></span>
</div>
""",
        unsafe_allow_html=True,
    )


with tab_chat:
    backend_ready = (
        (st.session_state.backend == "LLM Gateway" and gateway_info["available"])
        or (
            st.session_state.backend == "Local (Ollama)"
            and ollama_info["available"]
            and selected_model
        )
        or (st.session_state.backend == "Cloud (Gemini)" and api_key)
    )

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

    # Render history (with routing footer if the message was answered by the gateway,
    # and door inspection photos if the question referenced any door units).
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("meta"):
                _render_route_footer(message["meta"])
            if message["role"] == "assistant" and message.get("door_photos"):
                render_door_photos(
                    message["door_photos"],
                    bulk=message.get("door_photos_bulk", False),
                )

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
                if st.session_state.backend == "LLM Gateway":
                    st.warning(
                        f"Gateway isn't reachable at `{st.session_state.gateway_url}`. "
                        "In the `llm-gateway-platform` folder, run `docker-compose up -d` "
                        "and refresh this page."
                    )
                elif st.session_state.backend == "Local (Ollama)":
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

- DOOR INSPECTION QUERIES (any prompt mentioning a unit_id like G1EE, G2EE, F1EE, F2EE, D1EE, D2EE — or asking about door status/condition):
  You MUST report the FULL inspection record from `door_access_qc`, not a one-line summary.
  Format the answer as a labeled bullet list of every available field for that unit, in this order:
    • **Unit ID** · **Location** · **Inspection date** · **Inspection time** · **PIC**
    • **Lock status**
    • **Break-glass status**
    • **Controller status**
    • **Power adapter status**
    • **Card reader status**
    • **Remarks** (quote verbatim if present, otherwise write "—")
  Skip any field that is null / None / empty in the row.
  After the bullets, add a short **Issues** call-out listing every component whose value is anything other than "Working" / "Online" / "Normal" / "Intact"
  (e.g. "Not Working", "Faulty", "Offline", "Triggered", "Damaged", "No Power").
  If the **Remarks** appear to contradict the dropdown values (e.g. remarks mention a fault on a component
  the form lists as Working), flag it explicitly under Issues as a **Data inconsistency**.
  When multiple inspection rows exist for the same unit, default to the most recent inspection_date,
  and mention in the closing line how many historical inspections exist.

- If neither source contains the answer, say so plainly. Do not guess.
- Tone: senior network engineer — concise, technical, no filler. Use bullets where it helps.
"""

            full_prompt = f"{system_prompt}\n\nUser question: {prompt}"

            try:
                with st.chat_message("assistant"):
                    if st.session_state.backend == "LLM Gateway":
                        # Gateway is non-streaming. Show a spinner during the call,
                        # then render the answer + routing telemetry footer.
                        with st.spinner("Routing through gateway…"):
                            try:
                                payload = gateway_call(
                                    full_prompt,
                                    st.session_state.gateway_url,
                                    st.session_state.team_id,
                                )
                            except GatewayError as ge:
                                st.error(str(ge))
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": f"⚠️ {ge}"}
                                )
                                st.stop()

                        full_response = payload.get("output_text", "") or (
                            "_(Gateway returned no text.)_"
                        )
                        meta = {
                            "model_used": payload.get("model_used", "?"),
                            "latency_ms": payload.get("latency_ms", 0),
                            "prompt_tokens": payload.get("prompt_tokens", 0),
                            "completion_tokens": payload.get("completion_tokens", 0),
                            "total_tokens": payload.get("total_tokens", 0),
                            "cost_usd": payload.get("cost_usd", 0),
                        }
                        st.markdown(full_response)
                        _render_route_footer(meta)
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": full_response,
                                "meta": meta,
                            }
                        )
                    else:
                        # Streaming path for Ollama / Gemini
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
                            full_response = (
                                "_(Model returned no text — possibly a safety filter "
                                "or empty response.)_"
                            )
                        placeholder.markdown(full_response)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": full_response}
                        )

                    # FR-06: Door inspection photo display.
                    # Runs inside the assistant message block so photos render
                    # directly below the text answer. We also stash the photo
                    # metadata on the chat_history entry so the next Streamlit
                    # rerun re-renders the same images via the history loop.
                    uids = extract_unit_ids(prompt)
                    is_broad = bool(BROAD_DOOR_RE.search(prompt or ""))
                    if is_broad and not uids:
                        uids = fetch_all_door_units_with_photos()
                    photo_groups = fetch_door_photos_for_units(uids)
                    if photo_groups:
                        render_door_photos(photo_groups, bulk=is_broad)
                        if (
                            st.session_state.chat_history
                            and st.session_state.chat_history[-1]["role"] == "assistant"
                        ):
                            st.session_state.chat_history[-1]["door_photos"] = photo_groups
                            st.session_state.chat_history[-1]["door_photos_bulk"] = is_broad
            except Exception as e:
                import traceback

                st.error(f"Model error: {type(e).__name__}: {e}")
                st.code(traceback.format_exc(), language="text")

# ==============================================================================
# TAB: INVENTORY
# ==============================================================================
with tab_inventory:
    st.markdown(
        "<div class='section-label'>Inspection ledger</div>"
        "<div class='section-title'>QC records</div>"
        "<div class='section-sub'>Browse, search, and edit data ingested from your QC and inventory spreadsheets.</div>",
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

        # FR-07: Door inspection rows show a 📷 indicator next to rows that
        # have an image_filename reference. We only inject it into the
        # read-only display path so the data_editor save-roundtrip never
        # accidentally writes the indicator column back into SQLite.
        is_door_table = (
            st.session_state.selected_table == "door_access_qc"
            and "image_filename" in df_current.columns
        )

        if filtering:
            df_display = df_view.copy()

            # Hide columns that are entirely empty within the filtered subset.
            # door_access_qc unions the schemas of every uploaded source file, so
            # a filter scoped to one upload would otherwise show a long tail of
            # empty columns inherited from unrelated source files.
            def _col_is_empty(series):
                s = series.astype(object)
                s = s.where(pd.notna(s), None)
                return all(
                    v is None
                    or (isinstance(v, str) and v.strip() in ("", "None", "nan", "NaN", "NaT"))
                    for v in s.tolist()
                )

            empty_cols = [c for c in df_display.columns if _col_is_empty(df_display[c])]
            if empty_cols:
                df_display = df_display.drop(columns=empty_cols)

            if is_door_table and "image_filename" in df_display.columns:
                df_display.insert(
                    0,
                    "📷",
                    df_display["image_filename"].apply(
                        lambda v: "📷" if (pd.notna(v) and str(v).strip()) else ""
                    ),
                )

            info_line = (
                f"Showing **{len(df_view)} of {len(df_current)}** rows. "
                "Editing is locked while filtering — clear filters to edit."
            )
            if empty_cols:
                info_line += f" Hiding {len(empty_cols)} column(s) that are empty in this view."
            st.info(info_line)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
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

        # FR-07: Thumbnail panel for door_access_qc. One expander per unit;
        # newest inspection date first. Max thumbnail width 400px per spec.
        if is_door_table and "unit_id" in df_current.columns:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-label'>Door photos</div>"
                "<div class='section-title'>Inspection photos</div>"
                "<div class='section-sub'>"
                "Expand a unit to view its most recent inspection photo. "
                "Older photos are nested inside each expander."
                "</div>",
                unsafe_allow_html=True,
            )
            try:
                with get_sql_connection() as conn:
                    qc_cols = {
                        row[1]
                        for row in conn.execute(
                            "PRAGMA table_info(door_access_qc);"
                        ).fetchall()
                    }
                    has_date = "inspection_date" in qc_cols
                    has_pic = "pic" in qc_cols
                    date_sel = (
                        "inspection_date" if has_date else "NULL AS inspection_date"
                    )
                    pic_sel = "pic" if has_pic else "NULL AS pic"
                    order = (
                        "ORDER BY inspection_date DESC, rowid DESC"
                        if has_date
                        else "ORDER BY rowid DESC"
                    )
                    photo_rows = pd.read_sql_query(
                        f"SELECT unit_id, image_filename, {date_sel}, {pic_sel} "
                        f"FROM door_access_qc "
                        f"WHERE image_filename IS NOT NULL "
                        f"AND TRIM(image_filename) != '' "
                        f"{order}",
                        conn,
                    )
            except Exception as e:
                photo_rows = pd.DataFrame()
                st.caption(f"Could not load photo metadata: {e}")

            if photo_rows.empty:
                st.caption(
                    "No inspection photos referenced yet. Upload a `door_*.xlsx` "
                    "file with an `image_filename` column to populate this section."
                )
            else:
                photo_rows["_unit_key"] = (
                    photo_rows["unit_id"].astype(str).str.upper().str.strip()
                )
                for uid, group in photo_rows.groupby("_unit_key", sort=True):
                    newest = group.iloc[0]
                    label = f"📷  {uid}  —  {newest['inspection_date'] or '—'}"
                    if has_pic and pd.notna(newest["pic"]):
                        label += f"  ·  {newest['pic']}"
                    label += f"  ·  {len(group)} photo(s)"
                    with st.expander(label):
                        for _, row in group.iterrows():
                            path = door_image_path(uid, row["image_filename"])
                            cap = (
                                f"{uid} · "
                                f"{row['inspection_date'] or '—'} · "
                                f"Submitted by: {row['pic'] if has_pic else '—'}"
                            )
                            if path and os.path.isfile(path):
                                st.image(path, caption=cap, width=400)
                            else:
                                st.caption(
                                    f"📷 {row['image_filename']} — "
                                    "Photo not available — file missing."
                                )

# ==============================================================================
# TAB: KNOWLEDGE BASE
# ==============================================================================
with tab_kb:
    st.markdown(
        "<div class='section-label'>Ingestion</div>"
        "<div class='section-title'>Knowledge base</div>"
        "<div class='section-sub'>Upload SOPs and QC spreadsheets. Everything stays on this machine.</div>",
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
            "QC spreadsheets</div>"
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

                    # FR-05: door inspection image-reference validation
                    if (
                        target_table == "door_access_qc"
                        and "image_filename" in df_upload.columns
                        and "unit_id" in df_upload.columns
                    ):
                        rows_with_ref = 0
                        rows_image_ok = 0
                        missing = []
                        for _, row in df_upload.iterrows():
                            fn = row.get("image_filename")
                            uid = row.get("unit_id")
                            if (
                                pd.notna(fn)
                                and str(fn).strip()
                                and pd.notna(uid)
                                and str(uid).strip()
                            ):
                                rows_with_ref += 1
                                if door_image_exists(uid, fn):
                                    rows_image_ok += 1
                                else:
                                    missing.append(
                                        f"{str(uid).strip()}/{str(fn).strip()}"
                                    )
                        st.success(
                            f"Ingested **{len(df_upload)}** rows into "
                            f"`{target_table}` from **{file.name}** · "
                            f"{rows_with_ref} with image refs · "
                            f"{rows_image_ok} found on disk · "
                            f"{len(missing)} missing."
                        )
                        if missing:
                            with st.expander(
                                f"⚠️ {len(missing)} referenced image(s) not found "
                                f"under `door_images/`"
                            ):
                                for m in missing[:50]:
                                    st.code(f"door_images/{m}", language="text")
                                if len(missing) > 50:
                                    st.caption(f"…and {len(missing) - 50} more")
                    else:
                        st.toast(
                            f"Merged {len(df_upload)} rows into "
                            f"`{target_table}` from {file.name}"
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
            "QC spreadsheets</div>",
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
