# NOC.AI — Network Operations Copilot

A local-first AI assistant for Network Operations Centers. Combines a searchable inventory dashboard with a chatbot grounded in your SOPs and QC spreadsheets. Designed for campus networks managing **Wi-Fi, switching, door access, and CCTV** infrastructure.

Everything runs on your machine. Your network data never leaves the host.

---

## Features

- **Local chatbot** grounded in your SOPs and inventory — answers cite the source PDF or spreadsheet
- **Semantic SOP search** via local sentence-transformers (no API key, no cloud)
- **Multi-sheet Excel ingestion** with smart header detection — weekly QC logs across multiple sheets are fully consolidated
- **Interactive inventory grid** with source-file filter, free-text search across all columns, and double-click editing
- **Two backend options**:
  - **Ollama** (default) — free, runs entirely on your machine, no API key
  - **Gemini** — cloud fallback if you prefer not to install Ollama
- **NOC mission-control UI** — dark theme with live status indicators

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/mohammedfaisalsalim/noc-ai-copilot.git
cd noc-ai-copilot
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

### 2. Set up a chat backend (pick one)

**Option A — Ollama (recommended, no API key):**

1. Download from [ollama.com/download](https://ollama.com/download) and install
2. Pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. That's it — the app will auto-detect Ollama on `localhost:11434`.

**Option B — Gemini (cloud, requires free API key):**

1. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Paste it into the sidebar after launching the app

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Using the app

### Chat tab
Ask questions like:
- *"Which doors are flagged 'not working' in the on-campus QC form?"*
- *"Walk me through the off-campus mesh router pairing procedure."*
- *"A CCTV camera shows no signal. What's the standard troubleshooting flow?"*

Answers cite the source PDF (for procedures) or the source spreadsheet + sheet name (for inventory data).

### Inventory tab
- Pick a table (`wifi_qc`, `cctv_qc`, `door_access_qc`, `general_inventory`)
- Filter by source file and/or search across all columns
- Clear filters to enter edit mode — make changes, click **Save changes**

### Knowledge base tab
- **SOP manuals** column: upload PDF/TXT/MD files — chunked and indexed for semantic search
- **Inventory spreadsheets** column: upload CSV/XLSX — auto-routed to the right table by filename:
  - `wifi*` → `wifi_qc`
  - `cctv*`, `nvr*`, `camera*` → `cctv_qc`
  - `on campus*`, `door*`, `qc form*` → `door_access_qc`
  - `inventory*`, `materials*` → `general_inventory`
- Multi-sheet `.xlsx` files are fully ingested; each row tracks its `source_file` and `sheet_name`
- Remove any indexed source via the `✕` button

---

## Architecture

| Layer       | Stack                                              |
|-------------|----------------------------------------------------|
| Frontend    | Streamlit                                          |
| Chat LLM    | Ollama (local) or Google Gemini (cloud)            |
| Embeddings  | ChromaDB default — `all-MiniLM-L6-v2` (local ONNX) |
| Vector DB   | ChromaDB (persistent, on disk)                     |
| Relational  | SQLite with WAL mode                               |
| PDF / Excel | pypdf, pandas, openpyxl                            |

### Data privacy
- Ingested data lives in `noc_inventory.db` (SQLite) and `./chroma_db/` (vectors) — both local files
- With the Ollama backend, **zero data leaves your machine**
- With Gemini, only the question + retrieved context is sent to Google's API

---

## Configuration

Edit the constants at the top of `app.py`:

```python
DB_PATH = "noc_inventory.db"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "noc_sops_v2"

OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
```

---

## Troubleshooting

**Ollama shows "OFFLINE" in the sidebar**
Make sure Ollama is running (`ollama serve` or start the desktop app), then refresh the page.

**Chat returns no text**
Usually a safety filter (Gemini) or an empty model response. The UI surfaces the feedback reason — if it persists, try a different question or switch backends.

**Spreadsheet ingestion shows "No usable data rows detected"**
The auto-detector couldn't find a header row in the first 10 rows of any sheet. Check that at least one row in the first 10 contains short text labels (typical column headers).

---

## License

Internal / educational use.
