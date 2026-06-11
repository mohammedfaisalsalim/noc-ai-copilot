# NOC.AI Admin Guide

Practical reference for the person running the chatbot. Covers the manual work you need to do, and what kinds of questions the bot will answer well.

---

## Part 1 — Manual admin work

### A. One-time setup (already done once)

| Task | Where | Notes |
|------|-------|-------|
| Create MS Forms door inspection form | forms.office.com | Unit ID is pre-filled via URL parameter |
| Generate one QR code per door | Any free QR generator | Each QR encodes the form URL with `…=G1EE` etc. |
| Print + stick QR codes on doors | Physical | Done by you, not the app |
| Create `door_images/{UNIT}/` subfolders | Local disk | Already done for G1EE, G2EE, F1EE, F2EE, D1EE, D2EE |
| Install Ollama (optional) | `ollama.com/download` | Only needed for the local backend |

### B. Adding a new door (rare)

When a new door (e.g. `G3EE`) goes live:

1. Append `…=G3EE` to the form's pre-fill base URL
2. Generate a QR code from that URL, print, stick on the door
3. Create a new local subfolder:
   ```powershell
   mkdir "D:\VS Code\noc-ai-copilot\door_images\G3EE"
   New-Item "D:\VS Code\noc-ai-copilot\door_images\G3EE\.gitkeep" -ItemType File
   ```
4. No code changes. The chatbot picks up the new Unit ID automatically once it appears in an uploaded Excel.

### C. Weekly QC inspection workflow (recurring)

This is the main loop. Repeat every time you have new MS Forms submissions to load.

**Step 1 — Get the response file from MS Forms**
- Open MS Forms → **Responses** → **Open in Excel**
- Save the downloaded file to your Downloads folder

**Step 2 — Clean up the columns in Excel**

| MS Forms column | Action | Result |
|-----------------|--------|--------|
| `Id`, `Start time`, `Completion time`, `Email`, `Name` | Delete (optional) | Removes noise |
| `Break Glass Status` | Rename to `Breakglass Status` (one word) | Matches chatbot schema |
| `Photos` | Rename to `Image Filename` | Matches chatbot schema |
| `Photos` cell value (SharePoint URL) | Replace with the local filename, e.g. `G1EE_20260609_01.jpg` | Required for image lookup |

**Step 3 — Download and rename the photo(s)**
- In the original response Excel, Ctrl-click each photo link → it opens in OneDrive → **Download**
- Rename each photo to: `{unit_id}_{YYYYMMDD}_{NN}.jpg`
  - Example: `G1EE_20260609_01.jpg`
- Move to `D:\VS Code\noc-ai-copilot\door_images\{unit_id}\`

**Step 4 — Save the cleaned file**
- File → **Save As** → **CSV UTF-8** (this avoids the openpyxl xlsx bug)
- Filename: `door_{UNIT_ID}_{YYYY-MM-DD}.csv`
  - Example: `door_G1EE_2026-06-09.csv`
  - Any name starting with `door_` routes to the correct table

**Step 5 — Upload to the chatbot**
1. Launch:
   ```powershell
   cd "D:\VS Code\noc-ai-copilot"
   venv\Scripts\activate
   streamlit run app.py
   ```
2. **Knowledge base** tab → drag the CSV onto **QC spreadsheets**
3. Look for the green banner:
   ```
   ✓ Ingested 1 rows into `door_access_qc` from door_G1EE_2026-06-09.csv
     · 1 with image refs · 1 found on disk · 0 missing.
   ```
4. If it says **"X missing"**, expand the warning and fix the filename mismatch (Excel value must equal the actual file in `door_images/{unit_id}/` character-for-character).

### D. Updating SOPs

- Upload new SOPs via **Knowledge base** → **SOP manuals** (PDF, TXT, or MD)
- To remove an outdated SOP, click the **✕** next to its filename in the **Registry** section below

### E. Choosing a backend

| Backend | When to use | Setup needed |
|---------|-------------|--------------|
| **Cloud (Gemini)** | Default — works anywhere, fast, free tier | Paste API key in sidebar |
| **Local (Ollama)** | Air-gapped or no internet | Run `ollama pull llama3.1:8b` first |
| **LLM Gateway** | Production routing with rate limits + budget caps | Requires the companion `llm-gateway-platform` Docker stack |

### F. What to back up

Everything below is on your local disk only (gitignored). Copy to external storage on whatever schedule you need:

- `noc_inventory.db` — all ingested QC data
- `chroma_db/` — vector index of SOPs
- `door_images/` — inspection photos

### G. Fixing data mistakes

If a form submission has a typo (e.g. dropdown says "Working" but Remarks say "not working"):

1. Open the chatbot → **QC Records** tab → select `door_access_qc`
2. Clear all filters (editing is locked while filtering)
3. Click into the cell, edit the value
4. Click **Save changes**

Alternative: fix the source CSV and re-upload — the chatbot replaces existing rows for that same `source_file`.

---

## Part 2 — Queries that work well

The chatbot grounds answers in two sources: (1) indexed SOP PDFs and (2) the SQLite tables. It does **not** make calls to live infrastructure.

### Door inspection queries

Best when you mention a Unit ID like `G1EE`, `F2EE`, etc.

| Example prompt | What you'll get |
|----------------|-----------------|
| `What's the status of door G1EE?` | Full field breakdown (lock, breakglass, controller, power, card reader, remarks) + most recent photo inline + auto-flagged issues |
| `Show me the latest inspection for F2EE` | Newest row only, with photo |
| `Show me door status` / `List all doors` | Bulk render of every photographed unit |
| `Which doors have card_reader_status Not Working?` | Filtered list across all door rows |
| `Are there any contradictions in the G1EE remarks vs the status fields?` | Highlights dropdown ↔ remarks mismatches |

### SOP / procedure queries

These pull from the PDFs you've indexed under **SOP manuals**.

| Example prompt |
|----------------|
| `How do I pair an off-campus mesh router?` |
| `What's the troubleshooting procedure when a CCTV camera shows no signal?` |
| `Walk me through resetting a door access controller` |
| `What are the steps to swap out a faulty AP?` |

Answers cite the source PDF, e.g. *"(per APU Off-Campus mesh pairing SOP.pdf)"*.

### WiFi / CCTV / Inventory QC queries

| Example prompt | Table queried |
|----------------|---------------|
| `Which APs at GYM have signal below -75 dBm?` | `wifi_qc` |
| `Show me all access points currently in Maintenance status` | `wifi_qc` / `general_inventory` |
| `Which cameras on Block G are offline?` | `cctv_qc` |
| `What's the IP address of switch SW-B2-01?` | `general_inventory` |
| `List all faulty NVRs flagged in the last upload` | `cctv_qc` filtered by source_file |

### Cross-source queries

The bot can mix table data with SOPs in one answer:

| Example prompt |
|----------------|
| `The controller for G1EE is offline — what's the recovery procedure?` |
| `Camera 5 is faulty per the QC log — which SOP covers replacement?` |

---

## Part 3 — Queries that will NOT work well

Be aware of these limits so you don't trust a wrong answer:

| Type | Example | Why it fails |
|------|---------|--------------|
| **Image analysis** | "Does this G1EE photo show damage?" | The LLM doesn't see the photo — it's rendered by Streamlit, never sent to the model |
| **Live device status** | "Is G2EE online RIGHT NOW?" | Data is only as fresh as the last Excel upload — no live polling |
| **Predictions** | "When will the lock fail?" | No predictive model; only descriptive data |
| **Write operations** | "Mark G1EE as repaired" | Chatbot is read-only — use the QC Records tab to edit, then Save |
| **Data outside your sources** | "What's the warranty on a Cisco 9120?" | Only knows what's in your SOPs / QC tables |
| **Cross-tenant data** | "Compare us to other campuses" | No external data sources connected |
| **HR / staff queries** | "Who worked the most inspections this month?" | No HR data linked — only the `pic` field on each row |

---

## Part 4 — Common error messages

| Banner / error | What it means | Fix |
|---------------|---------------|-----|
| `Y found on disk · Z missing` after upload | Excel `image_filename` doesn't match a real file under `door_images/{unit_id}/` | Open the expander, fix spelling / case / extension, re-upload |
| `HTTPError 500 from /api/generate` | Ollama out of memory or context overflow | Switch to Gemini backend, OR pull `llama3.2:3b` and pick it in the sidebar |
| `ValueError: Sheet name is an empty list` | xlsx file was saved from a CSV in Excel and has invalid sheet metadata | Save as CSV UTF-8 instead, OR copy data into a fresh blank workbook first |
| `Gateway · OFFLINE` | LLM Gateway Docker stack isn't running | `docker-compose start` in the `llm-gateway-platform` folder |
| `Photo not available — file missing.` in chat | File listed in Excel doesn't exist locally | Place the file at `door_images/{unit_id}/{filename}` and re-ask |

---

## Part 5 — Quick reference: file naming conventions

| Type | Pattern | Example |
|------|---------|---------|
| Door inspection upload | `door_{UNIT}_{YYYY-MM-DD}.csv` | `door_G1EE_2026-06-09.csv` |
| Door inspection photo | `{UNIT}_{YYYYMMDD}_{NN}.jpg` | `G1EE_20260609_01.jpg` |
| Photo location | `door_images/{UNIT}/{filename}` | `door_images/G1EE/G1EE_20260609_01.jpg` |
| WiFi QC upload | Filename containing `wifi` | `wifi_qc_blockG_2026-06.xlsx` |
| CCTV QC upload | Filename containing `cctv`, `nvr`, or `camera` | `cctv_qc_2026-06.xlsx` |
| Inventory upload | Filename containing `inventory` or `materials` | `noc_materials.xlsx` |
| SOP document | Any descriptive name, PDF/MD/TXT | `APU Off-Campus mesh pairing SOP.pdf` |

The chatbot uses the filename to decide which SQLite table to write into. Stick to the patterns above and routing is automatic.
