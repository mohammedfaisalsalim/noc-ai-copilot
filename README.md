# 🌐 Standalone NOC AI Copilot & Knowledge Management System

A production-grade, localized engineering runtime designed for Network Operations Centers (NOC) to monitor, manage, and audit campus infrastructure across **Wi-Fi, Core/Distribution Switching, Door Access Systems, and Hikvision CCTV Networks**.

This platform combines a high-fidelity relational data dashboard (SQLite) with an advanced retrieval-augmented generation (RAG) vector space (ChromaDB) powered by the **Gemini 2.5 Flash** model.

---

## 🚀 System Architecture & Key Modules

### 1. 📊 Consolidated Inventory Grid (CRUD)
* Highly interactive frontend asset data sheets built with dynamic Streamlit data editors.
* Direct data manipulation connection with **SQLite Write-Ahead Logging (WAL)** transaction isolation to prevent runtime database engine locks.
* Granular database table commit layer for real-time asset tracking.

### 2. 💬 In-Terminal AI Copilot
* Leverages **Gemini 2.5 Flash** for rapid infrastructure diagnosis, network mapping configuration queries, and error log tracking.
* Fully contextualized through dual-engine intelligence: pulls real-time hardware status metrics from relational SQL layouts alongside semantic technical SOPs.

### 3. 🗂️ Knowledge Base Central Terminal
* **Intelligent Form Parser Layer:** Automatically processes stylized, human-readable operations spreadsheets and weekly checklist report forms. It dynamically skips administrative metadata headers and isolates the active structural logging matrix.
* **Granular Network Router Isolation Engine:** Intelligently maps files into four isolated database tables (`wifi_qc`, `cctv_qc`, `door_access_qc`, `general_inventory`) to guarantee complete cross-contamination prevention.
* **In-Memory Anti-Duplication Controls:** Scans primary keys (`asset_id`, `door_no`) and eliminates row stacking on the fly.
* **File Operations Management Console:** Integrated single-click row deletion buttons to purge specific datasets from either SQL or ChromaDB registries instantly.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend Engine:** Streamlit
* **AI Core LLM Engine:** Google Generative AI (`gemini-2.5-flash`)
* **Relational Storage Matrix:** SQLite3 (WAL Mode Enabled)
* **Vector Vector Space Index:** ChromaDB
* **Data Processing Pipeline:** Pandas, OpenPyXL (Excel Processing Core)
* **Unstructured Ingestion Engine:** PyPDF (PDF Processing Core)

---

## ⚙️ Local Development Installation

### 1. Navigate to Your Environment
### Example:
```bash
cd "D:\VS Code\noc-ai-copilot"