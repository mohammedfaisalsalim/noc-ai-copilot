import os
import sqlite3
import pandas as pd
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.api.types import EmbeddingFunction
from pypdf import PdfReader

# ==============================================================================
# DECOUPLED PROXY CONFIGURATION (FUTURE GATEWAY HOOK)
# ==============================================================================
API_GATEWAY_URL = None  
DB_PATH = "noc_inventory.db"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "noc_sops"

st.set_page_config(page_title="NOC AI Copilot & KM System", layout="wide")

# ==============================================================================
# LOCAL ZERO-COST EMBEDDING ENGINE
# ==============================================================================
class SimpleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            vec = [0.0] * 384
            for word in text.lower().split():
                idx = hash(word) % 384
                vec[idx] += 1.0
            mag = sum(x**2 for x in vec) ** 0.5
            if mag > 0:
                vec = [x / mag for x in vec]
            embeddings.append(vec)
        return embeddings

@st.cache_resource
def get_vector_db():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = SimpleEmbeddingFunction()
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

def get_sql_connection():
    """Returns a connection configured with WAL mode to prevent database locks."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0) 
    conn.execute("PRAGMA journal_mode=WAL;")      
    return conn

chroma_collection = get_vector_db()

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_table" not in st.session_state:
    st.session_state.selected_table = "wifi_qc"

# ==============================================================================
# CLEANED SIDEBAR: CONTROL ACCESS CORES ONLY
# ==============================================================================
with st.sidebar:
    st.header("⚙️ NOC Ingestion Control")
    
    api_key = st.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.warning("Please input your Gemini API Key to enable the Copilot layer.")

# ==============================================================================
# MAIN ENGINE LAYOUT
# ==============================================================================
st.title("🌐 Standalone NOC AI Copilot & Knowledge Management")
st.caption("Production-grade localized engineering runtime for Campus Wi-Fi, Switching, Door Access, and CCTV infrastructure.")

tab1, tab2, tab3 = st.tabs([
    "📊 Consolidated Inventory Grid (CRUD)", 
    "💬 Copilot Terminal", 
    "🗂️ Knowledge Base Directory & Uploader"
])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE DATA EDITOR LAYER
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Aggregated Campus Asset Matrix")
    
    tables = []
    with get_sql_connection() as conn:
        tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';").fetchall()]
    
    if tables:
        if st.session_state.selected_table not in tables:
            st.session_state.selected_table = tables[0]
            
        st.session_state.selected_table = st.selectbox("Select Operational Environment Database Table:", tables)
        
        df_current = pd.DataFrame()
        with get_sql_connection() as conn:
            df_current = pd.read_sql_query(f"SELECT * FROM {st.session_state.selected_table}", conn)
        
        st.write(f"Active records inside **{st.session_state.selected_table}**: **{len(df_current)} rows**.")
        
        dynamic_widget_key = f"editor_{st.session_state.selected_table}_{len(df_current)}"
        edited_df = st.data_editor(
            df_current,
            num_rows="dynamic",
            use_container_width=True,
            key=dynamic_widget_key
        )
        
        if st.button("Commit Table Changes", type="primary"):
            try:
                with get_sql_connection() as conn:
                    conn.execute(f"DROP TABLE IF EXISTS {st.session_state.selected_table};")
                    edited_df.to_sql(st.session_state.selected_table, conn, if_exists="replace", index=False)
                    conn.commit()
                st.success(f"Changes permanently synchronized inside '{st.session_state.selected_table}' database engine!")
                st.rerun()
            except Exception as e:
                st.error(f"CRUD Sync Failure: {e}")
    else:
        st.info("No active network data discovered. Head over to the 'Knowledge Base Directory & Uploader' tab to load your files.")

# ------------------------------------------------------------------------------
# TAB 2: TERMINAL INTERFACE
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("NOC Network Engine Terminal")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Query aggregated topologies, IP ranges, SOP entries, or specific NVR logs..."):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        if not api_key:
            with st.chat_message("assistant"):
                st.error("Operation Aborted: Gemini API authentication token missing.")
        else:
            rag_context = ""
            try:
                query_results = chroma_collection.query(query_texts=[prompt], n_results=4)
                if query_results and 'documents' in query_results and query_results['documents'][0]:
                    rag_context = "\n\n".join(query_results['documents'][0])
            except Exception as e:
                rag_context = "No historical SOP vectors loaded."

            db_context = ""
            try:
                with get_sql_connection() as conn:
                    chat_tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';").fetchall()]
                    if chat_tables:
                        for table in chat_tables:
                            df_table_snapshot = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 15", conn)
                            db_context += f"\nTABLE NAME: {table}\n"
                            db_context += df_table_snapshot.to_string() + "\n---"
                    else:
                        db_context = "No active relational data tables present in database storage."
            except Exception as db_err:
                db_context = f"Database read failure: {db_err}"

            system_prompt = f"""You are the expert Principal NOC Systems and Software Engineer Copilot. 
Your duties are assisting operators managing campus Wi-Fi uptime, switch configurations, door access control maps, and Hikvision IP surveillance systems.

The operator has loaded multiple infrastructure datasets cleanly split across specific environment tables:
- 'wifi_qc': Campus Access Point signal mapping and parameters.
- 'cctv_qc': Hikvision NVR streams, channel logs, and camera status records.
- 'door_access_qc': On-campus door control panels, card readers, and access logs.
- 'general_inventory': The master hardware and asset registry data sheets.

Use the 'source_file' column within the tables to identify the precise file origin of the data.

1. GROUNDED VECTOR SOP MANUAL CONTEXT (RAG):
---
{rag_context}
---

2. ACCUMULATED RELATIONAL TABLES SNAPSHOT (SQL LOGS):
---
{db_context}
---

INSTRUCTIONS:
- Directly read across all relational tables to pull asset statistics, channel maps, or camera flags.
- Trace specific device parameters back to their respective 'source_file'.
"""

            try:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    messages_payload = [
                        {"role": "user", "parts": [f"{system_prompt}\n\nUser Question: {prompt}"]}
                    ]
                    
                    full_response = ""
                    response = model.generate_content(messages_payload, stream=True)
                    
                    for chunk in response:
                        full_response += chunk.text
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Execution Error within AI Engine: {e}")

# ------------------------------------------------------------------------------
# TAB 3: CONSOLIDATED KNOWLEDGE BASE DIRECTORY, UPLOADER & GRANULAR DELETION
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🗂️ Knowledge Base Central Management Terminal")
    st.write("Upload or remove operational templates, manuals, and data sheets directly from your system memory indices.")
    
    # 📥 THE DECOUPLED FILE INGESTION CONTROLLERS
    upload_col1, upload_col2 = st.columns(2)
    
    with upload_col1:
        st.markdown("#### 📚 Ingest Operational Manuals (Chroma Vector DB)")
        sop_files = st.file_uploader("Upload SOPs (.txt, .md, .pdf)", accept_multiple_files=True, type=["txt", "md", "pdf"], key="kb_tab_sop_uploader")
        
        if sop_files:
            for file in sop_files:
                if file.size == 0:
                    continue
                    
                file_id = f"{file.name}_{file.size}"
                existing = chroma_collection.get(ids=[f"{file_id}_0"])
                
                if not existing or not existing['ids']:
                    content = ""
                    if file.name.endswith('.pdf'):
                        try:
                            pdf_reader = PdfReader(file)
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    content += text + "\n"
                        except Exception as e:
                            st.error(f"Error parsing PDF {file.name}: {e}")
                            continue
                    else:
                        content = file.read().decode("utf-8")
                    
                    if content.strip():
                        chunks = []
                        size, overlap = 600, 120
                        i = 0
                        while i < len(content):
                            chunks.append(content[i:i+size])
                            i += size - overlap
                        
                        ids = [f"{file_id}_{idx}" for idx in range(len(chunks))]
                        metadatas = [{"source": file.name} for _ in chunks]
                        
                        chroma_collection.add(documents=chunks, metadatas=metadatas, ids=ids)
                        st.toast(f"Indexed {len(chunks)} blocks from {file.name}", icon="📚")
                        st.rerun()

    with upload_col2:
        st.markdown("#### 📊 Ingest Network Asset Inventories (SQLite Engine)")
        excel_files = st.file_uploader("Upload Data Sheets (.csv, .xlsx)", accept_multiple_files=True, type=["csv", "xlsx"], key="kb_tab_excel_uploader")
        
        if excel_files:
            for file in excel_files:
                if file.size == 0:
                    continue
                    
                try:
                    if file.name.endswith('.csv'):
                        df_upload = pd.read_csv(file)
                    else:
                        # 🧠 INTELLIGENT QC FORM PARSER LAYER
                        filename_lower = file.name.lower()
                        
                        if "on_campus" in filename_lower or "door" in filename_lower or "access_qc" in filename_lower or "qc_form" in filename_lower:
                            df_raw = pd.read_excel(file)
                            
                            # Dynamically look for the table header row row
                            header_row_idx = 0
                            for idx, row in df_raw.iterrows():
                                row_str = " ".join([str(val).lower() for val in row.values if pd.notna(val)])
                                if "door" in row_str or "lock" in row_str or "reader" in row_str or "no." in row_str:
                                    header_row_idx = idx + 1
                                    break
                            
                            if header_row_idx > 0:
                                df_upload = pd.read_excel(file, skiprows=header_row_idx)
                            else:
                                df_upload = pd.read_excel(file)
                        else:
                            df_upload = pd.read_excel(file)
                    
                    # HARDENED CLEANING STRUCTURAL GUARDRAILS
                    df_upload = df_upload.dropna(how='all')
                    df_upload.columns = [str(c).strip().replace(' ', '_').lower() for c in df_upload.columns]
                    df_upload = df_upload.loc[:, ~df_upload.columns.str.contains('^unnamed')]
                    
                    if len(df_upload) == 0:
                        st.error(f"Skipping empty or malformed dataset grid: {file.name}")
                        continue
                        
                    df_upload['source_file'] = file.name
                    
                    # 🛠️ EXACT ISOLATION SWITCH ROUTING ENGINE
                    filename_lower = file.name.lower()
                    
                    if "wifi" in filename_lower or "bssid" in df_upload.columns or "ssid" in df_upload.columns:
                        target_table = "wifi_qc"
                        pk_column = 'asset_id' if 'asset_id' in df_upload.columns else ('device_name' if 'device_name' in df_upload.columns else None)
                    elif "cctv" in filename_lower or "nvr" in filename_lower or "camera" in filename_lower:
                        target_table = "cctv_qc"
                        pk_column = 'asset_id' if 'asset_id' in df_upload.columns else ('camera_name' if 'camera_name' in df_upload.columns else None)
                    elif "on_campus" in filename_lower or "door" in filename_lower or "access_qc" in filename_lower or "qc_form" in filename_lower:
                        target_table = "door_access_qc"
                        pk_column = 'door_no' if 'door_no' in df_upload.columns else (df_upload.columns[0] if len(df_upload.columns) > 0 else None)
                    elif "inventory" in filename_lower or "materials" in filename_lower:
                        target_table = "general_inventory"
                        pk_column = 'asset_id' if 'asset_id' in df_upload.columns else None
                    else:
                        clean_name = ''.join(e for e in filename_lower.split('.')[0] if e.isalnum() or e == ' ').strip().replace(' ', '_')
                        target_table = f"table_{clean_name}"
                        pk_column = None
                    
                    # 🛡️ LOCALIZED ANTI-DUPLICATION GUARDRAY LAYER
                    if pk_column and pk_column in df_upload.columns:
                        df_upload = df_upload.dropna(subset=[pk_column])
                        df_upload = df_upload.drop_duplicates(subset=[pk_column], keep='first')
                    else:
                        df_upload = df_upload.drop_duplicates(keep='first')
                    
                    with get_sql_connection() as conn:
                        try:
                            cursor = conn.cursor()
                            cursor.execute(f"PRAGMA table_info({target_table});")
                            existing_columns = {row[1] for row in cursor.fetchall()}
                            
                            for col in df_upload.columns:
                                if col not in existing_columns:
                                    conn.execute(f"ALTER TABLE {target_table} ADD COLUMN {col} TEXT;")
                                    existing_columns.add(col) 
                        except Exception:
                            pass 

                        # Isolated file transaction check to eliminate cross-refresh duplication
                        try:
                            df_existing = pd.read_sql_query(f"SELECT * FROM {target_table} WHERE source_file != ?", conn, params=(file.name,))
                            df_final = pd.concat([df_existing, df_upload], ignore_index=True)
                            df_final.to_sql(target_table, conn, if_exists="replace", index=False)
                        except Exception:
                            df_upload.to_sql(target_table, conn, if_exists="append", index=False)
                            
                        conn.commit()
                    st.toast(f"Merged {len(df_upload)} unique rows into '{target_table}' from {file.name}", icon="➕")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

    st.markdown("---")
    st.subheader("🔍 Active Infrastructure Registry Audits & File Operations")
    
    display_col1, display_col2 = st.columns(2)
    
    # --------------------------------------------------------------------------
    # COLUMN 1 (LEFT): CHROMADB VECTOR REGISTRY WITH GRANULAR DELETION
    # --------------------------------------------------------------------------
    with display_col1:
        st.markdown("### Processed Operational SOP Documents")
        try:
            all_vectors = chroma_collection.get(include=["metadatas"])
            
            if all_vectors and all_vectors['metadatas']:
                unique_sources = {}
                for doc_id, meta in zip(all_vectors['ids'], all_vectors['metadatas']):
                    src = meta.get("source", "Unknown Manual Source")
                    if src not in unique_sources:
                        unique_sources[src] = []
                    unique_sources[src].append(doc_id)
                
                for s_name, id_list in unique_sources.items():
                    c_doc, c_doc_del = st.columns([4, 1])
                    c_doc.write(f"📚 {s_name} ({len(id_list)} text chunks indexed)")
                    
                    if c_doc_del.button("🗑️ Delete", key=f"del_chroma_{s_name}", type="secondary"):
                        chroma_collection.delete(ids=id_list)
                        st.toast(f"Purged {s_name} from knowledge base vector index.", icon="🗑️")
                        st.rerun()
            else:
                st.info("No PDF or Markdown SOP manuals have been processed into the local vector index directory yet.")
        except Exception as e:
            st.info("Vector space index initialized but empty. Ready for document uploads.")
            
    # --------------------------------------------------------------------------
    # COLUMN 2 (RIGHT): SQL FILE REGISTRY WITH TARGETED ROW DELETION
    # --------------------------------==========================================
    with display_col2:
        st.markdown("### Compiled Spreadsheet Data Layers")
        try:
            with get_sql_connection() as conn:
                check_tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';").fetchall()]
                
                if check_tables:
                    for t in check_tables:
                        sources = pd.read_sql_query(f"SELECT DISTINCT source_file FROM {t}", conn)
                        st.markdown(f"**Database Table Reference:** `{t}`")
                        
                        for idx, row in sources.iterrows():
                            fname = row['source_file']
                            count_res = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE source_file = ?", (fname,)).fetchone()[0]
                            
                            c_file, c_del = st.columns([4, 1])
                            c_file.write(f"📄 {fname} ({count_res} records)")
                            
                            if c_del.button("🗑️ Delete", key=f"del_sql_{t}_{fname}", type="secondary"):
                                with get_sql_connection() as rm_conn:
                                    rm_conn.execute(f"DELETE FROM {t} WHERE source_file = ?", (fname,))
                                    rm_conn.commit()
                                    
                                    remaining = rm_conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                                    if remaining == 0:
                                        rm_conn.execute(f"DROP TABLE IF EXISTS {t};")
                                        rm_conn.commit()
                                        
                                st.toast(f"Removed data records for {fname} successfully.", icon="🗑️")
                                st.rerun()
                        st.markdown("---")
                else:
                    st.info("No spreadsheet files are currently mapped into the SQLite tracking layout.")
        except Exception as e:
            st.error(f"Failed to query SQL registry: {e}")