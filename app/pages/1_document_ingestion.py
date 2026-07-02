import sys
import os

# Walk up to project root: pages/ → app/ → project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st
from pathlib import Path

from ingestion.kb_builder import KBBuilder, COLUMNS

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Ingestion",
    page_icon="📂",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_DIR = str(PROJECT_ROOT / "data" / "mock_organization")
OUTPUT_PATH = PROJECT_ROOT / "data" / "Mock_Org_KB.xlsx"

# ── Session state ─────────────────────────────────────────────────────────────

if "ingestion_df" not in st.session_state:
    st.session_state.ingestion_df = None

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📂 Document Ingestion Engine")
st.caption(
    "Scans local corporate security documents and extracts structured controls "
    "using a **local Ollama model** — no data ever leaves your machine."
)
st.divider()

# ── Sidebar — configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    scan_dir = st.text_input(
        "Directory to scan",
        value=DEFAULT_SCAN_DIR,
        help="Absolute path to the folder containing .txt policy files.",
    )

    ollama_model = st.text_input(
        "Ollama model tag",
        value="gemma",
        placeholder="gemma, gemma2, llama3, mistral …",
        help="Must be pulled locally first: `ollama pull <tag>`",
    )

    ollama_url = st.text_input(
        "Ollama server URL",
        value="http://localhost:11434",
        help="Default Ollama address. Change only if running on a custom port.",
    )

    st.divider()
    st.info(
        "🔒 **Privacy guarantee**: extraction runs entirely on this machine "
        "via Ollama. No document content is sent to any external API."
    )

# ── File preview ──────────────────────────────────────────────────────────────

scan_path = Path(scan_dir)
if scan_path.exists():
    txt_files = sorted(scan_path.glob("*.txt"))
    if txt_files:
        with st.expander(f"📄 Files found in directory ({len(txt_files)} files)", expanded=True):
            for f in txt_files:
                size_kb = f.stat().st_size / 1024
                st.markdown(f"- `{f.name}` &nbsp; _({size_kb:.1f} KB)_")
    else:
        st.warning("No `.txt` files found in the selected directory.")
else:
    st.error(f"Directory not found: `{scan_dir}`")

st.divider()

# ── Scan button ───────────────────────────────────────────────────────────────

col_btn, col_status = st.columns([2, 5])

with col_btn:
    scan_clicked = st.button(
        "🔍 Scan Local Directory",
        type="primary",
        width="stretch",
        disabled=not (scan_path.exists() and txt_files),
    )

if scan_clicked:
    st.session_state.ingestion_df = None  # reset previous run

    builder = KBBuilder(
        directory=scan_dir,
        model=ollama_model,
        base_url=ollama_url,
    )

    files = sorted(scan_path.glob("*.txt"))
    total = len(files)

    progress_bar = st.progress(0, text="Starting…")
    status_box = st.empty()

    def _on_progress(idx: int, total_files: int, filename: str):
        pct = idx / total_files if total_files else 1.0
        label = f"Processing `{filename}` ({idx}/{total_files})"
        progress_bar.progress(pct, text=label)
        status_box.info(label)

    try:
        with st.spinner("Extracting controls — this may take a minute per file…"):
            df = builder.build(progress_callback=_on_progress)

        progress_bar.progress(1.0, text="Extraction complete ✓")
        status_box.empty()
        st.session_state.ingestion_df = df

    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        progress_bar.empty()
        status_box.empty()

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.ingestion_df is not None:
    df: pd.DataFrame = st.session_state.ingestion_df

    st.subheader(f"Extracted Controls — {len(df)} total")

    if df.empty:
        st.warning("No controls were extracted. Try a different model or check the documents.")
    else:
        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Controls", len(df))
        c2.metric("Unique Domains", df["domain"].nunique())
        c3.metric("Documents Processed", df["source_document"].nunique())

        st.divider()

        # Domain filter
        domains = ["All"] + sorted(df["domain"].unique().tolist())
        selected_domain = st.selectbox("Filter by domain", domains)
        view_df = df if selected_domain == "All" else df[df["domain"] == selected_domain]

        # Table
        st.dataframe(
            view_df,
            use_container_width=True,
            height=450,
            column_config={
                "domain":            st.column_config.TextColumn("Domain", width="medium"),
                "control_statement": st.column_config.TextColumn("Control Statement", width="large"),
                "source_document":   st.column_config.TextColumn("Source Document", width="medium"),
            },
        )

        st.divider()

        # Save to Excel
        col_save, col_msg = st.columns([2, 5])
        with col_save:
            if st.button("💾 Save KB to Excel", type="secondary"):
                df.to_excel(OUTPUT_PATH, index=False)
                st.success(f"Saved to `{OUTPUT_PATH}`")

        # Download button (no save needed)
        import io
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Mock_Org_KB.xlsx",
            data=buf,
            file_name="Mock_Org_KB.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
