import sys
import os
import io

# Walk up to project root: pages/ → app/ → project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st
from pathlib import Path

from ingestion.kb_builder import KBBuilder, COLUMNS
from agents.interviewer_agent import find_baseline_gaps, BASELINE_TOPICS, TOTAL_BASELINE

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Ingestion",
    page_icon="📂",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_DIR = str(PROJECT_ROOT / "data" / "mock_organization")

# ── Session state ─────────────────────────────────────────────────────────────

if "kb_df" not in st.session_state:
    st.session_state.kb_df = None
if "kb_gaps" not in st.session_state:
    st.session_state.kb_gaps = None

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
txt_files = []
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
        disabled=not (scan_path.exists() and bool(txt_files)),
    )

if scan_clicked:
    st.session_state.kb_df   = None
    st.session_state.kb_gaps = None

    builder = KBBuilder(
        directory=scan_dir,
        model=ollama_model,
        base_url=ollama_url,
    )

    files = sorted(scan_path.glob("*.txt"))
    progress_bar = st.progress(0, text="Starting…")
    status_box   = st.empty()

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

        # ── Phase 1: store KB in memory, compute gaps ─────────────────────────
        st.session_state.kb_df   = df
        st.session_state.kb_gaps = find_baseline_gaps(df, [])

    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        progress_bar.empty()
        status_box.empty()

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.kb_df is not None:
    df: pd.DataFrame = st.session_state.kb_df

    st.subheader(f"Extracted Controls — {len(df)} total")

    if df.empty:
        st.warning("No controls were extracted. Try a different model or check the documents.")
    else:
        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Controls",      len(df))
        c2.metric("Unique Domains",       df["domain"].nunique())
        c3.metric("Documents Processed",  df["source_document"].nunique())

        st.divider()

        with st.expander("View Full KB Document", expanded=False):
            domains = ["All"] + sorted(df["domain"].unique().tolist())
            selected_domain = st.selectbox("Filter by domain", domains)
            view_df = df if selected_domain == "All" else df[df["domain"] == selected_domain]

            st.dataframe(
                view_df,
                use_container_width=True,
                height=400,
                column_config={
                    "domain":            st.column_config.TextColumn("Domain",            width="medium"),
                    "control_statement": st.column_config.TextColumn("Control Statement", width="large"),
                    "source_document":   st.column_config.TextColumn("Source Document",   width="medium"),
                },
            )

        st.divider()

        # ── Phase 2: Gap Dashboard ────────────────────────────────────────────

        gaps: list[dict] = st.session_state.kb_gaps or []
        covered_count = TOTAL_BASELINE - len(gaps)
        coverage_pct  = int(covered_count / TOTAL_BASELINE * 100)

        st.subheader("🔍 Baseline Profile Gap Analysis")

        g1, g2, g3 = st.columns(3)
        g1.metric("Topics Covered", f"{covered_count} / {TOTAL_BASELINE}")
        g2.metric("Gaps Found",     len(gaps))
        g3.metric("Coverage",       f"{coverage_pct}%")

        if gaps:
            st.warning(
                f"**{len(gaps)} baseline topic(s) are missing** from the extracted Knowledge Base. "
                "The CISO Agent can interview you to fill them in."
            )

            # List missing topics grouped by domain
            by_domain: dict[str, list[str]] = {}
            for g in gaps:
                by_domain.setdefault(g["domain"], []).append(g["topic"])

            with st.expander("Missing topics", expanded=True):
                for domain, topics in by_domain.items():
                    bullet_list = "\n".join(f"  - {t}" for t in topics)
                    st.markdown(f"**{domain}**\n{bullet_list}")

            if st.button("💬 Complete missing info via CISO Agent →", type="primary"):
                st.switch_page("pages/2_ciso_interviewer.py")

        else:
            st.success(
                "✅ **All 14 baseline topics are already covered** by the extracted documents. "
                "Your Knowledge Base is ready — download it below and upload it to the Assessor."
            )

        st.divider()

        # ── Phase 4 (early exit path): download KB ────────────────────────────

        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Knowledge Base (Excel)",
            data=buf,
            file_name="knowledge_base.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Downloads the current in-memory KB. "
                 "Run the CISO Agent first to enrich it with the missing baseline topics.",
        )
