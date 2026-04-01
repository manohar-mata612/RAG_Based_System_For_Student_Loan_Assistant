"""
app.py
------
International Student Loan Assistant — Streamlit UI.

Mode A — Policy Q&A:       Questions answered from indexed lender corpus.
Mode B — Contract Analyzer: User uploads one or MORE PDFs; answers come
                             from all uploaded documents combined.
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from rag.pipeline  import build_groq_client, stream_answer, get_source_summary
from rag.retriever import build_retriever, build_vectorstore, delete_namespace
from utils.pdf_parser import extract_text_from_pdf
from utils.chunker    import chunk_pages


load_dotenv()

PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Student Loan Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resources ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_groq_client():
    return build_groq_client()

@st.cache_resource(show_spinner=False)
def get_policy_retriever():
    return build_retriever(namespace="policy-corpus")

# ── Session state ─────────────────────────────────────────────────────

def _init():
    defaults = {
        "mode":            "Policy Q&A",
        "messages":        [],
        "upload_ns":       None,         # single Pinecone namespace for ALL uploads
        "indexed_files":   [],           # list of filenames already indexed
        "last_sources":    [],
        "pending_query":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 Loan Assistant")
    st.caption("Free · Groq + MiniLM + Pinecone")
    st.divider()

    new_mode = st.radio(
        "Mode",
        ["Policy Q&A", "Contract Analyzer"],
        index=0 if st.session_state.mode == "Policy Q&A" else 1,
    )

    if new_mode != st.session_state.mode:
        st.session_state.mode          = new_mode
        st.session_state.messages      = []
        st.session_state.last_sources  = []
        st.session_state.pending_query = None

    st.divider()

    # ── Mode B — multi-file upload ────────────────────────────────────
    if new_mode == "Contract Analyzer":
        st.markdown("### 📄 Upload loan documents")
        st.caption("Upload one or more PDFs — sanction letters, agreements, T&C sheets.")

        uploaded_files = st.file_uploader(
            "Select PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can select multiple files at once or add them one by one.",
        )

        if uploaded_files:
            # Find files not yet indexed in this session
            current_names  = [f.name for f in uploaded_files]
            new_files      = [f for f in uploaded_files
                              if f.name not in st.session_state.indexed_files]

            if new_files:
                # Create session namespace on first upload
                if st.session_state.upload_ns is None:
                    st.session_state.upload_ns = f"upload-{uuid.uuid4().hex[:12]}"

                ns = st.session_state.upload_ns

                with st.spinner(f"Indexing {len(new_files)} new file(s) …"):
                    vs = build_vectorstore(ns)
                    for file in new_files:
                        pages = extract_text_from_pdf(file)
                        docs  = chunk_pages(pages, {
                            "source":    file.name,
                            "doc_type":  "loan_contract",
                            "namespace": ns,
                        })
                        vs.add_documents(docs)
                        st.session_state.indexed_files.append(file.name)

                st.session_state.messages = []   # reset chat for new doc set

            # Show indexed file list
            st.markdown("**Indexed documents:**")
            for fname in st.session_state.indexed_files:
                # only show files still present in uploader
                icon = "✅" if fname in current_names else "🗑️"
                st.markdown(f"{icon} `{fname}`")

            # Button to clear all uploads and start fresh
            if st.button("🗑️ Remove all documents", use_container_width=True):
                if st.session_state.upload_ns:
                    try:
                        delete_namespace(st.session_state.upload_ns)
                    except Exception:
                        pass
                st.session_state.upload_ns     = None
                st.session_state.indexed_files = []
                st.session_state.messages      = []
                st.session_state.last_sources  = []
                st.rerun()

        else:
            # No files in uploader — clean up if there were previous uploads
            if st.session_state.upload_ns and not st.session_state.indexed_files:
                st.session_state.upload_ns = None
            st.info("Upload one or more PDFs to start.")

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.last_sources  = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown(
        "<small>Policy sources: Avanse · MPOWER · Prodigy Finance · "
        "studentaid.gov · USCIS</small>",
        unsafe_allow_html=True,
    )

# ── Main content ──────────────────────────────────────────────────────

upload_ready = (
    new_mode == "Contract Analyzer"
    and len(st.session_state.indexed_files) > 0
)

if new_mode == "Policy Q&A":
    st.title("🎓 International Student Loan Assistant")
    st.markdown(
        "Ask about **Avanse**, **MPOWER**, and **Prodigy Finance** policies — "
        "repayment rules, collateral requirements, OPT/CPT impact, and more."
    )
else:
    st.title("📄 Loan Contract Analyzer")
    if upload_ready:
        n = len(st.session_state.indexed_files)
        names = ", ".join(f"**{f}**" for f in st.session_state.indexed_files)
        st.markdown(
            f"Analyzing {n} document{'s' if n > 1 else ''}: {names}. "
            "Ask me to explain any clause, rate, fee, or repayment term."
        )
    else:
        st.markdown("Upload one or more loan PDFs in the sidebar to get started.")

# ── Example chips ─────────────────────────────────────────────────────

POLICY_EX = [
    "Which banks give loans without collateral?",
    "What are the repayment rules after graduation?",
    "Does CPT/OPT affect loan eligibility?",
    "What is MPOWER's interest rate?",
    "How long is the moratorium at Avanse?",
]
CONTRACT_EX = [
    "What is my interest rate?",
    "When do I start repaying?",
    "Are there prepayment penalties?",
    "What fees am I charged?",
    "What happens if I miss a payment?",
]

examples = POLICY_EX if new_mode == "Policy Q&A" else CONTRACT_EX

if not st.session_state.messages and st.session_state.pending_query is None:
    st.markdown("**Try asking:**")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_query = ex

# ── Chat history ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Sources panel ─────────────────────────────────────────────────────

if st.session_state.last_sources and st.session_state.messages:
    if st.session_state.messages[-1]["role"] == "assistant":
        with st.expander("📚 Sources used", expanded=False):
            for src in st.session_state.last_sources:
                st.markdown(f"**{src['label']}**")
                if src["url"]:
                    st.markdown(f"🔗 [{src['url']}]({src['url']})")
                st.caption(src["snippet"])
                st.divider()

# ── Chat input ────────────────────────────────────────────────────────

blocked = new_mode == "Contract Analyzer" and not upload_ready

typed_input = st.chat_input(
    placeholder="Upload PDFs in the sidebar first…" if blocked else "Ask a question…",
    disabled=blocked,
)

active_query = typed_input or st.session_state.pending_query

# ── Generate answer ───────────────────────────────────────────────────

if active_query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    if new_mode == "Policy Q&A":
        retriever = get_policy_retriever()
        rag_mode  = "policy"
    else:
        retriever = build_retriever(namespace=st.session_state.upload_ns)
        rag_mode  = "contract"

    client = get_groq_client()
    with st.chat_message("assistant"):
        try:
            token_gen, source_docs = stream_answer(
                query     = active_query,
                retriever = retriever,
                client    = client,
                history   = st.session_state.messages[:-1],
                mode      = rag_mode,
            )
            full_reply = st.write_stream(token_gen)
        except Exception as e:
            full_reply = f" Error: {e}\n\nCheck your API keys in the `.env` file."
            st.error(full_reply)
            source_docs = []

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
    st.session_state.last_sources = get_source_summary(source_docs)
    st.rerun()
