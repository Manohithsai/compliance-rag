# frontend/app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title = "ComplianceRAG",
    page_icon  = "⚖️",
    layout     = "wide"
)

st.title("⚖️ ComplianceRAG")
st.markdown("**AI-powered compliance document Q&A system**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Status")

    try:
        status = requests.get(f"{API_URL}/status").json()
        st.success("✅ API Connected")
        st.metric(
            "Child Chunks Indexed",
            status["child_chunks"]["total_chunks"]
        )
        st.metric(
            "Parent Chunks Indexed",
            status["parent_chunks"]["total_chunks"]
        )
    except:
        st.error("❌ API not running. Start uvicorn first.")

    st.markdown("---")
    st.header("📤 Upload New Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type = ["pdf"]
    )

    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Ingesting document..."):
                files    = {"file": uploaded_file}
                response = requests.post(
                    f"{API_URL}/ingest",
                    files = files
                )
                if response.status_code == 200:
                    st.success("✅ Document ingested successfully")
                    st.json(response.json()["summary"])
                else:
                    st.error(f"Error: {response.text}")

# ── Main Query Interface ──────────────────────────────────────
st.header("💬 Ask a Compliance Question")

# example questions
st.markdown("**Try these example questions:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What is the right to erasure?"):
        st.session_state.question = "What is the right to erasure under GDPR?"

with col2:
    if st.button("Maximum GDPR fine?"):
        st.session_state.question = "What is the maximum fine for a GDPR violation?"

with col3:
    if st.button("Data controller obligations?"):
        st.session_state.question = "What are the obligations of a data controller?"

# query input
question = st.text_area(
    "Your question:",
    value  = st.session_state.get("question", ""),
    height = 100,
    placeholder = "e.g. What are the data subject rights under GDPR?"
)

top_k = st.slider("Number of sources to retrieve", 3, 10, 5)

if st.button("🔍 Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json = {
                        "question": question,
                        "top_k":    top_k
                    }
                )

                if response.status_code == 200:
                    result = response.json()

                    # ── Answer ────────────────────────────────
                    st.markdown("---")
                    st.subheader("📋 Answer")
                    st.markdown(result["answer"])

                    # ── Sources ───────────────────────────────
                    st.markdown("---")
                    st.subheader("📚 Sources Used")

                    for source in result["sources"]:
                        with st.expander(
                            f"Source {source['source_num']} — "
                            f"Page {source['page']} — "
                            f"{source['doc']}"
                        ):
                            st.markdown(source["preview"])

                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Could not connect to API: {e}")
                st.info("Make sure uvicorn is running in another terminal.")