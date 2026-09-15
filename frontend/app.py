"""
Barça History RAG Assistant -- Streamlit frontend.

A chat-style interface: the user types a question, sees a loading state
while the backend retrieves + generates, then sees the grounded answer
with its cited sources. Falls back to a friendly error message if the
backend is unreachable or returns an error.
"""

import streamlit as st

from api_client import API_BASE_URL, ApiError, ask_question, check_health

st.set_page_config(page_title="Barça History RAG Assistant", page_icon="🔵🔴", layout="centered")

st.title("🔵🔴 Barça History RAG Assistant")
st.caption(
    "Ask about FC Barcelona's history, signings, and stats -- answers are grounded "
    "in scraped club and Wikipedia sources, with cited sources shown below each answer."
)

# --- Sidebar: backend status ---
with st.sidebar:
    st.subheader("Backend status")
    st.caption(f"API: `{API_BASE_URL}`")
    if st.button("Check connection"):
        try:
            health = check_health()
            if health.get("vector_store_loaded"):
                st.success(f"Connected -- {health.get('total_chunks', '?')} chunks loaded")
            else:
                st.warning("Backend reachable, but the vector store isn't loaded yet.")
        except ApiError as exc:
            st.error(str(exc))

    st.divider()
    st.caption(
        "Example questions:\n"
        "- When did Johan Cruyff sign for Barcelona?\n"
        "- What is Camp Nou and when did it open?\n"
        "- What role did Ronaldinho play in the 2000s?"
    )

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str, "sources": list[str] | None}

# --- Render existing chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

# --- Chat input ---
question = st.chat_input("Ask about FC Barcelona history...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = ask_question(question)
                answer = result.get("answer", "")
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- `{source}`")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except ApiError as exc:
                error_message = f"⚠️ {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": None}
                )
