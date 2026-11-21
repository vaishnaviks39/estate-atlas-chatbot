"""Streamlit chat UI for Property Concierge. Talks to the FastAPI backend over HTTP."""
import json
import os
import re

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
APP_NAME = "Estate Atlas"

EXAMPLES = [
    "When will Trump Tower Jeddah be completed?",
    "Villas for sale in Riyadh under 3 million",
    "Which DarGlobal projects are in Oman?",
    "Apartments for rent in Jeddah with 3 bedrooms",
    "فيلا للبيع في جدة بأقل من مليونين",
]

st.set_page_config(page_title=APP_NAME, page_icon="🏛️", layout="centered")

st.markdown("""
<style>
h1 { font-weight: 400 !important; letter-spacing: 0.18em; text-transform: uppercase; }
.gold-rule { height: 1px; background: #C8A97E; opacity: 0.6; margin: 0.2rem 0 1.4rem 0; }
[data-testid="stChatMessage"] { border: 1px solid #2A2824; border-radius: 6px; }
section[data-testid="stSidebar"] { border-right: 1px solid #2A2824; }
</style>
""", unsafe_allow_html=True)


def cited_sources(answer, sources):
    """Keep only the sources the answer actually cites as [n]."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [s for s in sources if s["n"] in cited]


def show_sources(sources):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"[{s['n']}] **{s['source']}** · [{s['title']}]({s['url']})")


def stream_reply(question, history, meta):
    """Call the backend and yield answer pieces; sources / errors are stored in `meta`."""
    with requests.post(f"{API_URL}/api/chat", json={"question": question, "history": history},
                       stream=True, timeout=180) as r:
        r.raise_for_status()
        r.encoding = "utf-8"                      # keep Arabic text intact
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "meta":
                meta.update(event)
            elif event["type"] == "token":
                yield event["text"]
            elif event["type"] == "error":
                meta["error"] = event["message"]


# ---------- sidebar ----------
with st.sidebar:
    st.subheader(APP_NAME)
    st.caption("Luxury developments from DarGlobal and properties listed on Wasalt.")
    try:
        health = requests.get(f"{API_URL}/api/health", timeout=5).json()
        st.success("Online")
        st.caption(f"Model: {health['model']}")
    except requests.RequestException:
        st.error("Service offline")

    st.markdown("**Try asking**")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending = example
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ---------- chat ----------
st.title(APP_NAME)
st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
st.caption("Ask about DarGlobal branded residences or Wasalt properties across Saudi Arabia, "
           "in English or Arabic.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources", []))

question = st.chat_input("Ask about a project, area or price…") or st.session_state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-6:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        meta = {}
        try:
            placeholder, answer = st.empty(), ""
            placeholder.markdown("_Searching projects and listings…_")
            for piece in stream_reply(question, history, meta):
                answer += piece
                placeholder.markdown(answer + "▌")      # live typing with a cursor
            placeholder.markdown(answer)
        except requests.RequestException:
            answer, meta["error"] = "", "Could not reach the service. Please try again shortly."
        if meta.get("error"):
            st.error(meta["error"])
            answer = answer or meta["error"]
        sources = cited_sources(answer, meta.get("sources", []))
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})