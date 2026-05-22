import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API = os.getenv("FLASK_API_URL", "http://localhost:5001")

st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("🧠 RAG Chat Assistant")

# Sidebar
with st.sidebar:
    if st.button("Ingest Docs"):
        res = requests.post(f"{API}/ingest")
        st.success(res.json()["status"])

    if st.button("Clear Chat"):
        requests.post(f"{API}/clear")
        st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        res = requests.post(f"{API}/ask", json={"question": prompt})
        data = res.json()

        answer = data.get("answer", "Error")
        sources = data.get("sources", [])

    except Exception as e:
        answer = str(e)
        sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    if sources:
        with st.expander("Sources"):
            st.write(sources)