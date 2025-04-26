from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_local_storage import LocalStorage
from utils import load_documents, get_vectorstore, get_retriever, get_qa_chain
from prompt import get_analysis_prompt
import os

st.set_page_config(page_title="Prompt Analysis", layout="wide")
st.title("Prompt Analysis")

storage = LocalStorage()

env_api_key = os.getenv("GEMINI_API_KEY", "")

if "api_key" not in st.session_state:
    saved_key = storage.getItem("api_key")
    st.session_state["api_key"] = saved_key if saved_key else env_api_key

with st.sidebar:
    st.markdown("[Get your Gemini API key here](https://aistudio.google.com/apikey)")
    
    api_key_input = st.text_input(
        "Gemini API Key", value=st.session_state["api_key"], type="password", key="api_key_input"
    )
    
    st.session_state["api_key"] = api_key_input
    
    # Check if a key is already saved in local storage
    saved_key = storage.getItem("api_key")
    
    # Show either Save or Clear button based on whether a key is already saved
    if saved_key:
        if st.button("Clear Key"):
            storage.deleteItem("api_key")
            st.session_state["api_key"] = ""
            st.success("API key cleared")
            st.rerun()  # Rerun to update the button
    else:
        if st.button("Save Key"):
            storage.setItem("api_key", st.session_state["api_key"])
            st.success("API key saved to browser storage")
            st.warning("⚠️ Note: The API key is only saved in your browser's local storage. If you clear your browser data or use a different browser, you'll need to enter it again.")
            st.rerun()  # Rerun to update the button

if not st.session_state["api_key"]:
    st.warning("Enter your Gemini API key.")
    st.stop()

@st.cache_resource
def get_chain(api_key):
    docs = load_documents()
    vec = get_vectorstore(docs, api_key)
    retriever = get_retriever(vec)
    return get_qa_chain(api_key, retriever)

chain = get_chain(st.session_state["api_key"])

prompt = st.text_area("Your prompt")

col1, col2 = st.columns([1, 3])
with col1:
    analyze_button = st.button("Analyze")
with col2:
    st.markdown("<span style='color: #888; padding-top: 5px;'>Note: Results are not 100% accurate. I recommend running the analysis multiple times.</span>", unsafe_allow_html=True)

if analyze_button and prompt:
    with st.spinner("Thinking..."):
        out = chain({"query": get_analysis_prompt(prompt)})
        st.write(f"**Answer:**\n\n{out['result']}")