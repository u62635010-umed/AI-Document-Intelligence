import streamlit as st
import os
import tempfile
from utils import process_pdf, get_answer

# page config
st.set_page_config(page_title="RAG-QA Groq Pro", page_icon="⚡", layout="wide")

# Custom CSS for Glassmorphism UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: rgba(51, 65, 85, 0.4);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
    }
    
    /* Header styling */
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(to right, #22d3ee, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #0ea5e9, #2563eb);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
    }
    
    /* Input box */
    .stChatInputContainer {
        border-radius: 20px;
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Source documents section */
    .source-box {
        background-color: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #0ea5e9;
        padding: 10px;
        margin-top: 5px;
        font-size: 0.85em;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Title
st.markdown("<h1>AI Document Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; margin-top: -20px;'>Super-fast RAG using Llama3 on Groq Cloud</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/lightning-bolt.png", width=100)
    st.title("Control Panel")
    
    uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Process Document
if uploaded_file:
    if st.session_state.qa_chain is None or st.session_state.get("last_uploaded") != uploaded_file.name:
        with st.status("⚡ Turbo-charging Intelligence...", expanded=True) as status:
            st.write("Extracting patterns...")
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.write("Chunking and Vectorizing (HuggingFace)...")
            try:
                st.session_state.qa_chain = process_pdf(tmp_path)
                st.session_state.last_uploaded = uploaded_file.name
                status.update(label="✅ Document Ready!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                status.update(label="❌ Processing Failed", state="error")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Show Sources"):
                for doc in message["sources"]:
                    st.markdown(f"<div class='source-box'>{doc.page_content[:300]}...</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.qa_chain:
        st.error("Please upload a document first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with Llama3..."):
                try:
                    answer, sources = get_answer(st.session_state.qa_chain, prompt)
                    st.markdown(answer)
                    
                    with st.expander("Show Sources"):
                        for doc in sources:
                            st.markdown(f"<div class='source-box'>{doc.page_content[:300]}...</div>", unsafe_allow_html=True)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
