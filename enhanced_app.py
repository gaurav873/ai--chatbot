import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from pydantic.v1 import SecretStr

# Use a faster embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# --- SIDEBAR ---
st.sidebar.title("RAG Chatbot")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
chunk_size = st.sidebar.slider("Chunk size (characters)", min_value=256, max_value=2048, value=1000, step=128)
process_clicked = st.sidebar.button("Process PDFs", type="primary")

# --- SESSION STATE INIT ---
def ensure_session_state():
    for key, default in [
        ("processing_done", False),
        ("documents", []),
        ("vector_store", None),
        ("retriever", None),
        ("chat_history", []),
        ("model_provider", "OpenAI")
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
ensure_session_state()

# --- PROCESSING LOGIC ---
if process_clicked and uploaded_files:
    with st.spinner("Extracting text and chunking..."):
        texts = []
        for pdf_file in uploaded_files:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        # Chunking with progress bar
        chunks = []
        total = len(texts)
        progress = st.progress(0, text="Chunking documents...")
        for idx, text in enumerate(texts):
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            progress.progress((idx+1)/total, text=f"Chunked {idx+1}/{total} documents")
        progress.empty()
    st.session_state.documents = [Document(page_content=chunk) for chunk in chunks]
    # Embedding with progress bar
    with st.spinner("Embedding and indexing chunks..."):
        progress = st.progress(0, text="Embedding chunks...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        st.session_state.vector_store = FAISS.from_documents(st.session_state.documents, embedding=embeddings)
        progress.progress(1.0, text="Embedding complete!")
        progress.empty()
    st.session_state.retriever = st.session_state.vector_store.as_retriever()
    st.session_state.processing_done = True
    st.session_state.chat_history = []  # Reset chat on new process
    st.success("Processing done!")

if st.session_state.processing_done:
    st.sidebar.success("Processing done!")

# --- MAIN AREA ---
st.title("Chat with your PDFs")

# Clear Chat button
col_clear, col_title = st.columns([1, 8])
with col_clear:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
with col_title:
    st.markdown("<h3 style='margin-top:0.2em'>Conversation</h3>", unsafe_allow_html=True)

# Chat bar: [Model][Input][Send]
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    model_provider = st.selectbox(
        "Model",
        ["OpenAI", "Groq"],
        key="model_provider",
        label_visibility="visible",
        disabled=not st.session_state.processing_done
    )
with col2:
    user_input = st.text_input(
        "Ask a question about your PDFs:",
        key="user_input",
        disabled=not st.session_state.processing_done,
        placeholder="Type your question and press Enter or click Send..."
    )
with col3:
    send_clicked = st.button("Send", use_container_width=True, disabled=not st.session_state.processing_done)

# LLM selection and API key checks
llm = None
api_key_missing = False
if st.session_state.processing_done:
    if model_provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            api_key_missing = True
            st.warning("OpenAI API key is missing. Please set OPENAI_API_KEY in your environment.")
        else:
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=SecretStr(api_key))
    elif model_provider == "Groq":
        api_key = os.getenv("GROQ_API_KEY") or ""
        if not api_key:
            api_key_missing = True
            st.warning("Groq API key is missing. Please set GROQ_API_KEY in your environment.")
        else:
            llm = ChatOpenAI(model="llama3-8b-8192", api_key=SecretStr(api_key), base_url="https://api.groq.com/openai/v1")
    if llm:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=st.session_state.retriever, memory=memory, return_source_documents=True)

# Handle question submission with error handling
if st.session_state.processing_done and (user_input or send_clicked) and not api_key_missing:
    if user_input:
        try:
            response = chain({"question": user_input})
            st.session_state.chat_history.append((user_input, response["answer"]))
        except Exception as e:
            st.error(f"Error from LLM: {e}")
        st.rerun()

# Display chat history with modern bubbles
st.markdown("<div style='margin-top:1em'></div>", unsafe_allow_html=True)
for q, a in st.session_state.chat_history:
    st.markdown(f"<div style='background:#e3f2fd;padding:0.75em 1em;border-radius:8px;margin-bottom:0.5em'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#f5f5f5;padding:0.75em 1em;border-radius:8px;margin-bottom:1.5em'><b>Bot:</b> {a}</div>", unsafe_allow_html=True) 