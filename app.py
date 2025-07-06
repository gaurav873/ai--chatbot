import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import List
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Load environment variables
load_dotenv()

# Check if GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found! Please set it in your .env file.")
    st.stop()

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents and show detailed debugging info"""
    text = ""
    total_pdfs = len(pdf_docs)
    st.write(f"📊 Processing {total_pdfs} PDF document(s)...")
    
    processed_docs = 0
    failed_docs = 0
    
    for i, pdf in enumerate(pdf_docs):
        try:
            pdf_reader = PdfReader(pdf)
            total_pages = len(pdf_reader.pages)
            pdf_text = ""
            pages_with_text = 0
            
            st.write(f"📄 **Document {i+1}**: {getattr(pdf, 'name', 'Unknown')}")
            st.write(f"   - Pages: {total_pages}")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pdf_text += page_text + "\n"
                    pages_with_text += 1
            
            st.write(f"   - Pages with text: {pages_with_text}/{total_pages}")
            st.write(f"   - Characters extracted: {len(pdf_text)}")
            
            if len(pdf_text) == 0:
                st.error(f"❌ No text extracted from {getattr(pdf, 'name', 'this PDF')}. This PDF is likely:")
                st.error("   • A scanned document (image-based)")
                st.error("   • Password protected")
                st.error("   • Corrupted or empty")
                st.error("   • Contains only images")
                failed_docs += 1
            elif len(pdf_text) < 100:
                st.warning(f"⚠️ Very little text extracted from {getattr(pdf, 'name', 'this PDF')} ({len(pdf_text)} chars). Content might be limited.")
                text += pdf_text + "\n\n"
                processed_docs += 1
            else:
                st.success(f"✅ Successfully extracted {len(pdf_text)} characters from {getattr(pdf, 'name', 'this PDF')}")
                text += pdf_text + "\n\n"
                processed_docs += 1
            
        except Exception as e:
            st.error(f"❌ Error processing {getattr(pdf, 'name', 'PDF')}: {str(e)}")
            failed_docs += 1
    
    st.write(f"📝 **Total text length**: {len(text)} characters")
    st.write(f"✅ Successfully processed: {processed_docs} documents")
    st.write(f"❌ Failed to process: {failed_docs} documents")
    
    if failed_docs > 0:
        st.warning("⚠️ Some documents could not be processed. Only successfully processed documents will be used for answering questions.")
    
    if len(text.strip()) == 0:
        st.error("❌ No text could be extracted from any of the uploaded PDFs. Please check your documents.")
        return ""
    
    return text

def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create vector store from text chunks"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Using FAISS for vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    """Create conversation chain with the vector store using Groq LLM with strict document focus"""
    system_template = """You are a helpful assistant that answers questions based STRICTLY on the provided documents.
    If the answer cannot be found in the documents, say "I cannot answer this question based on the provided documents."
    Do not make up or infer information that is not directly supported by the documents.
    Use the following pieces of context to answer the user's question:
    {context}"""

    human_template = """Question: {question}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.error("Please set your Groq API key in the Streamlit secrets or environment variables.")
        return None
    
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        openai_api_key=groq_api_key,  # Use openai_api_key instead of api_key
        openai_api_base="https://api.groq.com/openai/v1",  # Use openai_api_base instead of base_url
        temperature=0.1  # Lower temperature for more focused answers
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'  # Specify which output to store in memory
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,  # This will help track which documents were used
        output_key='answer'  # Match the memory output_key
    )

    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.error("Please process documents before asking questions!")
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_question})

    response = st.session_state.conversation({"question": user_question})

    answer_text = response['answer']
    st.write("Answer:", answer_text)

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    
    # Show source documents in an expander
    if response.get('source_documents'):
        with st.expander("View Source Documents"):
            for i, doc in enumerate(response['source_documents']):
                st.write(f"Source {i+1}:")
                st.write(doc.page_content)
                st.write("---")

    # Update chat history (for memory)
    st.session_state.chat_history = response['chat_history']

def main():
    # Configure page - must be the first Streamlit command
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    
    st.title("📚 RAG ChatBot - Multiple PDF Chat")
    st.markdown("Upload multiple PDF documents and ask questions about their content!")
    
    # Sidebar for document upload
    with st.sidebar:
        st.subheader("📁 Document Management")
        
        # File upload
        pdf_docs = st.file_uploader(
            "Upload PDF files", 
            accept_multiple_files=True, 
            type=['pdf']
        )
        
        if pdf_docs:
            st.write(f"📄 {len(pdf_docs)} files uploaded")
            for pdf in pdf_docs:
                st.write(f"• {pdf.name}")
        
        # Process button
        if st.button("🔄 Process Documents"):
            with st.spinner("Processing..."):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vector_store = get_vector_store(text_chunks)
                
                # Create conversation chain
                conversation_chain = get_conversation_chain(vector_store)
                if conversation_chain is not None:
                    st.session_state.conversation = conversation_chain
                    st.success("Documents processed! You can now ask questions.")
                else:
                    st.error("Failed to create conversation chain. Please check your API key configuration.")
        
        # Clear button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation = None
            st.success("Conversation cleared!")

    # Main chat interface
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_user_input(user_question)

    # Display chat history
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Instructions
    with st.expander("📋 How to use"):
        st.markdown("""
        1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
        2. **Process Documents**: Click "Process Documents" to analyze and index your files
        3. **Ask Questions**: Type your questions in the chat input
        4. **Get Answers**: The AI will provide answers based on your document content
        
        **Features:**
        - Upload unlimited PDF documents
        - Intelligent text chunking and indexing
        - Context-aware responses
        - Conversation memory
        - Fast similarity search using FAISS
        """)

if __name__ == "__main__":
    main() 