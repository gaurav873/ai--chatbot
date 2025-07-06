# 📚 RAG ChatBot - Multiple PDF Document Chat

A powerful Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that allows you to upload multiple PDF documents and ask questions about their content using advanced language models.

## ✨ Features

- **📄 Multiple PDF Support**: Upload and process multiple PDF documents simultaneously
- **🤖 Groq LLM Integration**: Powered by Groq's fast LLM models (llama3-8b-8192)
- **🔍 Intelligent Search**: Advanced vector search using FAISS and sentence transformers
- **💬 Conversational Memory**: Maintains context across multiple questions
- **📊 Detailed Processing**: Shows detailed information about document processing
- **🎯 Document-Focused Answers**: Strictly answers based on uploaded documents only
- **📱 Modern UI**: Clean, responsive Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd project_ai_chat
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the template
   cp env_template.txt .env
   
   # Edit .env file and add your Groq API key
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the app**
   - Open your browser and go to `http://localhost:8501`
   - Upload your PDF documents and start chatting!

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required: Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Model Configuration
GROQ_MODEL=llama3-8b-8192
TEMPERATURE=0.1

# Optional: Text Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optional: Vector Store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SEARCH_K=5
```

### Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## 📖 How to Use

### 1. Upload Documents
- Use the sidebar to upload one or more PDF files
- Supported formats: PDF only
- Maximum file size: 100MB per file

### 2. Process Documents
- Click "🔄 Process Documents" button
- The app will show detailed processing information:
  - Number of pages per document
  - Pages with extractable text
  - Character count extracted
  - Success/failure status

### 3. Ask Questions
- Type your questions in the chat input
- The AI will search through all uploaded documents
- Answers are based strictly on document content
- Use "View Source Documents" to see which parts were used

### 4. Example Questions
```
- "What are the main topics in the documents?"
- "What does the second document say about [topic]?"
- "Can you summarize the key points?"
- "What are the differences between the documents?"
```

## 🏗️ Project Structure

```
project_ai_chat/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration management
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── env_template.txt      # Environment template
├── README.md            # This file
└── rag_env_new/         # Virtual environment
```

## 🔍 Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Python with LangChain framework
- **LLM**: Groq API (llama3-8b-8192 model)
- **Vector Store**: FAISS for fast similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **PDF Processing**: PyPDF2 for text extraction

### Processing Pipeline
1. **Text Extraction**: Extract text from PDF pages
2. **Chunking**: Split text into manageable chunks (1000 chars)
3. **Embedding**: Convert chunks to vector embeddings
4. **Indexing**: Store embeddings in FAISS vector database
5. **Retrieval**: Search for relevant chunks when questions are asked
6. **Generation**: Use LLM to generate answers from retrieved context

## 🛠️ Troubleshooting

### Common Issues

**1. "GROQ_API_KEY not found"**
- Ensure your `.env` file exists and contains the API key
- Check that the key is correctly formatted

**2. "No text extracted from PDF"**
- The PDF might be scanned (image-based)
- Try using a PDF with selectable text
- Check if the PDF is password-protected

**3. "App not responding"**
- Check if Groq API is accessible
- Verify your internet connection
- Ensure you haven't exceeded API limits

**4. "Streamlit set_page_config error"**
- Restart the Streamlit app
- Clear browser cache
- Check for multiple Streamlit processes

### Performance Tips

- **Large Documents**: Break into smaller PDFs for better processing
- **Multiple Documents**: Process them together for better context
- **Question Specificity**: Be specific in your questions for better answers
- **Clear Chat**: Use "Clear Conversation" to reset context when needed

## 🔄 Alternative LLM Providers

The app is configured for Groq by default, but you can easily switch to other providers:

### OpenAI
```python
# In app.py, replace Groq configuration with:
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1
)
```

### Google Gemini
```python
# Uncomment in requirements.txt and use:
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the error logs in the terminal
3. Ensure all dependencies are properly installed
4. Verify your API keys are correctly configured

## 🎯 Roadmap

- [ ] Support for more document formats (DOCX, TXT)
- [ ] Document comparison features
- [ ] Export chat history
- [ ] Advanced filtering options
- [ ] Multi-language support
- [ ] Document annotation features

---

**Happy Document Chatting! 📚✨** 