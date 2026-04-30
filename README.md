# 🏥 Medical Chatbot — RAG-Powered Q&A

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pinecone-Vector%20DB-6C5CE7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  A production-ready <b>Retrieval-Augmented Generation (RAG)</b> medical chatbot that answers clinical questions by retrieving relevant context from a curated medical knowledge base, powered by Google Gemini and Pinecone vector search.
</p>

---

## 📖 About

This project is a **full-stack AI medical chatbot** built using state-of-the-art LLM and vector database technologies. It ingests medical reference PDFs, indexes them into a Pinecone vector store, and uses Google Gemini 2.5 Flash to answer clinical questions with context-grounded, hallucination-resistant responses.

The system follows a clean **RAG (Retrieval-Augmented Generation)** architecture:
- **Ingest**: Medical PDFs are chunked, embedded, and stored in Pinecone
- **Retrieve**: User queries are matched against stored vectors using cosine similarity
- **Generate**: Gemini LLM synthesizes a concise, grounded answer using the retrieved context
- **Serve**: A Flask REST API and HTML chat UI provide a complete user interface

Built as a portfolio project to demonstrate expertise in **LangChain, LLMs, vector databases, and production-grade Python web applications**.

---

## 🧠 How It Works — RAG Pipeline

```
Medical PDFs  ──▶  load_pdf_file()          ← LangChain DirectoryLoader + PyPDFLoader
                       │
                   filter_to_minimal_docs()  ← strips metadata noise
                       │
                   text_split()             ← RecursiveCharacterTextSplitter (500 tokens, 20 overlap)
                       │
                   download_embedding()     ← sentence-transformers/all-MiniLM-L6-v2 (384d)
                       │
                   PineconeVectorStore      ← stores + indexes embeddings
                       │
              ┌────────┘
User Query ──▶│  similarity_search (k=3)
              └──▶  create_retrieval_chain()
                       │
                   ChatGoogleGenerativeAI   ← Gemini 2.5 Flash (temp=0.3)
                       │
                   Flask API  /chat         ← JSON response
                       │
                   chat.html UI             ← real-time chat interface
```

---

## ✨ Features

- 📄 **PDF Knowledge Ingestion** — bulk-load any medical reference PDFs into Pinecone
- 🔍 **Semantic Search** — cosine similarity retrieval (k=3 most relevant chunks)
- 🤖 **Gemini 2.5 Flash LLM** — fast, accurate, grounded answers with source context
- 🛡️ **Hallucination Guard** — explicitly says "I don't know" when context is insufficient
- 🌐 **Flask REST API** — lightweight `/chat` endpoint for easy integration
- 💬 **Chat UI** — clean HTML/JS chat interface

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Ashwin14101/Medical-Chatbot-With-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Medical-Chatbot-With-LLMs-LangChain-Pinecone-Flask-AWS
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

> **Get your keys:**
> - Pinecone → https://app.pinecone.io
> - Google AI Studio → https://aistudio.google.com/app/apikey

### 4. Add your medical PDFs

Place your PDF files in the `data/` directory.

### 5. Build the vector index (run once)

```bash
python store_index.py
```

This will:
- Load & chunk your PDFs
- Generate embeddings with `all-MiniLM-L6-v2`
- Create a Pinecone index named `medical-bot`
- Upsert all vectors

### 6. Launch the chatbot

```bash
python app.py
```

Open http://localhost:8080 in your browser.

---

## 📁 Project Structure

```
Medical-Chatbot-With-LLMs-LangChain-Pinecone-Flask-AWS/
├── app.py                  # Original Flask app (basic version)
├── app1.py                 # Updated Flask app with JSON API + improved comments
├── store_index.py          # One-time PDF ingestion + Pinecone indexing script
├── src/
│   ├── helper.py           # PDF loader, text splitter, embedding model
│   └── prompt.py           # System prompt for the medical assistant
├── templates/
│   └── chat.html           # Frontend chat UI
├── Static/                 # CSS / JS assets
├── data/                   # Place your medical PDFs here (gitignored)
├── research/
│   └── trials.ipynb        # Jupyter notebook for experimentation
├── requirement.txt         # Python dependencies
├── setup.py                # Package setup
├── template.sh             # Project scaffolding script
├── .env.example            # Environment variable template
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask 3.x |
| **LLM** | Google Gemini 2.5 Flash via `langchain-google-genai` |
| **Vector DB** | Pinecone (Serverless, AWS us-east-1, cosine, dim=384) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **RAG Framework** | LangChain (`create_retrieval_chain` + `create_stuff_documents_chain`) |
| **PDF Parsing** | LangChain `PyPDFLoader` + `DirectoryLoader` |

---

## ⚙️ Configuration

| Parameter | Value | Notes |
|---|---|---|
| Chunk size | 500 tokens | Balances context vs. retrieval precision |
| Chunk overlap | 20 tokens | Prevents information loss at boundaries |
| Retrieval k | 3 | Top-3 most similar chunks per query |
| LLM temperature | 0.3 | Low temperature for factual, deterministic answers |
| Embedding dim | 384 | Matches `all-MiniLM-L6-v2` output size |

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first for major changes.

---

## 👤 Author

**Ashwin Kotha**
- GitHub: [@Ashwin14101](https://github.com/Ashwin14101)
- Project: [Medical-Chatbot-With-LLMs-LangChain-Pinecone-Flask-AWS](https://github.com/Ashwin14101/Medical-Chatbot-With-LLMs-LangChain-Pinecone-Flask-AWS)

---

## 📄 License

MIT © [Ashwin14101](https://github.com/Ashwin14101)