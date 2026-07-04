# 📄 Document Analysis RAG System

A Retrieval-Augmented Generation (RAG) system for querying and summarizing PDF documents using Large Language Models (LLMs). The system extracts text from digital and scanned PDFs, builds a semantic search index, retrieves the most relevant document sections, and generates grounded answers based only on the provided document.

---

## ✨ Features

- 📄 PDF text extraction using PyMuPDF
- 🔍 OCR fallback for scanned PDFs using PaddleOCR
- 🧹 Automatic document cleaning and normalization
- 🗑️ Header and footer removal
- 📚 Parent–Child chunking strategy
- 🔢 Dense vector embeddings using BAAI BGE
- ⚡ FAISS vector database for similarity search
- 🔍 Multi-query retrieval for comparison-style questions
- 🤖 Answer generation using Qwen 3 4B Instruct
- 📝 Hierarchical document summarization
- 🏗️ Modular architecture for easy extension

---

# Architecture

```
                    PDF
                     │
                     ▼
           Text Extraction (PyMuPDF)
                     │
      ┌──────────────┴──────────────┐
      │                             │
Digital PDF                  Scanned PDF
      │                             │
      ▼                             ▼
 Direct Text                  PaddleOCR
      │                             │
      └──────────────┬──────────────┘
                     ▼
             Document Cleaning
                     ▼
        Remove Headers / Footers
                     ▼
         Parent–Child Chunking
                     ▼
        BGE Embedding Generation
                     ▼
             FAISS Vector Index
                     │
────────────────────────────────────────
                     │
              User Question
                     ▼
           Multi-Query Expansion
                     ▼
             Semantic Retrieval
                     ▼
          Parent Context Assembly
                     ▼
             Prompt Construction
                     ▼
          Qwen 3 4B Instruct LLM
                     ▼
            Grounded Answer
```

---

# Project Structure

```
Document-Analysis-RAG/
│
├── main.py                  # FastAPI application
├── requirements.txt
├── README.md
│
├── rag/
│   ├── config.py            # Configuration and constants
│   ├── pipeline.py          # Main RAG pipeline
│   ├── preprocessing.py     # PDF processing, OCR, text cleaning
│   ├── indexing.py          # Chunking, embeddings, FAISS indexing
│   ├── retrieval.py         # Query expansion and semantic retrieval
│   └── generation.py        # Prompt construction, LLM inference, summarization
│
└── uploads/                 # Uploaded PDF documents
```

---

# Retrieval Pipeline

1. Load PDF document
2. Extract text (OCR when necessary)
3. Clean and normalize document text
4. Remove repeated headers and footers
5. Create parent chunks
6. Create child chunks
7. Generate dense embeddings
8. Build FAISS index
9. Expand user query (for comparison questions)
10. Retrieve relevant parent contexts
11. Construct prompt
12. Generate answer using the LLM

---

# Technologies Used

| Component | Technology |
|----------|------------|
| LLM | Qwen3-4B-Instruct |
| Framework | Unsloth |
| OCR | PaddleOCR |
| PDF Processing | PyMuPDF |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | FAISS |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Backend | FastAPI *(in progress)* |

---

# Installation

Clone the repository

```bash
git clone https://github.com/Codex-P01/Doc-Analyzer.git

cd Doc-Analyzer
```

# Prerequisites

Before installing the project, ensure your environment meets the following requirements:

- **Python 3.11 or below**
  - Python 3.12+ is **not currently supported** due to compatibility issues with some dependencies.

- **NVIDIA GPU (recommended)**
  - CUDA-compatible GPU is recommended for inference.

- **PyTorch**
  - After installing the project requirements, install the appropriate **PyTorch** version that matches your CUDA version.
  - Refer to the official PyTorch installation guide:
    https://pytorch.org/get-started/locally/

---

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

Build the document index

```python
rag.index_pdf("document.pdf")
```

Ask questions

```python
response = rag.query(
    "Explain the Mamdani Fuzzy Inference System."
)

print(response["answer"])
```

Generate a document summary

```python
summary = rag.summarize()

print(summary)
```

---

# Current Features

- ✅ Digital PDF support
- ✅ Scanned PDF support
- ✅ OCR fallback
- ✅ Parent–Child Retrieval
- ✅ Multi-query Retrieval
- ✅ Hierarchical Summarization
- ✅ Modular RAG Pipeline

---

# Planned Improvements

- Hybrid Retrieval (BM25 + Dense Retrieval)
- Cross-Encoder Re-ranking
- Maximal Marginal Relevance (MMR)
- Metadata Filtering
- Persistent FAISS Index
- Streaming Responses
- Conversation Memory
- Source Citation in Responses
- Docker Support

---

# Example Query

**Question**

```
What is the difference between Competitive Learning and Hebbian Learning?
```

**Pipeline**

```
Question
      │
      ▼
Query Expansion
      ▼
Semantic Retrieval
      ▼
Relevant Parent Chunks
      ▼
Prompt Construction
      ▼
LLM
      ▼
Answer
```

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

- Unsloth
- Qwen
- PaddleOCR
- PyMuPDF
- Sentence Transformers
- FAISS
- LangChain