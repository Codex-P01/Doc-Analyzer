# 📄 Document Analysis RAG System

A modular **Retrieval-Augmented Generation (RAG)** system for document question answering and summarization. The system extracts text from digital and scanned PDFs, builds a semantic search index, retrieves the most relevant document sections, and generates grounded answers using a Large Language Model (LLM).

---

## ✨ Features

* 📄 PDF text extraction using **PyMuPDF**
* 🔍 OCR fallback for scanned PDFs using **PaddleOCR**
* 🧹 Document cleaning and normalization
* 🗑️ Automatic header and footer removal
* 📚 Parent–Child chunking strategy
* 🔢 Dense embeddings using **BAAI/bge-small-en-v1.5**
* ⚡ FAISS vector database for semantic search
* 🔍 Multi-query retrieval for comparison-based questions
* 🎯 Cross-Encoder re-ranking for improved retrieval accuracy
* 💬 Conversation memory for follow-up questions
* 🔄 Query rewriting for conversational retrieval
* 🤖 Answer generation using **Qwen3-4B-Instruct**
* 📝 Hierarchical document summarization
* 🏗️ Modular and extensible architecture

---

# Architecture

```text
                     PDF
                      │
                      ▼
         Text Extraction (PyMuPDF)
                      │
        ┌─────────────┴─────────────┐
        │                           │
   Digital PDF                Scanned PDF
        │                           │
        ▼                           ▼
  Direct Text                 PaddleOCR
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Text Cleaning
                      ▼
      Header / Footer Removal
                      ▼
          Parent-Child Chunking
                      ▼
          Embedding Generation
                      ▼
               FAISS Index
──────────────────────────────────────────────

                User Question
                      │
                      ▼
          Conversation Memory
                      ▼
             Query Rewriting
                      ▼
          Multi-Query Expansion
                      ▼
        Dense Vector Retrieval
                      ▼
        Cross-Encoder Re-ranking
                      ▼
       Parent Context Retrieval
                      ▼
          Prompt Construction
                      ▼
         Qwen3-4B-Instruct LLM
                      ▼
          Grounded Response
```

---

# Project Structure

```text
Document-Analysis-RAG/
│
├── main.py
├── requirements.txt
├── README.md
│
├── rag/
│   ├── config.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── generation.py
│   └── memory.py
│
└── uploads/
```

---

# Retrieval Pipeline

1. Load PDF document
2. Extract text using PyMuPDF
3. Apply OCR for scanned pages when required
4. Clean and normalize extracted text
5. Remove repeated headers and footers
6. Create parent and child chunks
7. Generate dense embeddings
8. Build the FAISS vector index
9. Rewrite conversational queries using chat history
10. Expand comparison-based queries
11. Retrieve relevant child chunks
12. Re-rank retrieved chunks using a Cross-Encoder
13. Aggregate relevant parent chunks
14. Construct the final prompt
15. Generate the answer using Qwen3-4B-Instruct

---

# Technologies Used

| Component       | Technology                               |
| --------------- | ---------------------------------------- |
| LLM             | Qwen3-4B-Instruct                        |
| Framework       | Unsloth                                  |
| OCR             | PaddleOCR                                |
| PDF Processing  | PyMuPDF                                  |
| Embedding Model | BAAI/bge-small-en-v1.5                   |
| Re-ranking      | CrossEncoder                             |
| Vector Database | FAISS                                    |
| Chunking        | LangChain RecursiveCharacterTextSplitter |
| Backend         | FastAPI                                  |

---

# Prerequisites

Before installing the project, ensure your environment satisfies the following requirements.

## Python

* **Python 3.11 or below**
* Python 3.12+ is currently **not supported** due to dependency compatibility.

## GPU (Recommended)

* NVIDIA GPU with CUDA support is recommended for faster inference.

## PyTorch

After installing the project requirements, install the correct **PyTorch** version for your CUDA installation.

For example, for **CUDA 12.1**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, refer to the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository>.git

cd <repository>
```

Install the project dependencies

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch version for your CUDA environment.

---

# Running the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

If the server starts successfully, you should see output similar to:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## API Documentation

FastAPI automatically generates interactive API documentation.

* **Swagger UI**

```
http://127.0.0.1:8000/docs
```

* **ReDoc**

```
http://127.0.0.1:8000/redoc
```

---

## Typical Workflow

1. Start the FastAPI server.
2. Open the Swagger UI (`/docs`).
3. Upload a PDF document using the upload endpoint.
4. Ask questions about the uploaded document using the query endpoint.
5. Generate a document summary using the summary endpoint.

---

## Example

Run the server:

```bash
uvicorn main:app --reload
```

Open your browser and navigate to:

```
http://127.0.0.1:8000/docs
```

From there, you can test all available API endpoints directly from the interactive interface.

---

# Current Features

* ✅ Digital PDF support
* ✅ Scanned PDF support
* ✅ OCR fallback
* ✅ Parent–Child Retrieval
* ✅ Multi-query Retrieval
* ✅ Cross-Encoder Re-ranking
* ✅ Conversational Memory
* ✅ Query Rewriting
* ✅ Hierarchical Summarization
* ✅ Modular RAG Pipeline

---

# Example Query

**Question**

```text
What is the difference between Competitive Learning and Hebbian Learning?
```

**Pipeline**

```text
Question
     │
     ▼
Conversation Memory
     ▼
Query Rewriting
     ▼
Multi-Query Expansion
     ▼
Dense Retrieval
     ▼
Cross-Encoder Re-ranking
     ▼
Parent Context
     ▼
Prompt Construction
     ▼
LLM
     ▼
Answer
```

---

# Compatibility

| Component        | Requirement              |
| ---------------- | ------------------------ |
| Python           | ≤ 3.11                   |
| GPU              | NVIDIA GPU (Recommended) |
| CUDA             | Supported                |
| Operating System | Windows / Linux          |

---

# License

This project is released under the **MIT License**.

---

# Acknowledgements

This project makes use of the following open-source projects:

* Unsloth
* Qwen
* PaddleOCR
* PyMuPDF
* Sentence Transformers
* FAISS
* LangChain
* FastAPI
