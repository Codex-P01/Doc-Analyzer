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
* Python 3.12+ is **not currently supported** due to dependency compatibility.

## GPU (Recommended)

* NVIDIA GPU with CUDA support is recommended for faster inference.

## PyTorch

If you are running the project locally (without Docker), install the appropriate **PyTorch** version for your CUDA installation after installing the project requirements.

For example, for **CUDA 12.1**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, visit:

https://pytorch.org/get-started/locally/

> **Note:** If you are using Docker, PyTorch is already included in the provided Docker image and no additional installation is required.

---

# Installation (Local)

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository>.git

cd <repository>
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install the correct PyTorch version for your CUDA installation.

---

# Running the API (Local)

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Once the server is running, open:

**Swagger UI**

```
http://127.0.0.1:8080/docs
```

**ReDoc**

```
http://127.0.0.1:8080/redoc
```

---

# Running with Docker

## Build the Docker image

```bash
docker build -t document-analysis-rag .
```

## Run the container

```bash
docker run --gpus all -p 8080:8080 document-analysis-rag
```

If your system does not support GPU acceleration, you can omit the `--gpus all` flag:

```bash
docker run -p 8080:8080 document-analysis-rag
```

Once the container is running, access the API at:

```
http://localhost:8080/docs
```

or

```
http://localhost:8080/redoc
```

---

# Docker Image

The provided Docker image includes:

* Python with PyTorch 2.6.0
* CUDA 12.4 Runtime
* cuDNN 9
* FastAPI
* Uvicorn
* All project dependencies
* OpenCV system libraries
* PaddleOCR dependencies

No additional setup is required other than building and running the container.

---

# Typical Workflow

1. Start the API (locally or using Docker).
2. Open the Swagger UI.
3. Upload a PDF document.
4. Ask questions about the uploaded document.
5. Generate a summary of the document.

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
