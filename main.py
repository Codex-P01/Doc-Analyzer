from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag.pipeline import RAGPipeline
import os
import shutil

app = FastAPI(title = "RAG API")
rag = RAGPipeline()

class Query(BaseModel):
    query: str

class Answer(BaseModel):
    answer: str

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to the RAG API. Use /upload to upload a PDF and /query to ask questions."}  

@app.get("/status")
def status():
    return {"Status": "Running Ok."}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag.index_pdf(path)
    return {"message": "PDF indexed successfully."}

@app.post("/query")
async def query(request: Query):
    answer = rag.query(request.query)
    return {"answer": answer}

@app.get("/summary")
async def summary():
    return {
        "summary": rag.summarize()
    }
