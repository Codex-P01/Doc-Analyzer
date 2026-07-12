import faiss
import numpy as np
from rag.config import EMB_MODEL
from rag.config import (PARENT_CHUNK_SIZE,
                            PARENT_OVERLAP,
                            CHILD_CHUNK_SIZE,
                            CHILD_OVERLAP
                            )
from rag.preprocessing import Preprocessing
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class IndexData:
    def __init__(self):
        self.index = None
        self.parent_info = {}
        self.child_doc = []
        self.parents = []

class Indexer:
    def __init__(self):
        self.preprocessing = Preprocessing()
        self.emb_model = SentenceTransformer(EMB_MODEL)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size = PARENT_CHUNK_SIZE,
            chunk_overlap = PARENT_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHILD_CHUNK_SIZE,
            chunk_overlap = CHILD_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
    def build_index(self, path):
        indexdata = IndexData()
        doc = self.preprocessing.process_pdf(path)
        page_texts = [page.page_content for page in doc]
        page_texts = self.preprocessing.clean_text(page_texts)
        repeated = self.preprocessing.detect_repeated_lines(page_texts)
        page_texts = self.preprocessing.remove_repeated_lines(page_texts, repeated)
        for page, cleaned_text in zip(doc, page_texts):
            page.page_content = cleaned_text
        indexdata.parents = self.parent_splitter.split_documents(doc)
        indexdata.child_doc, indexdata.parent_info = self.parent_child_mapping(indexdata.parents)
        child_text = [doc["text"].page_content for doc in indexdata.child_doc]
        indexdata.index = self.create_vector_index(child_text)
        return indexdata

    def parent_child_mapping(self, parents):
        child_doc = []
        parent_info = {}
        for parent_idx, parent in enumerate(parents):
            parent_id = f"parent_{parent_idx}"
            parent_info[parent_id] = {
                "page_content": parent.page_content,
                "page_label": parent.metadata["page_label"]
            }
            children = self.child_splitter.split_documents([parent])
            for child_idx, child in enumerate(children):
                child_doc.append({
                    "text": child,
                    "metadata": {
                        "parent_id": parent_id,
                        "child_id": child_idx,
                        "page_label": parent.metadata.get("page_label")
                    }
                })
        return (child_doc, parent_info)

    def create_vector_index(self, text):
        doc_embeddings = self.emb_model.encode(
            text,
            normalize_embeddings=True
            )
        doc_embeddings = np.array(doc_embeddings).astype("float32")
        dims = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dims)
        index.add(doc_embeddings)
        return index
