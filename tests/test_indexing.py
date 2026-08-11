import pytest
import numpy as np
import unittest
from unittest.mock import MagicMock
from rag.indexing import Indexer, IndexData
from langchain_core.documents import Document

def test_parent_child_mapping():
    indexer = Indexer()
    parents = [
        Document(
            page_content="This is the first parent document." * 30,
            metadata = {"page_label": "1"}
            ),
        Document(
            page_content="This is the second parent document." * 30,
            metadata = {"page_label": "2"}
            ),
    ]
    child_doc, parent_info = indexer.parent_child_mapping(parents)
    assert len(parent_info) == 2
    assert parent_info["parent_0"]["page_content"] == parents[0].page_content
    assert parent_info["parent_1"]["page_content"] == parents[1].page_content
    assert parent_info["parent_0"]["page_label"] == "1"
    assert parent_info["parent_1"]["page_label"] == "2"
    assert len(child_doc) > 0
    for child in child_doc:
        assert "text" in child
        assert "metadata" in child
        assert "parent_id" in child["metadata"]
        assert "child_id" in child["metadata"]

class MockIndexer:
    def encode(self, text, normalize_embeddings=True):
        return np.random.rand(len(text), 384).astype('float32')
def test_create_vector_index():
    indexer = Indexer()
    indexer.emb_model = MockIndexer()
    texts = [
        "This is the first document.",
        "This is the second document.",
        "This is the third document."
    ]
    index = indexer.create_vector_index(texts)
    assert index.ntotal == len(texts)

def test_build_index():
    fake_emb_model = MagicMock()
    indexer = Indexer(emb_model = fake_emb_model)

    page1 = MagicMock()
    page1.page_content = "This is page 1"
    page1.metadata = {"page_label": "1"}
    page2 = MagicMock()
    page2.page_content = "This is page 2"
    page2.metadata = {"page_label": "2"}

    indexer.preprocessing.process_pdf = MagicMock(
        return_value = [page1, page2]
        )
    indexer.preprocessing.clean_text = MagicMock(
        return_value = [
        "This is cleaned page 1",
        "This is cleaned page 2"
    ])
    indexer.preprocessing.detect_repeated_lines = MagicMock(return_value = {"header"})
    indexer.preprocessing.remove_repeated_lines = MagicMock(
        return_value = [
        "Final page 1",
        "Final page 2"
    ])

    parent1 = MagicMock()
    parent2 = MagicMock()
    parents = [parent1, parent2]

    child1 = MagicMock()
    child1.page_content = "child text 1"
    child2 = MagicMock()
    child2.page_content = "child text 2"

    child_docs = [
    {
        "text": child1,
        "metadata": {
            "parent_id": "parent_0"
        }
    },
    {
        "text": child2,
        "metadata": {
            "parent_id": "parent_1"
        }
    }
    ]
    parent_info = {
    "parent_0": {
        "page_content": "Parent text 1",
        "page_label": "1"
    },
    "parent_1": {
        "page_content": "Parent text 2",
        "page_label": "2"
    }
    }

    indexer.parent_splitter.split_documents = MagicMock(return_value = parents)
    indexer.parent_child_mapping = MagicMock(return_value = (child_docs, parent_info))
    fake_index = MagicMock()
    indexer.create_vector_index = MagicMock(return_value = fake_index)

    result = indexer.build_index("test.pdf")

    indexer.preprocessing.process_pdf.assert_called_once_with("test.pdf")
    indexer.preprocessing.clean_text.assert_called_once_with([
        "This is page 1",
        "This is page 2"
    ])
    indexer.preprocessing.detect_repeated_lines.assert_called_once_with([
        "This is cleaned page 1",
        "This is cleaned page 2"
    ])
    indexer.preprocessing.remove_repeated_lines.assert_called_once_with([
        "This is cleaned page 1",
        "This is cleaned page 2"], {"header"}
    )

    assert page1.page_content == "Final page 1"
    assert page2.page_content == "Final page 2"
    
    indexer.parent_splitter.split_documents.assert_called_once_with([page1, page2])
    indexer.parent_child_mapping.assert_called_once_with(parents)
    indexer.create_vector_index.assert_called_once_with([
        "child text 1",
        "child text 2"
    ])

    assert result.index is fake_index
    assert result.parents == parents
    assert result.child_doc == child_docs
    assert result.parent_info == parent_info
