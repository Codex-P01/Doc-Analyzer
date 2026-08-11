import numpy as np
import pytest
from unittest.mock import MagicMock
from rag.retrieval import Retriever
from langchain_core.documents import Document

class Fake_enc_model:
    def predict(self, pair):
        return np.array([0.2, 0.9, 0.5])

class Fake_emb_model:
    def encode(self, queries, normalize_embeddings = True):
        return np.random.uniform(0, 1, (len(queries), 384))

class FakeIndexData:
    index = None
    parent_info = None
    child_doc = None
    parents = None
    
def test_reRank():
    retriever = Retriever(FakeIndexData, Fake_emb_model(), Fake_enc_model())
    query = "Is this test correct"
    children = [
        Document(
            page_content = "This is 1st child",
            metadata = {"page_label": 1}
        ),
        Document(
                    page_content = "This is 2nd child",
                    metadata = {"page_label": 2}
        ),
        Document(
                    page_content = "This is 3rd child",
                    metadata = {"page_label": 3}
        )
    ]

    child_doc = []
    for child_idx, child in enumerate(children):
        child_doc.append({
            "text": child,
            "metadata": {
                    "parent_id": f"parent_{child_idx}",
                    "child_id": child_idx,
                    "page_label": child_idx
                }
        })
    results = retriever.reRanker(query, child_doc)

    assert len(results) == 3
    assert [result["score"] for result in results] == [0.9, 0.5, 0.2]
    assert [result["child"]["text"].page_content for result in results] == [
        "This is 2nd child",
        "This is 3rd child",
        "This is 1st child"
    ]

def test_emb_search():
    retrieval = Retriever(FakeIndexData, Fake_emb_model(), Fake_enc_model())

    query = "Is this test correct"
    children = [
        Document(
            page_content = "This is 1st child",
            metadata = {"page_label": 1}
        ),
        Document(
                    page_content = "This is 2nd child",
                    metadata = {"page_label": 2}
        ),
        Document(
                    page_content = "This is 3rd child",
                    metadata = {"page_label": 3}
        )
    ]

    child_doc = []
    for child_idx, child in enumerate(children):
        child_doc.append({
            "text": child,
            "metadata": {
                    "parent_id": f"parent_{child_idx}",
                    "child_id": child_idx,
                    "page_label": child_idx
                }
        })

    retrieval.child_doc = child_doc
    retrieval.index = MagicMock()
    retrieval.index.search.return_value = (
        np.array([[0.4, 0.6, 0.8]]),
        np.array([[2, 1, 0]])
        )

    children_scores = [
        {
            "child": child_doc[1],
            "score": 0.9
        },
        {
            "child": child_doc[2],
            "score": 0.5
        },
        {
            "child": child_doc[0],
            "score": 0.2
        }
    ]
    retrieval.reRanker = MagicMock(
        return_value = children_scores
    )

    retrieval.parent_info = {
        "parent_0": {
            "page_content": "Parent 0",
            "page_label": 1
        },
        "parent_1": {
            "page_content": "Parent 1",
            "page_label": 2
        },
        "parent_2": {
            "page_content": "Parent 2",
            "page_label": 3
        }
    }

    results = retrieval.emb_search(query, k=3)

    assert len(results) == 3
    assert [result["score"] for result in results] == [
        0.9,
        0.5,
        0.2
    ]
    assert [result["page_label"] for result in results] == [
        2,
        3,
        1
    ]

def test_extract_comparison_terms():
    retriever = Retriever(FakeIndexData, Fake_emb_model(), Fake_enc_model())
    query1 = "Difference between A and B"
    query2 = "Compare A and B"
    query3 = "Comparison between A and B"
    query4 = "Distinguish between A and B"

    result1 = retriever.extract_comparison_terms(query1)
    result2 = retriever.extract_comparison_terms(query2)
    result3 = retriever.extract_comparison_terms(query3)
    result4 = retriever.extract_comparison_terms(query4)

    assert result1 == ['A', 'B']
    assert result2 == ['A', 'B']
    assert result3 == ['A', 'B']
    assert result4 == ['A', 'B']

def test_expand_query():
    retriever = Retriever(FakeIndexData, Fake_emb_model(), Fake_enc_model())
    query = "Difference between A and B"
    terms = ['A', 'B']
    retriever.extract_comparison_terms = MagicMock()
    retriever.extract_comparison_terms.return_value = terms
    expanded_terms = [
        "Difference between A and B",
        'A',
        'A definition',
        'A characteristics',
        'A properties',
        'A examples',
        'B',
        'B definition',
        'B characteristics',
        'B properties',
        'B examples'
    ]
    result = retriever.expand_query(query)

    assert result == expanded_terms

def test_retrieve_multi_query():
    retriever = Retriever(FakeIndexData, Fake_emb_model(), Fake_enc_model())
    query = "Difference between A and B"
    expanded_terms = [
        "Difference between A and B",
        'A',
        'A definition',
        'A characteristics',
        'A properties',
        'A examples',
        'B',
        'B definition',
        'B characteristics',
        'B properties',
        'B examples'
    ]
    retriever.expand_query = MagicMock(
        return_value = expanded_terms
    )

    parent_info = {
        "parent_0": {
            "page_content": "Parent 0",
            "page_label": "1"
        },
        "parent_1": {
            "page_content": "Parent 1",
            "page_label": "2"
        },
        "parent_2": {
            "page_content": "Parent 2",
            "page_label": "3"
        }
    }

    emb_result = [
        {
            "doc": parent_info["parent_1"],
            "score": 0.9,
            "page_label": parent_info["parent_1"]["page_label"]
        },
        {
            "doc": parent_info["parent_2"],
            "score": 0.5,
            "page_label": parent_info["parent_2"]["page_label"]
        },
        {
            "doc": parent_info["parent_0"],
            "score": 0.2,
            "page_label": parent_info["parent_0"]["page_label"]
        },
    ]
    retriever.emb_search = MagicMock(
        return_value = emb_result
    )
    results = retriever.retrieve_multi_query(query, 3, 3)

    assert [result["doc"]["page_content"] for result in results] == [
        "Parent 1",
        "Parent 2",
        "Parent 0"
    ]
