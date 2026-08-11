import torch
import pytest
from unittest.mock import MagicMock
from rag.generation import Generator

class FakeDoc:
    def __init__(self, text):
        self.page_content = text

def fake_generator():
    generator = Generator.__new__(Generator)
    generator.tokenizer = MagicMock()
    generator.model = MagicMock()
    generator.device = "cpu"
    return generator

def test_count_tokens():
    generator = fake_generator()
    generator.tokenizer.return_value = {
        "input_ids": ["10", "20", "30", "40"]
        }
    result = generator.count_tokens("This is a test")
    assert result == 4

def test_select_parents():
    generator = fake_generator()
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
    fake_results = [
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
        }
    ]
    generator.count_tokens = MagicMock()
    generator.count_tokens.side_effect = [500, 600, 200]
    results = generator.select_parent(fake_results, max_token_set = 1000)

    assert [result["doc"]["page_content"] for result in results] == [
        "Parent 1",
        "Parent 0"
    ] 

def test_build_prompt():
    generator = fake_generator()
    generator.tokenizer.apply_chat_template.return_value = "FINAL PROMPT"
    result = generator.build_prompt(
        query="What is RAG?",
        context="RAG retrieves relevant documents.",
        history="User: Explain retrieval."
    )
    assert result == "FINAL PROMPT"

def test_build_query_prompt():
    generator = fake_generator()
    generator.tokenizer.apply_chat_template.return_value = "QUERY PROMPT"
    result = generator.build_query_prompt(
        query="What is RAG?",
        history="User: Explain retrieval."
    )
    assert result == "QUERY PROMPT"

def test_generate():
    generator = fake_generator()
    inputs = MagicMock()
    inputs["input_ids"] = torch.tensor([10, 20, 30, 40])
    inputs["attention_mask"] = torch.tensor([1, 1, 1, 1])
    generator.tokenizer.return_value = inputs
    generator.model.generate.return_value = [
        [10, 20, 30, 40, 50, 60, 70, 80]
    ]
    generator.tokenizer.decode.return_value = "Generated Answer"
    result = generator.generate("This is a test")
    assert result == "Generated Answer"

def test_generate_summary_batches_documents():
    generator = fake_generator()

    docs = [
        FakeDoc("Document 1"),
        FakeDoc("Document 2"),
        FakeDoc("Document 3"),
        FakeDoc("Document 4"),
        FakeDoc("Document 5"),
    ]

    generator.tokenizer.apply_chat_template.side_effect = [
        "prompt 1",
        "prompt 2",
        "prompt 3",
        "final prompt",
    ]

    generator.generate = MagicMock()
    generator.generate.side_effect = [
        "summary 1",
        "summary 2",
        "summary 3",
        "final summary",
    ]

    result = generator.generate_summary(docs, batch=2)
    assert result == "final summary"
    assert generator.generate.call_count == 4
    assert generator.tokenizer.apply_chat_template.call_count == 4

def test_rewite_query_without_history():
    generator = fake_generator()
    result = generator.rewrite_query(
        "What is it?",
        history=""
    )
    assert result == "What is it?"

def test_rewite_query_with_history():
    generator = fake_generator()
    generator.build_query_prompt = MagicMock(
        return_value = "Query Prompt"
    )
    generator.generate = MagicMock(
        return_value = "What is RAG?"
    )
    result = generator.rewrite_query(
        "What is it?",
        history="User: Tell me about RAG."
    )
    assert result == "What is RAG?"