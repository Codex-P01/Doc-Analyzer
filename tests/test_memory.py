import pytest
from unittest.mock import MagicMock
from rag.memory import ChatMemory

def fake_chatMemory(maxHistory = 5, maxHistoryTokens = 300):
    tokenizer = MagicMock()
    chatMemory = ChatMemory(tokenizer, maxHistory, maxHistoryTokens)
    return chatMemory

def test_count_tokens():
    chatMemory = fake_chatMemory()
    chatMemory.tokenizer.return_value = {
        "input_ids": [10, 20, 30, 40]
    }
    result = chatMemory.count_tokens("This is a test")
    assert result == 4

def test_add_within_limits():
    chatMemory = fake_chatMemory()
    query = "What is RAG"
    ans = "This is RAG"
    chatMemory.add(query, ans)
    assert chatMemory.history[0]["User"] == query
    assert chatMemory.history[0]["Assistant"] == ans

def test_add():
    chatMemory = fake_chatMemory(maxHistory = 2)
    queries = ["What is RAG", "What is LLM", "What is it"]
    answers = ["This is RAG", "This is LLM", "This is it"]
    for query, ans in zip(queries, answers):
        chatMemory.add(query, ans)
    assert chatMemory.history[1]["User"] == queries[2]
    assert chatMemory.history[1]["Assistant"] == answers[2]

def test_get_history_without_history():
    chatMemory = fake_chatMemory()
    result = chatMemory.getHistory()
    assert result == "No previous conversation."

def test_get_history_with_history():
    chatMemory = fake_chatMemory(maxHistory = 2, maxHistoryTokens = 10)
    chatMemory.tokenizer.return_value = {
        "input_ids": [1, 2, 3, 4, 5, 6]
    }
    chatMemory.history = [
        {
            "User": "What is rag",
            "Assistant": "This is rag"
        },
        {
            "User": "What is llm",
            "Assistant": "This is llm"
        },
        {
            "User": "What is it",
            "Assistant": "This is it"
        },
    ]
    result = chatMemory.getHistory()
    assert result == (
        "User: What is it\n"
        "Assistant: This is it\n"
    )