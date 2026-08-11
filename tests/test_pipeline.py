from unittest.mock import MagicMock
from rag.pipeline import RAGPipeline

def test_index_pdf():
    indexer = MagicMock()
    retriever = MagicMock()
    chatmemory = MagicMock()
    generator = MagicMock()

    pipeline = RAGPipeline(indexer, chatmemory, generator)
    pipeline.retriever = retriever

    pipeline.indexer = MagicMock()

    index_data = MagicMock()
    pipeline.indexer.build_index.return_value = index_data

    pipeline.index_pdf("document.pdf")

    assert pipeline.indexData == index_data
    assert pipeline.retriever is not None

def test_query():
    indexer = MagicMock()
    retriever = MagicMock()
    chatmemory = MagicMock()
    generator = MagicMock()

    pipeline = RAGPipeline(indexer, chatmemory, generator)
    pipeline.retriever = retriever

    pipeline.chatmemory.getHistory.return_value = "previous history"
    pipeline.generator.rewrite_query.return_value = "rewritten question"

    pipeline.retriever.retrieve_multi_query.return_value = [
        {"page_label": "1", "doc": {"page_content": "RAG information"}}
    ]
    pipeline.generator.select_parent.return_value = [
        {"page_label": "1", "doc": {"page_content": "RAG information"}}
    ]

    pipeline.generator.build_prompt.return_value = "final prompt"
    pipeline.generator.generate.return_value = "This is the answer"

    result = pipeline.query("What is RAG?")

    assert result == "This is the answer"

    pipeline.chatmemory.getHistory.assert_called_once()
    pipeline.generator.rewrite_query.assert_called_once_with(
        "What is RAG?",
        "previous history"
    )
    pipeline.retriever.retrieve_multi_query.assert_called_once_with(
        "rewritten question"
    )
    pipeline.generator.generate.assert_called_once_with(
        "final prompt",
        False
    )
    pipeline.chatmemory.add.assert_called_once_with(
        "What is RAG?",
        "This is the answer"
    )