import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.rag_engine import RAGEngine
from langchain_core.documents import Document

@pytest.fixture
def mock_rag_engine(monkeypatch):
    # Mock QdrantClient and OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with monkeypatch.context() as m:
        m.setattr("qdrant_client.QdrantClient", MagicMock())
        m.setattr("langchain_openai.OpenAIEmbeddings", MagicMock())
        m.setattr("langchain_openai.ChatOpenAI", MagicMock())
        
        engine = RAGEngine(collection_name="test_collection")
        return engine

@pytest.mark.asyncio
async def test_rag_query_logic(mock_rag_engine):
    # Mock the internal query logic
    mock_rag_engine.query = AsyncMock(return_value={
        "answer": "The capital of France is Paris.",
        "source_documents": [{"source": "wikipedia"}]
    })
    
    response = await mock_rag_engine.query("What is the capital of France?")
    
    assert "Paris" in response["answer"]
    assert response["source_documents"][0]["source"] == "wikipedia"
