import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Advanced RAG Engine with Hybrid Search capabilities.
    """
    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        
        # Initialize Vector Store
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    async def add_documents(self, documents: List[Document]):
        """Adds documents to the vector store with metadata."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to {self.collection_name}")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def get_hybrid_retriever(self, k: int = 5):
        """
        Returns a hybrid retriever that combines vector search with LLM-based compression.
        """
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return compression_retriever

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Executes a RAG query using the hybrid search retriever.
        """
        logger.info(f"Processing query: {question}")
        retriever = self.get_hybrid_retriever()
        docs = retriever.get_relevant_documents(question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use the following context to answer the question.
        Context: {context}
        Question: {question}
        Answer:"""
        
        response = await self.llm.ainvoke(prompt)
        
        return {
            "answer": response.content,
            "source_documents": [doc.metadata for doc in docs]
        }
