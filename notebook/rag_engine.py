import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_loader import pdf_loader, create_chunks
from embeddings import EmbeddingsManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
from llm_pipeline import GroqLLM, RAGPipeline


class RAGEngine:
    def __init__(self):
        self.embedding_manager = EmbeddingsManager()
        self.vector_store = VectorStore()
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
        self.llm = GroqLLM()
        self.pipeline = RAGPipeline(self.retriever, self.llm)
        print("RAGEngine initialized.")

    def ingest(self, pdf_directory: str) -> int:
        documents = pdf_loader(pdf_directory)
        chunks = create_chunks(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vector_store.add_documents(chunks, embeddings)
        return len(chunks)

    def query(self, question: str) -> dict:
        return self.pipeline.run(question)
