import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_loader import pdf_loader, create_chunks
from embeddings import EmbeddingsManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever


def run_pipeline(pdf_directory: str, query: str):
    # Step 1: Load PDFs
    documents = pdf_loader(pdf_directory)

    # Step 2: Create chunks
    chunks = create_chunks(documents)

    # Step 3: Generate embeddings
    embedding_manager = EmbeddingsManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Step 4: Store in vector store
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)

    # Step 5: Retrieve relevant documents for query
    retriever = RAGRetriever(vectorstore, embedding_manager)
    results = retriever.retrieve(query)
    print(f"\nQuery: {query}")
    for doc in results:
        print(f"\nRank {doc['rank']} | Score: {doc['similarity_score']:.4f}")
        print(f"Content: {doc['content'][:200]}...")


if __name__ == "__main__":
    run_pipeline("data/pdf/", query="What is this document about?")
