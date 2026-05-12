import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = ".chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_vectorstore():
    """Load the existing ChromaDB vectorstore"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    return vectorstore

def retrieve(query: str, k: int = 3):
    """
    Search ChromaDB for most relevant chunks
    
    Args:
        query: The user's question
        k: Number of chunks to retrieve (default 3)
    
    Returns:
        List of relevant document chunks
    """
    vectorstore = load_vectorstore()
    
    results = vectorstore.similarity_search(query, k=k)
    
    return results

def main():
    print("=== Testing Retriever ===\n")
    
    # Test query
    query = "What is the memory bandwidth of H100?"
    print(f"Query: {query}\n")
    
    results = retrieve(query)
    
    print(f"Top {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content)
        print(f"Source: Page {doc.metadata.get('page', 'unknown')}")
        print()

if __name__ == "__main__":
    main()