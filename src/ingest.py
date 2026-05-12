import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = "data"
CHROMA_DIR = ".chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_documents(data_dir: str):
    """Load all PDFs from the data directory"""
    documents = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    
    print(f"Total pages loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # 500 characters per chunk
        chunk_overlap=50,      # 50 character overlap between chunks
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def store_in_chromadb(chunks):
    """Convert chunks to vectors and store in ChromaDB"""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    print("Storing vectors in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def main():
    print("=== Starting Document Ingestion ===\n")
    
    # Step 1: Load PDFs
    documents = load_documents(DATA_DIR)
    
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Store in ChromaDB
    vectorstore = store_in_chromadb(chunks)
    
    print("\n=== Ingestion Complete ===")
    print(f"Vector DB saved to: {CHROMA_DIR}")

if __name__ == "__main__":
    main()