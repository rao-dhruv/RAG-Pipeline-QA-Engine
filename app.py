from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Example

if __name__ == "__main__":
    # Activate virtual envirnment if necessary
    # documents = load_all_documents("data")
    # print(f"{len(documents)} documents found") # for testing data_loader.py
    
    # documents = load_all_documents("data")
    # chunks = EmbeddingPipeline().chunk_documents(documents)
    # chunk_vectors = EmbeddingPipeline().embed_chunks(chunks)
    # print(chunk_vectors) 
    # print(f"{len(chunks)} chunks created")
    # print(f"Chunk vectors shape: {chunk_vectors.shape}") # for testing embedding.py if chunking is happening correctly and vectors are being generated

    # documents = load_all_documents("data")
    # store =  FaissVectorStore("faiss_store")
    # store.build_from_documents(documents) # testing vectorestore.py if the index is being built and saved correctly, this create the faiss_store directory

    # store =  FaissVectorStore("faiss_store")
    # store.load() # If already built and saved as faiss_store is already created, load the index and metadata
    # print(store.query("What is RAG?", top_k=3))
    
    
    store =  FaissVectorStore("faiss_store")
    store.load()
    rag_search = RAGSearch()
    query = "What is RAG?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)