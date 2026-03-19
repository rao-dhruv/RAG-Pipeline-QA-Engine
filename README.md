# RAG-Pipeline--QA-Engine 

> (FAISS + Sentence Transformers + Groq)

A modular Retrieval-Augmented Generation (RAG) project built with LangChain components, Sentence Transformers embeddings, FAISS vector search, and Groq for final answer generation.

## Overview
This project does the following:

- Loads documents from a local data folder
- Splits them into chunks
- Generates embeddings with a Sentence Transformer model
- Stores vectors in FAISS
- Retrieves top-k relevant chunks for a query
- Uses a Groq LLM to summarize retrieved context
