Repository Description
A lightweight Retrieval-Augmented Generation (RAG) project that turns local text corpora into a searchable chat API.

Uses FAISS to index document chunks from docs
Generates query embeddings with OpenAI
Retrieves relevant passages from faiss_index
Answers user questions via a FastAPI chat endpoint
Includes both chat_API.py and chat_CLI.py examples for API and terminal use
Ideal for exploring local knowledge search, building a simple RAG assistant, and experimenting with OpenAI embeddings + vector retrieval.