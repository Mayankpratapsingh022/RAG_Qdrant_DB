# Basic RAG with LangChain and Qdrant

![ragg](https://github.com/user-attachments/assets/46f6a510-6ec4-4c4c-a232-c7d35dd6b49a)

This repository contains a minimal and clear implementation of **Retrieval-Augmented Generation (RAG)** using [LangChain](https://www.langchain.com/), [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings), and [Qdrant](https://qdrant.tech/).

**Purpose**  
This code was developed as part of a blog series to demonstrate the core concepts behind RAG. It is designed for learners and developers who are exploring semantic search, vector databases, and LLM-based retrieval systems.

---

## How It Works

1. **Load PDF Documents**  
   The code uses `PyPDFLoader` to load research papers in PDF format.

2. **Split Text into Chunks**  
   Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter` (chunk size: 1000 characters, overlap: 200 characters). This allows for better contextual embedding and retrieval.

3. **Embed Chunks**  
   Each chunk is converted into a vector embedding using OpenAI’s `text-embedding-3-large` model.

4. **Store in Vector Database**  
   Embeddings are either:
   - Stored in Qdrant via `from_documents`, or
   - Retrieved from an existing collection using `from_existing_collection`.

5. **Perform Semantic Search**  
   A query is embedded and matched against the stored chunks using cosine similarity, returning the most relevant pieces of text.

---

## Setup Instructions

Install the required packages:

```bash
pip install langchain langchain-community langchain-openai langchain-qdrant qdrant-client
