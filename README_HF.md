---
title: RAG Project - Learn with Transformers
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# RAG Project - Learn with Transformers

A Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and FastAPI.

## Features
- 🔍 Document retrieval with FAISS vector store
- 🎯 FlashRank reranking for improved relevance
- 🤖 Corrective RAG with LangGraph agent workflow
- 🧠 Powered by Groq LLM
- 📄 Built on "Attention Is All You Need" paper

## Tech Stack
- **LLM**: Groq (openai/gpt-oss-120b)
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
- **Vector Store**: FAISS
- **Reranker**: FlashRank (rank-T5-flan)
- **Framework**: LangChain + LangGraph
- **Web**: FastAPI + Jinja2

## Environment Variables Required
Set these in your Space settings:
- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
- `LANGSMITH_API_KEY` (optional)
- `TAVILY_API_KEY` (optional)
