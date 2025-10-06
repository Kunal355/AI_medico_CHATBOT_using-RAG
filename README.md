# MediBot - AI Medical Chatbot using RAG

## Overview
MediBot is an AI-powered medical chatbot that utilizes Retrieval Augmented Generation (RAG) to provide intelligent responses based on medical documents. The system processes PDF documents, creates vector embeddings, and leverages the Mistral LLM to deliver accurate and context-aware medical information.

## Project Structure
The project is organized into three main phases:

### Phase 1: Setup Memory for LLM (Vector Database)
- Load raw PDF documents
- Create text chunks for processing
- Generate vector embeddings
- Store embeddings in FAISS vector database

### Phase 2: Connect Memory with LLM
- Setup Mistral LLM using HuggingFace
- Integrate LLM with FAISS vector store
- Create processing chain for seamless operation

### Phase 3: Setup UI for the Chatbot
- Develop chatbot interface using Streamlit
- Load vector store (FAISS) in cache
- Implement Retrieval Augmented Generation (RAG) pipeline

## Tools & Technologies
- **Langchain** - AI Framework for LLM applications
- **HuggingFace** - ML/AI Hub for model access
- **Mistral** - Large Language Model
- **FAISS** - Vector Database for efficient similarity search
- **Streamlit** - Framework for chatbot UI
- **Python** - Primary programming language
- **VS Code** - Integrated Development Environment

## Future Improvements
- Add authentication system in the UI
- Implement self-upload document functionality
- Support for multiple documents and batch embedding
- Add comprehensive unit testing for RAG applications

## Summary
MediBot represents a modern AI chatbot solution for medical document processing with:
- Modular 3-phase implementation
- Integration of Streamlit, Langchain, and HuggingFace
- RAG methodology with vector embeddings
- End-to-end RAG pipeline implementation

All code and dependencies will be available in the repository. Feedback and contributions are welcome!

