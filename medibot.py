from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
import sys
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.documents import Document

# --- Configuration Constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"
OLLAMA_MODEL_NAME = "mistral"  # <-- Use your local model name

# --- Utility Functions ---

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store once and caches it."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load the vector store from {DB_FAISS_PATH}. Ensure the creation script was run.")
        st.error(f"Details: {e}")
        return None

def set_custom_prompt(custom_prompt_template: str) -> PromptTemplate:
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def get_qa_chain(vectorstore):
    """Initializes the LLM and the RetrievalQA chain using Ollama."""
    CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
    qa_chain = RetrievalQA.from_chain_type(
        llm=Ollama(model=OLLAMA_MODEL_NAME),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

# Add this to help debug blank screen issues
def main():
    st.set_page_config(page_title="Medico RAG Chatbot", layout="wide")
    st.title("👨‍⚕️ Medico RAG Chatbot")
    st.markdown("Ask questions about your documents (PDFs) and get grounded answers using your local LLM.")
    st.info("Medibot main function started!")  # Debug line

    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Vector store could not be loaded. Please run the memory creation script and check your data.")
        return

    # ...rest of your Streamlit app code...

if __name__ == "__main__":
    main()