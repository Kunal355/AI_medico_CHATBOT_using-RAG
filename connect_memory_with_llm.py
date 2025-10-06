import os
import sys # Import sys for potential exit on error

# Import necessary components
# *** CHANGE 1: Replace HuggingFaceEndpoint with Ollama ***
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (optional for .env)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

# --- Configuration Constants ---
# *** CHANGE 2: Define Ollama model name and remove HF_TOKEN ***
OLLAMA_MODEL_NAME = "mistral" # Use the model you pulled with 'ollama pull mistral'
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Setup LLM (Updated for Ollama)
def load_llm():
    """Initializes and returns the Ollama LLM instance."""
    try:
        # Initialize Ollama LLM, assuming Ollama server is running on localhost:11434
        llm = Ollama(
            model=OLLAMA_MODEL_NAME,
            # Add other Ollama parameters if needed, like temperature=0.0
        )
        print(f"Ollama LLM initialized with model: {OLLAMA_MODEL_NAME}")
        return llm
    except Exception as e:
        print(f"ERROR: Could not initialize Ollama LLM. Is the Ollama server running?")
        print(f"Details: {e}")
        sys.exit(1)


# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    """Creates the PromptTemplate object."""
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print(f"FAISS Vector Store loaded successfully from: {DB_FAISS_PATH}")
except Exception as e:
    print(f"ERROR: Could not load FAISS database from {DB_FAISS_PATH}. Ensure the creation script ran.")
    sys.exit(1)


# Create QA chain
try:
    # *** CHANGE 3: Call the updated load_llm function ***
    qa_chain=RetrievalQA.from_chain_type(
        llm=load_llm(), 
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    # Now invoke with a single query
    print("\n" + "="*50)
    user_query=input("Write Query Here: ")
    print("Searching and generating response using Ollama Mistral, please wait...")

    response=qa_chain.invoke({'query': user_query})
    
    print("\n--- RESPONSE ---")
    print("RESULT: ", response["result"])
    print("\nSOURCE DOCUMENTS: ")
    for doc in response["source_documents"]:
        print(f"- Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Snippet: {doc.page_content[:150]}...")
    print("\n" + "="*50)

except Exception as e:
    print(f"\nFATAL ERROR during execution: {e}").py
    