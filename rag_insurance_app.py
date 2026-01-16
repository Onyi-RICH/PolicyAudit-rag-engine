# Install required libraries before running this script
# pip install langchain-google-genai langchain chromadb python-dotenv

import os
import shutil
from dotenv import load_dotenv

# Import PDF loader and text splitter from LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Import Gemini chat and embedding classes
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_classic.chains import RetrievalQA

# Load API keys secretly from .env file (never hard-code API keys!)
load_dotenv()

MODEL_NAME = "models/gemini-flash-lite-latest"          # Chosen Gemini LLM model
EMBEDDING_MODEL = "models/text-embedding-004"           # Chosen embedding model

# System prompt: tells the AI to only answer from the document
SYSTEM_PROMPT = (
    "You are an insurance policy assistant. "
    "Answer strictly based on the provided policy text. "
    "If the answer is not in the document, say so."
)

DB_PATH = "./insurance_db"  # Where the local vector DB is saved

def setup_vector_db(pdf_path="policy.pdf"):
    """Load PDF, split text, create embeddings, and store in vector DB."""

    # Step 1: Load PDF document- data ingestion
    print("--- Step 1: Loading Insurance Policy Document ---")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2:  TRANSFORMATION: Breaking/ Split text into small chunks so the AI can read it (Standard Data Engineering chunking)
    print("--- Step 2: Chunking Text ---")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Max size of each chunk
        chunk_overlap=100,  # Overlap for better context
    )
    chunks = splitter.split_documents(documents)

    # Step 3: EMBEDDINGS: Convert each chunk into mathematical vectors/coordinates with Google's model
    print("--- Step 3: Creating Google Embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Step 4:VECTOR STORAGE- Store all vectors in a local vector database (ChromaDB)
    print("--- Step 4: Building Vector Database ---")
    if os.path.exists(DB_PATH):      # Remove old DB if it exists
        shutil.rmtree(DB_PATH)

    vector_db = Chroma.from_documents(
        documents=chunks,                        # The chunked text
        embedding=embeddings,                    # Embedding model
        persist_directory=DB_PATH,               # Save DB locally
    )

    return vector_db

def query_policy(question, vector_db):
    """Ask a question using Gemini LLM, return answer and source text."""
    #  RETRIEVAL & AI: Connecting the Database to the Gemini 'Brain'
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,                             # Predictable answers
        system_prompt=SYSTEM_PROMPT                # Restricts to doc only
    )

    # Link the AI brain with the vector DB so it searches your document
    # This chain handles: 1. User Question -> 2. Search DB -> 3. Send to AI -> 4. Answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        return_source_documents=True,              # Also return source text
    )

    # Ask the question and get an answer and supporting text
    response = qa_chain.invoke({"query": question})   # The 'invoke' command runs the full RAG process

    answer = response["result"]
    snippets = [
        {"page": doc.metadata.get("page"), "text": doc.page_content}
        for doc in response["source_documents"]
    ]

    return answer, snippets

if __name__ == "__main__":
    db = setup_vector_db()  # 1st: Build the vector DB from PDF

    # Example question
    question = "What does the policy say about 'unexpected serious illness'?"
    answer, snippets = query_policy(question, db)

    print("\nAI Answer based on Insurance Document:")
    print(answer)

    print("\nSource Text Snippets:")
    for s in snippets:
        print(f"\n--- Page {s['page']} ---\n{s['text']}")