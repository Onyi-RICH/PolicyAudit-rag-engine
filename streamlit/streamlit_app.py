# pip install -r ../requirements.txt

import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import RetrievalQA

# Load secrets
load_dotenv()

MODEL_NAME = "models/gemini-flash-lite-latest"
EMBEDDING_MODEL = "models/text-embedding-004"

# -----------------------------
# Streamlit UI
# -----------------------------
# Sidebar: Suggested questions
st.sidebar.header("Suggested Questions")
suggested_questions = [
    "What does the policy say about 'unexpected serious illness'?",
    "How do I file a claim for hospitalization?",
    "Are pre-existing conditions covered?",
    "Are accidents during extreme sports covered?"
]
for q in suggested_questions:
    if st.sidebar.button(q, key=q):
        st.session_state['question'] = q

# Title
st.title("🛡️ PolicyAudit Assistant")
st.write("Automated Legal Document Verification & RAG Pipeline")

# -----------------------------
# PDF upload slot
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document", type="pdf", key="upload_pdf"
)

# User question input
user_question = st.text_input(
    "Ask a question about your insurance/legal policy:",
    value=st.session_state.get('question', ""),
    key="user_question"
)

if uploaded_file and user_question:
    # Save uploaded file locally
    pdf_path = os.path.join("./", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Step 3: Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Step 4: Vector store
    vector_db_path = "./insurance_db"
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vector_db_path,
    )

    # Step 5: Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,   # ensure the AI remains predictable and doesn't get 'creative' or hallucinate
    )
     # Build RetrievalQA + "stuff" chain  - Protects privacy by only showing the LLM the necessary data fragments.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",     # we are not just "sending data to AI," but injecting specific, audited context
        retriever=vector_db.as_retriever(),
        return_source_documents=True,
    )

    st.info("Running query, please wait...")

    # Run query
    response = qa_chain.invoke({"query": user_question})

    # Display AI answer
    st.markdown("## ✅ AI Answer")
    st.write(response["result"])

    # Display exact text snippets
    st.markdown("## 📄 Source Snippets")
    for i, doc in enumerate(response["source_documents"]):
        snippet_text = doc.page_content
        page_num = doc.metadata.get("page", "unknown")
        st.text_area(
            label=f"Snippet {i+1} (Page {page_num})",
            value=snippet_text,
            height=150,
            key=f"snippet_{i}"
        )
