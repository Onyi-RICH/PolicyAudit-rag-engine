# 🛡️ PolicyAudit-RAG-Pipeline  
### Automated Policy Ingestion, Verification & AI-Powered Audit System

![Streamlit UI Screenshot](streamlit/Streamlit-screenshot.png)

---

## 🎯 Project Goal
Design and implement a **production-ready Retrieval-Augmented Generation (RAG) system** that enables accurate, explainable question-answering over insurance policy documents using **Google Gemini, LangChain, and ChromaDB**.

The system ensures **zero hallucination risk** by grounding every AI response in verified source text extracted directly from policy PDFs.

---

## 🚀 Project Overview
Large Language Models often hallucinate when answering questions about legal or medical documents.  

This project solves that problem by:

- Converting insurance policy PDFs into a **vectorized knowledge base**
- Retrieving **only relevant policy clauses**
- Generating answers **strictly grounded in source text**
- Displaying **exact text snippets** used to generate each answer

The result is an **audit-friendly AI assistant** suitable for insurance, legal, and compliance workflows.

---

## 🧠 Key Engineering Highlights
- **End-to-End RAG Pipeline**  
  PDF ingestion → chunking → embeddings → vector search → grounded LLM response
- **Exact Source Attribution**  
  Displays the precise policy text used by the model, not just page numbers
- **Hybrid Embedding Strategy**  
  Supports both Google Generative AI embeddings and local HuggingFace embeddings for cost and quota optimization
- **Robust Dependency Management**  
  Resolved real-world conflicts between Keras 3, TensorFlow, and LangChain Google GenAI
- **Interactive Streamlit UI**  
  Upload documents, ask questions, and view cited policy clauses in real time

---

## 🏗️ Tech Stack
- **LLM:** Google Gemini 1.5 Flash (`langchain-google-genai`)
- **RAG Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** Google Generative AI / HuggingFace (`all-MiniLM-L6-v2`)
- **UI:** Streamlit
- **Language:** Python 3.12
- **Config & Secrets:** python-dotenv

---

## 📂 Project Structure

```text
Insurance-RAG-Engine/
│
├── streamlit/                      # Streamlit UI
│   ├── streamlit_app.py            # Main Streamlit application
│   ├── Streamlit-screenshot.png    # UI screenshot (used in README)
│   └── insurance_db/               # Persisted vector store (optional)
│
├── insurance_db/                   # Local ChromaDB storage
├── policy.pdf                      # Sample insurance policy document
├── rag_app.py                      # Core RAG pipeline logic
├── example_questions.md            # Sample insurance questions
├── requirements.txt                # Python dependencies
├── .env                            # API keys (gitignored)
└── README.md
```
---

## 👥 End / Intended Users
- Insurance/legal customers seeking quick clarifications
- Insurance/legal professionals reviewing contracts (policy details)
- Compliance and audit teams
- Engineers learning real-world RAG system design

---

## 💼 Practical Use Cases
- Insurance policy interpretation
- Legal contract analysis
- Compliance and audit verification
- Internal document Q&A systems
- Enterprise AI assistants grounded in proprietary documents

---

## ⚡ Key Challenges Solved
- Preventing LLM hallucinations in legal/medical contexts
- Maintaining semantic accuracy during text chunking
- Handling API quotas (Google Gemini) and dependency instability
- Returning **verbatim source text** for explainability
- Designing a clean, usable AI interface for non-technical users

---

## 🛣 Roadmap
- [x] End-to-end RAG pipeline with Gemini
- [x] Streamlit-based interactive UI
- [ ] Multi-document ingestion
- [ ] PDF text highlighting within UI
- [ ] Cloud deployment (GCP / AWS)
- [ ] Role-based access & logging

---

## 📬 Career Note
This project was built as a real-world RAG system, not a toy demo.

I’m actively seeking:
- AI Engineer / ML Engineer roles
- Data Engineering roles with GenAI focus
- Full-time positions

Feel free to connect or reach out if this project aligns with your team’s work.
