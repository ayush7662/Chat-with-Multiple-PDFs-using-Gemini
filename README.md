**📄 Chat with Multiple PDFs using Gemini (Streamlit + LangChain)**

An interactive AI-powered PDF Question Answering application built using Streamlit, LangChain, FAISS, and Google Gemini.
Users can upload multiple PDF documents and ask natural language questions to retrieve accurate answers directly from the document content.


**🚀 Features**

📂 Upload multiple PDF files

🔍 Extract and process text from PDFs

🧠 Convert text into vector embeddings

⚡ Store embeddings using FAISS vector database

🤖 Ask questions using Google Gemini (LLM)

💬 Chat-like interface with conversation history

🔐 Secure API key management using .env




**🛠️ Tech Stack**


| Category               | Technology                       |
| ---------------------- | -------------------------------- |
| Frontend               | Streamlit                        |
| LLM                    | Google Gemini (gemini-1.5-flash) |
| NLP Framework          | LangChain                        |
| Vector Database        | FAISS                            |
| Embeddings             | HuggingFace (MiniLM)             |
| PDF Processing         | PyPDF2                           |
| Environment Management | python-dotenv                    |




**📦 Dependencies**

streamlit
google-generativeai
python-dotenv
langchain
langchain-community
langchain-google-genai
PyPDF2
faiss-cpu
sentence-transformers



**🏗️ Project Architecture**

├── app.py
├── faiss_index/
├── .env
├── requirements.txt
└── README.md




**🔄 Application Workflow**


1️⃣ PDF Upload

Users upload one or more PDF files through the Streamlit sidebar.

2️⃣ Text Extraction

PDFs are read using PyPDF2

Text is extracted page by page

3️⃣ Text Chunking

Large text is split into chunks

Uses RecursiveCharacterTextSplitter

Ensures context is preserved using overlap

4️⃣ Embedding Generation

Uses HuggingFace sentence-transformers

Model: all-MiniLM-L6-v2

Embeddings are normalized for better similarity search

5️⃣ Vector Storage

FAISS stores embeddings locally

Enables fast semantic search

6️⃣ Question Answering

User question → similarity search

Relevant chunks passed to Gemini LLM

Context-aware response generated


**🧠 Core Components Explained**

🔹 get_pdf_text()

Extracts text from uploaded PDF files safely.

🔹 get_text_chunks()

Splits large text into manageable overlapping chunks to preserve meaning.

🔹 get_vectorstore()

Creates and saves a FAISS vector index using HuggingFace embeddings.

🔹 get_conversational_chain()

Defines a custom prompt and initializes the Gemini LLM using LangChain.

🔹 user_input()

Performs similarity search

Passes documents + question to LLM

Stores conversation history



**🔐 Environment Setup**

Create a .env file:

GOOGLE_API_KEY=your_google_gemini_api_key

**▶️ How to Run the Project**

Step 1: Install Dependencies

pip install -r requirements.txt


Step 2: Run Streamlit App

streamlit run app.py


Step 3: Use the App

Upload PDF files

Click Submit & Process

Ask questions from the document content

📸 UI Preview

Main Chat Window for Q&A

Sidebar for PDF uploads

Chat history maintained across interactions

📈 Use Cases

Research paper analysis

Legal document review

Study notes Q&A

Resume or report analysis

Enterprise document intelligence

**🧪 Future Improvements**

Multi-PDF source citation

Streaming responses

Authentication system

Cloud-based vector storage

Chat memory using LangChain Memory





