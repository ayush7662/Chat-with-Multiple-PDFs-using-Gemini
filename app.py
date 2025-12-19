import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)



def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible using the provided context.
If the answer is not available in the context, say:
"Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process PDFs first.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Save chat history
    st.session_state.chat.append(
        {"question": user_question, "answer": response["output_text"]}
    )



def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("📄 Chat with Multiple PDFs using Gemini")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    # Display chat history
    for chat in st.session_state.chat:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Gemini:** {chat['answer']}")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
