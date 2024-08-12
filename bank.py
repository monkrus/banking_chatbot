import streamlit as st
import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set API Key for NVIDIA
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    st.error("NVIDIA_API_KEY not found in environment variables.")
    st.stop()

# Function to clean text
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to process PDF
def process_pdf(file_path):
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found.")
        return None

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        st.error("No documents loaded.")
        return None

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.split_documents(docs)

    if not final_documents:
        st.error("No documents after splitting.")
        return None

    try:
        # Use HuggingFace Embeddings
        embeddings = HuggingFaceEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)
    except Exception as e:
        st.error(f"Vector embedding failed: {e}")
        return None

    return vectors

# Function to interact with LLM using ChatNVIDIA
def get_llm_response(prompt):
    try:
        client = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            api_key="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", 
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )

        response = ""
        for chunk in client.stream([{"role":"user","content":prompt}]):
            response += chunk.content

        if isinstance(response, str):
            return response
        else:
            raise TypeError(f"Expected string response, got {type(response)}")

    except TypeError as e:
        return f"TypeError occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Main script for Streamlit
if __name__ == "__main__":
    st.title("PDF Processor and Chatbot")

    pdf_file_path = "bank.pdf"
    vectors = process_pdf(pdf_file_path)

    if vectors:
        st.session_state.vectors = vectors
        st.success("PDF processed and vectors created.")
    else:
        st.stop()

    # Get user input
    user_query = st.text_input("Ask a question about the uploaded PDF or general knowledge")

    # Handle user query
    if st.button("Get Answer"):
        if "vectors" not in st.session_state:
            st.error("Vectors are not initialized. Please upload and process a PDF.")
        else:
            retriever = st.session_state.vectors.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(user_query)

            context = " ".join([doc.page_content for doc in retrieved_docs])

            # Create the prompt
            prompt = f"Question: {user_query}\nContext: {context}"
            response = get_llm_response(prompt)

            st.write(response)

            with st.expander("Related Document Content"):
                for doc in retrieved_docs:
                    st.write(doc.page_content)
                    st.write("-------------------------------")
