import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

# Cache the database load so it only happens once when the app starts
@st.cache_resource
def initialize_vector_database(pdf_path: str):
    if not os.path.exists(pdf_path):
        st.error(f"System Error: Target document '{pdf_path}' not found.")
        sys.exit(1)
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    return vectorstore

def main():
    # 1. UI Configuration: Clean and Minimalist
    st.set_page_config(page_title="POMELIQUID Research", layout="centered")
    
    # Hide Streamlit default menus for a professional look
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # 2. Page Header
    st.title("POMELIQUID")
    st.markdown("Search the *Pometia pinnata* antibacterial study.")

    # 3. Backend Initialization
    TARGET_DOCUMENT = "KTI_POMELIQUID Ekstrak Etanol Daun Matoa.pdf"
    vectorstore = initialize_vector_database(TARGET_DOCUMENT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model="llama3:8b")
    
    prompt = ChatPromptTemplate.from_template(
        """You are the official Bilingual Scientific AI Assistant for the POMELIQUID research project. 
        Your primary directive is to answer questions strictly based on the provided research context.
        The research context provided to you is in Indonesian. 

        CRITICAL INSTRUCTIONS:
        1. You MUST detect the language of the 'User Query'.
        2. If the user asks in English, translate the context in your mind and answer in English.
        3. If the user asks in Indonesian, answer in Indonesian.
        4. If the provided context does not contain the answer, you must state: 'This information is not covered in the current POMELIQUID documentation' (translate this sentence to match the user's language).

        Research Context:
        {context}

        User Query:
        {question}
        """
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # 4. Minimalist Search Interface
    user_query = st.text_input("", placeholder="Search the POMELIQUID research content...")

    if user_query:
        with st.spinner("Extracting scientific context..."):
            response = chain.invoke(user_query)
            st.markdown("### Output")
            st.write(response)

if __name__ == "__main__":
    main()