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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(script_dir, TARGET_DOCUMENT)
    vectorstore = initialize_vector_database(target_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set. Please add it in your Streamlit Cloud secrets.")
        st.stop()
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
    except ImportError:
        st.error("The langchain-groq package is missing. Add `langchain-groq` to requirements.txt and redeploy.")
        st.stop()
    
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