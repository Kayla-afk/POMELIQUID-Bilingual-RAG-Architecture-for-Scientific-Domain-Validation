# POMELIQUID: Bilingual Scientific RAG Architecture

## Credits & Acknowledgments
* **Software Engineering & AI Architecture:** Independently developed by Kayla Nuansa Ceria.
* **Underlying Scientific Research:** The foundational antibacterial study and ethical clearance for the Pometia pinnata ethanol extract were jointly researched by Kayla Nuansa Ceria and Mauhibatus Syifa during high school.
* **Original Concept Inspiration:** The foundational concept for the RAG implementation was inspired by the tutorial from [AmanxAI: Build a Real-Time AI Assistant using RAG & LangChain](https://amanxai.com/2025/11/18/build-a-real-time-ai-assistant-using-rag-langchain/).

## Original Concept vs. Custom Upgrades
The original idea from AmanxAI provided a strong foundation for building a basic, local RAG pipeline using LangChain and Ollama. However, to elevate this system into a production-ready tool for scientific validation, I engineered several major architectural upgrades:

1. **Bilingual Processing Pipeline (New Feature):** Unlike the base tutorial, this system features a dynamic prompt architecture capable of automatic language detection. It seamlessly translates complex Indonesian research contexts to accurately answer queries in English, and vice versa.
2. **Cloud API over Local LLM (Architecture Upgrade):** The original tutorial relied on local Ollama models, which limits deployment. To achieve ultra-low latency and enable seamless cloud hosting, the inference engine was rewritten to integrate the Groq API (llama3-8b-8192).
3. **Zero-Hallucination Mandate (New Feature):** Implemented strict contextual boundaries. If the requested data is absent from the target document, the system is hard-coded to refuse the prompt, ensuring absolute scientific integrity.
4. **Cloud-Native Deployment (Architecture Upgrade):** The local-only setup was heavily optimized for Streamlit Community Cloud, incorporating efficient dependency management and secure, environment-variable-based secrets handling.

## Executive Summary
This repository contains the source code for POMELIQUID, a Real-Time AI Assistant built to validate and answer queries strictly based on scientific research regarding the antibacterial properties of Matoa leaf ethanol extract (Pometia pinnata). 

Developed as a personal engineering project in college, this application serves as the advanced technical implementation of prior high school scientific research. The system enforces strict factual grounding to prevent AI hallucinations, ensuring that all generated responses are strictly tied to the provided documentation. 

## Live Demo
The system is deployed and ready for immediate evaluation without requiring local environment setup.

**[Access the Live Web Application Here](https://pomeliquid-bilingual-rag-architecture-for-scientific-domain-va.streamlit.app/)**

### Quick Test Prompts (Copy & Paste)
To evaluate the retrieval and bilingual reasoning capabilities, try pasting these exact queries into the live application:

1. **Indonesian Context Test:** "Apa kesimpulan utama dari uji antibakteri ekstrak etanol daun matoa pada penelitian ini?"
2. **English Translation & Reasoning Test:** "What are the main findings regarding the antibacterial properties of Pometia pinnata in this study?"
3. **Out-of-Scope Test (Hallucination Prevention):** "Bagaimana cara membuat aplikasi e-commerce?" 
   *(The system should correctly state that this information is not covered in the POMELIQUID documentation).*

## System Architecture & Tech Stack
* **Frontend:** Streamlit 
* **AI Framework:** LangChain Core & Community
* **LLM Engine:** Groq API (llama3-8b-8192)
* **Embeddings:** HuggingFace (paraphrase-multilingual-MiniLM-L12-v2)
* **Vector Database:** ChromaDB
* **Document Processing:** PyPDFLoader & RecursiveCharacterTextSplitter

## Local Installation
If you prefer to run the architecture locally:
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Configure your environment variables securely (e.g., create a `.streamlit/secrets.toml` file for your `GROQ_API_KEY`).
4. Execute the application: `streamlit run assistant.py`
