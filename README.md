# POMELIQUID: Bilingual RAG Architecture for Scientific Domain Validation

A localized, zero-hallucination Retrieval-Augmented Generation (RAG) architecture engineered to provide dynamic, bilingual (Indonesian/English) querying for the POMELIQUID scientific research and ethical-clearance documentation.

## Project Context & Attribution

This project is a fusion of empirical biological research and advanced AI software engineering. To ensure absolute transparency and academic integrity, the project is divided into two distinct domains:

### 1. The Scientific Research (The Data Domain)
The foundational scientific data, methodology, and ethical-clearance documentation utilized in this RAG system stem from the following research:
> **Title:** POMELIQUID: EKSTRAK ETANOL DAUN MATOA (Pometia Pinnata) SEBAGAI AGEN HAYATI ANTIBAKTERI *Aeromonas hydrophila* PADA BUDIDAYA IKAN MAS KOKI (*Carassius auratus*)
* **Research Team:** Kayla Nuansa Ceria & Mauhibatus Syifa.
* **Context:** This research successfully engineered an organic antibacterial formulation. The resulting document is utilized in this project with explicit permission as the closed-loop knowledge base.

### 2. The Software Architecture (The Engineering Domain)
* **AI Architecture & Software Engineering:** Kayla Nuansa Ceria (Solo Developer).
* **Context:** I engineered this application to demonstrate how domain-specific AI can securely query complex, localized scientific data without relying on hallucinatory open-web models.

## Architectural Evolution & Anti-Plagiarism Statement

The foundational logic for deploying a basic LangChain LCEL pipeline was inspired by the [AmanxAI Real-Time Assistant Framework](https://amanxai.com/2025/11/18/build-a-real-time-ai-assistant-using-rag-langchain/). 

However, open-web querying is unsafe for rigorous scientific validation due to LLM hallucination. To meet professional engineering standards, **I completely re-engineered the architecture** with the following advanced features:

1. **Closed-Loop Vectorization:** Removed the generic web-scraper and implemented a local Vector Database (ChromaDB) specifically mapped only to the POMELIQUID documentation.
2. **True Bilingual Intelligence:** The original paper is in Indonesian. I upgraded the embedding model to `paraphrase-multilingual-MiniLM-L12-v2`, allowing the AI to mathematically understand that English queries and Indonesian text share the same semantic meaning.
3. **Dynamic Translation Prompting:** Engineered a strict LangChain system prompt that forces the LLM to read the Indonesian context but dynamically formulate its response in the exact language the user types.
4. **Minimalist UI Integration:** Shifted from a terminal-only script to a sleek, Streamlit-powered web interface designed with Apple's Human Interface Guidelines in mind.

## Core Technology Stack
* **Framework:** LangChain & LangChain Expression Language (LCEL)
* **Interface:** Streamlit
* **Local LLM Inference:** Ollama (`llama3:8b`)
* **Vector Database:** ChromaDB
* **Multilingual Embeddings:** HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Document Processing:** PyPDFLoader & RecursiveCharacterTextSplitter
