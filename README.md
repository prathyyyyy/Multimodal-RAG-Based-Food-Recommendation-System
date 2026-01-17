# 🍽️ Multimodal RAG-Based Food Recommendation System

## Overview
This project implements a Multimodal Retrieval-Augmented Generation (RAG) system for personalized food recommendations by combining text and image understanding. It is designed for a restaurant aggregator use case and delivers scalable, real-time, context-aware recommendations.

## Features
- Multimodal retrieval using text and image embeddings  
- Context-aware generation with a vision-language model  
- Fast vector-based similarity search using FAISS  
- RAG orchestration with LangChain  
- Interactive Streamlit web application  
- Scalable deployment on AWS  

## Tech Stack
- **Vision-Language Model**: Meta Llama-4 Maverick  
- **Embeddings**: AWS Titan (text and image)  
- **Vector Store**: FAISS  
- **Framework**: LangChain  
- **Frontend**: Streamlit  
- **Cloud**: AWS  

## Architecture
1. Food images and text metadata are embedded using AWS Titan embeddings.
2. Embeddings are stored in a FAISS vector index for efficient multimodal search.
3. User queries are embedded and matched against the vector store.
4. Retrieved context is passed to the vision-language model for grounded generation.
5. Recommendations are displayed in real time via the Streamlit interface.
cd multimodal-rag-food-recommender
pip install -r requirements.txt
