# 🍽️ Multimodal RAG-Based Food Recommendation System
---
## Overview
This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** system for delivering **personalized food recommendations** by combining text and image understanding. It is designed for a restaurant aggregator use case and supports scalable, real-time inference.
<img width="1536" height="1024" alt="Multimodal RAG architecture for food recommendations" src="https://github.com/user-attachments/assets/f3ea9111-a24a-4ea2-bc30-abeeaf41454b" />

---
## Key Features
- Multimodal search using **text and image embeddings**
- Context-aware recommendations using a **vision-language model**
- Fast similarity search with **FAISS**
- Modular RAG pipeline with **LangChain**
- Interactive **Streamlit** web application
- Cloud-ready deployment on **AWS**
---
## Tech Stack
- **Vision-Language Model**: Meta Llama-4 Maverick  
- **Embeddings**: AWS Titan (Text + Image)  
- **Vector Store**: FAISS  
- **RAG Framework**: LangChain  
- **Frontend**: Streamlit  
- **Deployment**: AWS  
---
## Diagram Flow
```
User (Text / Image Query)
  ▼
Query Embedding (AWS Titan)
  ▼
FAISS Vector Store (Text + Image Embeddings)
  ▼
Relevant Multimodal Context (Text + Images)
  ▼
LangChain RAG Pipeline
  ▼
Meta Llama-4 Maverick (Vision-Language Model)
  ▼
Personalized Food Recommendation
  ▼
Streamlit UI (AWS Deployment)
```
---
## Architecture Description
Food images and textual metadata are embedded using AWS Titan embeddings and stored in a FAISS vector database. When a user submits a text or image query, the system retrieves the most relevant multimodal context and augments the prompt for the vision-language model, ensuring grounded and accurate recommendations.

---
## Proofs
<img width="1568" height="821" alt="Screenshot 2025-12-16 020613" src="https://github.com/user-attachments/assets/572d6626-478c-4a3d-ac8d-b9497d81660c" />
<img width="1640" height="343" alt="Screenshot 2025-12-16 020625" src="https://github.com/user-attachments/assets/72fb5bcd-3e5e-4ec9-99e4-00515bc859da" />
<img width="1508" height="792" alt="Screenshot 2025-12-16 020646" src="https://github.com/user-attachments/assets/41562477-79ac-4916-85e7-06182285f7a6" />


---

## Use Cases
- Personalized food discovery  
- Restaurant aggregation platforms  
- AI-powered menu recommendations  
- Multimodal semantic search  
---

## License
MIT License
