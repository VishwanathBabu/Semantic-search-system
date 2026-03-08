# Semantic Search System

## Deployed link : https://semantic-search-system-1jjg.onrender.com/

## Project Overview
This project is a lightweight, high-performance semantic search system built on the **20 Newsgroups dataset**. Unlike traditional keyword-matching systems, this engine understands the *context and meaning* of a user's query. 

It routes queries through a Gaussian Mixture Model (GMM) for soft clustering, utilizes FAISS for lightning-fast vector retrieval, and features a custom semantic cache that dramatically speeds up repeated or contextually similar queries.

## Tech Stack
* Backend: FastAPI, Python
* Machine Learning: PyTorch , Hugging Face (`all-MiniLM-L6-v2`), scikit-learn (GMM)
* Vector Database: FAISS 
* Deployment: Docker, Render
* Frontend: HTML, Vanilla JavaScript, CSS

---



## Local Installation & Setup

* python -m venv venv
* On Windows use: venv\Scripts\activate
* pip install -r requirements.txt
* uvicorn main:app --reload

## To run using Docker
* docker build -t semantic-search .
* docker run -p 8000:8000 semantic-search 
