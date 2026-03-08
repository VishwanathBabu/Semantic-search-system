# Semantic Search System

## Deployed link : https://semantic-search-system-1jjg.onrender.com/

## 1) Problem Statement
Traditional keyword-matching search engines (which rely on algorithms like TF-IDF or BM25) suffer from the "vocabulary mismatch" problem. They often fail to retrieve relevant information when users phrase their questions using different synonyms than the source text. Furthermore, applying standard clustering to unstructured, raw data like the **20 Newsgroups dataset** often results in models grouping documents by arbitrary metadata (like the sender's email address, organization, or timestamps) rather than the actual meaning of the text. 

The goal of this project is to build a lightweight, context-aware semantic search system. By converting text into dense mathematical vectors, the system understands the *underlying meaning* behind a query, routes it intelligently using probability distributions, and caches results to optimize latency and compute costs.

## 2) System Architecture


1. **Frontend UI:** A lightweight HTML/Vanilla JS interface for real-time user interaction.
2. **API Layer:** An  FastAPI backend to handle requests, manage state, and route traffic.
3. **Processing:** User text is vectorized into a dense numerical array using a Hugging Face Sentence Transformer.
4. **Routing:** A Gaussian Mixture Model (GMM) predicts the semantic neighborhood of the query.
5. **Caching:** A custom partitioned semantic cache intercepts repeat or contextually similar queries before they trigger database scans.
6. **Retrieval:** If a cache miss occurs, FAISS (Facebook AI Similarity Search) performs a lightning-fast nearest-neighbor search in memory to find the most relevant document.

## 3) Design Decisions

### 3.1 Data Processing & Aggressive Cleaning
The raw 20 Newsgroups dataset is highly noisy. If left uncleaned, the embedding model learns to group documents based on metadata rather than semantic meaning. I built custom preprocessing pipelines using Regular Expressions (RegEx) to systematically strip:
* Email headers, footers, and routing information.
* Quoted replies from previous threads.
* Signatures and excess whitespace.
This aggressive cleaning ensures the embeddings generated later represent the *actual conversational context* of the document, completely removing metadata bias.

### 3.2 Fuzzy (Soft) Clustering
A core requirement was avoiding "hard" clustering algorithms. Real-world topics naturally overlap (e.g., a query about "Apple" could belong to both computer hardware and agricultural news).
* **GMM over K-Means:** Instead of locking a document into a single rigid cluster centroid like K-Means, I implemented a Gaussian Mixture Model (GMM). GMM returns a probability distribution, allowing concepts to exist in multiple semantic neighborhoods simultaneously.
* **Spherical Covariance:** In a 384-dimensional vector space, calculating a full covariance matrix creates massive computational overhead. A full covariance matrix requires estimating $O(D^2)$ parameters per cluster, where $D$ is the dimensionality. I implemented a *spherical* covariance matrix, which restricts this to a single variance parameter per cluster. This simplifies the math, speeds up training drastically, and prevents the model from overfitting on high-dimensional data.

### 3.3 Semantic Cache Design
To optimize latency and save compute power, a custom semantic cache was built to intercept queries before they reach the heavy vector database.
* **Partitioned Search ($O(N)$ Reduction):** The cache dictionary is partitioned by the `dominant_cluster_id`. When a user submits a query, we predict its cluster first. Instead of scanning the entire cache history in $O(N)$ time, we only execute a targeted search within that specific semantic neighborhood. As the cache grows, this drastically reduces search time.
* **Cosine Similarity Threshold:** The cache operates on a semantic threshold of 0.88. If the threshold is set too high (e.g., 0.99), the cache demands a nearly exact string match and becomes useless. If set too low (e.g., 0.70), it returns irrelevant answers for vaguely similar queries. A score of 0.88 hits the perfect sweet spot for catching variations in phrasing while maintaining high accuracy.

### 3.4 Docker Size Optimization
The application is fully containerized. To optimize for cloud deployment (especially on free-tier platforms where servers sleep and cold-start wake times matter), the `Dockerfile` specifically forces the installation of the **CPU-only version of PyTorch**. By preventing the installation of heavy Nvidia GPU drivers (CUDA toolkits), the Docker image size was reduced from over 2GB down to under 500MB, resulting in faster builds and cheaper hosting.

## 4) Embedded Model Choice
For text vectorization, I utilized the **Hugging Face `all-MiniLM-L6-v2`** model. It is based on a Sentence-BERT architecture, making it highly optimized for lightweight systems. It is smart enough to capture deep semantic meaning and translate text into 384-dimensional vectors, but small enough to run highly efficiently on standard CPUs without requiring expensive GPU hardware.

## 5) Clustering (Execution)
During the training phase, the dataset cluster count was expanded from the default 20 up to 50. This expansion allows the GMM to identify granular micro-topics (e.g., separating "PC motherboards" from "Mac monitors" within the broader hardware category) rather than broad, generalized buckets. This vastly improves the routing accuracy and cache hit-rate of the user's search queries.

## 6) Semantic Cache (Execution)
The cache actively monitors and updates its internal state in real-time, storing the query embedding, the matched document, and the cluster ID. It provides dedicated API endpoints (`GET /cache/stats` and `DELETE /cache`) allowing administrators to monitor hit/miss ratios, track cache size, and flush the memory footprint without needing to restart the server.

## 7) Output
When a user submits a query, the API returns a comprehensive JSON response containing:
* **Status:** Whether the search resulted in a `Cache Hit` or `Cache Miss`.
* **Matched Document:** The cleaned text of the most semantically relevant document.
* **Cluster Routing:** The specific cluster ID the query was routed to.
* **Similarity Score:** The distance metric showing how closely the document matched the user's query.

---



## Local Installation & Setup

* python -m venv venv
* On Windows use: venv\Scripts\activate
* pip install -r requirements.txt
* uvicorn main:app --reload

## To run using Docker
* docker build -t semantic-search .
* docker run -p 8000:8000 semantic-search 
