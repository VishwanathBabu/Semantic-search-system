from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import FileResponse  
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from semantic_cache import SemanticCache

app = FastAPI(title="Semantic Search & Fuzzy Cache API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

'''# In a FastAPI service, loading ML models and vector databases from disk inside 
# the route handler (e.g., inside the /query function) would cause massive latency, 
# as disk I/O and model initialization would happen on every single API call. 
# 
# By loading the SentenceTransformer, GMM model, and FAISS index into global memory 
# at startup, they stay "warm." This allows the API to serve hundreds of concurrent 
# requests instantly using RAM, satisfying the requirement for a fast, responsive API.
# '''
print("Loading Models and Data into Memory...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
with open("gmm_model.pkl", "rb") as f:
    gmm_model = pickle.load(f)
with open("corpus_texts.pkl", "rb") as f:
    corpus_texts = pickle.load(f)
    
vector_index = faiss.read_index("corpus.index")
sem_cache = SemanticCache(similarity_threshold=0.88)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/query")
async def process_query(req: QueryRequest):
    q_emb = embed_model.encode([req.query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb) 
    
    '''# We predict the cluster distribution before checking the cache to exploit our 
    # cache's partitioned data structure. Knowing the dominant cluster upfront lets us 
    # instantly jump to the correct semantic neighborhood in the cache. This prevents 
    # a full-table scan and drastically reduces latency as the cache grows.'''
    probs = gmm_model.predict_proba(q_emb)[0]
    dominant_cluster = int(np.argmax(probs))
    
    flat_q_emb = q_emb.flatten()
    cache_hit = sem_cache.check_cache(req.query, flat_q_emb, dominant_cluster)
    
    if cache_hit:
        return {
            "query": req.query,
            "cache_hit": True,
            "matched_query": cache_hit["matched_query"],
            "similarity_score": cache_hit["similarity_score"],
            "result": cache_hit["result"],
            "dominant_cluster": dominant_cluster
        }
        
    
    distances, indices = vector_index.search(q_emb, k=1)
    retrieved_text = corpus_texts[indices[0][0]]
    
    sem_cache.store(req.query, flat_q_emb, dominant_cluster, retrieved_text)
    
    return {
        "query": req.query,
        "cache_hit": False,
        "result": retrieved_text,
        "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
async def get_stats():
    return sem_cache.get_stats()

@app.delete("/cache")
async def clear_cache():
    sem_cache.flush()
    return {"message": "Cache successfully flushed"}