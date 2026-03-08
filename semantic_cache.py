import numpy as np

class SemanticCache:
    def __init__(self, similarity_threshold=0.88):
        '''# In a 384-dimensional space (MiniLM), cosine similarity behaves differently 
        # than standard math. If we set this too high , the cache acts like 
        # an exact-match system—only returning hits for identical phrasing, which defeats 
        # the purpose of a semantic cache.
        # # If we set it too low, we get false positives (e.g., "How to buy a car" 
        # matches "How to repair a car").
        # So 0.88 is a good value which allows users to phrase the exact same intent differently, but remains 
        # strict enough to avoid serving the wrong answer'''
        self.cache = {}
        self.threshold = similarity_threshold
        
        self.total_entries = 0
        self.hit_count = 0
        self.miss_count = 0

    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def check_cache(self, query, query_embedding, dominant_cluster):
        '''# As the cache grows to thousands of queries, doing a cosine similarity check 
        # against every cached item becomes O(N) and creates a major bottleneck. 
        # By using the GMM clusters , we only compute similarity against 
        # queries in the exact same semantic neighborhood. 
        # This drops lookup time exponentially and ensures the cache remains lightning 
        # fast even at scale.'''
        if dominant_cluster not in self.cache:
            return None
            
        cluster_entries = self.cache[dominant_cluster]
        
        best_match = None
        highest_score = -1
        
        for entry in cluster_entries:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > highest_score:
                highest_score = score
                best_match = entry
                
        if highest_score >= self.threshold:
            self.hit_count += 1
            return {
                "matched_query": best_match["query"],
                "similarity_score": float(highest_score),
                "result": best_match["result"]
            }
            
        return None

    def store(self, query, query_embedding, dominant_cluster, result):
        if dominant_cluster not in self.cache:
            self.cache[dominant_cluster] = []
            
        self.cache[dominant_cluster].append({
            "query": query,
            "embedding": query_embedding,
            "result": result
        })
        self.total_entries += 1
        self.miss_count += 1

    def flush(self):
        self.cache = {}
        self.total_entries = 0
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self):
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 3)
        }