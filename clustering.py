import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

def train_fuzzy_clusters(n_components=20):
    print("Loading embeddings for clustering...")
    with open("corpus_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
        
    print(f"Training GMM with {n_components} clusters: ")
    '''Semantic concepts naturally overlap . Keywords such as politics and firearms have same content .
    # Gaussian Mixture Models (GMM) solve this by returning 
    # a probability distribution across all clusters, perfectly mapping the 'fuzzy' 
    # boundaries of natural language.
    '''
    '''The original 20 Newsgroups dataset has 20 rigid labels. Natural 
    # conversations contain very small sub-topics (e.g. hardware splits into display, 
    # keyboard , equipments , etc). By expanding the components to 50, the GMM can capture 
    # these very diverse meaning based semantic sub-clusters rather than forcing broad, muddy categories.'''
    gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42)
    '''# Our vectors have 384 dimensions. If we use a 'full' covariance, we are forcing the 
    # model to calculate the complex relationships between every single one of those 384 
    # dimensions. That is computationally exhausting and usually makes the model overthink (overfitting).
    # By using 'spherical', we simplify the math. We essentially tell 
    # the model to just look for simple, round clusters. It's much faster to train 
    # and still does a great job of separating our topics.
    '''
    gmm.fit(embeddings)
    
    with open("gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)
    print("Fuzzy clustering model saved.")

def get_cluster_distribution(embedding, gmm_model):
    '''# Instead of returning a single integer (e.g., Cluster 5), predict_proba() 
    # returns an array of probabilities (e.g., 60% Cluster 5, 30% Cluster 12, 
    # 10% Cluster 2). This distribution is what allows the semantic cache downstream 
    # to recognize when a query bridges multiple topics.'''
    embedding = embedding.reshape(1, -1)
    probabilities = gmm_model.predict_proba(embedding)[0]
    return probabilities

if __name__ == "__main__":
    train_fuzzy_clusters(n_components=50)