import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

'''The lightweight semantic search system is needed in the project.  
I chose all-MiniLM-L6-v2 because it is the most suitable combination of semantic accuracy and computational efficiency
It can generate 384-dimensional vectors and can continue to 
execute inference within a minute directly on a CPU, as it is not large like models such as BERT-base or RoBERTa.'''

model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    """
    The 20 Newsgroups data is noisy. If not cleaned then the embedding model will assume
    more weight to metadata such as similar domains like email address or same organization header 
    rather than the semantic text.
    What I did :
    ------------
    1. Strip all headers by skipping to the first blank line.
    2. Remove quoted replies (lines starting with '>') to prevent overlapping 
       conversation chains from artificially inflating similarity scores between 
       distinct documents.
    """
    lines = text.split('\n')
    cleaned_lines = []
    in_body = False
    for line in lines:
        if not in_body and line.strip() == '':
            in_body = True 
            continue
        if in_body and not line.startswith('>'): 
            cleaned_lines.append(line)
    return ' '.join(cleaned_lines).strip()

def build_vector_db(data_dir="./20_newsgroups/20_newsgroups"):
    print("Loading and cleaning dataset (with progress tracking)...")
    texts = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Could not find the dataset at {data_dir}. Check your folder structure.")
        
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for cat in categories:
        cat_path = os.path.join(data_dir, cat)
        for fname in os.listdir(cat_path):
            file_path = os.path.join(cat_path, fname)
            
            try:
                with open(file_path, 'r', encoding="latin1", errors="replace") as f:
                    doc = f.read()
                    
                cleaned = clean_text(doc)
                '''Removing documents under 50 characters because 
                # highly short texts (like "Thanks") lack sufficient 
                # semantic density to form meaningful clusters or search results.'''
                if len(cleaned) > 50: 
                    texts.append(cleaned)
                    
                    
                    if len(texts) % 500 == 0:
                        print(f"Processed {len(texts)} viable documents...")
                        
                if len(texts) >= 5000:
                    break
            except Exception as e:
                continue
        
        if len(texts) >= 5000:
            break
            
    print(f"Successfully retained {len(texts)} viable documents. Embedding now...")
    
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    '''The problem sttement asks for lightweight solution . Instead of using a heavyweight vector 
    # database container or writing a slow brute-force 
    # cosine similarity loop from scratch, I chose FAISS.
    #------------------------------------------------------------------------------------
    # I am using IndexFlatIP (Inner Product) with L2-normalized 
    # vectors, which is mathematically equivalent to Cosine Similarity. This provides 
    # exact nearest-neighbor search with millisecond retrieval time entirely 
    # in memory. Hence its suitable for lightweight solution.'''
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, "corpus.index")
    with open("corpus_texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    with open("corpus_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
        
    print("Vector DB successfully built and saved.")

if __name__ == "__main__":
    build_vector_db("./20_newsgroups/20_newsgroups")