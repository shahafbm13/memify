import json
import numpy as np
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
import os
import re

def embed_text_with_word2vec(text: str, model) -> np.ndarray:
    """Embed text using Word2Vec."""
    words = text.lower().split()
    embeddings = []
    
    for word in words:
        if word in model.key_to_index:
            embeddings.append(model[word])
    
    if not embeddings:
        return np.zeros(model.vector_size)
    
    return np.mean(embeddings, axis=0)
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_best_meme_description(prompt: str, meme_data: dict, model) -> dict:
    """
    Returns a dictionary containing:
      { 
        "best_meme_path": str,
        "similarity": float
      }
    """
    best_match = {
        "best_meme_path": None,
        "similarity": -1
    }
    # Create a new dict of meme_file_path: embedding
    file_path_to_embedding = {file_path: model.encode(description) for file_path, description in meme_data.items()}
    prompt_emb = model.encode(prompt)
    
    for meme_file_path, description_embedding in file_path_to_embedding.items():
        sim = cosine_similarity(prompt_emb, description_embedding)
        
        if sim > best_match["similarity"]:
            best_match["similarity"] = sim
            best_match["best_meme_path"] = meme_file_path
    
    return best_match

def clean_meme_name(file_name: str) -> str:
    """Clean meme file name by removing numbers, underscores, and extensions."""
    return re.sub(r"^\d+_", "", file_name).replace("_", " ").replace(".jpg", "")

