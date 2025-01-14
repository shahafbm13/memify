import json
import numpy as np
from gensim.models import KeyedVectors
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
        "best_meme_name": str,
        "best_description": str,
        "similarity": float
      }
    """
    best_match = {
        "best_meme_name": None,
        "best_description": None,
        "similarity": -1
    }
    
    prompt_emb = embed_text_with_word2vec(prompt, model)
    
    for meme_name, descriptions in meme_data.items():
        for description in descriptions:
            desc_emb = embed_text_with_word2vec(description, model)
            sim = cosine_similarity(prompt_emb, desc_emb)
            
            if sim > best_match["similarity"]:
                best_match["similarity"] = sim
                best_match["best_meme_name"] = meme_name
                best_match["best_description"] = description
    
    return best_match

def clean_meme_name(file_name: str) -> str:
    """Clean meme file name by removing numbers, underscores, and extensions."""
    return re.sub(r"^\d+_", "", file_name).replace("_", " ").replace(".jpg", "")

# if __name__ == "__main__":
#     # 1. Load your meme JSON
#     with open("/Users/royayalon/Downloads/meme_text_description.json", "r", encoding="utf-8") as f:
#         meme_data = json.load(f)

#     # 2. Get meme names from the meme_templates folder
#     meme_template_path = "/Users/royayalon/Documents/Academy/generative_project/for_project/meme_templates"
#     meme_names = os.listdir(meme_template_path)

#     # Clean meme names in the template folder
#     cleaned_meme_names = [clean_meme_name(name) for name in meme_names]

#     # Filter meme_data to only include memes present in the template folder
#     meme_data = {
#         clean_meme_name(meme_name): descriptions
#         for meme_name, descriptions in meme_data.items()
#         if clean_meme_name(meme_name) in cleaned_meme_names
#     }

#     # Print how many memes are in meme_data
#     print(f"Number of memes: {len(meme_data)}")

#     # 3. Load your Word2Vec model
#     model_path = "GoogleNews-vectors-negative300.bin"
#     model = KeyedVectors.load_word2vec_format(model_path, binary=True)

#     # 4. Your user prompt
#     user_prompt = "im tired. want to sleep"

#     # 5. Find the best match
#     result = find_best_meme_description(user_prompt, meme_data, model)
    
#     print("Best Meme Name:", result["best_meme_name"])
#     print("Best Description:", result["best_description"])
#     print("Similarity Score:", result["similarity"])