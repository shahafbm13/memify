import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from collections import Counter
from tqdm import tqdm  # Import tqdm for progress bars

# 1) Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2) Load your data (from "archive (5)" folder, all JSON files)
jsons = {}
print("Loading JSON files...")
for file in tqdm(os.listdir("archive (5)"), desc="Processing JSON files"):
    if file.endswith(".json"):
        with open(os.path.join("archive (5)", file), "r", encoding="utf-8") as f:
            # add to jsons, key is the template's filename (or you can strip .json if you prefer)
            jsons[file] = json.load(f)

# 3) Load all images (optional, just storing their filenames here)
images = {}
print("Loading image files...")
for file in tqdm(os.listdir("archive (5)"), desc="Processing image files"):
    if file.endswith(".jpg"):
        images[file] = file

# Show first 10 captions of the first template (as a sanity check)
first_json_key = list(jsons.keys())[0]
print("First template name:", first_json_key)
print("First 10 captions of this template:", jsons[first_json_key][:10])

# 4) Make every template name a unique ID if needed (optional)
template_id_to_index = {}
index_to_template_id = {}
template_counter = 0

for template_id in jsons:
    template_id_to_index[template_id] = template_counter
    index_to_template_id[template_counter] = template_id
    template_counter += 1

# 5) Create a FAISS index
dimension = 384  # The dimension for 'all-MiniLM-L6-v2' embeddings
faiss_index = faiss.IndexFlatIP(dimension)  # Using inner product

# We'll store the embeddings in a list, and also track which template each embedding belongs to
all_embeddings = []
embedding_to_template_id = []

# 6) Compute embeddings for each caption of each template, add to the index
print("Computing caption embeddings...")
for template_id, captions in tqdm(jsons.items(), desc="Processing templates", unit="template"):
    for caption in captions:
        caption_embedding = model.encode(caption, convert_to_numpy=True)
        # Normalize for cosine similarity
        caption_embedding /= np.linalg.norm(caption_embedding)
        caption_embedding = caption_embedding.astype('float32')
        
        all_embeddings.append(caption_embedding)
        embedding_to_template_id.append(template_id)

# Convert to a NumPy array
all_embeddings = np.array(all_embeddings, dtype='float32')

# Add to the FAISS index
print("Adding embeddings to FAISS index...")
faiss_index.add(all_embeddings)

print("Total embeddings added to FAISS:", faiss_index.ntotal)

# 7) Define a search function
def search_templates(prompt, top_k=10):
    # Embed the prompt
    prompt_embedding = model.encode(prompt, convert_to_numpy=True)
    prompt_embedding /= np.linalg.norm(prompt_embedding)
    prompt_embedding = prompt_embedding.astype('float32')
    
    # FAISS search
    D, I = faiss_index.search(np.array([prompt_embedding]), top_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        # idx is the embedding index within all_embeddings
        # Map it back to the template ID
        template_id = embedding_to_template_id[idx]
        results.append((template_id, float(score)))
    
    return results

# 8) Test retrieval with a user prompt
user_prompt = "I need a meme about feeling jealous and ignoring important stuff"
matches = search_templates(user_prompt, top_k=20)

# 9) For final selection, let's see which template appears most frequently in top_k
counter = Counter([m[0] for m in matches])
best_template = counter.most_common(1)[0][0]

print("Top-k matches:", matches)
print("Likely best template:", best_template)