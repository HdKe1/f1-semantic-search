"""
Pre-compute all embeddings and save to a single JSON file.
Run this locally once before deploying:

    pip install sentence-transformers
    python precompute_embeddings.py

This generates embeddings.json which includes:
- Driver vibe_embedding_text embeddings
- Query term embeddings (memes, nicknames, aliases, common searches)
"""

import json
from sentence_transformers import SentenceTransformer

# Load drivers
with open("drivers_v2.json") as f:
    drivers = json.load(f)

corpus = [d["vibe_embedding_text"] for d in drivers]

# Collect all curated query terms from drivers
query_terms = set()
for d in drivers:
    for field in ["memes", "aliases", "nicknames", "search_keywords"]:
        for term in d.get(field, []):
            query_terms.add(term.lower().strip())
    # Also add driver name
    query_terms.add(d["name"].lower().strip())

# Add demonyms and common descriptive queries
extra_terms = [
    "dutch", "british", "english", "german", "finnish", "spanish",
    "australian", "mexican", "brazilian", "french", "monegasque",
    "canadian", "italian", "aggressive", "smooth", "champion",
    "world champion", "rookie", "veteran", "fast", "slow",
    "aggressive dutch champion", "smooth operator", "iceman",
    "the honey badger", "driver who drinks from his shoe",
    "bwoah", "shoey", "grazie ragazzi", "valtteri its james",
    "is that glock", "no michael no", "get in there lewis",
]
for term in extra_terms:
    query_terms.add(term.lower().strip())

query_terms = sorted(query_terms)
print(f"Computing embeddings for {len(corpus)} drivers and {len(query_terms)} query terms...")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute driver embeddings
doc_embeddings = model.encode(corpus).tolist()

# Compute query term embeddings
term_embeddings = model.encode(query_terms).tolist()

# Build output
query_terms_dict = {}
for term, emb in zip(query_terms, term_embeddings):
    query_terms_dict[term] = emb

output = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "embeddings": doc_embeddings,
    "query_terms": query_terms_dict,
}

with open("embeddings.json", "w") as f:
    json.dump(output, f)

print(f"Saved to embeddings.json:")
print(f"  - {len(doc_embeddings)} driver embeddings (dim={len(doc_embeddings[0])})")
print(f"  - {len(query_terms_dict)} query term embeddings")
