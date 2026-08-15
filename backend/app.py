"""
FastAPI backend for F1 Driver Semantic Search.

Uses pre-computed embeddings for drivers AND a local cosine similarity
search. No external API calls at runtime.

Query embeddings are computed locally using a lightweight approach:
we pre-compute embeddings for all possible curated terms and use
fuzzy + keyword matching for everything else.

Run locally: python app.py
Deploy: set PORT env var (Render sets this automatically)
"""

import os
import json
import difflib
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="F1 Semantic Search", version="1.0.0")

# CORS: allow specific origins in production, fallback to * for local dev
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.environ.get("DRIVERS_JSON", None)
if DATA_PATH is None:
    if os.path.exists("drivers_v2.json"):
        DATA_PATH = "drivers_v2.json"
    elif os.path.exists("../drivers_v2.json"):
        DATA_PATH = "../drivers_v2.json"
    else:
        raise FileNotFoundError(
            "drivers_v2.json not found. Place it in the backend/ directory "
            "or set the DRIVERS_JSON environment variable."
        )

with open(DATA_PATH) as f:
    drivers = json.load(f)

names_by_id = {d["id"]: d["name"] for d in drivers}
ids = [d["id"] for d in drivers]
corpus = [d["vibe_embedding_text"] for d in drivers]

# ---------------------------------------------------------------------------
# Load pre-computed embeddings
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_JSON", None)
if EMBEDDINGS_PATH is None:
    if os.path.exists("embeddings.json"):
        EMBEDDINGS_PATH = "embeddings.json"
    elif os.path.exists("../embeddings.json"):
        EMBEDDINGS_PATH = "../embeddings.json"
    else:
        raise FileNotFoundError(
            "embeddings.json not found. Run: python precompute_embeddings.py"
        )

with open(EMBEDDINGS_PATH) as f:
    embeddings_data = json.load(f)

doc_embeddings = np.array(embeddings_data["embeddings"])

# Also load query term embeddings (pre-computed for common search terms)
query_term_embeddings = {}
if "query_terms" in embeddings_data:
    for term, emb in embeddings_data["query_terms"].items():
        query_term_embeddings[term.lower()] = np.array(emb)

print(f"Loaded {len(drivers)} drivers, {doc_embeddings.shape[1]}-dim embeddings.")
print(f"Loaded {len(query_term_embeddings)} pre-computed query term embeddings.")

# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def get_query_embedding(query: str) -> np.ndarray | None:
    """
    Try to find a pre-computed embedding for the query.
    Checks exact match first, then finds the closest pre-computed term.
    """
    query_lower = query.lower().strip()

    # Exact match
    if query_lower in query_term_embeddings:
        return query_term_embeddings[query_lower]

    # Find best matching pre-computed term via fuzzy match
    best_term = None
    best_ratio = 0.0
    for term in query_term_embeddings:
        ratio = difflib.SequenceMatcher(None, query_lower, term).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_term = term

    # Only use if it's a close match (>70% similar)
    if best_ratio > 0.7 and best_term:
        return query_term_embeddings[best_term]

    return None


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = 0.75
KEYWORD_BOOST = 0.6
NATIONALITY_BOOST = 0.35
CONFIDENCE_FLOOR = 0.30

DEMONYMS = {
    "dutch": "Netherlands",
    "british": "United Kingdom",
    "english": "United Kingdom",
    "german": "Germany",
    "finnish": "Finland",
    "finn": "Finland",
    "spanish": "Spain",
    "australian": "Australia",
    "mexican": "Mexico",
    "brazilian": "Brazil",
    "french": "France",
    "monegasque": "Monaco",
    "canadian": "Canada",
    "italian": "Italy",
}


def curated_terms(driver: dict) -> list[str]:
    terms = []
    terms += driver.get("memes", [])
    terms += driver.get("aliases", [])
    terms += driver.get("nicknames", [])
    terms += driver.get("search_keywords", [])
    return terms


def nationality_match(query: str, driver_country: str) -> bool:
    query_lower = query.lower()
    for demonym, country in DEMONYMS.items():
        if demonym in query_lower and country == driver_country:
            return True
    return False


def best_fuzzy_match(query: str, terms: list[str]) -> float:
    query = query.lower().strip()
    best = 0.0
    for term in terms:
        ratio = difflib.SequenceMatcher(None, query, term.lower().strip()).ratio()
        best = max(best, ratio)
    return best


def hybrid_rank(query: str, top_k: int = 10) -> list[tuple[int, float]]:
    """Rank drivers using semantic + fuzzy + nationality matching."""

    # Try to get semantic scores from pre-computed query embeddings
    query_embedding = get_query_embedding(query)
    if query_embedding is not None:
        embed_scores = cosine_similarity(query_embedding, doc_embeddings).tolist()
    else:
        # No semantic embedding available — use fuzzy matching only
        embed_scores = [0.0] * len(drivers)

    results = []
    for i, driver in enumerate(drivers):
        embed_score = embed_scores[i]
        fuzzy_score = best_fuzzy_match(query, curated_terms(driver))
        keyword_boost = KEYWORD_BOOST if fuzzy_score >= FUZZY_THRESHOLD else 0.0
        nat_boost = NATIONALITY_BOOST if nationality_match(query, driver.get("country", "")) else 0.0
        final_score = embed_score + keyword_boost + nat_boost
        results.append((i, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/search")
def search(q: str = Query(default="", description="Search query"), top_k: int = Query(default=5, ge=1, le=50)):
    if not q.strip():
        return {"results": [], "query": ""}

    ranked = hybrid_rank(q.strip(), top_k=top_k)

    results = []
    for idx, score in ranked:
        driver = drivers[idx]
        results.append({
            "id": driver["id"],
            "name": driver["name"],
            "country": driver.get("country", ""),
            "team": driver.get("team", ""),
            "nicknames": driver.get("nicknames", []),
            "score": round(score, 4),
            "confident": score >= CONFIDENCE_FLOOR,
        })

    return {"results": results, "query": q}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "pre-computed (all-MiniLM-L6-v2)",
        "drivers_loaded": len(drivers),
        "embedding_dim": doc_embeddings.shape[1],
        "query_terms_loaded": len(query_term_embeddings),
    }


# ---------------------------------------------------------------------------
# Run directly with: python app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
