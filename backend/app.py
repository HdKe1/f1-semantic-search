"""
FastAPI backend for F1 Driver Semantic Search.

Exposes the hybrid_rank search logic as a REST endpoint.
Expects drivers_v2.json in the same directory (or parent directory).
Optionally uses a fine-tuned model from ./f1_finetuned_model/ if available.

Run with: uvicorn app:app --reload --port 5000
"""

import os
import json
import difflib
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

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
# Load model (fine-tuned if available, otherwise base)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", None)
if MODEL_PATH is None:
    if os.path.exists("f1_finetuned_model"):
        MODEL_PATH = "f1_finetuned_model"
    elif os.path.exists("../f1_finetuned_model"):
        MODEL_PATH = "../f1_finetuned_model"
    else:
        MODEL_PATH = "all-MiniLM-L6-v2"

print(f"Loading model from: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH)
doc_embeddings = model.encode(corpus, convert_to_tensor=True)
print(f"Loaded {len(drivers)} drivers, embeddings ready.")

# ---------------------------------------------------------------------------
# Search logic (same as Colab notebook)
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
    query_embedding = model.encode(query, convert_to_tensor=True)
    embed_scores = util.cos_sim(query_embedding, doc_embeddings)[0].tolist()

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
        "model": MODEL_PATH,
        "drivers_loaded": len(drivers),
    }


# ---------------------------------------------------------------------------
# Run directly with: python app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
