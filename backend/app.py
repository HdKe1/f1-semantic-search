"""
FastAPI backend for F1 Driver Semantic Search.

Uses Hugging Face Inference API for embeddings (no local model loading).
Expects drivers_v2.json in the same directory (or parent directory).

Run with: uvicorn app:app --reload --port 5001
"""

import os
import json
import difflib
import httpx
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
# Hugging Face Inference API config
# ---------------------------------------------------------------------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL = os.environ.get("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"

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
# Embedding via HF Inference API
# ---------------------------------------------------------------------------
def get_headers():
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return headers


async def get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings from HF Inference API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            HF_API_URL,
            headers=get_headers(),
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        return np.array(response.json())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)


# ---------------------------------------------------------------------------
# Pre-compute driver embeddings at startup
# ---------------------------------------------------------------------------
doc_embeddings: np.ndarray | None = None


@app.on_event("startup")
async def startup_event():
    """Compute driver embeddings once at startup via HF API."""
    global doc_embeddings
    print(f"Computing embeddings for {len(corpus)} drivers via HF API ({HF_MODEL})...")
    doc_embeddings = await get_embeddings(corpus)
    print(f"Embeddings ready: shape {doc_embeddings.shape}")


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


async def hybrid_rank(query: str, top_k: int = 10) -> list[tuple[int, float]]:
    query_embedding = (await get_embeddings([query]))[0]
    embed_scores = cosine_similarity(query_embedding, doc_embeddings).tolist()

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
async def search(q: str = Query(default="", description="Search query"), top_k: int = Query(default=5, ge=1, le=50)):
    if not q.strip():
        return {"results": [], "query": ""}

    ranked = await hybrid_rank(q.strip(), top_k=top_k)

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
        "model": HF_MODEL,
        "drivers_loaded": len(drivers),
        "embeddings_ready": doc_embeddings is not None,
    }


# ---------------------------------------------------------------------------
# Run directly with: python app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
