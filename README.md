# F1 Driver Semantic Search 🏎️

A web app that lets you search for F1 drivers by meme, nickname, vibe, or natural language description — powered by sentence-transformers and a hybrid ranking algorithm.

## Project Structure

```
semantic_search/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   ├── drivers_v2.json     # ← YOU NEED TO ADD THIS
│   └── f1_finetuned_model/ # ← Optional: your fine-tuned model from Colab
├── frontend/
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── App.js
│       ├── App.css
│       ├── index.js
│       ├── index.css
│       └── components/
│           ├── SearchBar.js
│           └── ResultsList.js
└── sematic_search_pynb.py  # Original Colab notebook (reference)
```

## Prerequisites

- Python 3.9+
- Node.js 16+
- `drivers_v2.json` (download from your Colab files)
- (Optional) `f1_finetuned_model/` directory (from your Colab training step)

## Setup

### 1. Backend

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your data file here
# cp /path/to/drivers_v2.json .

# (Optional) Place your fine-tuned model here
# cp -r /path/to/f1_finetuned_model .

# Start the API server
python app.py
```

The backend runs on http://localhost:5000. It will:
- Auto-detect `drivers_v2.json` in the current or parent directory
- Use `f1_finetuned_model/` if found, otherwise fall back to `all-MiniLM-L6-v2`

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm start
```

The frontend runs on http://localhost:3000 and proxies API requests to the backend.

## Usage

1. Start the backend first (`python app.py`)
2. Start the frontend (`npm start`)
3. Open http://localhost:3000
4. Type queries like:
   - `"bwoah"` → Kimi Räikkönen
   - `"shoey"` → Daniel Ricciardo
   - `"aggressive dutch champion"` → Max Verstappen
   - `"driver who drinks from his shoe"` → Daniel Ricciardo

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search?q=<query>&top_k=5` | GET | Returns ranked driver matches |
| `/api/health` | GET | Server status & model info |

## How It Works

The search uses a **hybrid ranking** approach:
1. **Semantic embeddings** — encodes your query with sentence-transformers and compares cosine similarity against each driver's "vibe text"
2. **Fuzzy keyword matching** — boosts drivers whose curated memes/nicknames/aliases closely match the query
3. **Nationality matching** — small boost when the query contains a demonym matching a driver's country

Scores below the confidence floor (0.30) are marked as low-confidence.
