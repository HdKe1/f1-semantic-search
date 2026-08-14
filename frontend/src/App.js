import React, { useState, useCallback } from "react";
import SearchBar from "./components/SearchBar";
import ResultsList from "./components/ResultsList";
import "./App.css";

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [error, setError] = useState(null);

  const handleSearch = useCallback(async (searchQuery) => {
    setQuery(searchQuery);
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const apiBase = process.env.REACT_APP_API_URL || "";
      const response = await fetch(
        `${apiBase}/api/search?q=${encodeURIComponent(searchQuery)}&top_k=5`
      );
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>F1 Semantic Search</h1>
        <p className="subtitle">search by meme, nickname, vibe, or description</p>
      </header>

      <main className="app-main">
        <SearchBar onSearch={handleSearch} />

        {loading && <div className="loading">searching...</div>}
        {error && <div className="error">{error}</div>}
        {!loading && !error && query && results.length > 0 && (
          <ResultsList results={results} />
        )}
        {!loading && !error && query && results.length === 0 && (
          <div className="no-results">no results</div>
        )}
      </main>

      <footer className="app-footer">
        <p>try: bwoah, shoey, aggressive dutch champion, smooth operator</p>
      </footer>
    </div>
  );
}

export default App;
