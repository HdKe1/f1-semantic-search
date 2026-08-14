import React, { useState, useEffect, useRef } from "react";

function SearchBar({ onSearch }) {
  const [input, setInput] = useState("");
  const debounceTimer = useRef(null);

  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    debounceTimer.current = setTimeout(() => {
      onSearch(input);
    }, 400);

    return () => clearTimeout(debounceTimer.current);
  }, [input, onSearch]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    onSearch(input);
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit} role="search">
      <label htmlFor="search-input" className="visually-hidden">
        Search F1 drivers
      </label>
      <input
        id="search-input"
        type="text"
        placeholder='Try "bwoah" or "driver who drinks from his shoe"'
        value={input}
        onChange={(e) => setInput(e.target.value)}
        autoFocus
        autoComplete="off"
      />
      <button type="submit" aria-label="Search">
        Search
      </button>
    </form>
  );
}

export default SearchBar;
