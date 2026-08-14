import React from "react";

function ResultsList({ results }) {
  if (!results.length) return null;

  // Opacity: #1 result is fully opaque, last result fades out
  const maxOpacity = 1.0;
  const minOpacity = 0.25;

  return (
    <ul className="results-list" aria-label="Search results">
      {results.map((driver, index) => {
        const opacity =
          results.length === 1
            ? maxOpacity
            : maxOpacity - ((maxOpacity - minOpacity) * index) / (results.length - 1);

        return (
          <li
            key={driver.id}
            className="result-item"
            style={{ opacity }}
          >
            <div className="result-name">{driver.name}</div>
            <div className="result-meta">
              {driver.country && <span>{driver.country}</span>}
              {driver.team && <span>{driver.team}</span>}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export default ResultsList;
