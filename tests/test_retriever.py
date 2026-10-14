import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from unittest import mock

# Import the retriever implementation from the project.
# Adjust the import path according to the actual package layout.
# It is assumed that the project exposes a `CodeRetriever` class with the
# methods used below.
from retriever import CodeRetriever


@pytest.fixture
def dummy_embeddings():
    """
    Returns a deterministic embedding function that maps known strings
    to fixed vectors. This avoids loading heavy transformer models
    during testing and makes similarity expectations explicit.
    """
    # Mapping from text to a 3‑dimensional one‑hot vector that encodes language.
    mapping = {
        "def add(a, b): return a + b": np.array([1.0, 0.0, 0.0]),
        "function add(a, b) { return a + b; }": np.array([0.0, 1.0, 0.0]),
        "int add(int a, int b) { return a + b; }": np.array([0.0, 0.0, 1.0]),
        "python add function": np.array([1.0, 0.0, 0.0]),
        "javascript add function": np.array([0.0, 1.0, 0.0]),
        "java add function": np.array([0.0, 0.0, 1.0]),
    }

    def encode(texts: List[str]) -> np.ndarray:
        """Encode a list of strings into deterministic vectors."""
        vectors = []
        for txt in texts:
            # Use the lower‑cased version as a key to be tolerant to case.
            key = txt.strip().lower()
            vec = mapping.get(key)
            if vec is None:
                # Unknown text -> zero vector (ensures no accidental similarity).
                vec = np.zeros(3, dtype=np.float32)
            vectors.append(vec)
        return np.stack(vectors)

    mock_model = mock.Mock()
    mock_model.encode.side_effect = encode
    return mock_model


@pytest.fixture
def code_snippets():
    """A small multilingual code corpus used for indexing."""
    return [
        "def add(a, b): return a + b",                     # Python
        "function add(a, b) { return a + b; }",           # JavaScript
        "int add(int a, int b) { return a + b; }",        # Java
    ]


def test_build_index_and_search(dummy_embeddings, code_snippets):
    """
    Verify that the retriever can build a FAISS index from a list of code
    snippets and retrieve the most relevant snippet for a given query.
    """
    # Use a temporary directory so the test does not leave artefacts.
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "faiss.index"

        # Initialise the retriever with the mocked embedding model.
        retriever = CodeRetriever(embedding_model=dummy_embeddings, index_path=str(index_path))

        # Build the index from the provided snippets.
        retriever.build_index(code_snippets)

        # Ensure that an index file was created.
        assert index_path.is_file(), "FAISS index file was not written to disk."

        # Perform a search that should match the Python snippet.
        query = "python add function"
        results = retriever.search(query, k=3)

        # The retriever is expected to return a list of (doc, score) tuples.
        assert isinstance(results, list), "search() should return a list."
        assert len(results) == 3, "search() should return k results."

        # The top‑ranked document must be the Python snippet.
        top_doc, top_score = results[0]
        assert top_doc == code_snippets[0], "Top result does not match expected Python snippet."
        # Scores should be non‑negative and decreasing.
        scores = [score for _, score in results]
        assert all(s >= 0 for s in scores), "All relevance scores should be non‑negative."
        assert scores == sorted(scores, reverse=True), "Scores should be sorted in descending order."


def test_compute_relevance_score(dummy_embeddings, code_snippets):
    """
    Directly test the relevance scoring function to ensure that it reflects
    the expected similarity between a query and each code snippet.
    """
    retriever = CodeRetriever(embedding_model=dummy_embeddings)

    query = "python add function"
    # Compute relevance for each snippet.
    scores = [retriever.compute_relevance(query, doc) for doc in code_snippets]

    # The score for the Python snippet should be the highest.
    python_score = scores[0]
    other_scores = scores[1:]

    assert all(python_score > s for s in other_scores), (
        "Python snippet should have a higher relevance score than other languages."
    )

    # Verify that scores are finite numbers.
    assert all(np.isfinite(scores)), "All relevance scores must be finite numbers."
