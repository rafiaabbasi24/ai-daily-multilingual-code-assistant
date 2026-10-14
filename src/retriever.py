"""
src/retriever.py

FAISS‑based vector store loader and similarity search for code snippets.

The module provides:
- :class:`Retriever`: a high‑level class that can embed a list of code
  snippets, build a FAISS index, persist it to disk and perform similarity
  queries.
- A small Click‑based CLI for building an index from a JSONL file and for
  querying the index.

The implementation is deliberately lightweight and does not depend on the
full ``sentence‑transformers`` library – it uses a HuggingFace
``AutoModel``/``AutoTokenizer`` pair together with mean‑pooling to obtain
embeddings.  This keeps the dependency footprint small while still offering
good quality embeddings for many code‑related models (e.g. ``microsoft/codebert-base``).

Typical usage::

    # Build an index
    python -m src.retriever build-index \
        --model-name microsoft/codebert-base \
        --corpus data/snippets.jsonl \
        --index-path data/faiss.index \
        --meta-path data/meta.pkl

    # Query the index
    python -m src.retriever query \
        --index-path data/faiss.index \
        --meta-path data/meta.pkl \
        --model-name microsoft/codebert-base \
        --query "def add(a, b):" \
        --top-k 5
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import click
import faiss
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _mean_pooling(
    model_output: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Perform mean pooling on token embeddings.

    Args:
        model_output: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_len)

    Returns:
        Tensor of shape (batch_size, hidden_dim) representing the pooled
        sentence embeddings.
    """
    token_embeddings = model_output  # (bs, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def _load_jsonl(path: Path) -> List[dict]:
    """
    Load a JSONL file where each line is a JSON object.

    Returns:
        List of dictionaries.
    """
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# --------------------------------------------------------------------------- #
# Retriever class
# --------------------------------------------------------------------------- #
class Retriever:
    """
    A class that encapsulates embedding generation, FAISS index construction,
    persistence and similarity search for code snippets.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device | None = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialise the Retriever.

        Args:
            model_name: HuggingFace model identifier (e.g. ``microsoft/codebert-base``).
            device: ``cpu`` or ``cuda``. If ``None`` auto‑detects CUDA if available.
            batch_size: Number of texts processed per forward pass.
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer
        logger.info("Loading model %s on %s", model_name, self.device)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        # FAISS index (initially empty)
        self.index: faiss.Index | None = None
        # Metadata: list of original snippets (or any user‑provided dict)
        self.metadata: List[dict] = []

    # --------------------------------------------------------------------- #
    # Embedding utilities
    # --------------------------------------------------------------------- #
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """
        Compute embeddings for an iterable of texts.

        Args:
            texts: Iterable of strings.

        Returns:
            Numpy array of shape (len(texts), hidden_dim) with dtype ``float32``.
        """
        embeddings: List[np.ndarray] = []
        batch: List[str] = []

        for text in texts:
            batch.append(text)
            if len(batch) == self.batch_size:
                embeddings.append(self._embed_batch(batch))
                batch = []

        if batch:
            embeddings.append(self._embed_batch(batch))

        return np.vstack(embeddings).astype(np.float32)

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        Internal helper to embed a single batch.

        Returns:
            Numpy array of shape (batch_size, hidden_dim).
        """
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            model_output = self.model(input_ids, attention_mask=attention_mask)[0]
            pooled = _mean_pooling(model_output, attention_mask)

        return pooled.cpu().numpy()

    # --------------------------------------------------------------------- #
    # Index handling
    # --------------------------------------------------------------------- #
    def build_index(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine",
        nlist: int = 100,
    ) -> None:
        """
        Build a FAISS index from pre‑computed embeddings.

        Args:
            embeddings: Numpy array of shape (n_items, dim) with dtype ``float32``.
            metric: Distance metric – ``"l2"``, ``"inner_product"``, or ``"cosine"``.
            nlist: Number of clusters for IVF index (only used for large corpora).
        """
        dim = embeddings.shape[1]
        if metric == "l2":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        elif metric in ("inner_product", "cosine"):
            # For cosine similarity we normalise vectors and use inner product.
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Train the IVF index (required before adding vectors)
        logger.info("Training IVF index with %d vectors", embeddings.shape[0])
        index.train(embeddings)

        # Add vectors
        logger.info("Adding vectors to FAISS index")
        index.add(embeddings)

        # If cosine metric, ensure vectors are normalised (FAISS does not store that)
        if metric == "cosine":
            # Normalise the index vectors in place
            faiss.normalize_L2(index.xb)

        self.index = index

    def save_index(self, index_path: Path) -> None:
        """
        Persist the FAISS index to disk.

        Args:
            index_path: Destination path (will be overwritten if exists).
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        logger.info("Saving FAISS index to %s", index_path)
        faiss.write_index(self.index, str(index_path))

    def load_index(self, index_path: Path) -> None:
        """
        Load a FAISS index from disk.

        Args:
            index_path: Path to the saved FAISS index.
        """
        logger.info("Loading FAISS index from %s", index_path)
        self.index = faiss.read_index(str(index_path))

    def save_metadata(self, meta_path: Path) -> None:
        """
        Persist the metadata (list of dicts) to disk using pickle.

        Args:
            meta_path: Destination path.
        """
        logger.info("Saving metadata (%d items) to %s", len(self.metadata), meta_path)
        with meta_path.open("wb") as f:
            pickle.dump(self.metadata, f)

    def load_metadata(self, meta_path: Path) -> None:
        """
        Load metadata from a pickle file.

        Args:
            meta_path: Path to the pickle file.
        """
        logger.info("Loading metadata from %s", meta_path)
        with meta_path.open("rb") as f:
            self.metadata = pickle.load(f)

    # --------------------------------------------------------------------- #
    # Search
    # --------------------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[Tuple[dict, float]]:
        """
        Perform a similarity search for a single query string.

        Args:
            query: The query text (e.g., a code snippet).
            top_k: Number of nearest neighbours to return.
            metric: Distance metric used during index construction.

        Returns:
            List of tuples ``(metadata_item, score)`` ordered by decreasing
            similarity (or decreasing relevance for inner product).
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call load_index() first.")
        if not self.metadata:
            raise RuntimeError("Metadata not loaded. Call load_metadata() first.")

        query_emb = self.embed_texts([query])  # shape (1, dim)

        if metric == "cosine":
            # Normalise query for cosine similarity (inner product after normalisation)
            faiss.normalize_L2(query_emb)

        distances, indices = self.index.search(query_emb, top_k)
        distances = distances[0]  # (top_k,)
        indices = indices[0]      # (top_k,)

        results: List[Tuple[dict, float]] = []
        for idx, dist in zip(indices, distances):
            if idx == -1:
                continue  # FAISS returns -1 for empty results
            meta = self.metadata[idx]
            # For inner product / cosine, higher is better; for L2 lower is better.
            score = float(dist) if metric != "cosine" else float(dist)
            results.append((meta, score))

        return results

    # --------------------------------------------------------------------- #
    # Convenience pipeline
    # --------------------------------------------------------------------- #
    def build_from_corpus(
        self,
        corpus_path: Path,
        index_path: Path,
        meta_path: Path,
        metric: str = "cosine",
        nlist: int = 100,
    ) -> None:
        """
        End‑to‑end pipeline: load a JSONL corpus, embed, build index,
        and persist both index and metadata.

        The expected JSONL format is one object per line with at least a
        ``"code"`` field containing the snippet.  All fields are stored as
        metadata.

        Args:
            corpus_path: Path to the JSONL file.
            index_path: Destination for the FAISS index.
            meta_path: Destination for the metadata pickle.
            metric: Distance metric for the index.
            nlist: IVF cluster count.
        """
        logger.info("Loading corpus from %s", corpus_path)
        records = _load_jsonl(corpus_path)

        if not records:
            raise ValueError("Corpus is empty.")

        # Store the full records as metadata
        self.metadata = records

        # Extract the code field for embedding
        codes = [rec.get("code", "") for rec in records]

        logger.info("Embedding %d code snippets", len(codes))
        embeddings = self.embed_texts(tqdm(codes, desc="Embedding"))

        self.build_index(embeddings, metric=metric, nlist=nlist)
        self.save_index(index_path)
        self.save_metadata(meta_path)
        logger.info("Indexing complete.")


# --------------------------------------------------------------------------- #
# Click CLI
# --------------------------------------------------------------------------- #
@click.group()
def cli() -> None:
    """FAISS Retriever command‑line interface."""
    pass


@cli.command(name="build-index")
@click.option(
    "--model-name",
    required=True,
    help="HuggingFace model identifier for embedding generation.",
)
@click.option(
    "--corpus",
    "corpus_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a JSONL file containing code snippets (field ``code``).",
)
@click.option(
    "--index-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Destination path for the FAISS index.",
)
@click.option(
    "--meta-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Destination path for the metadata pickle.",
)
@click.option(
    "--metric",
    type=click.Choice(["cosine", "l2", "inner_product"], case_sensitive=False),
    default="cosine",
    show_default=True,
    help="Distance metric for the FAISS index.",
)
@click.option(
    "--nlist",
    type=int,
    default=100,
    show_default=True,
    help="Number of IVF clusters (higher -> faster queries, more memory).",
)
def build_index_cli(
    model_name: str,
    corpus_path: Path,
    index_path: Path,
    meta_path: Path,
    metric: str,
    nlist: int,
) -> None:
    """Build a FAISS index from a corpus."""
    retriever = Retriever(model_name=model_name)
    retriever.build_from_corpus(
        corpus_path=corpus_path,
        index_path=index_path,
        meta_path=meta_path,
        metric=metric,
        nlist=nlist,
    )
    click.echo(f"Index saved to {index_path}")
    click.echo(f"Metadata saved to {meta_path}")


@cli.command(name="query")
@click.option(
    "--model-name",
    required=True,
    help="HuggingFace model identifier (must match the one used for indexing).",
)
@click.option(
    "--index-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the FAISS index file.",
)
@click.option(
    "--meta-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the metadata pickle file.",
)
@click.option(
    "--query",
    "query_text",
    required=True,
    help="Query string (e.g., a code snippet).",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    show_default=True,
    help="Number of nearest neighbours to retrieve.",
)
@click.option(
    "--metric",
    type=click.Choice(["cosine", "l2", "inner_product"], case_sensitive=False),
    default="cosine",
    show_default=True,
    help="Distance metric used during index construction.",
)
def query_cli(
    model_name: str,
    index_path: Path,
    meta_path: Path,
    query_text: str,
    top_k: int,
    metric: str,
) -> None:
    """Query a FAISS index."""
    retriever = Retriever(model_name=model_name)
    retriever.load_index(index_path)
    retriever.load_metadata(meta_path)

    results = retriever.search(query_text, top_k=top_k, metric=metric)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Top {len(results)} results:")
    for rank, (meta, score) in enumerate(results, start=1):
        # Show a short preview of the code snippet
        code_preview = meta.get("code", "").strip().replace("\n", " ")[:200]
        click.echo(f"{rank}. Score: {score:.4f}")
        click.echo(f"   Code: {code_preview}")
        # Optionally display additional metadata fields
        extra = {k: v for k, v in meta.items() if k != "code"}
        if extra:
            click.echo(f"   Meta: {extra}")
        click.echo("-" * 40)


if __name__ == "__main__":
    cli()