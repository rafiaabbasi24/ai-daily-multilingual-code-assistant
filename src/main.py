import os
import json
import click
import torch
import faiss
import numpy as np
from typing import List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)

# ----------------------------------------------------------------------
# Helper functions for retrieval and generation
# ----------------------------------------------------------------------


def _load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a sentence‑transformer model used to embed queries and documents.

    Returns
    -------
    tokenizer, model, device
        The tokenizer, the model (in eval mode) and the torch device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model, device


def _embed_text(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute L2‑normalized embeddings for a list of texts.

    Parameters
    ----------
    texts : List[str]
        Input strings.
    tokenizer, model, device : as returned by ``_load_embedding_model``.
    batch_size : int, optional
        Number of texts processed at once.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), hidden_dim)``.
    """
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            outputs = model(**encoded, output_hidden_states=True)
            # Use the CLS token (first token) from the last hidden state
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def load_faiss_index(
    index_path: str, embeddings_path: str
) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Load a FAISS index and the corresponding document corpus.

    The ``embeddings_path`` is expected to be a ``.npz`` file containing:
        - ``embeddings``: a 2‑D float32 array.
        - ``documents``: a JSON‑encoded list of strings.

    Parameters
    ----------
    index_path : str
        Path to the FAISS index file (produced with ``faiss.write_index``).
    embeddings_path : str
        Path to the ``.npz`` file storing embeddings and documents.

    Returns
    -------
    index : faiss.IndexFlatL2
        The loaded FAISS index.
    documents : List[str]
        The corpus of raw documents (code snippets, explanations, …).
    """
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

    index = faiss.read_index(index_path)

    data = np.load(embeddings_path, allow_pickle=True)
    if "documents" not in data:
        raise KeyError("Embeddings file must contain a 'documents' entry")
    documents = json.loads(data["documents"].tobytes().decode("utf-8"))
    return index, documents


def retrieve_documents(
    query: str,
    top_k: int,
    index: faiss.IndexFlatL2,
    documents: List[str],
    embed_tokenizer,
    embed_model,
    embed_device: torch.device,
) -> List[str]:
    """
    Retrieve the *top_k* most similar documents for a given query.

    Parameters
    ----------
    query : str
        The user query.
    top_k : int
        Number of documents to return.
    index : faiss.IndexFlatL2
        Pre‑built FAISS index.
    documents : List[str]
        Corpus aligned with the index.
    embed_tokenizer, embed_model, embed_device : as returned by ``_load_embedding_model``.

    Returns
    -------
    List[str]
        The retrieved documents ordered by relevance (most relevant first).
    """
    query_emb = _embed_text([query], embed_tokenizer, embed_model, embed_device)
    distances, indices = index.search(query_emb, top_k)
    retrieved = [documents[idx] for idx in indices[0] if idx < len(documents)]
    return retrieved


def _load_generator_model(
    model_name: str = "bigcode/starcoderbase",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """
    Load a causal language model for code generation.

    Returns
    -------
    tokenizer, model, device
        The tokenizer, the model (in eval mode) and the torch device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def generate_code(
    query: str,
    language: str,
    retrieved_docs: List[str],
    gen_tokenizer,
    gen_model,
    gen_device: torch.device,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate code using retrieved context.

    The prompt is built as:

    ```
    # Language: <language>
    # Question: <query>
    # Context:
    <doc1>
    <doc2>
    ...
    # Answer:
    ```

    Parameters
    ----------
    query : str
        User's request.
    language : str
        Target programming language (e.g., ``python``).
    retrieved_docs : List[str]
        Contextual snippets to prepend to the prompt.
    gen_tokenizer, gen_model, gen_device : as returned by ``_load_generator_model``.
    max_new_tokens : int, optional
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The generated code (raw string, without the prompt).
    """
    prompt_lines = [
        f"# Language: {language}",
        f"# Question: {query}",
        "# Context:",
    ]
    prompt_lines.extend(retrieved_docs)
    prompt_lines.append("# Answer:")
    prompt = "\n".join(prompt_lines)

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_device)

    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=gen_tokenizer.eos_token_id,
        )
    generated_text = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the original prompt to keep only the model's continuation
    answer = generated_text[len(prompt) :].strip()
    return answer


# ----------------------------------------------------------------------
# Click command line interface
# ----------------------------------------------------------------------


@click.group()
def cli():
    """Multilingual Code Assistant with Retrieval‑Augmented Generation."""
    pass


@cli.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="Natural‑language description of the desired code.",
)
@click.option(
    "--language",
    "-l",
    default="python",
    show_default=True,
    help="Target programming language for the generated code.",
)
@click.option(
    "--top-k",
    "-k",
    default=5,
    show_default=True,
    type=int,
    help="Number of retrieved documents to use as context.",
)
@click.option(
    "--index-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the FAISS index file.",
)
@click.option(
    "--embeddings-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the .npz file containing document embeddings and texts.",
)
@click.option(
    "--model-name",
    default="bigcode/starcoderbase",
    show_default=True,
    help="HuggingFace model name for code generation.",
)
@click.option(
    "--embed-model-name",
    default="sentence-transformers/all-MiniLM-L6-v2",
    show_default=True,
    help="Model name for embedding queries and documents.",
)
@click.option(
    "--max-new-tokens",
    default=256,
    show_default=True,
    type=int,
    help="Maximum number of tokens to generate for the answer.",
)
def code(
    query,
    language,
    top_k,
    index_path,
    embeddings_path,
    model_name,
    embed_model_name,
    max_new_tokens,
):
    """
    Generate code from a natural‑language description using retrieval‑augmented generation.
    """
    click.echo("Loading embedding model...")
    embed_tokenizer, embed_model, embed_device = _load_embedding_model(embed_model_name)

    click.echo("Loading FAISS index and document corpus...")
    index, documents = load_faiss_index(index_path, embeddings_path)

    click.echo(f"Retrieving top {top_k} relevant documents...")
    retrieved = retrieve_documents(
        query,
        top_k,
        index,
        documents,
        embed_tokenizer,
        embed_model,
        embed_device,
    )
    if not retrieved:
        click.echo("No relevant documents were found; proceeding without context.", err=True)

    click.echo("Loading generation model...")
    gen_tokenizer, gen_model, gen_device = _load_generator_model(model_name)

    click.echo("Generating code...")
    answer = generate_code(
        query,
        language,
        retrieved,
        gen_tokenizer,
        gen_model,
        gen_device,
        max_new_tokens=max_new_tokens,
    )

    click.secho("\n--- Generated Code ---\n", fg="green")
    click.echo(answer)


if __name__ == "__main__":
    cli()