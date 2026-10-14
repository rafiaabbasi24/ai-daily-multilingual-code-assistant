import json
import os
from typing import List, Tuple, Optional

import click
import faiss
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


class RetrievalAugmentedGenerator:
    """
    Core class that combines a retrieval system (FAISS) with a language model
    to produce answers based on retrieved context.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the generative LLM (e.g., "meta-llama/Meta-Llama-3-8B").
    embed_model_name : str
        HuggingFace model identifier for the embedding model (e.g., "sentence-transformers/all-MiniLM-L6-v2").
    faiss_index_path : str
        Path to a pre‑built FAISS index file (produced with `faiss.write_index`).
    docs_path : str
        Path to a JSON file containing a list of documents (strings) that correspond
        to the vectors stored in the FAISS index.
    device : Optional[str], default ``None``
        Torch device to use (e.g., "cpu", "cuda"). If ``None``, automatically selects CUDA if available.
    prompt_template : Optional[str], default ``None``
        Template used to construct the final prompt. The template may contain
        ``{question}`` and ``{context}`` placeholders. If ``None`` a sensible default
        is used.
    """

    def __init__(
        self,
        model_name: str,
        embed_model_name: str,
        faiss_index_path: str,
        docs_path: str,
        device: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load the generative model and its tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

        # Load the embedding model (used for query encoding)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name, use_fast=True)
        self.embed_model = AutoModel.from_pretrained(embed_model_name)
        self.embed_model.to(self.device)
        self.embed_model.eval()

        # Load FAISS index
        if not os.path.isfile(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
        self.index = faiss.read_index(faiss_index_path)

        # Load documents (must be a JSON array of strings)
        if not os.path.isfile(docs_path):
            raise FileNotFoundError(f"Documents file not found at {docs_path}")
        with open(docs_path, "r", encoding="utf-8") as f:
            self.documents: List[str] = json.load(f)

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                f"Number of vectors in FAISS index ({self.index.ntotal}) does not match "
                f"number of documents ({len(self.documents)})."
            )

        # Prompt template
        self.prompt_template = prompt_template or (
            "You are a multilingual code assistant. Use the following context to answer the question.\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> torch.Tensor:
        """
        Compute L2‑normalized embeddings for a batch of texts using the embedding model.

        Returns
        -------
        torch.Tensor
            Tensor of shape (len(texts), hidden_size) on ``self.device``.
        """
        encoded = self.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        outputs = self.embed_model(**encoded)
        # For most sentence‑transformer models the CLS token representation is suitable.
        embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        # L2‑normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()  # FAISS expects numpy arrays on CPU

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the top‑k most similar documents for a given query.

        Parameters
        ----------
        query : str
            The user query.
        k : int
            Number of documents to retrieve.

        Returns
        -------
        List[Tuple[str, float]]
            List of (document, score) tuples ordered by decreasing similarity.
        """
        query_emb = self._embed([query])  # shape (1, d)
        distances, indices = self.index.search(query_emb, k)  # distances are L2 squared
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            # Convert L2 distance to a similarity score (higher is better)
            similarity = 1 / (1 + dist)
            results.append((doc, similarity))
        return results

    def _build_prompt(self, question: str, retrieved_docs: List[Tuple[str, float]]) -> str:
        """
        Construct the final prompt by injecting the question and the retrieved context.

        Parameters
        ----------
        question : str
            The user query.
        retrieved_docs : List[Tuple[str, float]]
            List of retrieved documents with their scores.

        Returns
        -------
        str
            The formatted prompt.
        """
        # Concatenate documents; optionally we could include scores.
        context = "\n---\n".join(doc for doc, _ in retrieved_docs)
        prompt = self.prompt_template.format(question=question, context=context)
        return prompt

    @torch.no_grad()
    def generate(
        self,
        query: str,
        k: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **generation_kwargs,
    ) -> str:
        """
        Generate an answer for ``query`` using retrieved context.

        Parameters
        ----------
        query : str
            The user question.
        k : int
            Number of retrieved documents.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling probability.
        repetition_penalty : float
            Penalty to avoid repeated tokens.
        generation_kwargs : dict
            Additional arguments passed to ``model.generate``.

        Returns
        -------
        str
            The generated answer (text after the prompt).
        """
        retrieved = self.retrieve(query, k=k)
        prompt = self._build_prompt(query, retrieved)

        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids
        input_ids = input_ids.to(self.device)

        # Generation configuration
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **generation_kwargs,
        )

        output_ids = self.model.generate(input_ids, generation_config=gen_cfg, do_sample=True)
        # Remove the prompt part
        generated_text = self.tokenizer.decode(output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)
        return generated_text.strip()


@click.command()
@click.option(
    "--model-name",
    default="meta-llama/Meta-Llama-3-8B",
    help="HuggingFace identifier for the generative model.",
)
@click.option(
    "--embed-model-name",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="HuggingFace identifier for the embedding model.",
)
@click.option("--faiss-index", required=True, type=click.Path(exists=True), help="Path to FAISS index file.")
@click.option("--docs", required=True, type=click.Path(exists=True), help="Path to JSON file with documents.")
@click.option("--query", required=True, help="User question to answer.")
@click.option("--top-k", default=5, help="Number of retrieved documents.")
@click.option("--max-new-tokens", default=256, help="Maximum tokens to generate.")
@click.option("--temperature", default=0.7, help="Sampling temperature.")
@click.option("--top-p", default=0.9, help="Top‑p nucleus sampling.")
@click.option("--device", default=None, help="Torch device (e.g., cpu, cuda).")
def main(
    model_name: str,
    embed_model_name: str,
    faiss_index: str,
    docs: str,
    query: str,
    top_k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: Optional[str],
) -> None:
    """
    CLI entry point for the multilingual code assistant.
    """
    rag = RetrievalAugmentedGenerator(
        model_name=model_name,
        embed_model_name=embed_model_name,
        faiss_index_path=faiss_index,
        docs_path=docs,
        device=device,
    )
    answer = rag.generate(
        query,
        k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    click.echo("\n=== Answer ===\n")
    click.echo(answer)


if __name__ == "__main__":
    main()
