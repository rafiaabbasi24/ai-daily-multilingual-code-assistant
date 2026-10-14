# Multilingual Code Assistant with Retrieval‑Augmented Generation

## Project Overview

The **Multilingual Code Assistant** is an open‑source LLM‑powered tool that helps developers write, understand, and refactor code across multiple programming languages. It combines a large language model (LLM) with a vector store (FAISS) to retrieve relevant code snippets, documentation, and examples, then augments generation with this context.  

Key features:

- **Multilingual support** – works with Python, JavaScript, Java, C++, Rust, Go, and more.
- **Retrieval‑augmented generation (RAG)** – fast similarity search over a local FAISS index of curated codebases and documentation.
- **Interactive CLI** – built with Click for a smooth terminal experience.
- **Extensible architecture** – plug in any Hugging Face Transformer model and customize the index pipeline.

The project is built on **Python**, **PyTorch**, **🤗 Transformers**, **FAISS**, and **Click**. It is ready for production use and can be integrated into IDE extensions, CI pipelines, or used as a standalone assistant.

## Quick Start

### Prerequisites

- Python 3.9 or newer
- A CUDA‑enabled GPU (optional but recommended for faster inference)
- Git (to clone the repository)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/multilingual-code-assistant.git
cd multilingual-code-assistant

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and its dependencies
pip install -e .
```

The `-e` flag installs the package in editable mode, allowing you to modify the source code without reinstalling.

### Model & Index Setup

1. **Download a pretrained model** (e.g., `bigcode/starcoder`):
   ```bash
   python -m multilingual_code_assistant download-model --model bigcode/starcoder
   ```

2. **Build a FAISS index** from a directory of source files (e.g., a local copy of the Linux kernel):
   ```bash
   python -m multilingual_code_assistant build-index \
       --source-dir /path/to/codebase \
       --index-path data/faiss_index.bin
   ```

   The command extracts code chunks, encodes them with the model’s embedding head, and stores them in a FAISS index for fast similarity search.

## Usage Examples

### Interactive CLI

```bash
python -m multilingual_code_assistant chat \
    --model-dir models/starcoder \
    --index-path data/faiss_index.bin \
    --language python
```

The assistant will:

1. Retrieve the top‑k most relevant code snippets from the FAISS index.
2. Append them as context to the LLM prompt.
3. Generate a response that is grounded in real code.

### One‑off Generation

```bash
python -m multilingual_code_assistant generate \
    --model-dir models/starcoder \
    --index-path data/faiss_index.bin \
    --prompt "Write a Rust function that parses a CSV file into a struct." \
    --max-new-tokens 256
```

### Python API

```python
from multilingual_code_assistant import Assistant, RetrievalEngine, ModelLoader

# Load model and tokenizer
loader = ModelLoader.from_pretrained("bigcode/starcoder")
model, tokenizer = loader.load()

# Load FAISS index
retriever = RetrievalEngine(
    index_path="data/faiss_index.bin",
    embedding_model=model,
    tokenizer=tokenizer,
)

assistant = Assistant(model=model, tokenizer=tokenizer, retriever=retriever)

response = assistant.ask(
    prompt="Explain the time complexity of quicksort in JavaScript.",
    language="javascript",
    top_k=5,
)

print(response)
```

## Configuration

All CLI commands accept a `--config` option pointing to a YAML file. Example `config.yaml`:

```yaml
model:
  name_or_path: bigcode/starcoder
  device: cuda
retrieval:
  index_path: data/faiss_index.bin
  top_k: 5
generation:
  max_new_tokens: 256
  temperature: 0.7
```

Run with:

```bash
python -m multilingual_code_assistant chat --config config.yaml
```

## Contribution Guide

We welcome contributions! Please follow these steps:

1. **Fork the repository** and clone your fork.
2. **Create a feature branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```
   This installs testing tools (`pytest`), linting (`ruff`), and type checking (`mypy`).

4. **Write tests** in the `tests/` directory. All new functionality must be covered by unit tests.

5. **Run the test suite**:
   ```bash
   pytest -q
   ```

6. **Lint and type‑check**:
   ```bash
   ruff check .
   mypy multilingual_code_assistant
   ```

7. **Commit** with a clear message and **push** to your fork.
8. **Open a Pull Request** targeting the `main` branch. Include a concise description of the change and reference any related issues.

### Code Style

- Follow PEP 8 (use `ruff` for automatic formatting).
- Use type hints throughout the codebase.
- Keep docstrings in Google style; they are rendered in the generated documentation.

### Release Process

Releases are managed via GitHub Actions:

1. Merge all approved PRs into `main`.
2. Tag a new version:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
3. The CI pipeline will publish the package to PyPI and generate release notes.

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.

## Acknowledgements

- The **🤗 Transformers** library for model handling.
- **FAISS** for efficient similarity search.
- **Click** for the command‑line interface.
- The open‑source community for the many code corpora used to build the retrieval index.