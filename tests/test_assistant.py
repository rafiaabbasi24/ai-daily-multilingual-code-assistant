import pytest

# The project package is assumed to be named `multilingual_code_assistant`.
# It provides two public utilities:
#   - `build_prompt(user_query: str, language: str) -> str`
#   - `parse_response(raw_output: str) -> str`
# Adjust the import paths if the actual module layout differs.

from multilingual_code_assistant.prompt import build_prompt
from multilingual_code_assistant.response import parse_response


def test_build_prompt_basic():
    """
    Verify that a normal prompt contains the expected markers,
    language specification and the original user query.
    """
    user_query = "def hello():\n    print('Hello, world!')"
    language = "python"
    prompt = build_prompt(user_query, language=language)

    # The prompt should start with a system instruction.
    assert prompt.lower().startswith("you are a multilingual code assistant")

    # The language must be indicated inside a fenced code block.
    assert f"```{language}" in prompt
    # The user query must appear verbatim inside the prompt.
    assert user_query in prompt
    # The prompt should end with a clear instruction delimiter.
    assert prompt.strip().endswith("```")


@pytest.mark.parametrize(
    "invalid_query",
    [None, "", "   "],
)
def test_build_prompt_invalid_query_raises(invalid_query):
    """
    An empty or non‑string query should raise a ValueError.
    """
    with pytest.raises(ValueError):
        build_prompt(invalid_query, language="python")  # type: ignore[arg-type]


def test_build_prompt_unsupported_language():
    """
    Supplying a language that the assistant does not recognise must raise a ValueError.
    """
    with pytest.raises(ValueError):
        build_prompt("print('test')", language="unknown_lang")


def test_parse_response_strips_whitespace_and_markers():
    """
    The response parser should remove surrounding whitespace,
    markdown fences and any leading explanatory text.
    """
    raw_output = """
Sure! Here is the solution:

```python
def add(a, b):
    return a + b
```

Let me know if you need anything else.
"""
    expected = "def add(a, b):\n    return a + b"
    cleaned = parse_response(raw_output)
    assert cleaned == expected


def test_parse_response_multiple_code_blocks():
    """
    If the model returns several fenced blocks, only the first block of code
    should be extracted.
    """
    raw_output = """
Here are two snippets:

```javascript
console.log('first');
```

```javascript
console.log('second');
```
"""
    expected = "console.log('first');"
    cleaned = parse_response(raw_output)
    assert cleaned == expected


def test_parse_response_no_code_block_returns_stripped_text():
    """
    When no markdown fence is present, the parser should simply strip surrounding
    whitespace and return the raw string.
    """
    raw_output = "   This is a plain answer without code fences.   "
    assert parse_response(raw_output) == "This is a plain answer without code fences."


def test_parse_response_invalid_input_type():
    """
    Passing a non‑string (e.g., None) to the parser should raise a TypeError.
    """
    with pytest.raises(TypeError):
        parse_response(None)  # type: ignore[arg-type]