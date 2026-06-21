from __future__ import annotations

import re


TOKEN_RE = re.compile(
    r"""
    [a-zA-Z_][a-zA-Z0-9_]*                    # python identifiers
    |
    [a-zA-Z0-9]+(?:[-./:][a-zA-Z0-9_{}]+)+    # technical tokens with separators
    |
    \d+(?:\.\d+)*                             # numbers / versions
    |
    [a-zA-Zа-яА-ЯёЁ0-9]+                       # regular words
    """,
    re.VERBOSE,
)


def tokenize_for_search(text: str) -> list[str]:
    """
    Tokenizer for technical documentation search.

    Keeps and expands tokens such as:
    - allow_origins
    - CORSMiddleware
    - api-key
    - /items/{item_id}
    - http://127.0.0.1:8000
    """
    text = text.lower()
    tokens = TOKEN_RE.findall(text)

    expanded: list[str] = []

    for token in tokens:
        expanded.append(token)

        parts = re.split(r"[-./:_{}/]+", token)
        for part in parts:
            if part and part != token:
                expanded.append(part)

    return expanded
