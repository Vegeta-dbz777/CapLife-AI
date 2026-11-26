"""Small helpers for user-facing wording."""
from __future__ import annotations


def count(number: int, word: str, plural: str | None = None) -> str:
    """'1 capacitor', '3 capacitors' (or the given plural), with a thousands separator."""
    return f"{number:,} {word if number == 1 else (plural or word + 's')}"
