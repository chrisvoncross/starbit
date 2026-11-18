"""The next instruction is read from the two stores, not from the next candle."""

from __future__ import annotations


def instruction(kept: list[dict], rejected: list[dict]) -> str:
    if not kept and not rejected:
        return "Die Speicher sind leer. Die Rechnung ist frei."
    kept_names = ", ".join(item["name"] for item in kept) or "keine"
    rejected_names = ", ".join(item["name"] for item in rejected) or "keine"
    return (
        "Lies beide Speicher. Die nächste Rechnung ist eine Lücke, ein Störfaktor, "
        "oder eine Rechnung aus mehreren bestandenen. "
        f"Bestanden: {kept_names}. Verworfen: {rejected_names}."
    )
