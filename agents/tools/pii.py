"""Détection heuristique et masquage des données personnelles."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


NAME_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"email|e_mail", re.I),
    "phone": re.compile(r"phone|mobile|telephone|tel", re.I),
    "name": re.compile(r"(^|_)name($|_)|first_name|last_name", re.I),
    "address": re.compile(r"address|street|postal|zip_code", re.I),
    "customer_identifier": re.compile(r"customer.*id|subscriber.*id", re.I),
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")


def detect_pii_column(column_name: str, series: pd.Series) -> list[str]:
    categories = [
        category
        for category, pattern in NAME_PATTERNS.items()
        if pattern.search(column_name)
    ]

    # Heuristiques sur valeurs, échantillon plafonné et uniquement en mémoire.
    values = series.dropna().astype(str).head(500)
    if len(values):
        if values.map(lambda value: bool(EMAIL_RE.search(value))).mean() >= 0.5:
            categories.append("email")
        if values.map(lambda value: bool(PHONE_RE.search(value))).mean() >= 0.5:
            categories.append("phone")

    return sorted(set(categories))


def mask_pii(value: Any, categories: list[str]) -> str:
    """Masque les exemples avant leur inclusion dans un rapport."""
    if value is None or pd.isna(value):
        return "<NULL>"
    text = str(value)

    if "email" in categories:
        return "<EMAIL_MASKED>"
    if "phone" in categories:
        return "<PHONE_MASKED>"
    if categories:
        return "<PII_MASKED>"
    return text
