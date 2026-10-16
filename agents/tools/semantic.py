"""Suggestion de rôle sémantique basée sur des règles explicables."""

from __future__ import annotations

import re

import pandas as pd

from agents.tools.schema import infer_column_type


ID_RE = re.compile(r"(^id$|_id$|^id_|key$|identifier)", re.I)
DATE_RE = re.compile(r"date|time|timestamp|month|year", re.I)
MEASURE_RE = re.compile(
    r"amount|charge|revenue|usage|minutes|bytes|count|score|tenure|"
    r"monthly|total|balance|rate",
    re.I,
)


def suggest_semantic_role(column_name: str, series: pd.Series) -> str:
    name = column_name.lower()
    inferred = infer_column_type(series)

    if ID_RE.search(name):
        return "id"
    if inferred == "datetime" or DATE_RE.search(name):
        return "date"
    if inferred in {"integer", "numeric"} and MEASURE_RE.search(name):
        return "measure"
    if inferred in {"categorical", "string", "boolean"}:
        return "dimension"
    if inferred in {"integer", "numeric"}:
        return "measure"
    return "unknown"
