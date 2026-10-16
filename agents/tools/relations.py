"""Relations candidates et intégrité référentielle entre tables."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_relation(
    table_a: str,
    column_a: str,
    frame_a: pd.DataFrame,
    table_b: str,
    column_b: str,
    frame_b: pd.DataFrame,
) -> dict[str, Any]:
    """Mesure la correspondance dans les deux sens, sans faire de jointure destructive."""
    values_a = frame_a[column_a].dropna()
    values_b = frame_b[column_b].dropna()

    set_a = set(values_a.tolist())
    set_b = set(values_b.tolist())

    matched_a = values_a.isin(set_b)
    matched_b = values_b.isin(set_a)

    rate_a = float(matched_a.mean() * 100.0) if len(values_a) else 0.0
    rate_b = float(matched_b.mean() * 100.0) if len(values_b) else 0.0

    unique_a = values_a.is_unique
    unique_b = values_b.is_unique
    if unique_a and unique_b:
        relation_type = "1-1"
    elif unique_a:
        relation_type = "1-N"
    elif unique_b:
        relation_type = "N-1"
    else:
        relation_type = "N-N"

    return {
        "table_a": table_a,
        "column_a": column_a,
        "table_b": table_b,
        "column_b": column_b,
        "relationship_type": relation_type,
        "match_rate_pct": min(rate_a, rate_b),
        "orphan_count_a": int((~matched_a).sum()),
        "orphan_count_b": int((~matched_b).sum()),
        "evidence": (
            f"Correspondance A→B: {rate_a:.2f}% ({int(matched_a.sum())}/"
            f"{len(values_a)}); B→A: {rate_b:.2f}% ({int(matched_b.sum())}/"
            f"{len(values_b)})."
        ),
    }


def candidate_column_pairs(
    frames: dict[str, pd.DataFrame],
) -> list[tuple[str, str, str, str]]:
    """Propose des paires par nom identique ou suffixe `_id`.

    Cette fonction propose des candidates ; elle ne certifie pas une relation.
    """
    pairs: list[tuple[str, str, str, str]] = []
    names = list(frames)

    for i, table_a in enumerate(names):
        for table_b in names[i + 1:]:
            for column_a in frames[table_a].columns:
                for column_b in frames[table_b].columns:
                    a, b = str(column_a).lower(), str(column_b).lower()
                    if a == b or (
                        a.endswith("_id") and b.endswith("_id")
                        and a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0]
                    ):
                        pairs.append((table_a, str(column_a), table_b, str(column_b)))
    return pairs
