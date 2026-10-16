"""Chargement en mémoire de sources CSV, Excel, SQL et API."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class LoaderError(RuntimeError):
    """Erreur de chargement signalable à l'Orchestrator."""


def _sample_if_needed(
    frame: pd.DataFrame,
    *,
    threshold: int = 1_000_000,
    sample_size: int = 100_000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, bool, int]:
    """Retourne une copie entière ou un échantillon aléatoire sans modifier l'entrée."""
    row_count = len(frame)
    if row_count <= threshold:
        return frame.copy(deep=True), False, row_count

    size = min(sample_size, row_count)
    sampled = frame.sample(n=size, random_state=random_seed)
    return sampled.reset_index(drop=True), True, size


def load_source(
    config: Any,
    *,
    threshold: int = 1_000_000,
    sample_size: int = 100_000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Charge une source configurée et retourne (DataFrame, métadonnées).

    `config` est un SourceConfig Pydantic. Les erreurs attendues sont emballées
    dans LoaderError pour que l'agent puisse les convertir en erreur de rapport.
    """
    options: dict[str, Any] = config.options or {}
    source_type = str(config.type.value if hasattr(config.type, "value") else config.type)

    try:
        if source_type == "csv":
            frame = pd.read_csv(
                config.uri,
                sep=options.get("sep", ","),
                encoding=options.get("encoding", "utf-8"),
                **options.get("read_csv_kwargs", {}),
            )

        elif source_type == "excel":
            frame = pd.read_excel(
                config.uri,
                sheet_name=options.get("sheet_name", 0),
                **options.get("read_excel_kwargs", {}),
            )

        elif source_type == "sql":
            query = options.get("query")
            if not query:
                if not config.table:
                    raise LoaderError("Source SQL : fournir `table` ou `options.query`.")
                # Le nom de table est validé avant interpolation : pas d'identifiant SQL libre.
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", config.table):
                    raise LoaderError("Nom de table SQL invalide.")
                query = f"SELECT * FROM {config.table}"

            # Bloque les commandes non-SELECT. À compléter par des droits SQL read-only.
            normalized = re.sub(r"/\*.*?\*/|--[^\n]*", "", query, flags=re.S).strip()
            if not re.match(r"^(SELECT|WITH)\b", normalized, flags=re.I):
                raise LoaderError("Seules les requêtes SELECT sont autorisées.")
            if ";" in normalized.rstrip(";"):
                raise LoaderError("Une seule requête SQL est autorisée.")

            engine: Engine = create_engine(config.uri)
            try:
                with engine.connect() as connection:
                    frame = pd.read_sql_query(text(query), connection)
            finally:
                engine.dispose()

        elif source_type == "api":
            response = requests.get(
                config.uri,
                params=options.get("params"),
                headers=options.get("headers"),
                timeout=float(options.get("timeout", 30)),
            )
            response.raise_for_status()
            payload = response.json()

            records_path = options.get("records_path")
            if records_path:
                for key in records_path.split("."):
                    payload = payload[key]

            if not isinstance(payload, list):
                raise LoaderError(
                    "La réponse API doit être une liste d'enregistrements "
                    "ou définir un `records_path` vers une liste."
                )
            frame = pd.DataFrame.from_records(payload)

        else:
            raise LoaderError(f"Type de source non supporté : {source_type}")

        if not isinstance(frame, pd.DataFrame):
            raise LoaderError("Le connecteur n'a pas retourné de DataFrame.")

        original_rows = len(frame)
        frame, sampled, sample_rows = _sample_if_needed(
            frame,
            threshold=threshold,
            sample_size=sample_size,
            random_seed=random_seed,
        )
        metadata = {
            "source_name": config.name,
            "source_type": source_type,
            "row_count": original_rows,
            "column_count": len(frame.columns),
            "sampled": sampled,
            "sample_rows": sample_rows,
        }
        return frame, metadata

    except LoaderError:
        raise
    except UnicodeDecodeError as exc:
        raise LoaderError(
            f"Encodage illisible pour la source {config.name!r}; "
            "indiquer options.encoding."
        ) from exc
    except Exception as exc:
        raise LoaderError(f"Échec de chargement de {config.name!r}: {exc}") from exc
