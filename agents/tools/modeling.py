"""Modélisation en étoile déterministe (proposition, construction, contrôles).

Principe : à partir d'une table plate nettoyée, on décide
  1. le GRAIN (une ligne de fait = une entité, ex. un client) ;
  2. les MESURES (colonnes numériques agrégeables) ;
  3. les DIMENSIONS (colonnes descriptives regroupées par thème : géographie,
     contrat, services, résultat churn...) ;
puis on construit fact_* / dim_* avec des clés de substitution et on PROUVE que
l'étoile est sans perte (on re-joint tout et on retrouve exactement la table d'origine).

Aucun LLM : chaque choix a une justification écrite dans le plan.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from agents.tools.cleaning import is_numeric, looks_like_identifier

GEO_TOKENS = {"country", "state", "province", "region", "city", "zip", "zipcode", "postal",
              "lat", "latitude", "lon", "long", "longitude", "lng", "geo", "location", "county"}

# Vocabulaire télécom : thème -> mots-clés (comparés aux « mots » du nom de colonne).
DOMAIN_CLUSTERS: list[tuple[str, set[str], str]] = [
    ("churn", {"churn", "attrition", "cancel", "cancellation", "termination", "reason"},
     "résultat / motif de départ"),
    ("contract", {"contract", "payment", "billing", "plan", "tariff", "subscription",
                  "offer", "bundle", "paperless"}, "contrat et facturation"),
    ("service", {"phone", "line", "internet", "streaming", "security", "backup", "protection",
                 "support", "tv", "movie", "roaming", "voice", "sms", "data", "device",
                 "service", "fiber", "broadband"}, "services souscrits"),
    ("customer", {"gender", "sex", "senior", "age", "partner", "dependent", "married",
                  "segment", "citizen", "occupation", "education"}, "profil client"),
]

ADDITIVE_TOKENS = {"charge", "revenue", "amount", "price", "cost", "fee", "spend", "payment",
                   "minute", "byte", "usage", "call", "sms", "volume", "balance"}


def _tokens(name: str) -> set[str]:
    words = re.split(r"[^a-z0-9]+", name.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w}


def _is_geo(name: str) -> bool:
    return bool(_tokens(name) & GEO_TOKENS)


def _entity_name(grain_key: str | None, table: str) -> str:
    if not grain_key:
        return table
    base = re.sub(r"(_id|_key|_number|_no|_code)$", "", grain_key.lower())
    return base if base and base != grain_key.lower() else table


def _is_flag(series: pd.Series) -> bool:
    values = set(series.dropna().unique().tolist())
    return bool(values) and values <= {0, 1, 0.0, 1.0, True, False}


def _measure(series: pd.Series, name: str) -> dict[str, Any]:
    if _is_flag(series):
        return {"column": name, "kind": "flag", "suggested_aggregations": ["avg"],
                "note": "indicateur 0/1 : la moyenne donne un taux (ex. taux de churn)"}
    if _tokens(name) & ADDITIVE_TOKENS:
        return {"column": name, "kind": "metric", "suggested_aggregations": ["sum", "avg"],
                "note": "montant / volume additif"}
    return {"column": name, "kind": "metric", "suggested_aggregations": ["avg"],
            "note": "valeur non additive (score, durée, ancienneté) : on moyenne"}


# --------------------------------------------------------------------------- #
# 1. Proposition
# --------------------------------------------------------------------------- #
def propose_star_schema(
    frame: pd.DataFrame,
    roles: dict[str, str],
    *,
    table: str,
    business_key: list[str] | None = None,
    spec: Any | None = None,
    entity_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Retourne un plan d'étoile explicable. `spec` (StarSchemaSpec) a priorité sur l'auto."""
    cols = [str(c) for c in frame.columns]
    rows = len(frame)
    excluded: list[dict[str, str]] = []
    forced_excl = set(getattr(spec, "exclude", []) or [])

    constants = {c for c in cols if frame[c].nunique(dropna=True) <= 1}
    for c in cols:
        if c in forced_excl:
            excluded.append({"column": c, "reason": "exclue par configuration"})
        elif c in constants:
            excluded.append({"column": c, "reason": "colonne constante : aucune valeur analytique"})
    skip = {e["column"] for e in excluded}

    # ---- Grain
    grain: str | None = getattr(spec, "grain_key", None)
    grain_reason = "défini par configuration" if grain else ""
    if not grain:
        keys = [k for k in (business_key or []) if k in cols]
        if len(keys) == 1:
            grain, grain_reason = keys[0], "clé métier fournie"
        else:
            for c in cols:
                s = frame[c]
                if (c not in skip and looks_like_identifier(c) and not _is_geo(c)
                        and s.notna().all() and s.is_unique):
                    grain, grain_reason = c, "premier identifiant unique et non nul"
                    break
    entity = _entity_name(grain, table)

    # ---- Mesures
    spec_measures = getattr(spec, "measures", None)
    measures: list[dict[str, Any]] = []
    for c in cols:
        if c in skip or c == grain:
            continue
        if spec_measures is not None:
            if c in spec_measures:
                measures.append(_measure(frame[c], c))
            continue
        if is_numeric(frame[c]) and roles.get(c) != "id" and not _is_geo(c):
            measures.append(_measure(frame[c], c))
    measure_cols = {m["column"] for m in measures}

    # ---- Dimensions
    spec_dims = getattr(spec, "dimensions", None)
    dims: dict[str, dict[str, Any]] = {}

    def put(name: str, column: str, role: str, why: str) -> None:
        d = dims.setdefault(name, {"name": name, "role": role, "columns": [], "rationale": why})
        d["columns"].append(column)

    date_cols: list[str] = []
    if spec_dims is not None:
        for dim_name, dim_cols in spec_dims.items():
            for c in dim_cols:
                if c in cols and c not in skip:
                    put(dim_name, c, "configured", "regroupement défini par configuration")
    else:
        for c in cols:
            if c in skip or c in measure_cols or c == grain:
                continue
            series = frame[c]
            if pd.api.types.is_datetime64_any_dtype(series):
                date_cols.append(c)
                continue
            if _is_geo(c):
                put("dim_location", c, "geography", "attributs géographiques")
                continue
            if entity_columns and c in entity_columns:
                put(f"dim_{entity}", c, "entity", "donnée personnelle : attribut de l'entité")
                continue
            tokens = _tokens(c)
            placed = False
            for cluster, words, why in DOMAIN_CLUSTERS:
                if tokens & words:
                    if cluster == "customer":
                        put(f"dim_{entity}", c, "entity", "profil client")
                    else:
                        put(f"dim_{cluster}", c, cluster, why)
                    placed = True
                    break
            if placed:
                continue
            high_card = series.nunique(dropna=True) > max(50, rows * 0.5)
            if roles.get(c) == "id" or (not is_numeric(series) and high_card):
                put(f"dim_{entity}", c, "entity", "identifiant / attribut propre à l'entité")
            else:
                put("dim_attributes", c, "attributes", "autres attributs descriptifs")

    # La dimension de l'entité porte la clé du grain en première colonne.
    if grain:
        ent = dims.setdefault(f"dim_{entity}", {"name": f"dim_{entity}", "role": "entity",
                                                "columns": [], "rationale": "l'entité du grain"})
        ent["columns"] = [grain] + [c for c in ent["columns"] if c != grain]

    ordered = list(dims.values())
    ordered.sort(key=lambda d: (d["role"] != "entity", d["name"]))
    for d in ordered:
        d["key"] = d["name"].removeprefix("dim_") + "_key"

    # Un identifiant d'entité non unique ne doit pas rester seul dans un fait sans grain.
    return {
        "source": "configured" if spec else "auto",
        "fact_name": getattr(spec, "fact_name", None) or f"fact_{entity}",
        "grain_key": grain,
        "grain": (f"1 ligne = 1 {entity} ({grain})" if grain
                  else "1 ligne = 1 ligne de la table source"),
        "grain_reason": grain_reason or "aucun identifiant unique trouvé : clé de ligne créée",
        "measures": measures,
        "dimensions": ordered,
        "date_columns": date_cols,
        "excluded": excluded,
    }


# --------------------------------------------------------------------------- #
# 2. Construction
# --------------------------------------------------------------------------- #
def _date_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")


def build_star_tables(
    frame: pd.DataFrame, plan: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Construit (fait, {nom_dim: dim}). Les clés de substitution sont déterministes."""
    fact = pd.DataFrame(index=frame.index)
    dims: dict[str, pd.DataFrame] = {}

    for d in plan["dimensions"]:
        cols, key = d["columns"], d["key"]
        keys = frame.groupby(cols, dropna=False, sort=True).ngroup() + 1
        table = frame[cols].copy()
        table.insert(0, key, keys.astype("int64"))
        dims[d["name"]] = (table.drop_duplicates(subset=key).sort_values(key)
                           .reset_index(drop=True))
        fact[key] = keys.astype("int64")

    if not plan["grain_key"]:
        # Nom dédié : ne peut jamais entrer en collision avec la clé d'une dimension.
        fact.insert(0, "fact_row_key", np.arange(1, len(frame) + 1))

    if plan["date_columns"]:
        stamps = pd.concat([frame[c] for c in plan["date_columns"]]).dropna().dt.normalize()
        unique = pd.Series(sorted(stamps.unique()))
        dims["dim_date"] = pd.DataFrame({
            "date_key": _date_key(unique), "date": unique.dt.date,
            "year": unique.dt.year, "quarter": unique.dt.quarter, "month": unique.dt.month,
            "day": unique.dt.day, "weekday": unique.dt.day_name(),
        })
        for c in plan["date_columns"]:
            fact[f"{c}_date_key"] = _date_key(frame[c])

    for m in plan["measures"]:
        fact[m["column"]] = frame[m["column"]]
    return fact, dims


# --------------------------------------------------------------------------- #
# 3. Contrôles : l'étoile doit être sans perte
# --------------------------------------------------------------------------- #
def check_star(
    frame: pd.DataFrame, fact: pd.DataFrame, dims: dict[str, pd.DataFrame], plan: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, violations: int, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "violations": int(violations),
                       "detail": detail})

    add("star_fact_row_count", len(fact) == len(frame), abs(len(fact) - len(frame)),
        f"{len(frame)} lignes dans la table nettoyée, {len(fact)} dans {plan['fact_name']}.")

    null_fk = orphan = 0
    for d in plan["dimensions"]:
        key = d["key"]
        null_fk += int(fact[key].isna().sum())
        orphan += int((~fact[key].isin(dims[d["name"]][key])).sum())
    add("star_no_null_foreign_key", null_fk == 0, null_fk, f"{null_fk} clé(s) étrangère(s) nulle(s).")
    add("star_no_orphan_foreign_key", orphan == 0, orphan,
        f"{orphan} ligne(s) du fait pointent vers une dimension inexistante.")

    dup = sum(int(dims[d["name"]][d["key"]].duplicated().sum()) for d in plan["dimensions"])
    add("star_dimension_keys_unique", dup == 0, dup, f"{dup} clé(s) de dimension en double.")

    covered = [c for d in plan["dimensions"] for c in d["columns"]]
    covered += [m["column"] for m in plan["measures"]] + plan["date_columns"]
    covered += [e["column"] for e in plan["excluded"]]
    if plan["grain_key"] and not any(plan["grain_key"] in d["columns"] for d in plan["dimensions"]):
        covered.append(plan["grain_key"])
    source_cols = [str(c) for c in frame.columns]
    missing = sorted(set(source_cols) - set(covered))
    repeated = sorted({c for c in covered if covered.count(c) > 1})
    add("star_column_coverage", not missing and not repeated, len(missing) + len(repeated),
        f"Colonnes non placées : {missing or 'aucune'} ; placées deux fois : {repeated or 'aucune'}.")

    bad_sum = 0
    for m in plan["measures"]:
        a, b = float(frame[m["column"]].sum()), float(fact[m["column"]].sum())
        if not np.isclose(a, b, rtol=1e-9, atol=1e-9):
            bad_sum += 1
    add("star_measure_reconciliation", bad_sum == 0, bad_sum,
        f"{bad_sum} mesure(s) dont la somme diffère entre table nettoyée et fait.")

    if len(fact) != len(frame):
        # Comparaison cellule à cellule impossible : l'échec est déjà signalé ci-dessus.
        add("star_lossless_rejoin", False, abs(len(fact) - len(frame)),
            "Re-jointure impossible : le fait et la table nettoyée n'ont pas le même nombre de lignes.")
        return checks

    mismatched = 0
    for d in plan["dimensions"]:
        dim = dims[d["name"]].set_index(d["key"])
        for c in d["columns"]:
            rebuilt = fact[d["key"]].map(dim[c])
            same = (rebuilt == frame[c]) | (rebuilt.isna() & frame[c].isna())
            mismatched += int((~same.fillna(False)).sum())
    for m in plan["measures"]:
        a, b = fact[m["column"]], frame[m["column"]]
        mismatched += int((~((a == b) | (a.isna() & b.isna()))).sum())
    add("star_lossless_rejoin", mismatched == 0, mismatched,
        f"En re-joignant le fait et les dimensions, {mismatched} cellule(s) diffèrent de la "
        "table nettoyée (0 = l'étoile ne perd et ne déforme aucune donnée).")
    return checks


# --------------------------------------------------------------------------- #
# 4. Diagramme
# --------------------------------------------------------------------------- #
def to_mermaid(plan: dict[str, Any], fact: pd.DataFrame, dims: dict[str, pd.DataFrame]) -> str:
    lines = ["erDiagram", f"    {plan['fact_name']} {{"]
    for c in fact.columns:
        is_fk = any(c == d["key"] for d in plan["dimensions"]) or c.endswith("_date_key")
        kind = "int" if is_fk or str(fact[c].dtype).startswith(("int", "Int")) else "float"
        lines.append(f"        {kind} {c}{' FK' if is_fk else ''}")
    lines.append("    }")
    for name, table in dims.items():
        lines.append(f"    {name} {{")
        for c in table.columns:
            lines.append(f"        {'int' if c.endswith('_key') else 'string'} {c}"
                         f"{' PK' if c.endswith('_key') else ''}")
        lines.append("    }")
    for d in plan["dimensions"]:
        lines.append(f"    {d['name']} ||--o{{ {plan['fact_name']} : {d['key']}")
    if "dim_date" in dims:
        lines.append(f"    dim_date ||--o{{ {plan['fact_name']} : date_key")
    return "\n".join(lines) + "\n"
