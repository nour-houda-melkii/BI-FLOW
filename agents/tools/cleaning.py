"""Outils de nettoyage déterministes de l'ETL / Data Quality Agent.

Chaque fonction :
  - ne modifie jamais son entrée (travaille sur une copie) ;
  - retourne (DataFrame, liste d'actions) ; une action est un dict contenant
    rule / column / rows_affected / before / after / rationale ;
  - ne fait appel à aucun LLM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from agents.tools.anomalies import iqr_outliers, zscore_outliers

Action = dict[str, Any]

ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|^key$|_key$|identifier|zip|postal|_code$)")
DATE_NAME_RE = re.compile(r"date|timestamp|_at$|^dt_", re.I)
COORD_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$")
THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_text(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    )


def is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    )


def looks_like_identifier(name: str) -> bool:
    """Nom évoquant un identifiant / code postal (jamais imputé ni converti)."""
    return bool(ID_NAME_RE.search(to_snake_case(name)))


def to_snake_case(name: str) -> str:
    """'Total Charges' -> 'total_charges', 'CustomerID' -> 'customer_id'."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name).strip())
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or "column"


def snake_case_mapping(columns: list[str]) -> dict[str, str]:
    """Mapping original -> snake_case, collisions résolues par un suffixe."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for column in columns:
        base = to_snake_case(column)
        candidate, index = base, 2
        while candidate in used:
            candidate = f"{base}_{index}"
            index += 1
        used.add(candidate)
        mapping[str(column)] = candidate
    return mapping


def _action(rule: str, column: str | None, rows: int, before: str, after: str,
            rationale: str, source_issue_id: str | None = None) -> Action:
    return {
        "rule": rule, "column": column, "rows_affected": int(rows),
        "before": before, "after": after, "rationale": rationale,
        "source_issue_id": source_issue_id,
    }


# --------------------------------------------------------------------------- #
# 1. Normalisation du texte
# --------------------------------------------------------------------------- #
def normalize_text_columns(
    frame: pd.DataFrame,
    *,
    null_tokens: list[str],
    strip: bool = True,
) -> tuple[pd.DataFrame, list[Action]]:
    """Supprime les espaces parasites et convertit les 'blancs' en valeur manquante."""
    out = frame.copy()
    actions: list[Action] = []
    tokens = {token.strip().lower() for token in null_tokens}

    for column in out.columns:
        if not is_text(out[column]):
            continue

        text = out[column].astype("string")
        cleaned = text.str.replace(r"\s+", " ", regex=True).str.strip() if strip else text

        null_like = cleaned.str.lower().isin(tokens).fillna(False).astype(bool)
        # Cellules réellement présentes (pas déjà nulles) devenues nulles.
        hidden_nulls = int((null_like & text.notna()).sum())
        changed = (cleaned != text).fillna(False).astype(bool)
        whitespace_fixed = int((changed & ~null_like).sum())

        out[column] = cleaned.mask(null_like)

        if whitespace_fixed:
            actions.append(_action(
                "strip_whitespace", str(column), whitespace_fixed,
                f"{whitespace_fixed} valeur(s) avec espaces en trop.",
                "Espaces de début/fin/doublons supprimés.",
                "Des espaces invisibles créent de fausses catégories et cassent les jointures.",
            ))
        if hidden_nulls:
            actions.append(_action(
                "blank_to_null", str(column), hidden_nulls,
                f"{hidden_nulls} cellule(s) vide(s) ou égale(s) à un jeton nul "
                "(ex. ' ', 'N/A') non vues comme manquantes.",
                "Converties en valeur manquante réelle.",
                "Un blanc est une valeur manquante déguisée : il empêche la détection "
                "des nulls et la conversion en numérique.",
            ))
    return out, actions


# --------------------------------------------------------------------------- #
# 2. Typage
# --------------------------------------------------------------------------- #
def _numeric_candidate(text: pd.Series) -> pd.Series:
    """Prépare du texte pour to_numeric : symboles monétaires, séparateurs de milliers."""
    prepared = text.str.replace(r"[$€£\s]", "", regex=True)
    thousands = prepared.str.match(THOUSANDS_RE).fillna(False).astype(bool)
    return prepared.mask(thousands, prepared.str.replace(",", "", regex=False))


def _has_leading_zero(text: pd.Series) -> bool:
    values = text.dropna()
    return bool(values.str.match(r"^0\d+").fillna(False).any())


def numeric_like_columns(frame: pd.DataFrame, *, min_ratio: float = 0.9,
                         protected: set[str] | None = None) -> list[str]:
    """Colonnes texte dont ≥ min_ratio des valeurs non nulles sont numériques."""
    protected = protected or set()
    found: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if not is_text(series) or str(column) in protected:
            continue
        text = series.astype("string").str.strip()
        text = text.mask(text == "")
        present = text.dropna()
        if present.empty or _has_leading_zero(present):
            continue
        parsed = pd.to_numeric(_numeric_candidate(present), errors="coerce")
        if parsed.notna().mean() >= min_ratio:
            found.append(str(column))
    return found


def coerce_numeric_columns(
    frame: pd.DataFrame,
    *,
    min_ratio: float,
    protected: set[str],
    issue_ids: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[Action]]:
    """Convertit en numérique les colonnes texte quasi entièrement numériques."""
    out = frame.copy()
    actions: list[Action] = []
    issue_ids = issue_ids or {}

    for column in numeric_like_columns(out, min_ratio=min_ratio, protected=protected):
        text = out[column].astype("string")
        present_before = int(text.notna().sum())
        parsed = pd.to_numeric(_numeric_candidate(text), errors="coerce")
        failed = int((text.notna() & parsed.isna()).sum())

        if (parsed.dropna() % 1 == 0).all() and failed == 0 and parsed.notna().all():
            parsed = parsed.astype("int64")

        out[column] = parsed
        actions.append(_action(
            "coerce_to_numeric", column, present_before - failed,
            f"Colonne stockée en texte ({text.dtype}) mais numérique à "
            f"{(present_before - failed) / max(present_before, 1) * 100:.1f}%.",
            f"Convertie en {parsed.dtype}.",
            "Un montant stocké en texte ne peut être ni sommé ni moyenné par le KPI Agent.",
            issue_ids.get(column),
        ))
        if failed:
            actions.append(_action(
                "unparseable_to_null", column, failed,
                f"{failed} valeur(s) non numérique(s) dans une colonne numérique.",
                "Converties en valeur manquante.",
                "Une valeur non interprétable ne doit pas être devinée.",
            ))
    return out, actions


def coerce_datetime_columns(
    frame: pd.DataFrame,
    *,
    hinted: set[str],
    min_ratio: float,
) -> tuple[pd.DataFrame, list[Action]]:
    """Convertit en datetime les colonnes texte signalées comme dates."""
    out = frame.copy()
    actions: list[Action] = []
    for column in out.columns:
        series = out[column]
        if not is_text(series):
            continue
        name = str(column)
        if name not in hinted and not DATE_NAME_RE.search(name):
            continue
        text = series.astype("string")
        present = text.dropna()
        if present.empty:
            continue
        parsed = pd.to_datetime(present, errors="coerce", format="mixed")
        if parsed.notna().mean() < min_ratio:
            continue
        full = pd.to_datetime(text, errors="coerce", format="mixed")
        failed = int((text.notna() & full.isna()).sum())
        out[column] = full
        actions.append(_action(
            "coerce_to_datetime", name, int(full.notna().sum()),
            "Colonne de dates stockée en texte.",
            f"Convertie en datetime ({failed} valeur(s) illisible(s) mises à null).",
            "Les dates textuelles empêchent tri, filtres temporels et séries chronologiques.",
        ))
    return out, actions


def harmonize_categories(
    frame: pd.DataFrame,
    *,
    protected: set[str],
    max_cardinality: int = 100,
) -> tuple[pd.DataFrame, list[Action]]:
    """Fusionne les variantes de casse ('yes', 'YES', 'Yes') vers la forme la plus fréquente."""
    out = frame.copy()
    actions: list[Action] = []
    for column in out.columns:
        series = out[column]
        if not is_text(series) or str(column) in protected:
            continue
        text = series.astype("string")
        if text.nunique(dropna=True) > max_cardinality:
            continue
        lowered = text.str.lower()
        counts = text.value_counts(dropna=True)
        mapping: dict[str, str] = {}
        for lower_value in lowered.dropna().unique():
            variants = counts[(counts.index.to_series().str.lower() == lower_value).to_numpy()]
            if len(variants) > 1:
                canonical = variants.sort_values(ascending=False, kind="stable").index[0]
                mapping.update({v: canonical for v in variants.index if v != canonical})
        if mapping:
            affected = int(text.isin(list(mapping)).sum())
            out[column] = text.replace(mapping)
            actions.append(_action(
                "harmonize_case", str(column), affected,
                f"{len(mapping)} variante(s) de casse d'une même modalité.",
                "Fusionnées vers la forme la plus fréquente.",
                "Des variantes de casse fragmentent les agrégations (ex. 'yes' ≠ 'Yes').",
            ))
    return out, actions


# --------------------------------------------------------------------------- #
# 3. Valeurs manquantes
# --------------------------------------------------------------------------- #
def detect_structural_missing(
    frame: pd.DataFrame, *, max_driver_cardinality: int = 10, min_support: int = 5
) -> dict[str, dict[str, Any]]:
    """Repère les nulls « logiques » : colonne nulle ssi `driver == valeur`.

    Exemple : Churn Reason est nul exactement quand Churn Label == 'No'.
    Ce n'est pas un défaut de qualité mais une information absente par construction.
    """
    result: dict[str, dict[str, Any]] = {}
    total = len(frame)
    for column in frame.columns:
        mask = frame[column].isna()
        nulls = int(mask.sum())
        if nulls < min_support or nulls == total:
            continue
        for driver in frame.columns:
            if driver == column:
                continue
            drv = frame[driver]
            if drv.isna().any() or not (2 <= drv.nunique() <= max_driver_cardinality):
                continue
            grouped = mask.groupby(drv, observed=True).agg(["sum", "count"])
            hit = grouped[(grouped["sum"] == grouped["count"]) & (grouped["sum"] == nulls)]
            if len(hit):
                value = hit.index[0]
                result[str(column)] = {
                    "driver": str(driver),
                    "value": value.item() if hasattr(value, "item") else value,
                    "null_count": nulls,
                }
                break
    return result


def apply_conditional_fills(
    frame: pd.DataFrame,
    rules: list[Any],
    *,
    issue_ids: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[Action]]:
    """Applique les règles métier de remplissage conditionnel."""
    out = frame.copy()
    actions: list[Action] = []
    issue_ids = issue_ids or {}
    for rule in rules:
        if rule.column not in out.columns or rule.when_column not in out.columns:
            continue
        condition = out[rule.when_column] == rule.when_equals
        target = out[rule.column].isna() & condition.fillna(False)
        count = int(target.sum())
        if not count:
            continue
        out.loc[target, rule.column] = rule.value
        actions.append(_action(
            "conditional_fill", rule.column, count,
            f"{count} valeur(s) nulle(s) alors que {rule.when_column} == {rule.when_equals!r}.",
            f"Remplies par {rule.value!r}.",
            rule.rationale, issue_ids.get(rule.column),
        ))
    return out, actions


def handle_missing_values(
    frame: pd.DataFrame,
    *,
    structural: dict[str, dict[str, Any]],
    protected: set[str],
    numeric_strategy: str,
    categorical_strategy: str,
    unknown_label: str,
    not_applicable_label: str,
    drop_threshold_pct: float,
    issue_ids: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[Action]]:
    """Traite les valeurs manquantes colonne par colonne, avec une justification."""
    out = frame.copy()
    actions: list[Action] = []
    issue_ids = issue_ids or {}
    total = len(out)

    for column in list(out.columns):
        series = out[column]
        missing = int(series.isna().sum())
        if missing == 0:
            continue
        name = str(column)
        pct = missing / total * 100.0 if total else 0.0
        issue = issue_ids.get(name)

        if name in structural:
            info = structural[name]
            if is_text(series):
                out[column] = series.astype("string").fillna(not_applicable_label)
                actions.append(_action(
                    "structural_missing_fill", name, missing,
                    f"{missing} null(s), présents exactement quand {info['driver']} == "
                    f"{info['value']!r}.",
                    f"Remplis par {not_applicable_label!r}.",
                    "Valeur absente par construction (ex. pas de motif de départ pour un client "
                    "resté) : ce n'est pas une erreur, on l'étiquette explicitement.",
                    issue,
                ))
            else:
                actions.append(_action(
                    "structural_missing_keep", name, 0,
                    f"{missing} null(s) liés à {info['driver']} == {info['value']!r}.",
                    "Nulls conservés.",
                    "Imputer une valeur numérique fausserait les moyennes ; la valeur est "
                    "non applicable.",
                    issue,
                ))
            continue

        if name in protected:
            # Identifiants / clés : on n'invente jamais une valeur.
            continue

        if pct >= drop_threshold_pct:
            out = out.drop(columns=[column])
            actions.append(_action(
                "drop_sparse_column", name, missing,
                f"{pct:.1f}% de valeurs manquantes (seuil {drop_threshold_pct:.0f}%).",
                "Colonne supprimée.",
                "Trop vide pour être exploitable ; l'imputer reviendrait à inventer la donnée.",
                issue,
            ))
            continue

        if is_numeric(series):
            if numeric_strategy == "keep":
                continue
            fill = {"median": series.median(), "mean": series.mean(), "zero": 0.0}[numeric_strategy]
            if pd.isna(fill):
                continue
            out[column] = series.fillna(fill)
            actions.append(_action(
                f"impute_{numeric_strategy}", name, missing,
                f"{missing} valeur(s) manquante(s) ({pct:.2f}%).",
                f"Imputées par {numeric_strategy} = {float(fill):.4g}.",
                "Stratégie robuste choisie pour ne pas supprimer de lignes ; la médiane est peu "
                "sensible aux valeurs extrêmes.",
                issue,
            ))
        elif is_text(series):
            if categorical_strategy == "keep":
                continue
            text = series.astype("string")
            if categorical_strategy == "mode":
                modes = text.mode(dropna=True)
                if modes.empty:
                    continue
                fill_value, label = modes.iloc[0], f"mode = {modes.iloc[0]!r}"
            else:
                fill_value, label = unknown_label, f"{unknown_label!r}"
            out[column] = text.fillna(fill_value)
            actions.append(_action(
                f"impute_{categorical_strategy}", name, missing,
                f"{missing} valeur(s) manquante(s) ({pct:.2f}%).",
                f"Remplies par {label}.",
                "Une modalité explicite évite de perdre les lignes tout en gardant "
                "l'information qu'elle était inconnue.",
                issue,
            ))
        # datetime : on garde NaT, aucune date n'est inventée.
    return out, actions


# --------------------------------------------------------------------------- #
# 4. Doublons
# --------------------------------------------------------------------------- #
def drop_exact_duplicates(
    frame: pd.DataFrame, *, table: str
) -> tuple[pd.DataFrame, list[Action]]:
    count = int(frame.duplicated(keep="first").sum())
    if not count:
        return frame.copy(), []
    out = frame.drop_duplicates(keep="first").reset_index(drop=True)
    return out, [_action(
        "drop_exact_duplicates", None, count,
        f"{count} ligne(s) identique(s) à une autre.",
        f"{count} ligne(s) supprimée(s), première occurrence conservée.",
        "Des lignes strictement identiques gonflent les comptages et les sommes.",
        f"{table}.exact_duplicates",
    )]


def resolve_key_duplicates(
    frame: pd.DataFrame, *, key: list[str], strategy: str, table: str
) -> tuple[pd.DataFrame, list[Action]]:
    if not key or any(column not in frame.columns for column in key):
        return frame.copy(), []
    count = int(frame.duplicated(subset=key, keep="first").sum())
    if not count:
        return frame.copy(), []
    if strategy == "flag":
        return frame.copy(), [_action(
            "flag_key_duplicates", None, 0,
            f"{count} ligne(s) partagent une clé métier {key} déjà vue.",
            "Aucune ligne supprimée (stratégie 'flag').",
            "Deux lignes différentes pour la même clé exigent une décision métier ; "
            "l'ETL ne choisit pas à votre place.",
            f"{table}.business_key_duplicates",
        )]
    keep = "first" if strategy == "keep_first" else "last"
    out = frame.drop_duplicates(subset=key, keep=keep).reset_index(drop=True)
    return out, [_action(
        "drop_key_duplicates", None, count,
        f"{count} ligne(s) en doublon sur la clé {key}.",
        f"Supprimées, occurrence '{keep}' conservée.",
        f"Règle de résolution configurée : {strategy}.",
        f"{table}.business_key_duplicates",
    )]


# --------------------------------------------------------------------------- #
# 5. Valeurs aberrantes
# --------------------------------------------------------------------------- #
def handle_outliers(
    frame: pd.DataFrame,
    *,
    protected: set[str],
    strategy: str,
    iqr_multiplier: float,
    issue_ids: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[Action], dict[str, int]]:
    """Signale (par défaut) ou borne les outliers IQR ∪ z-score. Retourne aussi les comptes."""
    out = frame.copy()
    actions: list[Action] = []
    counts: dict[str, int] = {}
    issue_ids = issue_ids or {}

    for column in out.columns:
        series = out[column]
        name = str(column)
        if not is_numeric(series) or name in protected or series.nunique(dropna=True) <= 2:
            continue
        mask = iqr_outliers(series, multiplier=iqr_multiplier) | zscore_outliers(series)
        count = int(mask.sum())
        if not count:
            continue
        counts[name] = count
        if strategy == "clip":
            q1, q3 = series.quantile([0.25, 0.75])
            low, high = q1 - iqr_multiplier * (q3 - q1), q3 + iqr_multiplier * (q3 - q1)
            out[column] = series.clip(lower=low, upper=high)
            actions.append(_action(
                "clip_outliers", name, count,
                f"{count} valeur(s) hors de [{low:.4g}, {high:.4g}].",
                "Bornées aux limites IQR.",
                "Winsorisation demandée par configuration ; les valeurs d'origine sont "
                "perdues dans la table nettoyée.",
                issue_ids.get(name),
            ))
        else:
            actions.append(_action(
                "report_outliers", name, 0,
                f"{count} valeur(s) extrême(s) (IQR ou z-score).",
                "Aucune modification.",
                "Un outlier peut être un vrai gros client : on le signale sans le supprimer.",
                issue_ids.get(name),
            ))
    return out, actions, counts


# --------------------------------------------------------------------------- #
# 6. PII
# --------------------------------------------------------------------------- #
DATE_VALUE_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}|^\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}")

NON_NEGATIVE_TOKENS = {
    "minute", "mb", "gb", "kb", "byte", "data", "usage", "duration", "charge", "amount",
    "price", "cost", "fee", "revenue", "tenure", "quantity", "volume", "sms", "call",
}


def is_non_negative_name(name: str) -> bool:
    """Nom évoquant une quantité qui ne peut pas être négative (heuristique, jamais certaine)."""
    words = re.split(r"[^a-z0-9]+", to_snake_case(name))
    return any((w[:-1] if len(w) > 3 and w.endswith("s") else w) in NON_NEGATIVE_TOKENS
               for w in words if w)


def is_pii_false_positive(name: str, series: pd.Series, categories: list[str]) -> str | None:
    """Retourne la raison si la détection PII du Profiler est un faux positif."""
    present = series.dropna()
    if present.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        return "colonne de dates (pas un numéro de téléphone)"
    if (not is_numeric(series)
            and present.astype("string").str.match(DATE_VALUE_RE).fillna(False).mean() >= 0.9):
        return "valeurs au format date"
    if is_numeric(series) and not set(categories) & {"customer_identifier"}:
        return "colonne numérique (pas un identifiant personnel textuel)"
    text = present.astype("string")
    if text.str.match(COORD_RE).fillna(False).mean() >= 0.9:
        return "valeurs de type coordonnées GPS"
    if present.nunique() <= 20 and not set(categories) & {"customer_identifier"}:
        return "faible cardinalité (modalités, pas des personnes)"
    return None


def hash_value(value: Any, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()[:16]


def apply_pii_policy(
    frame: pd.DataFrame,
    pii_columns: dict[str, list[str]],
    *,
    pii_policy: str,
    identifier_policy: str,
    salt: str,
) -> tuple[pd.DataFrame, list[Action], dict[str, str]]:
    """Applique la politique PII. Retourne (frame, actions, traitement par colonne)."""
    out = frame.copy()
    actions: list[Action] = []
    treatment: dict[str, str] = {}

    for column, categories in pii_columns.items():
        if column not in out.columns:
            continue
        reason = is_pii_false_positive(column, out[column], categories)
        if reason:
            treatment[column] = "false_positive_ignored"
            actions.append(_action(
                "pii_false_positive", column, 0,
                f"Signalée PII par le Profiler ({', '.join(categories)}).",
                "Aucune modification.",
                f"Faux positif écarté : {reason}.",
            ))
            continue

        is_identifier = categories == ["customer_identifier"]
        policy = identifier_policy if is_identifier else pii_policy
        if policy == "keep":
            treatment[column] = "kept"
            actions.append(_action(
                "pii_kept", column, 0,
                f"Colonne PII ({', '.join(categories)}).",
                "Conservée en clair.",
                "Identifiant technique nécessaire aux jointures ; à pseudonymiser avant "
                "tout partage externe." if is_identifier else "Politique 'keep' configurée.",
            ))
        elif policy == "hash":
            present = int(out[column].notna().sum())
            out[column] = out[column].map(
                lambda v: hash_value(v, salt) if pd.notna(v) else v
            )
            treatment[column] = "hashed"
            actions.append(_action(
                "pii_hash", column, present,
                f"Colonne PII ({', '.join(categories)}).",
                "Pseudonymisée (SHA-256 salé, 16 car.).",
                "Les données personnelles ne doivent pas atteindre le dashboard ; le hash "
                "conserve l'unicité et les jointures.",
            ))
        else:
            out = out.drop(columns=[column])
            treatment[column] = "dropped"
            actions.append(_action(
                "pii_drop", column, len(out),
                f"Colonne PII ({', '.join(categories)}).", "Colonne supprimée.",
                "Politique 'drop' configurée : la donnée n'est pas nécessaire aux KPI.",
            ))
    return out, actions, treatment


# --------------------------------------------------------------------------- #
# 7. Règles de plausibilité et cohérence
# --------------------------------------------------------------------------- #
def apply_range_rules(frame: pd.DataFrame, rules: list[Any]) -> tuple[pd.DataFrame, list[Action]]:
    out = frame.copy()
    actions: list[Action] = []
    for rule in rules:
        if rule.column not in out.columns or not is_numeric(out[rule.column]):
            continue
        series = out[rule.column]
        bad = pd.Series(False, index=out.index)
        if rule.min is not None:
            bad |= series < rule.min
        if rule.max is not None:
            bad |= series > rule.max
        count = int(bad.sum())
        if count and rule.fix == "set_null":
            out.loc[bad, rule.column] = np.nan
            actions.append(_action(
                "range_violation_to_null", rule.column, count,
                f"{count} valeur(s) hors de [{rule.min}, {rule.max}].",
                "Mises à null (valeur impossible).",
                "Une valeur hors domaine métier est une erreur de saisie, pas une mesure.",
            ))
    return out, actions


def apply_mapping_rules(frame: pd.DataFrame, rules: list[Any]) -> tuple[pd.DataFrame, list[Action]]:
    out = frame.copy()
    actions: list[Action] = []
    for rule in rules:
        if rule.fix != "derive_b" or rule.column_a not in out or rule.column_b not in out:
            continue
        fixed = 0
        for a_value, b_value in rule.mapping.items():
            rows = out[rule.column_a].astype("string") == str(a_value)
            wrong = rows & (out[rule.column_b].astype("string") != str(b_value))
            n = int(wrong.fillna(False).sum())
            if n:
                out.loc[wrong.fillna(False), rule.column_b] = b_value
                fixed += n
        if fixed:
            actions.append(_action(
                "derive_from_reference", rule.column_b, fixed,
                f"{fixed} valeur(s) incohérente(s) avec {rule.column_a}.",
                f"{rule.column_b} recalculée depuis {rule.column_a}.",
                "Deux colonnes décrivant le même fait doivent être cohérentes.",
            ))
    return out, actions
