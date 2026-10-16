"""Graphe LangGraph de l'ETL / Data Quality Agent.

    load -> clean -> validate -> export -> summarize -> persist

Principe : toutes les décisions de nettoyage sont déterministes et tracées ; le LLM
(optionnel) ne fait que rédiger la synthèse à partir du journal d'actions.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.etl.models import (
    CleanedTable,
    RangeRule,
    CleaningAction,
    ColumnSummary,
    ETLOptions,
    ETLReport,
    ETLRequest,
    IssueResolution,
    QualityScores,
    StarDimension,
    StarMeasure,
    StarSchema,
    ValidationCheck,
)
from agents.etl.presets import apply_preset
from agents.etl.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from agents.profiler.models import Issue, TableProfile, TraceEntry
from agents.tools import cleaning as C
from agents.tools import modeling as M
from agents.tools import validation as V
from agents.tools.loader import LoaderError, load_source
from agents.tools.pii import detect_pii_column

REPORT_ONLY_RULES = {"report_outliers", "flag_key_duplicates", "structural_missing_keep"}


class ETLState(TypedDict, total=False):
    request: ETLRequest
    options: ETLOptions
    report: ETLReport
    raw: dict[str, pd.DataFrame]
    clean: dict[str, pd.DataFrame]
    ctx: dict[str, dict[str, Any]]
    final: dict[str, pd.DataFrame]


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _trace(tool: str, started: float, **params: Any) -> TraceEntry:
    return TraceEntry(
        timestamp=datetime.now(), tool=tool, parameters=params,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def _profile_for(request: ETLRequest, table: str) -> TableProfile | None:
    return next((t for t in request.profiling_report.tables if t.name == table), None)


def _resolve_key(options: ETLOptions, profile: TableProfile | None,
                 table: str, frame: pd.DataFrame) -> list[str]:
    candidates = (
        options.business_keys.get(table)
        or (profile.duplicates.business_key if profile else [])
        or options.business_keys.get("*")
        or []
    )
    return [c for c in candidates] if all(c in frame.columns for c in candidates) else []


def _issue_index(request: ETLRequest, table: str, rule: str) -> dict[str, str]:
    """column -> id du problème profiler pour une règle donnée."""
    return {
        i.column: i.id for i in request.profiling_report.issues
        if i.table == table and i.rule == rule and i.column
    }


def _etl_role(name: str, series: pd.Series, protected: set[str]) -> str:
    if name in protected:
        return "id"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if C.is_numeric(series):
        return "measure"
    return "dimension"


# --------------------------------------------------------------------------- #
# Noeuds
# --------------------------------------------------------------------------- #
def _load_node(state: ETLState) -> ETLState:
    request = state["request"]
    options = apply_preset(request.options)
    report = ETLReport(business_goal=request.profiling_report.business_goal)
    raw: dict[str, pd.DataFrame] = {}

    if request.profiling_report.status == "error" and not request.profiling_report.tables:
        report.errors.append("Le rapport du Profiler est en erreur : aucune table à nettoyer.")

    for source in request.sources:
        started = time.perf_counter()
        try:
            # threshold maximal : l'ETL travaille sur les données complètes, jamais un échantillon.
            frame, meta = load_source(source, threshold=sys.maxsize)
            raw[source.name] = frame
            report.trace.append(_trace("load_source", started, source_name=source.name,
                                       rows=meta["row_count"]))
            if _profile_for(request, source.name) is None:
                report.errors.append(
                    f"Source {source.name!r} absente du rapport du Profiler : "
                    "nettoyage par heuristiques uniquement."
                )
        except LoaderError as exc:
            report.errors.append(str(exc))
            report.trace.append(_trace("load_source", started, source_name=source.name,
                                       status="error"))
    return {"options": options, "report": report, "raw": raw}


def _clean_node(state: ETLState) -> ETLState:
    request, options, report = state["request"], state["options"], state["report"]
    clean: dict[str, pd.DataFrame] = {}
    ctx: dict[str, dict[str, Any]] = {}
    salt = os.environ.get(options.pii_salt_env, "biflow-default-salt")
    salt_is_default = options.pii_salt_env not in os.environ

    for table, raw_frame in state["raw"].items():
        started = time.perf_counter()
        profile = _profile_for(request, table)
        try:
            frame, info = _clean_one_table(
                table, raw_frame, profile, request, options, salt, salt_is_default
            )
        except Exception as exc:  # une table en échec ne doit pas bloquer les autres
            report.errors.append(f"Nettoyage de {table!r} impossible : {exc}")
            continue
        clean[table] = frame
        ctx[table] = info
        report.trace.append(_trace("clean_table", started, table=table,
                                   actions=len(info["actions"])))
    return {"clean": clean, "ctx": ctx}


def _clean_one_table(
    table: str, raw: pd.DataFrame, profile: TableProfile | None,
    request: ETLRequest, options: ETLOptions, salt: str, salt_is_default: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    rows_in = len(raw)

    def run(fn_result: tuple) -> pd.DataFrame:
        frame, new_actions = fn_result[0], fn_result[1]
        actions.extend(new_actions)
        return frame

    key = _resolve_key(options, profile, table, raw)
    profiler_roles = {c.name: c.semantic_role for c in profile.columns} if profile else {}
    protected = {c for c in raw.columns if C.looks_like_identifier(str(c))}
    protected |= {c for c, role in profiler_roles.items() if role == "id" and c in raw.columns}
    protected |= set(key)
    protected = {str(c) for c in protected}

    missing_ids = _issue_index(request, table, "missing_values")
    outlier_ids = _issue_index(request, table, "iqr_or_zscore_outlier")
    datetime_hints = {
        c.name for c in (profile.columns if profile else []) if c.inferred_type == "datetime"
    }

    # 1. Normalisation du texte
    frame = run(C.normalize_text_columns(
        raw, null_tokens=options.null_tokens, strip=options.strip_whitespace))

    # 2. Nulls structurels, mesurés avant toute imputation
    structural = C.detect_structural_missing(frame)

    # Règles de plausibilité : celles de l'utilisateur + « jamais négatif » déduit du nom
    # (minutes, data_mb, charges...) pour les colonnes numériques sans règle explicite.
    numeric_cols = {str(c) for c in raw.columns if C.is_numeric(raw[c])} | set(
        C.numeric_like_columns(frame, min_ratio=options.numeric_coercion_min_ratio,
                               protected=protected))
    explicit = {r.column for r in options.range_rules}
    range_rules = list(options.range_rules) + [
        RangeRule(column=c, min=0, auto=True) for c in raw.columns
        if str(c) in numeric_cols and str(c) not in explicit and str(c) not in protected
        and C.is_non_negative_name(str(c))
    ]

    # Mesure « avant » : sur les données brutes
    before = V.quality_snapshot(
        raw, null_tokens=options.null_tokens, structural=structural,
        range_rules=range_rules, mapping_rules=options.mapping_rules,
        min_ratio=options.numeric_coercion_min_ratio, protected=protected,
    )
    numeric_expected = list(before["text_numeric_columns"])

    # 3. Typage
    frame = run(C.coerce_numeric_columns(
        frame, min_ratio=options.numeric_coercion_min_ratio, protected=protected))
    frame = run(C.coerce_datetime_columns(
        frame, hinted=datetime_hints, min_ratio=options.numeric_coercion_min_ratio))
    if options.harmonize_case:
        frame = run(C.harmonize_categories(frame, protected=protected))

    # 4. Doublons
    rows_before_dedup = len(frame)
    if options.drop_exact_duplicates:
        frame = run(C.drop_exact_duplicates(frame, table=table))
    frame = run(C.resolve_key_duplicates(
        frame, key=key, strategy=options.key_duplicate_strategy, table=table))
    rows_removed = rows_in - len(frame)
    assert rows_before_dedup >= len(frame)

    # 5. Règles métier puis valeurs manquantes
    frame = run(C.apply_conditional_fills(frame, options.conditional_fills))
    frame = run(C.handle_missing_values(
        frame, structural=structural, protected=protected,
        numeric_strategy=options.numeric_missing_strategy,
        categorical_strategy=options.categorical_missing_strategy,
        unknown_label=options.unknown_label,
        not_applicable_label=options.not_applicable_label,
        drop_threshold_pct=options.drop_column_missing_pct,
        issue_ids=missing_ids,
    ))

    # 6. Cohérence / plausibilité (corrections uniquement si demandées par la règle)
    frame = run(C.apply_mapping_rules(frame, options.mapping_rules))
    frame = run(C.apply_range_rules(frame, range_rules))

    # 7. Valeurs aberrantes
    frame, outlier_actions, outlier_counts = C.handle_outliers(
        frame, protected=protected, strategy=options.outlier_strategy,
        iqr_multiplier=options.iqr_multiplier, issue_ids=outlier_ids)
    actions.extend(outlier_actions)

    # 8. PII : union des drapeaux du Profiler et de la détection propre à l'ETL
    # (défense en profondeur : une PII ne doit jamais passer parce que l'amont l'a ratée).
    pii_columns: dict[str, list[str]] = {
        c.name: list(c.pii_categories) for c in (profile.columns if profile else []) if c.is_pii
    }
    for column in raw.columns:
        own = detect_pii_column(str(column), raw[column])
        if own:
            pii_columns[str(column)] = sorted(set(pii_columns.get(str(column), [])) | set(own))
    frame, pii_actions, pii_treatment = C.apply_pii_policy(
        frame, pii_columns, pii_policy=options.pii_policy,
        identifier_policy=options.identifier_policy, salt=salt)
    actions.extend(pii_actions)

    # 9. Colonnes constantes (optionnel)
    if options.drop_constant_columns:
        constant = [c for c in frame.columns if frame[c].nunique(dropna=True) <= 1]
        if constant:
            frame = frame.drop(columns=constant)
            actions.append(C._action(
                "drop_constant_columns", None, 0,
                f"{len(constant)} colonne(s) constante(s) : {constant}.",
                "Supprimées.", "Une colonne constante n'apporte aucune information."))

    # Mesure « après » et contrôles
    after = V.quality_snapshot(
        frame, null_tokens=options.null_tokens, structural=structural,
        range_rules=range_rules, mapping_rules=options.mapping_rules,
        min_ratio=options.numeric_coercion_min_ratio, protected=protected,
    )
    checks = V.validate_clean_table(
        table, frame, key=key, structural=structural, expected_numeric=numeric_expected,
        range_rules=range_rules, mapping_rules=options.mapping_rules,
        product_rules=options.product_rules, rows_in=rows_in, rows_removed=rows_removed,
    )

    hashed = [c for c, t in pii_treatment.items() if t == "hashed"]
    return frame, {
        "actions": actions, "before": before, "after": after, "checks": checks,
        "key": key, "structural": structural, "protected": protected,
        "profiler_roles": profiler_roles, "pii_columns": pii_columns,
        "pii_treatment": pii_treatment, "rows_in": rows_in,
        "outlier_counts": outlier_counts,
        "default_salt_used": bool(hashed) and salt_is_default,
    }


def _validate_node(state: ETLState) -> ETLState:
    """Assemble actions, contrôles, résolutions de problèmes et statut global."""
    request, options, report = state["request"], state["options"], state["report"]
    started = time.perf_counter()

    for table, info in state["ctx"].items():
        for index, raw_action in enumerate(info["actions"]):
            base = f"{table}.{index:02d}.{raw_action['rule']}"
            if raw_action["column"]:
                base += f".{raw_action['column']}"
            report.actions.append(CleaningAction(
                id=base, step=index, table=table, **raw_action))
        report.validation_checks.extend(ValidationCheck(**c) for c in info["checks"])

        for check in info["checks"]:
            if not check["passed"] and check["severity"] != "info":
                report.remaining_issues.append(Issue(
                    id=f"{table}.remaining.{check['check']}"
                       + (f".{check['column']}" if check["column"] else ""),
                    severity=check["severity"], table=table,
                    column=check["column"] if check["column"] and "->" not in check["column"] else None,
                    rule=check["check"], evidence=check["detail"],
                    affected_rows=check["violations"],
                    suggested_action="Valider la règle métier avec le responsable de la donnée "
                                     "avant de la corriger.",
                ))
        for column, count in info["outlier_counts"].items():
            if options.outlier_strategy == "report":
                report.remaining_issues.append(Issue(
                    id=f"{table}.remaining.outliers.{column}", severity="low", table=table,
                    column=column, rule="outliers_reported",
                    evidence=f"{count} valeur(s) extrême(s) conservée(s) telles quelles.",
                    affected_rows=count,
                    suggested_action="Confirmer avec le métier qu'il s'agit de valeurs réelles.",
                ))
        if info["default_salt_used"]:
            report.remaining_issues.append(Issue(
                id=f"{table}.remaining.pii_default_salt", severity="medium", table=table,
                rule="pii_default_salt",
                evidence=f"Variable d'environnement {options.pii_salt_env} absente : "
                         "sel par défaut utilisé pour le hachage.",
                affected_rows=0,
                suggested_action=f"Définir {options.pii_salt_env} avant toute mise en production.",
            ))

    # Lien problème du Profiler -> action de l'ETL (traçabilité pour l'Auditor)
    by_issue: dict[str, list[CleaningAction]] = {}
    for action in report.actions:
        if action.source_issue_id:
            by_issue.setdefault(action.source_issue_id, []).append(action)
    for issue in request.profiling_report.issues:
        acts = by_issue.get(issue.id, [])
        if not acts:
            report.issue_resolutions.append(IssueResolution(
                issue_id=issue.id, status="not_addressed",
                note="Aucune règle de l'ETL ne couvre ce problème."))
        elif all(a.rule in REPORT_ONLY_RULES for a in acts):
            report.issue_resolutions.append(IssueResolution(
                issue_id=issue.id, status="reported_only", action_ids=[a.id for a in acts],
                note="Signalé sans modification : décision métier requise."))
        else:
            report.issue_resolutions.append(IssueResolution(
                issue_id=issue.id, status="resolved", action_ids=[a.id for a in acts],
                note=acts[0].rationale))

    report.trace.append(_trace("validate", started, checks=len(report.validation_checks)))
    return {"report": report}


def _export_node(state: ETLState) -> ETLState:
    """Renomme les colonnes, écrit les fichiers et construit le résumé par table."""
    options, report = state["options"], state["report"]
    final: dict[str, pd.DataFrame] = {}
    out_dir = Path(options.output_dir)

    for table, frame in state["clean"].items():
        started = time.perf_counter()
        info = state["ctx"][table]
        mapping = (
            C.snake_case_mapping([str(c) for c in frame.columns])
            if options.standardize_column_names else {str(c): str(c) for c in frame.columns}
        )
        out_frame = frame.rename(columns=mapping)

        path: str | None = None
        if options.write_outputs:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"{table}_clean.{options.output_format}"
                try:
                    if options.output_format == "parquet":
                        out_frame.to_parquet(target, index=False)
                    else:
                        out_frame.to_csv(target, index=False)
                except ImportError:
                    target = out_dir / f"{table}_clean.csv"
                    out_frame.to_csv(target, index=False)
                    report.errors.append("pyarrow absent : export CSV utilisé à la place de parquet.")
                path = str(target)
            except OSError as exc:
                report.errors.append(f"Écriture de {table!r} impossible : {exc}")

        columns: list[ColumnSummary] = []
        for original in frame.columns:
            name = str(original)
            role = _etl_role(name, frame[original], info["protected"])
            profiler_role = info["profiler_roles"].get(name)
            treatment = info["pii_treatment"].get(name)
            columns.append(ColumnSummary(
                original_name=name, name=mapping[name], dtype=str(frame[original].dtype),
                semantic_role=role,
                role_corrected_from=profiler_role if profiler_role and profiler_role != role else None,
                is_pii=treatment in {"hashed", "kept", "dropped"},
                pii_treatment=treatment,
            ))

        if options.build_star_schema:
            _build_star(table, out_frame, columns, info, options, report, out_dir)

        report.tables.append(CleanedTable(
            name=table, rows_in=info["rows_in"], rows_out=len(out_frame),
            columns_in=len(state["raw"][table].columns), columns_out=len(out_frame.columns),
            output_path=path, quality_before=QualityScores(**info["before"]),
            quality_after=QualityScores(**info["after"]), columns=columns,
        ))
        final[table] = out_frame
        report.trace.append(_trace("export", started, table=table, path=path))
    return {"report": report, "final": final}


def _build_star(table: str, frame: pd.DataFrame, columns: list[ColumnSummary],
                info: dict[str, Any], options: ETLOptions, report: ETLReport,
                out_dir: Path) -> None:
    """Propose, construit, vérifie et écrit le modèle en étoile d'une table."""
    started = time.perf_counter()
    try:
        roles = {c.name: c.semantic_role for c in columns}
        mapping = {c.original_name: c.name for c in columns}
        key = [mapping.get(k, k) for k in info["key"]]
        personal = {c.name for c in columns if c.pii_treatment in {"hashed", "kept"}}
        plan = M.propose_star_schema(frame, roles, table=table, business_key=key,
                                     spec=options.star_schemas.get(table),
                                     entity_columns=personal)
        if not plan["measures"]:
            report.remaining_issues.append(Issue(
                id=f"{table}.star_schema_skipped", severity="low", table=table,
                rule="star_schema_skipped", affected_rows=0,
                evidence="Aucune mesure numérique détectée : un fait sans mesure n'a pas d'intérêt BI.",
                suggested_action="Définir les mesures via ETLOptions.star_schemas si besoin."))
            return
        fact, dims = M.build_star_tables(frame, plan)
        checks = M.check_star(frame, fact, dims, plan)
    except Exception as exc:  # l'étoile est un bonus : elle ne doit jamais casser le nettoyage
        report.errors.append(f"Modèle en étoile impossible pour {table!r} : {exc}")
        return

    check_ids = []
    for c in checks:
        vc = ValidationCheck(
            id=f"{table}.{c['check']}", table=table, check=c["check"], passed=c["passed"],
            severity="info" if c["passed"] else "high", violations=c["violations"],
            detail=c["detail"])
        report.validation_checks.append(vc)
        check_ids.append(vc.id)
        if not c["passed"]:
            report.remaining_issues.append(Issue(
                id=f"{table}.remaining.{c['check']}", severity="high", table=table,
                rule=c["check"], evidence=c["detail"], affected_rows=c["violations"],
                suggested_action="Vérifier la définition de l'étoile (grain, dimensions)."))

    files: dict[str, str] = {}
    if options.write_outputs:
        try:
            target = out_dir / "model" / table
            target.mkdir(parents=True, exist_ok=True)
            for name, data in {plan["fact_name"]: fact, **dims}.items():
                path = target / f"{name}.csv"
                data.to_csv(path, index=False)
                files[name] = str(path)
            mermaid = M.to_mermaid(plan, fact, dims)
            (target / "star_schema.mmd").write_text(mermaid, encoding="utf-8")
            files["star_schema.mmd"] = str(target / "star_schema.mmd")
        except OSError as exc:
            report.errors.append(f"Écriture du modèle en étoile de {table!r} impossible : {exc}")
    else:
        mermaid = M.to_mermaid(plan, fact, dims)

    report.star_schemas.append(StarSchema(
        table=table, source=plan["source"], fact_name=plan["fact_name"], grain=plan["grain"],
        grain_reason=plan["grain_reason"], fact_rows=len(fact),
        measures=[StarMeasure(**m) for m in plan["measures"]],
        dimensions=[StarDimension(name=d["name"], key=d["key"], role=d["role"],
                                  columns=d["columns"], rows=len(dims[d["name"]]),
                                  rationale=d["rationale"]) for d in plan["dimensions"]],
        date_columns=plan["date_columns"], excluded_columns=plan["excluded"],
        files=files, check_ids=check_ids, mermaid=mermaid,
    ))
    report.trace.append(_trace("build_star_schema", started, table=table,
                               dimensions=len(plan["dimensions"])))


# --------------------------------------------------------------------------- #
# Synthèse
# --------------------------------------------------------------------------- #
def _fallback_summary(report: ETLReport) -> str:
    if not report.tables:
        return "Le nettoyage n'a pas abouti : " + ("; ".join(report.errors) or "aucune table.")
    lines = []
    for t in report.tables:
        lines.append(
            f"Table {t.name} : {t.rows_in} → {t.rows_out} ligne(s), score qualité "
            f"{t.quality_before.overall} → {t.quality_after.overall}/100."
        )
    lines.append("Corrections appliquées :")
    for a in report.actions:
        if a.rows_affected:
            col = f".{a.column}" if a.column else ""
            lines.append(f"- {a.table}{col} [{a.rule}] {a.rows_affected} ligne(s) : {a.rationale}")
    pending = [i for i in report.remaining_issues]
    if pending:
        lines.append("À surveiller :")
        lines += [f"- {i.severity} {i.table}" + (f".{i.column}" if i.column else "")
                  + f" : {i.evidence}" for i in pending]
    else:
        lines.append("Aucun problème restant selon les contrôles exécutés.")
    return "\n".join(lines)


def _load_llm():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=0)
    except Exception:
        return None


def _numbers_grounded(summary: str, source_json: str) -> bool:
    """Rejette une synthèse LLM qui contient un nombre absent des résultats."""
    allowed = set(re.findall(r"\d+(?:\.\d+)?", source_json))
    for number in re.findall(r"\d+(?:[.,]\d+)?", summary):
        if len(number.replace(",", ".").replace(".", "")) < 2:
            continue
        if number.replace(",", ".") not in allowed:
            return False
    return True


def _summarize(report: ETLReport, use_llm: bool) -> str:
    fallback = _fallback_summary(report)
    if not use_llm or not report.tables:
        return fallback
    llm = _load_llm()
    if llm is None:
        return fallback
    data = {
        "tables": [t.model_dump(mode="json", exclude={"columns"}) for t in report.tables],
        "actions": [a.model_dump(mode="json") for a in report.actions if a.rows_affected],
        "remaining_issues": [i.model_dump(mode="json") for i in report.remaining_issues],
        "errors": report.errors,
    }
    results = json.dumps(data, ensure_ascii=False)
    try:
        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", SUMMARY_PROMPT.format(business_goal=report.business_goal, results=results)),
        ])
        text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        return fallback
    # Garde-fou anti-hallucination : un chiffre inventé => on retombe sur la synthèse déterministe.
    return text if _numbers_grounded(text, results) else fallback


def _summary_node(state: ETLState) -> ETLState:
    report = state["report"]
    report.llm_summary = _summarize(report, state["options"].use_llm)

    options = state["options"]
    low_quality = any(t.quality_after.overall < options.min_quality_score for t in report.tables)
    blocking = any(i.severity in {"medium", "high"} for i in report.remaining_issues)
    if not report.tables:
        report.status = "error"
    elif report.errors or blocking or low_quality:
        report.status = "warning"
    else:
        report.status = "success"
    return {"report": report}


def _persist_node(state: ETLState) -> ETLState:
    options, report = state["options"], state["report"]
    if options.write_outputs:
        try:
            out_dir = Path(options.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "etl_report.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            report.errors.append(f"Écriture du rapport impossible : {exc}")
    return {"report": report}


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #
def build_graph():
    graph = StateGraph(ETLState)
    graph.add_node("load", _load_node)
    graph.add_node("clean", _clean_node)
    graph.add_node("validate", _validate_node)
    graph.add_node("export", _export_node)
    graph.add_node("summarize", _summary_node)
    graph.add_node("persist", _persist_node)
    graph.add_edge(START, "load")
    graph.add_edge("load", "clean")
    graph.add_edge("clean", "validate")
    graph.add_edge("validate", "export")
    graph.add_edge("export", "summarize")
    graph.add_edge("summarize", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


_etl_graph = build_graph()


def run_with_data(request: ETLRequest) -> tuple[ETLReport, dict[str, pd.DataFrame]]:
    """Exécute l'agent et retourne (rapport, DataFrames nettoyés en mémoire)."""
    try:
        result = _etl_graph.invoke({"request": request})
        return result["report"], result.get("final", {})
    except Exception as exc:  # dernier filet de sécurité pour l'Orchestrator
        return ETLReport(
            business_goal=request.profiling_report.business_goal, status="error",
            errors=[f"Erreur inattendue de l'ETL Agent : {exc}"],
        ), {}


def run(request: ETLRequest) -> ETLReport:
    """Point d'entrée public pour l'Orchestrator."""
    return run_with_data(request)[0]
