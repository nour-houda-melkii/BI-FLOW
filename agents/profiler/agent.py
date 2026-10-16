"""Graphe LangGraph du Data Profiler Agent."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.profiler.models import (
    ColumnProfile,
    DuplicateProfile,
    Issue,
    ProfileRequest,
    ProfilingReport,
    RelationshipProfile,
    TableProfile,
    TraceEntry,
)
from agents.profiler.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from agents.tools.anomalies import iqr_outliers, zscore_outliers
from agents.tools.loader import LoaderError, load_source
from agents.tools.pii import detect_pii_column, mask_pii
from agents.tools.quality import duplicate_profile, missing_by_column
from agents.tools.relations import candidate_column_pairs, profile_relation
from agents.tools.schema import (
    candidate_keys,
    declared_type,
    infer_column_type,
)
from agents.tools.semantic import suggest_semantic_role
from agents.tools.stats import profile_column


class ProfilerState(TypedDict, total=False):
    request: ProfileRequest
    report: ProfilingReport
    frames: dict[str, pd.DataFrame]


def _make_fallback_summary(report: ProfilingReport) -> str:
    """Synthèse sans LLM ; ne présente que les valeurs du rapport."""
    if report.status == "error" and not report.tables:
        details = "; ".join(report.errors) or "aucune table n'a pu être profilée"
        return f"Le profilage n'a pas abouti : {details}"

    table_descriptions = [
        f"{table.name} : {table.row_count} ligne(s), "
        f"{table.column_count} colonne(s)"
        + (
            f", échantillon de {table.sample_rows} ligne(s)"
            if table.sampled
            else ""
        )
        for table in report.tables
    ]
    issue_descriptions = [
        f"{issue.severity} — {issue.table}"
        + (f".{issue.column}" if issue.column else "")
        + f" : {issue.rule}. {issue.evidence} Action : {issue.suggested_action}"
        for issue in report.issues
    ]

    summary = "Tables profilées : " + (
        "; ".join(table_descriptions) if table_descriptions else "aucune"
    )
    if issue_descriptions:
        summary += "\nProblèmes prioritaires : " + " ".join(issue_descriptions)
    else:
        summary += "\nAucun problème couvert par les règles exécutées n'a été signalé."
    return summary


def _load_llm():
    """Retourne un Ollama local si configuré et disponible, sinon None."""
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model="llama3.1",
            temperature=0,
        )
    except Exception:
        # Dépendance, serveur ou modèle non disponibles : la synthèse de repli
        # permet à l'agent de retourner tout de même les résultats déterministes.
        return None


def _summarize_with_llm(report: ProfilingReport) -> str:
    """Demande au LLM d'interpréter le rapport, sans lui confier de calcul."""
    llm = _load_llm()
    if llm is None:
        return _make_fallback_summary(report)

    # Envoie uniquement les résultats synthétisés, pas les données sources.
    data = {
        "tables": [
            {
                "name": table.name,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "sampled": table.sampled,
                "sample_rows": table.sample_rows,
            }
            for table in report.tables
        ],
        "issues": [
            issue.model_dump(mode="json")
            for issue in report.issues
        ],
        "relationships": [
            relation.model_dump(mode="json")
            for relation in report.relationships
        ],
        "errors": report.errors,
    }
    prompt = SUMMARY_PROMPT.format(
        business_goal=report.business_goal,
        results=json.dumps(data, ensure_ascii=False),
    )

    try:
        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ])
        content = response.content
        return content if isinstance(content, str) else str(content)
    except Exception:
        return _make_fallback_summary(report)


def _profile_node(state: ProfilerState) -> ProfilerState:
    request = state["request"]
    report = ProfilingReport(business_goal=request.business_goal)
    frames: dict[str, pd.DataFrame] = {}
    trace: list[TraceEntry] = []

    for source in request.sources:
        started = time.perf_counter()
        try:
            frame, metadata = load_source(
                source,
                threshold=request.options.sample_threshold_rows,
                sample_size=request.options.sample_size,
                random_seed=request.options.random_seed,
            )
            frames[source.name] = frame
            trace.append(TraceEntry(
                timestamp=datetime.now(),
                tool="load_source",
                parameters={
                    "source_name": source.name,
                    "source_type": metadata["source_type"],
                    "sampled": metadata["sampled"],
                    "sample_rows": metadata["sample_rows"],
                },
                duration_ms=(time.perf_counter() - started) * 1000,
            ))

            if frame.empty:
                report.errors.append(f"Source {source.name!r} chargée, mais vide.")
                continue

            columns: list[ColumnProfile] = []
            table_issues: list[Issue] = []

            for column in frame.columns:
                name = str(column)
                series = frame[column]
                inferred = infer_column_type(series)
                stats = profile_column(
                    series,
                    rare_max_count=request.options.rare_category_max_count,
                    rare_max_fraction=request.options.rare_category_max_fraction,
                )

                pii_categories = (
                    detect_pii_column(name, series)
                    if request.options.detect_pii
                    else []
                )
                rare_values = stats.get("rare_values", [])
                if pii_categories:
                    rare_values = [
                        {
                            "value": mask_pii(item.get("value"), pii_categories),
                            "count": item.get("count"),
                        }
                        for item in rare_values
                    ]

                anomaly_mask = pd.Series(False, index=frame.index)
                if request.options.detect_anomalies and inferred == "numeric":
                    anomaly_mask = iqr_outliers(
                        series,
                        multiplier=request.options.iqr_multiplier,
                    ) | zscore_outliers(
                        series,
                        threshold=request.options.zscore_threshold,
                    )

                anomaly_count = int(anomaly_mask.sum())
                anomaly_examples: list[Any] = []
                if anomaly_count:
                    raw_examples = series.loc[anomaly_mask].head(5).tolist()
                    anomaly_examples = [
                        mask_pii(value, pii_categories)
                        for value in raw_examples
                    ]

                columns.append(ColumnProfile(
                    name=name,
                    declared_type=declared_type(series),
                    inferred_type=inferred,
                    semantic_role=suggest_semantic_role(name, series),
                    missing_count=int(stats["missing_count"]),
                    missing_pct=float(stats["missing_pct"]),
                    cardinality=int(stats["cardinality"]),
                    stats={
                        key: value
                        for key, value in stats.items()
                        if key not in {"missing_count", "missing_pct", "cardinality", "rare_values"}
                    },
                    rare_values=rare_values,
                    is_pii=bool(pii_categories),
                    pii_categories=pii_categories,
                    anomaly_count=anomaly_count,
                    anomaly_examples=anomaly_examples,
                ))

                missing = int(series.isna().sum())
                missing_pct = float(stats["missing_pct"])
                if missing:
                    severity = "high" if missing_pct >= 20 else (
                        "medium" if missing_pct >= 5 else "low"
                    )
                    table_issues.append(Issue(
                        id=f"{source.name}.{name}.missing",
                        severity=severity,
                        table=source.name,
                        column=name,
                        rule="missing_values",
                        evidence=f"{missing} valeur(s) manquante(s), soit {missing_pct:.2f}% des lignes profilées.",
                        affected_rows=missing,
                        suggested_action="Évaluer une règle métier de traitement des valeurs manquantes avant l'ETL.",
                    ))

                if anomaly_count:
                    table_issues.append(Issue(
                        id=f"{source.name}.{name}.iqr_zscore_outliers",
                        severity="medium",
                        table=source.name,
                        column=name,
                        rule="iqr_or_zscore_outlier",
                        evidence=(
                            f"{anomaly_count} ligne(s) signalée(s) par IQR ou z-score "
                            f"sur {len(frame)} lignes profilées."
                        ),
                        affected_rows=anomaly_count,
                        suggested_action="Vérifier les valeurs signalées avec les règles métier avant toute correction.",
                    ))

            key = request.options.business_keys.get(source.name, [])
            try:
                duplicates = duplicate_profile(frame, key)
            except ValueError as exc:
                duplicates = duplicate_profile(frame)
                report.errors.append(f"{source.name}: {exc}")

            table_profile = TableProfile(
                name=source.name,
                source_name=source.name,
                row_count=int(metadata["row_count"]),
                column_count=len(frame.columns),
                memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
                sampled=bool(metadata["sampled"]),
                sample_rows=int(metadata["sample_rows"]) if metadata["sampled"] else None,
                columns=columns,
                duplicates=DuplicateProfile(**duplicates),
            )
            report.tables.append(table_profile)
            report.issues.extend(table_issues)

            if duplicates["exact_count"]:
                report.issues.append(Issue(
                    id=f"{source.name}.exact_duplicates",
                    severity="medium",
                    table=source.name,
                    rule="exact_duplicate_rows",
                    evidence=(
                        f"{duplicates['exact_count']} ligne(s) dupliquée(s) "
                        f"exactement sur {len(frame)} lignes profilées."
                    ),
                    affected_rows=int(duplicates["exact_count"]),
                    suggested_action="Confirmer la règle de déduplication métier avant toute suppression par l'ETL.",
                ))

            if duplicates["business_key_duplicate_count"]:
                report.issues.append(Issue(
                    id=f"{source.name}.business_key_duplicates",
                    severity="high",
                    table=source.name,
                    rule="business_key_not_unique",
                    evidence=(
                        f"{duplicates['business_key_duplicate_count']} doublon(s) "
                        f"sur la clé métier {key!r}."
                    ),
                    affected_rows=int(duplicates["business_key_duplicate_count"]),
                    suggested_action="Valider la clé métier et définir la règle de résolution des doublons.",
                ))

        except LoaderError as exc:
            report.errors.append(str(exc))
            trace.append(TraceEntry(
                timestamp=datetime.now(),
                tool="load_source",
                parameters={"source_name": source.name, "status": "error"},
                duration_ms=(time.perf_counter() - started) * 1000,
            ))

    # Évalue les paires candidates entre les sources chargées.
    if request.options.detect_relationships and len(frames) > 1:
        started = time.perf_counter()
        for table_a, col_a, table_b, col_b in candidate_column_pairs(frames):
            try:
                relationship = profile_relation(
                    table_a, col_a, frames[table_a],
                    table_b, col_b, frames[table_b],
                )
                report.relationships.append(RelationshipProfile(**relationship))
            except (KeyError, TypeError, ValueError):
                # Une paire impossible à comparer ne doit pas interrompre les autres.
                continue
        trace.append(TraceEntry(
            timestamp=datetime.now(),
            tool="profile_candidate_relationships",
            parameters={"table_count": len(frames)},
            duration_ms=(time.perf_counter() - started) * 1000,
        ))

    if report.errors and not report.tables:
        report.status = "error"
    elif report.errors or report.issues:
        report.status = "warning"
    else:
        report.status = "success"

    report.trace = trace
    return {"report": report, "frames": frames}


def _summary_node(state: ProfilerState) -> ProfilerState:
    report = state["report"]
    report.llm_summary = _summarize_with_llm(report)
    return {"report": report}


def build_graph():
    """Construit le graphe LangGraph du profileur."""
    graph = StateGraph(ProfilerState)
    graph.add_node("profile", _profile_node)
    graph.add_node("summarize", _summary_node)
    graph.add_edge(START, "profile")
    graph.add_edge("profile", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


_profiler_graph = build_graph()


def run(profile_request: ProfileRequest) -> ProfilingReport:
    """Point d'entrée public : exécute le graphe et renvoie un rapport Pydantic."""
    try:
        result = _profiler_graph.invoke({"request": profile_request})
        return result["report"]
    except Exception as exc:
        # Dernier filet de sécurité pour l'Orchestrator.
        return ProfilingReport(
            business_goal=profile_request.business_goal,
            status="error",
            errors=[f"Erreur inattendue du Profiler Agent : {exc}"],
        )
