"""Graphe LangGraph du KPI & Semantic Agent.

    load -> compute -> validate -> summarize -> persist

Principe : la sélection des KPI (catalogue + colonnes disponibles) et leur calcul sont
déterministes et tracés ; le LLM (optionnel) ne fait que rédiger la synthèse.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.kpi.catalog import CATALOG, PREFERRED_DIMENSIONS
from agents.tools.kpi import (
    compute_breakdown,
    compute_value,
    pick_dimensions,
    resolve_inputs,
)
from agents.kpi.models import KPIOptions, KPIReport, KPIRequest, KPIResult
from agents.kpi.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from agents.profiler.models import TraceEntry


class KPIState(TypedDict, total=False):
    request: KPIRequest
    options: KPIOptions
    report: KPIReport
    frames: dict[str, pd.DataFrame]
    tables_meta: dict[str, dict[str, Any]]


def _trace(tool: str, started: float, **params: Any) -> TraceEntry:
    return TraceEntry(
        timestamp=datetime.now(), tool=tool, parameters=params,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


# --------------------------------------------------------------------------- #
# Noeuds
# --------------------------------------------------------------------------- #
def _load_node(state: KPIState) -> KPIState:
    request = state["request"]
    etl = request.etl_report
    report = KPIReport(business_goal=etl.get("business_goal", ""),
                       etl_status=etl.get("status", "unknown"))
    tables_meta = {t["name"]: t for t in etl.get("tables", [])}
    frames: dict[str, pd.DataFrame] = dict(state.get("frames") or {})

    if not tables_meta:
        report.errors.append("Le rapport de l'ETL ne contient aucune table nettoyée.")

    for name, meta in tables_meta.items():
        if name in frames:  # DataFrame déjà fourni en mémoire par l'Orchestrator
            continue
        started = time.perf_counter()
        path = meta.get("output_path")
        try:
            if not path:
                raise FileNotFoundError("aucun output_path dans le rapport de l'ETL")
            frames[name] = (pd.read_parquet(path) if path.endswith(".parquet")
                            else pd.read_csv(path))
            report.trace.append(_trace("load_table", started, table=name, rows=len(frames[name])))
        except Exception as exc:  # une table illisible ne doit pas bloquer les autres
            report.errors.append(f"Lecture de la table {name!r} impossible : {exc}")
    return {"options": request.options, "report": report, "frames": frames,
            "tables_meta": tables_meta}


def _compute_node(state: KPIState) -> KPIState:
    options, report = state["options"], state["report"]
    catalog = [k for k in CATALOG if not options.categories or k.category in options.categories]
    computed: set[str] = set()
    missing_by_kpi: dict[str, dict[str, list[str]]] = {}
    undefined: set[str] = set()

    for table, df in state["frames"].items():
        started = time.perf_counter()
        dims = pick_dimensions(state["tables_meta"][table], df, options.max_segments,
                               PREFERRED_DIMENSIONS)
        for defn in catalog:
            cols, missing = resolve_inputs(defn, df)
            if missing:
                missing_by_kpi.setdefault(defn.id, missing)
                continue
            value = compute_value(defn, df, cols)
            if value is None:
                undefined.add(defn.id)
                continue
            breakdowns = compute_breakdown(
                defn, df, cols, dims[: options.max_breakdown_dimensions], options.min_segment_rows)
            used = ", ".join(f"{role}={col}" for role, col in cols.items())
            report.kpis.append(KPIResult(
                id=defn.id, name=defn.name, category=defn.category, definition=defn.definition,
                formula=defn.formula, unit=defn.unit, status="computed", value=value,
                table=table, rows_used=len(df), columns_used=cols, breakdowns=breakdowns,
                justification=(f"Calculable : colonnes trouvées dans « {table} » ({used}). "
                               f"Calcul déterministe : {defn.formula}."),
            ))
            computed.add(defn.id)
        report.trace.append(_trace("compute_table", started, table=table, kpis=len(computed)))

    for defn in catalog:
        if defn.id in computed:
            continue
        if defn.id in undefined:
            why, missing = "valeur indéfinie (table vide ou dénominateur nul).", {}
        else:
            missing = missing_by_kpi.get(defn.id, {})
            roles = "; ".join(f"{role} ← {'/'.join(aliases)}" for role, aliases in missing.items())
            why = f"colonnes requises introuvables ou de mauvais type ({roles})."
        report.kpis.append(KPIResult(
            id=defn.id, name=defn.name, category=defn.category, definition=defn.definition,
            formula=defn.formula, unit=defn.unit, status="not_computable",
            missing_inputs=missing, justification=f"Non calculable : {why}"))
    return {"report": report}


def _validate_node(state: KPIState) -> KPIState:
    """Plausibilité des valeurs, cohérence avec l'ETL, réserves héritées de la qualité des données."""
    options, report, etl = state["options"], state["report"], state["request"].etl_report
    started = time.perf_counter()

    defs = {k.id: k for k in CATALOG}
    for kpi in report.kpis:
        if kpi.status != "computed" or kpi.value is None:
            continue
        low, high = defs[kpi.id].valid_range
        if (low is not None and kpi.value < low) or (high is not None and kpi.value > high):
            kpi.status = "suspect"
            kpi.warnings.append(
                f"Valeur {kpi.value} hors de la plage plausible [{low}, {high}] : à vérifier.")

    for name, meta in state["tables_meta"].items():
        frame = state["frames"].get(name)
        if frame is not None and meta.get("rows_out") is not None and len(frame) != meta["rows_out"]:
            report.caveats.append(
                f"Table {name} : {len(frame)} lignes lues, {meta['rows_out']} annoncées par l'ETL.")
        score = (meta.get("quality_after") or {}).get("overall")
        if score is not None and score < options.min_quality_score:
            report.caveats.append(
                f"Table {name} : score qualité après nettoyage {score}/100 "
                f"(seuil {options.min_quality_score}).")

    for issue in etl.get("remaining_issues", []):
        if issue.get("severity") in {"medium", "high"}:
            target = f"{issue.get('table')}" + (f".{issue['column']}" if issue.get("column") else "")
            report.caveats.append(f"ETL — {target} : {issue.get('evidence', '')}")

    report.trace.append(_trace("validate", started, kpis=len(report.kpis)))
    return {"report": report}


# --------------------------------------------------------------------------- #
# Synthèse
# --------------------------------------------------------------------------- #
def _fallback_summary(report: KPIReport) -> str:
    computed = [k for k in report.kpis if k.status in {"computed", "suspect"}]
    if not computed:
        return "Aucun KPI calculé : " + ("; ".join(report.errors) or "colonnes requises absentes.")
    lines = [f"{len(computed)} KPI calculé(s) sur {len(report.kpis)} du catalogue."]
    for k in computed:
        flag = " [SUSPECT]" if k.status == "suspect" else ""
        lines.append(f"- {k.name} ({k.category}) : {k.value} {k.unit}{flag} — {k.definition}")
    skipped = [k for k in report.kpis if k.status == "not_computable"]
    if skipped:
        lines.append(f"Non calculables avec ces données : {', '.join(k.name for k in skipped)}.")
    if report.caveats:
        lines.append("Réserves :")
        lines += [f"- {c}" for c in report.caveats]
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


def _summarize(report: KPIReport, use_llm: bool) -> str:
    fallback = _fallback_summary(report)
    if not use_llm or not any(k.status != "not_computable" for k in report.kpis):
        return fallback
    llm = _load_llm()
    if llm is None:
        return fallback
    data = {
        "kpis": [k.model_dump(mode="json", include={
            "name", "category", "definition", "unit", "status", "value", "warnings"})
            for k in report.kpis],
        "caveats": report.caveats,
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
    # Garde-fou anti-hallucination : un chiffre inventé => synthèse déterministe.
    return text if _numbers_grounded(text, results) else fallback


def _summary_node(state: KPIState) -> KPIState:
    report = state["report"]
    report.llm_summary = _summarize(report, state["options"].use_llm)

    has_value = any(k.status in {"computed", "suspect"} for k in report.kpis)
    suspect = any(k.status == "suspect" for k in report.kpis)
    if not has_value:
        report.status = "error"
    elif suspect or report.caveats or report.errors:
        report.status = "warning"
    else:
        report.status = "success"
    return {"report": report}


def _persist_node(state: KPIState) -> KPIState:
    """Écrit kpi_report.json (contrat complet) et kpi_values.csv (format tidy pour l'Analyst)."""
    options, report = state["options"], state["report"]
    if not options.write_outputs:
        return {"report": report}
    try:
        out_dir = Path(options.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path, csv_path = out_dir / "kpi_report.json", out_dir / "kpi_values.csv"

        rows = []
        for k in report.kpis:
            if k.status == "not_computable":
                continue
            base = {"kpi_id": k.id, "kpi_name": k.name, "category": k.category,
                    "unit": k.unit, "status": k.status, "table": k.table}
            rows.append({**base, "dimension": "all", "segment": "all",
                         "value": k.value, "rows": k.rows_used})
            for dim, segments in k.breakdowns.items():
                rows.extend({**base, "dimension": dim, "segment": s.segment,
                             "value": s.value, "rows": s.rows} for s in segments)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        report.output_files = {"report": str(json_path), "values": str(csv_path)}
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        report.errors.append(f"Écriture des résultats impossible : {exc}")
    return {"report": report}


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #
def build_graph():
    graph = StateGraph(KPIState)
    graph.add_node("load", _load_node)
    graph.add_node("compute", _compute_node)
    graph.add_node("validate", _validate_node)
    graph.add_node("summarize", _summary_node)
    graph.add_node("persist", _persist_node)
    graph.add_edge(START, "load")
    graph.add_edge("load", "compute")
    graph.add_edge("compute", "validate")
    graph.add_edge("validate", "summarize")
    graph.add_edge("summarize", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


_kpi_graph = build_graph()


def run(request: KPIRequest, frames: dict[str, pd.DataFrame] | None = None) -> KPIReport:
    """Point d'entrée public pour l'Orchestrator.

    `frames` (optionnel) : DataFrames nettoyés déjà en mémoire (ex. issus de
    `agents.etl.agent.run_with_data`) ; sinon les tables sont lues depuis `output_path`.
    """
    try:
        return _kpi_graph.invoke({"request": request, "frames": frames or {}})["report"]
    except Exception as exc:  # dernier filet de sécurité pour l'Orchestrator
        return KPIReport(
            business_goal=request.etl_report.get("business_goal", ""), status="error",
            errors=[f"Erreur inattendue du KPI Agent : {exc}"],
        )