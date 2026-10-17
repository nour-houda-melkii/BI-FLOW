"""CLI : python -m agents.kpi --etl-report output/etl/etl_report.json"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.kpi.agent import run
from agents.kpi.models import KPIOptions, KPIRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KPI & Semantic Agent")
    parser.add_argument("--etl-report", required=True, help="JSON produit par l'ETL Agent")
    parser.add_argument("--output-dir", default="output/kpi")
    parser.add_argument("--category", action="append", default=None,
                        help="limiter à une catégorie (répétable), ex. --category Churn")
    parser.add_argument("--no-llm", action="store_true", help="synthèse déterministe uniquement")
    args = parser.parse_args(argv)

    etl_report = json.loads(Path(args.etl_report).read_text(encoding="utf-8"))
    request = KPIRequest(
        etl_report=etl_report,
        options=KPIOptions(output_dir=args.output_dir, categories=args.category,
                           use_llm=not args.no_llm),
    )
    report = run(request)

    print(json.dumps({
        "status": report.status,
        "kpis": [{"id": k.id, "value": k.value, "unit": k.unit, "status": k.status}
                 for k in report.kpis if k.status != "not_computable"],
        "not_computable": [k.id for k in report.kpis if k.status == "not_computable"],
        "caveats": report.caveats, "errors": report.errors,
        "files": report.output_files,
    }, indent=2, ensure_ascii=False))
    print("\n" + report.llm_summary)
    return 0 if report.status != "error" else 1


if __name__ == "__main__":
    sys.exit(main())