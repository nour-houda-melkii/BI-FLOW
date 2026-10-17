# KPI & Semantic Agent

Calcule les KPI télécom à partir des tables nettoyées par l'ETL Agent, avec leur définition
métier, leur formule et leur traçabilité. Le LLM ne calcule jamais un chiffre.

```
Profiler -> ETL (etl_report.json + tables nettoyées) -> KPI Agent -> BI Analyst / Auditor XAI
```

## Lancer

```bash
python run_pipeline.py data/telco.csv --preset telco                  # Profiler + ETL + KPI
python -m agents.kpi --etl-report output/telco/etl_report.json --no-llm
python -m agents.kpi --etl-report output/telco/etl_report.json --category Churn
pytest tests/kpi
```

Depuis l'Orchestrator :

```python
from agents.kpi import KPIRequest, run

kpi_report = run(KPIRequest(etl_report=etl_report.model_dump(mode="json")), frames=etl_frames)
```

## Entrée

`KPIRequest.etl_report` = `ETLReport.model_dump(mode="json")` (ou le contenu de `etl_report.json`).
Champs utilisés : `business_goal`, `status`, `tables[].{name, rows_out, output_path, quality_after,
columns[].{name, semantic_role, is_pii, pii_treatment}}`, `remaining_issues`.
`frames` (optionnel) : DataFrames nettoyés en mémoire, sinon lecture depuis `output_path`.

## Sortie pour le BI Analyst Agent

`KPIReport` (`kpi_report.json`) :

| Champ | Contenu |
|---|---|
| `status` | `success` / `warning` / `error` |
| `kpis[]` | un KPI : `id`, `name`, `category`, `definition`, `formula`, `unit`, `status`, `value`, `table`, `rows_used`, `columns_used`, `missing_inputs`, `breakdowns`, `justification`, `warnings` |
| `caveats` | réserves héritées de la qualité des données (score faible, problèmes ETL medium/high, lignes incohérentes) |
| `llm_summary` | synthèse lisible (déterministe, ou LLM avec garde-fou anti-chiffres inventés) |
| `output_files` | chemins de `kpi_report.json` et `kpi_values.csv` |

Statut d'un KPI : `computed` (fiable), `suspect` (valeur hors plage plausible : à vérifier),
`not_computable` (colonnes requises absentes : jamais de valeur inventée).

`kpi_values.csv` (format tidy, prêt pour Plotly / Streamlit) :
`kpi_id, kpi_name, category, unit, status, table, dimension, segment, value, rows`
(`dimension = segment = "all"` pour la valeur globale).

Lecture côté Analyst :

```python
import pandas as pd
from agents.kpi.models import KPIReport

report = KPIReport.model_validate_json(open("output/telco/kpi/kpi_report.json", encoding="utf-8").read())
values = pd.read_csv("output/telco/kpi/kpi_values.csv")
churn_by_contract = values[(values.kpi_id == "churn_rate") & (values.dimension == "contract")]
```

## Ajouter un KPI

Ajouter une entrée dans `agents/kpi/catalog.py` (`agg` : mean, sum, count_distinct, rate,
count_flag, sum_where, ratio ; `inputs` : rôle -> alias de colonnes). Aucun autre code à toucher.

## Limites

- Les colonnes sont retrouvées par alias (casse et `_` ignorés) : ajouter l'alias si un jeu de
  données nomme autrement une colonne.
- Ventilations : 3 dimensions max, faible cardinalité, hors PII, segments de moins de 30 lignes ignorés.
- La table plate nettoyée est lue ; le modèle en étoile est produit par l'ETL mais pas encore consommé.