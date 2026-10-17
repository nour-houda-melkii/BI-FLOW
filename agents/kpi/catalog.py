"""Catalogue de KPI télécom = couche sémantique du KPI & Semantic Agent.

Chaque KPI porte : une définition métier, une formule lisible, les colonnes dont il
a besoin (par rôle, avec des alias de noms de colonnes) et une plage de plausibilité.
Pour ajouter un KPI : ajouter une entrée dans CATALOG, aucun autre code à toucher.

Types de calcul (`agg`) :
  mean            moyenne de inputs["value"]
  sum             somme de inputs["value"]
  count_distinct  nombre de valeurs distinctes de inputs["id"]
  rate            % de lignes où inputs["flag"] est vrai (Yes/1/True)
  count_flag      nombre de lignes où inputs["flag"] est vrai
  sum_where       somme de inputs["value"] là où inputs["flag"] est vrai
  ratio           somme(inputs["num"]) / somme(inputs["den"]) * scale
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KPIDefinition:
    id: str
    name: str
    category: str
    definition: str
    formula: str
    agg: str
    inputs: dict[str, list[str]]
    unit: str = ""
    scale: float = 1.0
    valid_range: tuple[float | None, float | None] = (0.0, None)
    tags: list[str] = field(default_factory=list)


# Colonnes de segmentation à privilégier pour les ventilations (noms après snake_case de l'ETL).
PREFERRED_DIMENSIONS = [
    "contract", "internet_service", "payment_method", "paperless_billing",
    "senior_citizen", "gender", "partner", "dependents", "phone_service",
]

_CHURN_FLAG = ["churn_value", "churn_flag", "churn", "is_churn", "churned", "churn_label"]
_MONTHLY = ["monthly_charges", "monthly_charge", "monthly_fee", "monthly_revenue"]

CATALOG: list[KPIDefinition] = [
    # ------------------------------------------------------------------ Churn
    KPIDefinition(
        id="churn_rate", name="Taux de churn", category="Churn", unit="%",
        definition="Part des clients ayant quitté l'opérateur.",
        formula="clients_churnés / clients_total × 100",
        agg="rate", inputs={"flag": _CHURN_FLAG}, valid_range=(0.0, 100.0)),
    KPIDefinition(
        id="churned_customers", name="Clients churnés", category="Churn", unit="clients",
        definition="Nombre de clients ayant quitté l'opérateur.",
        formula="somme(indicateur_churn)",
        agg="count_flag", inputs={"flag": _CHURN_FLAG}),
    KPIDefinition(
        id="avg_churn_score", name="Score de churn moyen", category="Churn", unit="points",
        definition="Score moyen de risque de départ (0 à 100) du portefeuille.",
        formula="moyenne(churn_score)",
        agg="mean", inputs={"value": ["churn_score"]}, valid_range=(0.0, 100.0)),
    KPIDefinition(
        id="monthly_revenue_lost", name="Revenu mensuel perdu par churn", category="Churn",
        unit="devise/mois",
        definition="Revenu mensuel récurrent des clients qui sont partis.",
        formula="somme(frais_mensuels) pour les clients churnés",
        agg="sum_where", inputs={"value": _MONTHLY, "flag": _CHURN_FLAG}),

    # ------------------------------------------------------- Base clients
    KPIDefinition(
        id="customer_count", name="Nombre de clients", category="Customer Base", unit="clients",
        definition="Nombre de clients distincts dans la table.",
        formula="nombre_distinct(identifiant_client)",
        agg="count_distinct", inputs={"id": ["customer_id", "customerid", "subscriber_id"]}),
    KPIDefinition(
        id="avg_tenure", name="Ancienneté moyenne", category="Customer Base", unit="mois",
        definition="Durée moyenne depuis la souscription.",
        formula="moyenne(ancienneté_en_mois)",
        agg="mean", inputs={"value": ["tenure_months", "tenure", "months_active"]}),

    # ------------------------------------------------ Facturation & revenu
    KPIDefinition(
        id="arpu", name="ARPU mensuel", category="Billing & Revenue", unit="devise/client/mois",
        definition="Revenu mensuel moyen par client.",
        formula="moyenne(frais_mensuels)",
        agg="mean", inputs={"value": _MONTHLY}),
    KPIDefinition(
        id="total_revenue", name="Revenu cumulé", category="Billing & Revenue", unit="devise",
        definition="Somme des montants facturés depuis la souscription.",
        formula="somme(total_facturé)",
        agg="sum", inputs={"value": ["total_charges", "total_revenue", "total_billed"]}),
    KPIDefinition(
        id="avg_cltv", name="CLTV moyen", category="Billing & Revenue", unit="devise",
        definition="Valeur vie client moyenne estimée.",
        formula="moyenne(cltv)",
        agg="mean", inputs={"value": ["cltv", "clv", "customer_lifetime_value"]}),
    KPIDefinition(
        id="unpaid_rate", name="Taux d'impayés", category="Billing & Revenue", unit="%",
        definition="Part du montant facturé restée impayée.",
        formula="somme(montant_impayé) / somme(montant_facturé) × 100",
        agg="ratio", scale=100.0, valid_range=(0.0, 100.0),
        inputs={"num": ["unpaid_amount", "overdue_amount"],
                "den": ["billed_amount", "invoice_amount"]}),

    # ------------------------------------------------------ Usage / conso
    KPIDefinition(
        id="avg_data_usage", name="Consommation data moyenne", category="Usage & Consumption",
        unit="Mo",
        definition="Volume de données moyen consommé par ligne.",
        formula="moyenne(volume_data)",
        agg="mean", inputs={"value": ["data_usage_mb", "data_mb", "data_usage", "internet_mb"]}),
    KPIDefinition(
        id="avg_call_minutes", name="Minutes d'appel moyennes", category="Usage & Consumption",
        unit="min",
        definition="Durée d'appel moyenne par ligne.",
        formula="moyenne(minutes_appel)",
        agg="mean", inputs={"value": ["call_minutes", "minutes", "call_duration", "voice_minutes"]}),

    # ------------------------------------------------- Performance réseau
    KPIDefinition(
        id="dropped_call_rate", name="Taux d'appels coupés", category="Network Performance",
        unit="%",
        definition="Part des appels interrompus avant la fin normale.",
        formula="somme(appels_coupés) / somme(appels_total) × 100",
        agg="ratio", scale=100.0, valid_range=(0.0, 100.0),
        inputs={"num": ["dropped_calls"], "den": ["total_calls", "calls"]}),
    KPIDefinition(
        id="avg_latency", name="Latence moyenne", category="Network Performance", unit="ms",
        definition="Latence réseau moyenne.",
        formula="moyenne(latence)",
        agg="mean", inputs={"value": ["latency_ms", "latency"]}),

    # ---------------------------------------------------- Support / opérations
    KPIDefinition(
        id="ticket_count", name="Tickets support", category="Operations & Support",
        unit="tickets",
        definition="Nombre de tickets distincts ouverts.",
        formula="nombre_distinct(ticket_id)",
        agg="count_distinct", inputs={"id": ["ticket_id", "case_id"]}),
    KPIDefinition(
        id="avg_resolution_time", name="Temps de résolution moyen",
        category="Operations & Support", unit="h",
        definition="Délai moyen de résolution d'un ticket.",
        formula="moyenne(temps_de_résolution)",
        agg="mean", inputs={"value": ["resolution_time_hours", "resolution_time",
                                      "resolution_hours"]}),
]