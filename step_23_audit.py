"""
step_23_audit.py — The audit tables: the satellites of the core (sff_nucleo) for the drill-down.

The core keeps, for every forecast series, what the report needs. When a number is in doubt,
these tables explain the whole way it came from, joined to the core by its keys:

  table                          one row per                                  key in the core
  sff_composition                id of any stage (0 raw … 3 composition)      s08_stage0_id … s08_stage3_id
  sff_composition_techniques     composition × horizon band × technique       s08_stage3_id
  sff_credibility                credibility reference                        s09_credibility_ref_id
  sff_credibility_members        reference × forecast series in its rate      s09_credibility_ref_id
  sff_series_dynamics            forecast series                              s03_fs_id
  sff_composition_forecast_all   composition × future horizon × technique     s08_stage3_id

EVERY TEST HAS ITS INTERVAL, built as the forecast builds its band: the error quantiles of the
technique in the backtest (step 12) × the binomial error with the series' own units due. A test
is IN THE INTERVAL when the real rate of the series falls inside it. The exam of every forecast
series (its chosen technique, every exam month and horizon) is summarised in sff_series_exam and
in the core: how much it erred and how many predictions were in the interval.

The 4 stages only give the forecast better conditions to predict: the backtest, the exam and the
forecast are referred back to every forecast series (its own units due, its own renewals).

NOTHING IS COUNTED TWICE: every table is checked against the table it comes from (the checks
below). A forecast series is in one id per stage, in one composition, and has one reference; a
reference may hold series of many compositions, so its members are not summed across references.

Actions (logged as they are done):
  1. sff_composition: every id of every stage                         checks 1-2
  2. sff_composition_techniques: every technique of every composition check 3
  3. sff_credibility and its members                                  check 4
  4. sff_series_dynamics                                              check 5
  5. sff_composition_forecast_all                                     check 6
  6. write the audit tables                                           checks 7-12
  7. count the checks; stop if any failed

The backtest of every technique referred back to every forecast series is NOT here: step 19 examines
every forecast series (sff_series_exam, sff_series_exam_detail) and the backtest of every composition is
sff_backtest_predicciones (step 12).

Checks (logged as they are made, numbered, at the level of their status):
   1. every stage is a partition: the units due of its ids add up to the raw's
   2. every stage holds every estimable series once
   3. every composition and band has exactly one chosen technique, the one of step 12
   4. the support and the rate of every reference, recomputed from its members, are the ones used
   5. one dynamics row per estimable forecast series
   6. the chosen technique gives the rate step 20 used (rows with no credibility shift)
   7-12. the six tables written and read back

Output: dict of the audit tables · the tables listed above.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, Config, GATE_LEVEL, PURPOSE_EXAM, PURPOSE_SELECTION,
                    RATE_COLUMN, RATE_FROM_POOL, SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_COMPOSITION,
                    TABLE_COMPOSITION_FORECAST_ALL, TABLE_COMPOSITION_TECHNIQUES, TABLE_CREDIBILITY,
                    TABLE_CREDIBILITY_MEMBERS, TABLE_SERIES_DYNAMICS, TECHNIQUE_COMPOSITION_BELOW_FLOOR,
                    TECHNIQUE_NOT_ENOUGH_HISTORY, TECHNIQUE_TESTED, TRUTH_ROLES)
from step_09_ladder import credibility_k_detail
from step_11_dynamics import dynamics_of_one_series
from step_12_backtest import band_of_horizon
from techniques import CATALOGUE, inverse_logit, predict_logit


STEP_LABEL = "23"
STEP_NAME = "AUDIT TABLES"
STEP_PURPOSE = ("build the satellites of the core for the drill-down: every id of every stage, every technique of every "
                "composition, every credibility reference and its members, the dynamics of every forecast series, the "
                "future rate of every technique; each one checked against the table it comes from, so nothing is counted twice")
STEP_ACTIONS = ["sff_composition: every id of every stage (checks 1-2)",
                "sff_composition_techniques: every technique of every composition (check 3)",
                "sff_credibility and its members (check 4)",
                "sff_series_dynamics (check 5)",
                "sff_composition_forecast_all (check 6)",
                "write the audit tables (checks 7-12)",
                "count the checks; stop if any failed"]
STEP_OUTPUT = "six audit tables joined to the core by fs_id, the stage ids and the credibility reference"

PERCENTAGE_POINTS = 100
STAGES = (0, 1, 2, 3)


def build_audit_tables(ladder: dict, series_rate: pd.DataFrame, series_estimate: pd.DataFrame,
                       rated_units: pd.DataFrame, pool_series: pd.DataFrame, pool_reference: pd.DataFrame,
                       pool_dynamics: pd.DataFrame, backtest: dict, forecast_rows: pd.DataFrame,
                       configuration: Config, rate_box=None) -> dict:
    """The six audit tables, checked against their sources and written.

    INPUT:   the ladder (step 08), series_rate (08), series_estimate (11), rated_units (08), the
             composition series and reference (12), their dynamics (13), the backtest (14) and the
             forecast rows (17).
    OUTPUT:  dict(composition, composition_techniques, credibility, credibility_members,
             series_dynamics, composition_forecast_all).
    RULES:   see the module header.
    EDGE CASES: a composition below the support floor is not judged by the backtest: its
             techniques have status composition_below_floor and its series have no backtest rows.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    history = real_history(rated_units, configuration)
    groups = ladder["groups"]

    # [1] every id of every stage
    composition = composition_table(ladder["stages"], groups, series_estimate, pool_reference, backtest, history,
                                    forecast_rows, configuration)
    configuration.log_action(STEP_LABEL, 1, f"{len(composition):,} ids over the 4 stages · "
                                            f"{int(composition['is_composition'].sum()):,} compositions")
    check_stages(ladder["stages"], history, configuration, check_log)

    # [2] every technique of every composition
    techniques = composition_techniques(pool_series, pool_reference, backtest, configuration)
    configuration.log_action(STEP_LABEL, 2, f"{len(techniques):,} composition × band × technique rows · status "
                                            f"{techniques['status'].value_counts().to_dict()}")
    check_chosen_once(techniques, backtest["decision"], [COMPOSITION_ID_COLUMN, "tramo_h"],
                      "every composition and band has exactly one chosen technique, the one of step 12",
                      configuration, check_log)

    # [3] the credibility references and their members
    credibility, members = credibility_tables(groups, ladder["reference_members"], series_rate, configuration)
    configuration.log_action(STEP_LABEL, 3, f"{len(credibility):,} references · {len(members):,} reference × series rows "
                                            f"({members['role'].value_counts().to_dict() if len(members) else {}})")
    check_references(credibility, members, history, configuration, check_log)

    # [4] the dynamics of every forecast series
    series_dynamics = series_dynamics_table(groups, series_rate, history, pool_dynamics, configuration)
    configuration.log_action(STEP_LABEL, 4, f"{len(series_dynamics):,} forecast series · measurable "
                                            f"{series_dynamics['measurable'].value_counts().to_dict()} · "
                                            f"{int(series_dynamics['differs_from_composition'].sum()):,} differ from their composition")
    configuration.log_check(STEP_LABEL, check_log, "one dynamics row per estimable forecast series",
                            len(series_dynamics) == len(groups) and series_dynamics[SERIES_ID_COLUMN].is_unique,
                            failure_detail=f"{len(series_dynamics):,} rows for {len(groups):,} series")

    # [5] the future rate of every technique
    if rate_box is None:
        from step_13_rate_box import RateBox
        rate_box = RateBox(rated_units, series_estimate, pool_series, backtest, ladder.get("reference_members"), configuration)
    forecast_all = composition_forecast_all(rate_box, backtest["decision"], forecast_rows, configuration)
    configuration.log_action(STEP_LABEL, 5, f"{len(forecast_all):,} composition × horizon × technique rates")
    check_forecast_all(forecast_all, forecast_rows, series_estimate, configuration, check_log)

    # [6] the tables
    configuration.log_action(STEP_LABEL, 6, "writing the audit tables")
    tables = dict(composition=composition, composition_techniques=techniques, credibility=credibility,
                  credibility_members=members, series_dynamics=series_dynamics, composition_forecast_all=forecast_all)
    for name, table_name in (("composition", TABLE_COMPOSITION), ("composition_techniques", TABLE_COMPOSITION_TECHNIQUES),
                             ("credibility", TABLE_CREDIBILITY), ("credibility_members", TABLE_CREDIBILITY_MEMBERS),
                             ("series_dynamics", TABLE_SERIES_DYNAMICS),
                             ("composition_forecast_all", TABLE_COMPOSITION_FORECAST_ALL)):
        configuration.write_table(STEP_LABEL, check_log, tables[name], table_name)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
    return tables


def real_history(rated_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The real closed months of every forecast series with something due: its units due and renewed."""
    real = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                       & (rated_units[configuration.pipeline_units_col] > 0)]
    return real[[SERIES_ID_COLUMN, configuration.period_col, CALENDAR_ROLE_COLUMN,
                 configuration.pipeline_units_col, configuration.renewed_units_col]].copy()


def monthly_support(history: pd.DataFrame, members: pd.DataFrame, key: str, configuration: Config) -> pd.DataFrame:
    """Per key: support (median of the monthly units due summed over its members), units due, renewed, rate."""
    joined = history.merge(members, on=SERIES_ID_COLUMN)
    monthly = joined.groupby([key, configuration.period_col])[[configuration.pipeline_units_col,
                                                               configuration.renewed_units_col]].sum().reset_index()
    grouped = monthly.groupby(key)
    result = pd.DataFrame({"support": grouped[configuration.pipeline_units_col].median(),
                           "units_due": grouped[configuration.pipeline_units_col].sum(),
                           "units_renewed": grouped[configuration.renewed_units_col].sum()})
    result["rate"] = result["units_renewed"] / result["units_due"].where(result["units_due"] > 0)
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# 1 · THE COMPOSITIONS (every id of every stage)
# ═══════════════════════════════════════════════════════════════════════════════════

def composition_table(stages: pd.DataFrame, groups: pd.DataFrame, series_estimate: pd.DataFrame,
                      pool_reference: pd.DataFrame, backtest: dict, history: pd.DataFrame, forecast_rows: pd.DataFrame,
                      configuration: Config) -> pd.DataFrame:
    """One row per id of any stage: the stages it appears in, its series, support, rate; and, for a
    composition (stage 3), how it is predicted."""
    long = pd.concat([pd.DataFrame({SERIES_ID_COLUMN: stages[SERIES_ID_COLUMN], "stage": stage,
                                    "id": stages[f"stage{stage}_id"]}) for stage in STAGES], ignore_index=True)
    # an id that repeats in several stages has the same series in all of them: one membership per id
    membership = long.drop_duplicates(["id", SERIES_ID_COLUMN])[["id", SERIES_ID_COLUMN]]
    support = monthly_support(history, membership, "id", configuration)
    table = (long.groupby("id").agg(first_stage=("stage", "min"), last_stage=("stage", "max"))
             .join(membership.groupby("id").size().rename("series")).join(support).reset_index())
    table["is_composition"] = (table["last_stage"] == 3).astype(int)

    # the compositions: how they are predicted
    per_composition = (series_estimate.dropna(subset=["credibility_ref_id"]).drop_duplicates(COMPOSITION_ID_COLUMN)
                       .set_index(COMPOSITION_ID_COLUMN)[["credibility_ref_id", "ref_support", "ref_rate", "k", "z"]])
    estimated = series_estimate.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["tasa_estimada"]
    table = table.join(per_composition, on="id").join(estimated.rename("estimated_rate"), on="id")
    table = table.join(pool_reference.set_index(COMPOSITION_ID_COLUMN)[["gate", "usd_por_predecir"]], on="id")
    decision = backtest["decision"].pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h", values="tecnica").add_prefix("technique_")
    exam = backtest["exam_by_pool"].pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h", values="elegida_err_pp_medio").add_prefix("exam_err_pp_")
    table = table.join(decision, on="id").join(exam, on="id")
    if forecast_rows is not None and len(forecast_rows):
        expected = forecast_rows.groupby([COMPOSITION_ID_COLUMN, "confidence"])["esperado_usd"].sum().unstack(fill_value=0.0)
        expected = expected.add_prefix("expected_usd_")
        table = table.join(expected, on="id")
    return table.rename(columns={"id": COMPOSITION_ID_COLUMN}).sort_values(["first_stage", COMPOSITION_ID_COLUMN]).reset_index(drop=True)


def check_stages(stages: pd.DataFrame, history: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 and 2: every stage is a partition of the raw."""
    raw_due = float(history[history[SERIES_ID_COLUMN].isin(stages[SERIES_ID_COLUMN])][configuration.pipeline_units_col].sum())
    due_by_stage = {}
    for stage in STAGES:
        members = stages[[SERIES_ID_COLUMN, f"stage{stage}_id"]].rename(columns={f"stage{stage}_id": "id"})
        due_by_stage[stage] = float(monthly_support(history, members, "id", configuration)["units_due"].sum())
    configuration.log_check(STEP_LABEL, check_log, "every stage is a partition: the units due of its ids add up to the raw's",
                            all(abs(due - raw_due) <= configuration.units_tolerance for due in due_by_stage.values()),
                            failure_detail=f"units due by stage {due_by_stage} vs {raw_due:,.0f}",
                            context=f"{raw_due:,.0f} units due in every stage")
    configuration.log_check(STEP_LABEL, check_log, "every stage holds every estimable series once",
                            stages[SERIES_ID_COLUMN].is_unique and stages[[f"stage{stage}_id" for stage in STAGES]].notna().all().all(),
                            failure_detail="a series without an id in some stage, or twice",
                            context=f"{len(stages):,} series")


# ═══════════════════════════════════════════════════════════════════════════════════
# 2 · THE TECHNIQUES OF EVERY COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════════

def composition_techniques(pool_series: pd.DataFrame, pool_reference: pd.DataFrame, backtest: dict,
                           configuration: Config) -> pd.DataFrame:
    """Composition × band × technique: status and why, months, errors in selection and exam, rank, chosen."""
    truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()]
    months_available = truth.groupby(COMPOSITION_ID_COLUMN).size()
    gate = pool_reference.set_index(COMPOSITION_ID_COLUMN)["gate"]
    predictions = backtest["predictions"].assign(abs_err_pp=lambda frame: frame["err_pp"].abs(),
                                                 abs_err_norm=lambda frame: frame["err_norm"].abs())
    by_purpose = (predictions.groupby([COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "proposito"])
                  [["abs_err_pp", "abs_err_norm"]].mean().unstack("proposito"))
    by_purpose.columns = [f"{purpose}_{measure}" for measure, purpose in by_purpose.columns]
    chosen = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
    # tested = it predicted in the selection months of that band (the history of the exam months does not count:
    # the choice is made before them)
    selected = set(map(tuple, predictions.loc[predictions["proposito"] == PURPOSE_SELECTION,
                                              [COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"]].drop_duplicates().to_numpy()))

    rows = []
    for composition_id in pool_reference[COMPOSITION_ID_COLUMN]:
        for band_name in configuration.horizon_bands:
            for technique_id, technique in CATALOGUE.items():
                required = int(technique[3])
                available = int(months_available.get(composition_id, 0))
                if gate.get(composition_id) != GATE_LEVEL:
                    status = TECHNIQUE_COMPOSITION_BELOW_FLOOR
                elif available < required or (composition_id, band_name, technique_id) not in selected:
                    status = TECHNIQUE_NOT_ENOUGH_HISTORY
                else:
                    status = TECHNIQUE_TESTED
                rows.append({COMPOSITION_ID_COLUMN: composition_id, "tramo_h": band_name, "tecnica": technique_id,
                             "status": status, "months_available": available, "months_required": required,
                             "is_chosen": int(chosen.get((composition_id, band_name)) == technique_id)})
    table = pd.DataFrame(rows).join(by_purpose, on=[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"])
    table = table.rename(columns={f"{PURPOSE_SELECTION}_abs_err_norm": "selection_err_norm",
                                  f"{PURPOSE_SELECTION}_abs_err_pp": "selection_err_pp",
                                  f"{PURPOSE_EXAM}_abs_err_norm": "exam_err_norm",
                                  f"{PURPOSE_EXAM}_abs_err_pp": "exam_err_pp"})
    tested = table["status"] == TECHNIQUE_TESTED
    table["rank"] = np.nan
    table.loc[tested, "rank"] = (table[tested].groupby([COMPOSITION_ID_COLUMN, "tramo_h"])["selection_err_norm"]
                                 .rank(method="min"))
    return table


def check_chosen_once(table: pd.DataFrame, decision: pd.DataFrame, keys: list, name: str, configuration: Config,
                      check_log: list) -> None:
    """Exactly one chosen technique per key, and it is the technique of step 12's decision."""
    chosen = table[table["is_chosen"] == 1]
    per_key = chosen.groupby(keys).size()
    matches = chosen.merge(decision[keys + ["tecnica"]], on=keys, suffixes=("", "_decision"))
    configuration.log_check(STEP_LABEL, check_log, name,
                            bool((per_key == 1).all()) and len(per_key) == len(decision)
                            and bool((matches["tecnica"] == matches["tecnica_decision"]).all()),
                            failure_detail=f"{int((per_key != 1).sum())} keys without exactly one chosen · "
                                           f"{len(per_key)} keys with a choice for {len(decision)} decisions",
                            context=f"{len(decision):,} composition × band")


# ═══════════════════════════════════════════════════════════════════════════════════
# 3 · THE CREDIBILITY
# ═══════════════════════════════════════════════════════════════════════════════════

def credibility_tables(groups: pd.DataFrame, reference_members: pd.DataFrame, series_rate: pd.DataFrame,
                       configuration: Config) -> tuple:
    """One row per reference (support, rate, k and its parts), and one per reference × forecast series."""
    references = (groups.dropna(subset=["credibility_ref_id"])
                  .drop_duplicates("credibility_ref_id")[["credibility_ref_id", "credibility_ref_step", "ref_series",
                                                          "ref_support", "ref_rate"]])
    k_detail = credibility_k_detail(groups, configuration)
    credibility = references.merge(k_detail, on="credibility_ref_id", how="left")
    credibility["compositions_using"] = credibility["compositions_using"].fillna(0).astype(int)
    credibility["k"] = credibility["k"].fillna(configuration.k_cred)
    credibility["k_source"] = credibility["k_source"].fillna("default")

    composition_of = groups.set_index(SERIES_ID_COLUMN)[COMPOSITION_ID_COLUMN]
    reference_of_composition = groups.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["credibility_ref_id"]
    members = reference_members.merge(series_rate[[SERIES_ID_COLUMN, "n_propio", "tasa_propia"]], on=SERIES_ID_COLUMN, how="left")
    members[COMPOSITION_ID_COLUMN] = members[SERIES_ID_COLUMN].map(composition_of)
    borrows = members[COMPOSITION_ID_COLUMN].map(reference_of_composition) == members["credibility_ref_id"]
    members["role"] = np.where(borrows, "borrower", "lender")
    members = members.rename(columns={"n_propio": "support", "tasa_propia": "own_rate"})
    return credibility, members


def check_references(credibility: pd.DataFrame, members: pd.DataFrame, history: pd.DataFrame, configuration: Config,
                     check_log: list) -> None:
    """Check 4: every reference recomputed from its members."""
    if credibility.empty:
        configuration.log_not_evaluated(STEP_LABEL, check_log, "every reference recomputed from its members", "no reference")
        return
    recomputed = monthly_support(history, members[["credibility_ref_id", SERIES_ID_COLUMN]], "credibility_ref_id", configuration)
    compared = credibility.join(recomputed, on="credibility_ref_id")
    compared["members"] = compared["credibility_ref_id"].map(members.groupby("credibility_ref_id").size())
    wrong = compared[((compared["support"] - compared["ref_support"]).abs() > configuration.units_tolerance)
                     | ((compared["rate"] - compared["ref_rate"]).abs() > configuration.rate_tolerance)
                     | (compared["members"] != compared["ref_series"])]
    configuration.log_check(STEP_LABEL, check_log, "the support, rate and series of every reference, recomputed from its members, "
                            "are the ones used", wrong.empty, failure_detail=f"{len(wrong)} references differ",
                            context=f"{len(credibility):,} references",
                            examples=wrong[["credibility_ref_id", "ref_support", "support", "ref_rate", "rate"]])


# ═══════════════════════════════════════════════════════════════════════════════════
# 4 · THE DYNAMICS OF EVERY FORECAST SERIES
# ═══════════════════════════════════════════════════════════════════════════════════

def series_dynamics_table(groups: pd.DataFrame, series_rate: pd.DataFrame, history: pd.DataFrame,
                          pool_dynamics: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """φ, trend and seasonality of every estimable forecast series on its own months, whether they can
    be trusted, and how they compare with its composition's."""
    support = series_rate.set_index(SERIES_ID_COLUMN)["n_propio"]
    monthly = history.rename(columns={configuration.pipeline_units_col: "vencen", configuration.renewed_units_col: "renovadas"})
    composition_dynamics = (pool_dynamics.set_index(COMPOSITION_ID_COLUMN)[["tendencia", "estacional"]]
                            if pool_dynamics is not None and len(pool_dynamics) else pd.DataFrame(columns=["tendencia", "estacional"]))
    # one sort of the whole history, then the blocks arrive already in order (no sort per series)
    monthly = monthly.sort_values([SERIES_ID_COLUMN, configuration.period_col])
    months_of_series = dict(iter(monthly.groupby(SERIES_ID_COLUMN, sort=False)))
    no_months = monthly.iloc[0:0]
    rows = []
    for series_id, composition_id in zip(groups[SERIES_ID_COLUMN], groups[COMPOSITION_ID_COLUMN]):
        months = months_of_series.get(series_id, no_months)
        own_support = float(support.get(series_id, 0.0))
        row = {SERIES_ID_COLUMN: series_id, COMPOSITION_ID_COLUMN: composition_id, "months": len(months),
               "own_support": own_support}
        if len(months) < configuration.audit_trend_min_months:
            row["measurable"] = "short_history"
        else:
            row["measurable"] = "yes" if own_support >= configuration.support_floor else "low_support"
            measured = dynamics_of_one_series(months, configuration.period_col, configuration)
            row.update(phi=measured["phi"], trend=measured["tendencia"], trend_pp_year=measured["tendencia_pp_ano"],
                       trend_p_value=measured["p_valor_tendencia"])
            if len(months) >= configuration.audit_seasonality_min_months:
                row.update(seasonal=int(measured["estacional"]), seasonal_p_value=measured["p_valor_mes"],
                           amplitude_pp=measured["amplitud_pp"], high_months=measured["meses_alto"],
                           low_months=measured["meses_bajo"])
        if composition_id in composition_dynamics.index:
            row["composition_trend"] = composition_dynamics.loc[composition_id, "tendencia"]
            row["composition_seasonal"] = int(composition_dynamics.loc[composition_id, "estacional"])
        rows.append(row)
    table = pd.DataFrame(rows)
    for column_name in ("phi", "trend", "trend_pp_year", "trend_p_value", "seasonal", "seasonal_p_value", "amplitude_pp",
                        "high_months", "low_months", "composition_trend", "composition_seasonal"):
        if column_name not in table.columns:
            table[column_name] = np.nan
    # a series that behaves differently from its composition (only when both are measured and trusted)
    trusted = (table["measurable"] == "yes") & table["composition_trend"].notna()
    different_trend = table["trend"].fillna(0).ne(0) & table["trend"].ne(table["composition_trend"])
    different_season = table["seasonal"].notna() & table["seasonal"].ne(table["composition_seasonal"])
    table["differs_from_composition"] = (trusted & (different_trend | different_season)).astype(int)
    return table


# ═══════════════════════════════════════════════════════════════════════════════════
# 5 · THE BACKTEST REFERRED BACK TO EVERY FORECAST SERIES
# ═══════════════════════════════════════════════════════════════════════════════════

def composition_forecast_all(rate_box, decision: pd.DataFrame, forecast_rows: pd.DataFrame,
                             configuration: Config) -> pd.DataFrame:
    """Composition × future horizon × technique: the rate each technique gives with the whole history
    (before the credibility shift), and which one was chosen for that horizon's band."""
    horizons = sorted(int(horizon) for horizon in forecast_rows["h"].dropna().unique()) if forecast_rows is not None else []
    chosen = decision.set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
    last_closed = configuration.calendar_boundaries()["current"] - 1
    rows = []
    # the history of every composition as THE BOX OF THE RATE knows it at the last closed month
    for composition_id in sorted(rate_box.composition_index):
        history_logit, months, _, _ = rate_box.composition_history(composition_id, last_closed)
        months_available = len(months)
        for horizon in horizons:
            band_name = band_of_horizon(horizon, configuration.horizon_bands)
            for technique_id, technique in CATALOGUE.items():
                enough = months_available >= int(technique[3])
                value = predict_logit(technique_id, history_logit, months, horizon) if enough else np.nan
                rows.append({COMPOSITION_ID_COLUMN: composition_id, "h": horizon, "tramo_h": band_name,
                             "tecnica": technique_id, "months_available": months_available,
                             "rate": float(inverse_logit(value)) if np.isfinite(value) else np.nan,
                             "is_chosen": int(chosen.get((composition_id, band_name)) == technique_id)})
    return pd.DataFrame(rows)


def check_forecast_all(forecast_all: pd.DataFrame, forecast_rows: pd.DataFrame, series_estimate: pd.DataFrame,
                       configuration: Config, check_log: list) -> None:
    """Check 6: the chosen technique gives the rate step 20 used, on the rows with no credibility shift."""
    z_of = series_estimate.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["z"]
    rows = forecast_rows[(forecast_rows["origen_tasa"] == RATE_FROM_POOL)]
    rows = rows[rows[COMPOSITION_ID_COLUMN].map(z_of).fillna(1) >= 1]
    chosen = forecast_all[forecast_all["is_chosen"] == 1].set_index([COMPOSITION_ID_COLUMN, "h"])["rate"].to_dict()
    expected = pd.Series([chosen.get(pair, np.nan) for pair in zip(rows[COMPOSITION_ID_COLUMN], rows["h"].astype(int))],
                         index=rows.index, dtype=float)
    differs = (expected - rows[RATE_COLUMN]).abs() > configuration.rate_tolerance
    configuration.log_check(STEP_LABEL, check_log, "the chosen technique gives the rate step 20 used (rows with no credibility shift)",
                            not bool(differs.any()), failure_detail=f"{int(differs.sum())} future rows differ",
                            context=f"{len(rows):,} future rows compared")
