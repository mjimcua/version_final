"""
block_3_price.py — BLOCK 3 · PRICE: at what price the renewers renew (the revaluation), with its band.

MISSION: give the uplift of every row of pipeline (renewed USD over the USD that was falling due, among those who
renew), the one that predicts the price best, and explain the price to business.

MECHANISM: two candidates for every row: the CONTRACT rule 1 / (1 − discount) where the discount is known, and the
STATISTICAL uplift of its cell (its past renewers, borrowing from the parent cell with few renewers, with a bootstrap
band and without the price increases, which are applied apart while their cycle lasts). A backtest on the exam
months, with the uplift estimated before them, decides which one the forecast uses and whether the price factor of
the customer states helps.

THE STEPS:
  3.1  uplift                 the statistical uplift of every cell and of every combination of states; the contract
                              rule against the past renewals; the price increases still in play
  3.2  price diagnostics      for business only: homogeneity of the cells, the contract rule, the revaluation of the
                              renewers licence by licence, the increases (no prediction reads it)
  3.3  uplift backtest        contract or statistical, and the state factor, judged on the exam months: the verdict

READS (block 1): fine_table.
GIVES: uplift_cells, uplift_state_factors, price_increases, uplift_verdict (block 7 values every row with them) ·
       contract_check, uplift_basis, uplift_homogeneity, uplift_backtest (the report and the audit).

THE INTERFACE:
  block = PriceBlock(configuration).run(fine_table)
  block.uplift_cells, block.uplift_verdict, …    block.results()    block.seconds
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import gc
import time

import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, Config, ISOLATED_RATIO_BANDS, PATH_CONTRACT, PATH_CONTRACT_NEUTRAL,
                    PATH_STATISTICAL, PATH_STATISTICAL_STATES, ROLE_PROJECTION, ROLE_TEST, SERIES_ID_COLUMN,
                    TABLE_CONTRACT_CHECK, TABLE_UPLIFT_BACKTEST, TABLE_UPLIFT_CELLS, TABLE_UPLIFT_HOMOGENEITY,
                    TRUTH_ROLES, UPLIFT_CELL, UPLIFT_CELL_ID_COLUMN, UPLIFT_GLOBAL, UPLIFT_OWN, UPLIFT_PARENT,
                    WILDCARD, has_no_negative_state, join_columns, negative_states, state_combination,
                    weighted_ratio_spread)
from price_increases import (active_increases, increase_calendar, increase_factor, pending_steps, price_group_columns,
                             price_group_of, price_monitor)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 3.1 · UPLIFT  (until 25-oct-2026: step_14_uplift.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The revaluation: at what price does a renewer renew?
#
# What falls due has a price (USD due / units due = the pipeline AUV of the row). A renewer
# pays a price too (renewed USD / renewed units). The UPLIFT is the second over the first:
# 1.0 = renews at the same price; 1.43 = pays 43 % more. The forecast of a row is
# USD due × renewal rate × uplift.
#
# Two ways to know the uplift of a future row:
#   · STATISTICAL (the default, and the only one where the discount is unknown): the uplift
#     observed in the past renewals of its UPLIFT CELL (uplift mandatory dims, revaluation
#     extras, discount bucket), as a ratio of sums over the renewers (a contract that did not
#     renew has no price), weighted by money:
#         Σ renewed USD / Σ (the USD that was falling due of the contracts that RENEWED)
#     when the extract carries that base (renewed_pipeline_usd_col): the revaluation pure,
#     conditional on renewing. Over the RECENT window only (uplift_window_months, 12 = one price
#     cycle; 0 = all the history), and WITHOUT the price increases (uplift_without_increases): a
#     month of the window inside the cycle of an increase detected in its price group
#     (price_increases.py) has its renewed USD divided by (1 + the size of the increase). The uplift
#     of the cell is then the one without increases; step 7.2 applies the increase again to the future
#     months still inside its cycle, and after the cycle its effect is lost. Without it, the base is approximated as renewed units × the
#     row's pipeline AUV, which is exact only if renewers were worth the row's average: the
#     difference between the two bases is the HOMOGENEITY of the cell, measured below.
#     Its n is the number of renewers. A cell with fewer than uplift_floor
#     renewers takes its PARENT (same mandatory dims and discount bucket, the other extras
#     set to '*'), then its mandatory cell, then the whole portfolio. The band of the
#     uplift: bootstrap of the renewer rows (p5-p95).
#   · CONTRACT (where the exact discount is known): a customer who paid list × (1 − d)
#     renews at list, so the uplift is 1 / (1 − d). This step checks the rule against the past
#     renewals with a known discount (how much of the renewed money it explains within ±2 %);
#     step 3.3 decides, in the exam months, which path predicts better.
#
# Actions (logged as they are done):
#   1. the renewer rows of the closed months, with the base of their uplift
#   2. the price increases of every price group (closed months), the window of the uplift and its
#      months deflated; the statistical uplift of every uplift cell: own, parent, cell, global; the band;
#      the price factor of every combination of negative customer states (what its renewers paid over their cell)
#   3. the contract rule against the past renewals with a known discount
#   4. check the uplifts                                               checks 1-4
#   5. write the tables                                                checks 5-6
#   6. count the checks; stop if any failed
#   7. show the uplift by origin and the price by customer states
# The diagnostics of the price for business (homogeneity, the contract rule read, the revaluation of the
# renewers, the price increases on screen) are the next step (3.2): no prediction
# reads them.
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every uplift cell of the fine table has an uplift (future cells included)
#    2. every uplift is inside [0.01, uplift_cap]
#    3. the band contains the uplift of every cell with its own or its parent's
#    4. no price group ends its history with a step not confirmed yet (WARN: if it is an increase, the
#       forecast does not carry it, and its months are in the window without being deflated)
#    5-6. tables sff_uplift_celda and sff_uplift_contrato_check written and read back
#
# Output: the uplift of every cell (with its origin and band), the price factor of every combination of customer
# states (the forecast applies it when step 3.3 says it predicts better), the contract check, the price increases in
# play (for the forecast) and the basis the diagnostics read · tables sff_uplift_celda, sff_uplift_contrato_check.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_3_1_LABEL = "3.1"
STEP_3_1_NAME = "UPLIFT"
STEP_3_1_PURPOSE = ("estimate at what price a renewer renews relative to what fell due: the statistical uplift of every "
                "uplift cell from its past renewals (borrowing from its parent when it has few renewers), and the check "
                "of the contract rule 1 / (1 − discount) where the discount is known")
STEP_3_1_ACTIONS = ["the renewer rows of the closed months, with the base of their uplift",
                "the price increases, the window of the uplift and its months deflated; the statistical uplift of "
                "every uplift cell: own, parent, cell, global; the band; the price factor of every combination of "
                "customer states",
                "the contract rule against the past renewals with a known discount",
                "check the uplifts (checks 1-4)",
                "write the tables (checks 5-6)",
                "count the checks; stop if any failed",
                "show the uplift by origin and the price by customer states"]
STEP_3_1_OUTPUT = ("the uplift of every uplift cell (origin, band) · the price factor of every combination of customer states · "
               "the contract check · the price increases in play · tables sff_uplift_celda, sff_uplift_contrato_check")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RENEWERS AND THE WINDOW OF THE UPLIFT
# ═══════════════════════════════════════════════════════════════════════════════════

def renewer_rows(fine_table: pd.DataFrame, configuration: Config, before_month=None) -> pd.DataFrame:
    """The rows of closed truth months that renewed (optionally only before a month), with
    their pipeline AUV, the two sides of the ratio and their parent and cell ids."""
    # the rows are chosen first and copied once, without the maturation columns (block 3 never reads them). Until
    # 26-oct-2026 the closed rows were copied whole and then filtered and copied again: three copies of most of the
    # fine table at the peak, called three times in the block (memory with the real extract)
    is_renewer = (fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                  & (fine_table[configuration.renewed_units_col].fillna(0) > 0)
                  & (fine_table[configuration.pipeline_units_col] > 0))
    if before_month is not None:
        is_renewer &= fine_table[configuration.period_col] < before_month
    if configuration.renewed_pipeline_usd_col:
        is_renewer &= fine_table[configuration.renewed_pipeline_usd_col] > 0   # a renewer row without its base cannot opine (warned in step 1.2)
    not_read_here = set(configuration.maturation_columns)
    columns_read = [column_name for column_name in fine_table.columns if column_name not in not_read_here]
    closed = fine_table.loc[is_renewer, columns_read].copy()
    closed["auv_pipeline"] = closed[configuration.pipeline_usd_col] / closed[configuration.pipeline_units_col]
    closed["_numerador"] = closed[configuration.renewed_usd_col]
    # the base of the uplift: what the RENEWERS were worth before renewing. Exact when the extract carries it;
    # otherwise the renewed units valued at the row's average due price (the whole row, renewers or not)
    closed["_denominador_aproximado"] = closed[configuration.renewed_units_col] * closed["auv_pipeline"]
    if configuration.renewed_pipeline_usd_col:
        closed["_denominador"] = closed[configuration.renewed_pipeline_usd_col]
    else:
        closed["_denominador"] = closed["_denominador_aproximado"]
    return add_parent_ids(closed, configuration, copy=False)        # closed is already this function's own copy


def closed_view_of_fine_table(fine_table: pd.DataFrame, configuration: Config, before_month=None) -> pd.DataFrame:
    """The closed pipeline months of the fine table (optionally only before a month) with the columns the price
    monitor reads, named as in the core: period, forecast_to_renew_units / _USD, forecast_renewed_units / _USD,
    the closed-month measures by their core names (CORE_*) and the price dims.

    The core carries exactly these values for its closed pipeline rows (step 8.3 copies the fine table), so the
    monitor read here and the one of step 8.4 (from the core) are the same; steps 3.1 and 3.3 cannot wait for the
    core, it is built after the forecast."""
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                        & (fine_table[configuration.pipeline_units_col] > 0)]
    if before_month is not None:
        closed = closed[closed[configuration.period_col] < before_month]
    view = pd.DataFrame({"period": closed[configuration.period_col],
                         "forecast_to_renew_units": closed[configuration.pipeline_units_col],
                         "forecast_to_renew_USD": closed[configuration.pipeline_usd_col],
                         "forecast_renewed_units": closed[configuration.renewed_units_col],
                         "forecast_renewed_USD": closed[configuration.renewed_usd_col]}, index=closed.index)
    for measure in configuration.closed_month_measures:
        view[measure.core_column] = closed[measure.raw_column]
    for column_name in price_group_columns(closed, configuration):
        view[column_name] = closed[column_name]
    return view


def price_increases_as_of(fine_table: pd.DataFrame, as_of, configuration: Config) -> dict:
    """The price increases as they are known at `as_of` (the current month, or the first exam month): built
    ONLY with the closed months before it. A dict: monitor (price group × month), calendar (every increase
    detected), pending (the steps at the end of the history not confirmed yet) and detector_series."""
    monitor = price_monitor(closed_view_of_fine_table(fine_table, configuration, before_month=as_of), configuration)
    return dict(monitor=monitor, calendar=increase_calendar(monitor, configuration),
                pending=pending_steps(monitor, configuration), detector_series=monitor.attrs.get("detector_series", ""))


def uplift_history(renewers: pd.DataFrame, as_of, increases: pd.DataFrame, window_months: int,
                   without_increases: bool, configuration: Config) -> pd.DataFrame:
    """The renewer rows the statistical uplift is estimated from.

    INPUT:   the renewer rows (renewer_rows), the first month that is not history, the increases in play
             (price_increases.active_increases), the window in months (0 = all the history) and whether the
             increases are removed.
    OUTPUT:  the renewer rows of the months as_of − window_months .. as_of − 1 (every one before as_of with
             window 0), with _factor_subida (the factor of the increases whose cycle contains the month of the
             row, in its price group; 1 when the increases are not removed) and _numerador_uplift = renewed USD
             / _factor_subida: the renewed USD the renewers would have paid without the increase.
    RULES:   the deflation divides only the numerator: the base is what fell due, it carries no increase."""
    period = configuration.period_col
    rows = renewers[renewers[period] < as_of]
    if window_months:
        rows = rows[rows[period] >= as_of - window_months]
    rows = rows.copy()
    if without_increases and len(increases):
        groups = price_group_of(rows, price_group_columns(rows, configuration))
        rows["_factor_subida"] = increase_factor(groups, rows[period], increases)
        rows["_numerador_uplift"] = rows["_numerador"] / rows["_factor_subida"]
    else:
        rows["_factor_subida"] = 1.0
        rows["_numerador_uplift"] = rows["_numerador"]
    return rows


def add_parent_ids(rows: pd.DataFrame, configuration: Config, copy: bool = True) -> pd.DataFrame:
    """The parent of an uplift cell (mandatory dims and discount bucket kept, the other extras
    '*') and its mandatory cell (only the uplift mandatory dims). copy=False adds the two columns to `rows` itself
    (when the caller already holds its own copy)."""
    uplift_mandatory = configuration.uplift_mandatory_dims or configuration.business_mandatory_dims
    bucket = [configuration.discount_bucket_column] if configuration.discount_value_column else []
    if copy:
        rows = rows.copy()
    rows["uplift_padre_id"] = join_columns(rows, list(uplift_mandatory) + bucket) + "|" + WILDCARD
    rows["uplift_celda_mandatory_id"] = join_columns(rows, list(uplift_mandatory))
    return rows



def estimate_cell_uplifts(fine_table: pd.DataFrame, renewers: pd.DataFrame, configuration: Config,
                          numerator_column: str = "_numerador", with_band: bool = True) -> pd.DataFrame:
    """The statistical uplift of every uplift cell of the fine table (future ones included):
    own if ≥ uplift_floor renewers, else parent, else mandatory cell, else global. Clipped to
    [0.01, uplift_cap]. Band: bootstrap of the renewer rows the uplift was computed from.
    numerator_column: the renewed USD of every row (_numerador_uplift: without the price increases).
    with_band=False skips the bootstrap (the exam of step 3.3 judges the uplift, not its band)."""
    renewed_units = configuration.renewed_units_col
    # one row per cell first, then its parent ids: the ids are made of the dims that define the cell, so every row
    # of a cell has the same ones. (Until 26-oct-2026 the ids were made on a copy of the whole fine table, one per
    # call: a transient copy of 1.3 GB with the extract of 1.1M rows, three times in step 3.3.)
    uplift_mandatory = configuration.uplift_mandatory_dims or configuration.business_mandatory_dims
    bucket = [configuration.discount_bucket_column] if configuration.discount_value_column else []
    cell_columns = list(dict.fromkeys([UPLIFT_CELL_ID_COLUMN, *uplift_mandatory, *bucket]))
    one_row_per_cell = fine_table[cell_columns].drop_duplicates(UPLIFT_CELL_ID_COLUMN)
    cells = add_parent_ids(one_row_per_cell, configuration)[[UPLIFT_CELL_ID_COLUMN, "uplift_padre_id", "uplift_celda_mandatory_id"]]
    cells = cells.reset_index(drop=True)

    def level(group_column):
        grouped = renewers.groupby(group_column)
        return pd.DataFrame({"uplift": grouped[numerator_column].sum() / grouped["_denominador"].sum(),
                             "renovadores": grouped[renewed_units].sum()})

    own, parent, cell = level(UPLIFT_CELL_ID_COLUMN), level("uplift_padre_id"), level("uplift_celda_mandatory_id")
    global_uplift = float(renewers[numerator_column].sum() / renewers["_denominador"].sum()) if len(renewers) else 1.0
    cells["renovadores"] = cells[UPLIFT_CELL_ID_COLUMN].map(own["renovadores"]).fillna(0.0)
    cells["uplift_propio"] = cells[UPLIFT_CELL_ID_COLUMN].map(own["uplift"])
    cells["renovadores_padre"] = cells["uplift_padre_id"].map(parent["renovadores"]).fillna(0.0)
    cells["uplift_padre"] = cells["uplift_padre_id"].map(parent["uplift"])
    cells["renovadores_celda"] = cells["uplift_celda_mandatory_id"].map(cell["renovadores"]).fillna(0.0)
    cells["uplift_celda"] = cells["uplift_celda_mandatory_id"].map(cell["uplift"])
    floor = configuration.uplift_floor
    conditions = [cells["renovadores"] >= floor, cells["renovadores_padre"] >= floor, cells["renovadores_celda"] > 0]
    cells["uplift_origen"] = np.select(conditions, [UPLIFT_OWN, UPLIFT_PARENT, UPLIFT_CELL], default=UPLIFT_GLOBAL)
    chosen = np.select(conditions, [cells["uplift_propio"], cells["uplift_padre"], cells["uplift_celda"]], default=global_uplift)
    cells["uplift"] = np.clip(chosen, configuration.min_uplift, configuration.uplift_cap)
    cells["recortado"] = (cells["uplift"] != chosen).astype(int)
    cells.attrs["uplift_global"] = global_uplift
    if not with_band:
        cells["uplift_bajo"] = np.nan
        cells["uplift_alto"] = np.nan
        return cells

    # the band: bootstrap of the renewer rows of the group the uplift came from
    generator = np.random.default_rng(configuration.random_seed)
    group_key_of = {UPLIFT_OWN: UPLIFT_CELL_ID_COLUMN, UPLIFT_PARENT: "uplift_padre_id", UPLIFT_CELL: "uplift_celda_mandatory_id"}
    row_groups = {origin: renewers.groupby(column).indices for origin, column in group_key_of.items()}
    numerator, denominator = renewers[numerator_column].to_numpy(dtype=float), renewers["_denominador"].to_numpy(dtype=float)
    lows, highs, cache = [], [], {}
    for _, cell_row in cells.iterrows():
        origin = cell_row["uplift_origen"]
        if origin == UPLIFT_GLOBAL:
            lows.append(np.nan); highs.append(np.nan); continue
        group_key = cell_row[group_key_of[origin]]
        if (origin, group_key) not in cache:
            positions = row_groups[origin].get(group_key, np.array([], dtype=int))
            if len(positions) < 2:
                cache[(origin, group_key)] = (np.nan, np.nan)
            else:
                picks = positions[generator.integers(0, len(positions), size=(configuration.uplift_bootstrap_samples, len(positions)))]
                samples = numerator[picks].sum(axis=1) / denominator[picks].sum(axis=1)
                cache[(origin, group_key)] = (float(np.quantile(samples, configuration.uplift_band_low)), float(np.quantile(samples, configuration.uplift_band_high)))
        low, high = cache[(origin, group_key)]
        lows.append(low); highs.append(high)
    cells["uplift_bajo"] = np.clip(lows, configuration.min_uplift, configuration.uplift_cap)
    cells["uplift_alto"] = np.clip(highs, configuration.min_uplift, configuration.uplift_cap)
    return cells


def state_uplift_factors(history: pd.DataFrame, cells: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The price of a renewal by its negative states: what the renewers of every combination (neutral, softcancel,
    dormant, softcancel+dormant…) paid, over what the uplift of their cell predicts for them.

    INPUT:   the renewer rows the cells were estimated with (history: the window, without the increases) and the
             cells (estimate_cell_uplifts).
    OUTPUT:  one row per combination: renewed units, renewed USD, the USD its cells predict, the factor measured and
             the factor applied.
    RULES:   the uplift cell has no states: its uplift comes mostly from the neutral renewers. A softcancel that renews
             does it with a retention offer, and may pay less than its cell says. For every combination
                 factor = Σ renewed USD (without the increases) / Σ (base of the uplift × uplift of its cell)
             over the same renewers, the same months and the same base the cells were estimated with: it is only
             the effect of the state. A combination with fewer than uplift_floor renewed units keeps factor 1."""
    states = negative_states(configuration)
    if not states or history.empty:
        return pd.DataFrame(columns=["combinacion", "renovadas", "usd_renovado", "usd_predicho_celda", "factor_medido", "factor"])
    rows = history.copy()
    rows["combinacion"] = state_combination(rows, states)
    cell_uplift = rows[UPLIFT_CELL_ID_COLUMN].map(cells.set_index(UPLIFT_CELL_ID_COLUMN)["uplift"])
    rows["_predicho_celda"] = rows["_denominador"] * cell_uplift
    rows = rows[rows["_predicho_celda"].notna()]
    factors = (rows.groupby("combinacion")
               .agg(renovadas=(configuration.renewed_units_col, "sum"), usd_renovado=("_numerador_uplift", "sum"),
                    usd_predicho_celda=("_predicho_celda", "sum"))
               .reset_index())
    factors["factor_medido"] = factors["usd_renovado"] / factors["usd_predicho_celda"]
    enough = factors["renovadas"] >= configuration.uplift_floor
    factors["factor"] = np.where(enough, factors["factor_medido"], 1.0)
    return factors.sort_values("usd_renovado", ascending=False).reset_index(drop=True)


def show_the_price_by_states(state_factors: pd.DataFrame, configuration: Config) -> None:
    """EL PRECIO POR ESTADO DEL CLIENTE on screen: the factor of every combination and what it says."""
    if state_factors is None or state_factors.empty:
        return
    doc = configuration.logger.doc
    doc(f"[{STEP_3_1_LABEL}] ══════════ EL PRECIO POR ESTADO DEL CLIENTE ══════════")
    doc(f"[{STEP_3_1_LABEL}] what the renewers of every combination of negative states paid over what the uplift of their cell "
        f"predicts (factor 1 = they renew at the price of their cell; below 1 = cheaper, e.g. with a retention offer; "
        f"fewer than {configuration.uplift_floor:.0f} renewed units: factor 1):")
    configuration.show_table(state_factors.round(3))
    neutral = state_factors[state_factors["combinacion"] == "neutral"]
    in_state = state_factors[(state_factors["combinacion"] != "neutral") & (state_factors["factor"] != 1.0)]
    if len(neutral) and len(in_state):
        lowest = in_state.sort_values("factor").iloc[0]
        doc(f"[{STEP_3_1_LABEL}] VALORACIÓN: the neutral renewers pay {neutral['factor'].iloc[0]:.1%} of their cell's price; "
            f"{lowest['combinacion']} the least, {lowest['factor']:.1%}. Step 3.3 decides whether the factor predicts better.")
    doc(f"[{STEP_3_1_LABEL}] ══════════ FIN DE EL PRECIO POR ESTADO DEL CLIENTE ══════════")


def state_factor_of_rows(rows: pd.DataFrame, state_factors: pd.DataFrame, configuration: Config) -> pd.Series:
    """The price factor of every row by its combination of negative states; 1 for a combination never measured."""
    if state_factors is None or state_factors.empty:
        return pd.Series(1.0, index=rows.index)
    combination = state_combination(rows, negative_states(configuration))
    return combination.map(state_factors.set_index("combinacion")["factor"]).fillna(1.0)


def contract_rule(discount_values: pd.Series, configuration: Config) -> pd.Series:
    """The contract rule: the uplift 1 / (1 − d) of a known discount d, capped at contract_discount_cap. The discount
    is measured against the MSRP of the contract at expiry, so the rule already carries the price increases between
    the purchase and the expiry."""
    return 1 / (1 - discount_values.clip(upper=configuration.contract_discount_cap))


def contract_check(renewers: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The contract rule against the past renewals with a known discount, per discount bucket and by who renews.

    COLUMNS: renovadores, usd_renovado, uplift_observado, uplift_regla, ratio_realizacion (observed / rule, all the
    renewers); cuota_neutras and realizacion_neutras (the rows without a negative state: the typical renewal the rule
    speaks of, the ones PATH_CONTRACT_NEUTRAL applies it to); realizacion_con_estado (softcancel, dormant…: offers and
    changes); realizacion_isolated (same product, no retention event, never a softcancel), when declared."""
    columns = ["tramo", "renovadores", "usd_renovado", "uplift_observado", "uplift_regla", "ratio_realizacion",
               "cuota_neutras", "realizacion_neutras", "realizacion_con_estado", "realizacion_isolated"]
    if not configuration.discount_value_column:
        return pd.DataFrame(columns=columns)
    known = renewers[renewers[configuration.discount_value_column].notna()].copy()
    known["_regla"] = contract_rule(known[configuration.discount_value_column], configuration)
    known["_regla_x_base"] = known["_regla"] * known["_denominador"]
    neutral = has_no_negative_state(known, configuration)
    bucket = configuration.discount_bucket_column

    def realization(rows: pd.DataFrame, numerator: str, base: str) -> pd.Series:
        grouped = rows.assign(_regla_x_esta_base=rows["_regla"] * rows[base]).groupby(bucket)
        return (grouped[numerator].sum() / grouped["_regla_x_esta_base"].sum().where(lambda value: value > 0))

    grouped = known.groupby(bucket)
    check = pd.DataFrame({"renovadores": grouped[configuration.renewed_units_col].sum(),
                          "usd_renovado": grouped["_numerador"].sum(),
                          "uplift_observado": grouped["_numerador"].sum() / grouped["_denominador"].sum(),
                          "uplift_regla": grouped["_regla_x_base"].sum() / grouped["_denominador"].sum()})
    check["ratio_realizacion"] = check["uplift_observado"] / check["uplift_regla"]
    check["cuota_neutras"] = (known[neutral].groupby(bucket)["_denominador"].sum() / grouped["_denominador"].sum()).reindex(check.index)
    check["realizacion_neutras"] = realization(known[neutral], "_numerador", "_denominador").reindex(check.index)
    check["realizacion_con_estado"] = realization(known[~neutral], "_numerador", "_denominador").reindex(check.index)
    if configuration.isolated_renewed_usd_col and configuration.isolated_renewed_pipeline_usd_col:
        known["_iso_usd"] = known[configuration.isolated_renewed_usd_col].fillna(0.0)
        known["_iso_base"] = known[configuration.isolated_renewed_pipeline_usd_col].fillna(0.0)
        check["realizacion_isolated"] = realization(known, "_iso_usd", "_iso_base").reindex(check.index)
    else:
        check["realizacion_isolated"] = np.nan
    return check.reset_index().rename(columns={bucket: "tramo"})[columns]


def estimate_uplift(fine_table: pd.DataFrame, configuration: Config) -> tuple:
    """The statistical uplift of every cell, the price factor of the customer states, the contract check and the price
    increases in play; checked and written. Returns (cells, state factors, contract check, increases, basis): the
    basis (renewer rows, the window of the uplift, the price monitor) is what the diagnostics of the price read."""
    configuration.log_step_start(STEP_3_1_LABEL, STEP_3_1_NAME, STEP_3_1_PURPOSE, STEP_3_1_ACTIONS, STEP_3_1_OUTPUT)
    check_log = []
    current_month = configuration.calendar_boundaries()["current"]

    # [1] the renewers
    renewers = renewer_rows(fine_table, configuration)
    base_text = (f"exact base: {configuration.renewed_pipeline_usd_col}" if configuration.renewed_pipeline_usd_col
                 else "approximate base: renewed units × the row's pipeline AUV (renewed_pipeline_usd_col not declared)")
    configuration.log_action(STEP_3_1_LABEL, 1, f"{len(renewers):,} renewer rows in the closed months "
                                            f"({renewers[configuration.renewed_units_col].sum():,.0f} renewed units, "
                                            f"${renewers['_numerador'].sum():,.0f}) · {base_text}")

    # [2] the price increases known today, the window of the uplift (its months inside a cycle deflated) and the
    #     statistical uplift of every cell over it
    window_months = configuration.uplift_window_months
    price = price_increases_as_of(fine_table, current_month, configuration)
    increases = active_increases(price["calendar"], current_month, window_months, configuration)
    history = uplift_history(renewers, current_month, increases, window_months, configuration.uplift_without_increases,
                             configuration)
    cells = estimate_cell_uplifts(fine_table, history, configuration, numerator_column="_numerador_uplift")
    # the price by states: what the renewers of every combination of negative states paid over their cell, over the
    # same window (step 7.2 applies it when step 3.3 finds it predicts better)
    state_factors = state_uplift_factors(history, cells, configuration)
    window_text = (f"window {history[configuration.period_col].min()}..{history[configuration.period_col].max()} "
                   f"({window_months} closed months)" if window_months and len(history) else "all the history")
    deflated = history[history["_factor_subida"] != 1.0]
    increases_text = (f"{len(increases)} price increases in play, removed from {len(deflated):,} renewer rows "
                      f"(${deflated['_numerador'].sum():,.0f} renewed USD deflated to ${deflated['_numerador_uplift'].sum():,.0f})"
                      if configuration.uplift_without_increases else
                      f"{len(increases)} price increases in play, NOT removed (uplift_without_increases off)")
    configuration.log_action(STEP_3_1_LABEL, 2, f"{window_text}: {len(history):,} renewer rows · {increases_text} · "
                                            f"{len(cells):,} uplift cells · origins {cells['uplift_origen'].value_counts().to_dict()} · "
                                            f"global uplift {cells.attrs['uplift_global']:.3f} · price factor by states "
                                            + " · ".join(f"{row.combinacion} {row.factor:.3f}" for row in state_factors.itertuples()))

    # [3] the contract rule
    check = contract_check(renewers, configuration)
    known_share = renewers[configuration.discount_value_column].notna().mean() if configuration.discount_value_column else 0.0
    configuration.log_action(STEP_3_1_LABEL, 3, f"contract rule checked on the {known_share:.0%} of renewer rows with a known "
                                            f"discount" if configuration.discount_value_column else "no exact discount declared")

    # [4] the checks
    configuration.log_action(STEP_3_1_LABEL, 4, "checking the uplifts")
    missing = set(fine_table[UPLIFT_CELL_ID_COLUMN]) - set(cells[UPLIFT_CELL_ID_COLUMN])
    configuration.log_check(STEP_3_1_LABEL, check_log, "every uplift cell of the fine table has an uplift (future cells included)",
                            not missing and cells["uplift"].notna().all(),
                            failure_detail=f"{len(missing)} cells without uplift", context=f"{len(cells):,} cells")
    configuration.log_check(STEP_3_1_LABEL, check_log, f"every uplift is inside [{configuration.min_uplift}, {configuration.uplift_cap}]",
                            cells["uplift"].between(configuration.min_uplift, configuration.uplift_cap).all(),
                            failure_detail="uplifts outside the range",
                            context=f"from {cells['uplift'].min():.3f} to {cells['uplift'].max():.3f}; {int(cells['recortado'].sum())} clipped")
    with_band = cells[cells["uplift_bajo"].notna() & (cells["recortado"] == 0)]
    outside = with_band[(with_band["uplift"] < with_band["uplift_bajo"] - 1e-9) | (with_band["uplift"] > with_band["uplift_alto"] + 1e-9)]
    configuration.log_check(STEP_3_1_LABEL, check_log, "the band contains the uplift of every cell that has one", outside.empty,
                            failure_detail=f"{len(outside)} cells whose uplift is outside its band",
                            context=f"{len(with_band):,} cells with a band", examples=outside)
    pending = price["pending"]
    configuration.log_check(STEP_3_1_LABEL, check_log,
                            f"no price group ends its history with a step not confirmed yet (above "
                            f"{configuration.price_step_threshold:.0%} for fewer than {configuration.price_persistence_months} months)",
                            pending.empty,
                            failure_detail=f"{len(pending)} groups with a step still unconfirmed "
                                           f"(${pending['usd_renovado'].sum() if len(pending) else 0:,.0f} renewed in its months): "
                                           f"if it is an increase, the forecast does not carry it · see SUBIDAS DE PRECIO",
                            blocking=False, context=f"{price['monitor']['price_group'].nunique():,} price groups watched",
                            examples=pending if len(pending) else None)

    # [5] the tables
    configuration.log_action(STEP_3_1_LABEL, 5, "writing the tables")
    configuration.write_table(STEP_3_1_LABEL, check_log, cells, TABLE_UPLIFT_CELLS)
    configuration.write_table(STEP_3_1_LABEL, check_log, check, TABLE_CONTRACT_CHECK)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_3_1_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_3_1_LABEL, STEP_3_1_NAME, check_log)

    # [7] the uplift by origin and the price by customer states (the diagnostics of the price for business: the next
    #     stage)
    configuration.log_action(STEP_3_1_LABEL, 7, "uplift cells by origin (propia: ≥ floor renewers; padre: its mandatory dims and "
                                            "discount bucket; celda: its mandatory dims; global):")
    configuration.show_table(cells.groupby("uplift_origen").agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"),
                                                                uplift_medio=("uplift", "mean"),
                                                                renovadores=("renovadores", "sum")).reset_index())
    show_the_price_by_states(state_factors, configuration)
    basis = dict(renewers=renewers, history=history, price=price, increases=increases, current_month=current_month)
    return cells, state_factors, check, increases, basis


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 3.2 · PRICE DIAGNOSTICS  (until 25-oct-2026: step_15_price_diagnostics.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE DIAGNOSTICS OF THE PRICE, for business: how the renewers' price behaves, read
# from the closed months. Nothing here changes a prediction: the uplift is estimated in step 3.1 and examined in step 3.3;
# this stage only explains it (what kind of stage: knowledge of the patterns).
#
# INPUT:   the fine table · the uplift estimation of step 3.1 (its renewer rows, the window of the uplift, the price
#          monitor and the increases in play) · its contract check.
# OUTPUT:  the homogeneity of every uplift cell (table sff_uplift_homogeneidad, exact base only) · on screen: the
#          contract rule read for business, the revaluation of the renewers, the price increases and the window.
#
# Actions (logged as they are done):
#   1. the homogeneity of every cell (only with the exact base): renewers' average due price against the cell's, and
#      the approximate uplift against the exact one; by discount bucket
#   2. write the homogeneity                                           check 1
#   3. count the checks; stop if any failed
#   4. the contract rule read for business (LA REGLA DEL CONTRATO): who meets it, and the list price implied by
#      every discount bucket
#   5. the revaluation of the renewers, read for business: the population of the claim (isolated renewals, no
#      acquisition), where its uplift sits (percentiles, histogram by series), the discount split and the monthly path
#      with its steps
#   6. the price increases and the window of the uplift, read for business (SUBIDAS DE PRECIO Y VENTANA DEL UPLIFT):
#      the uplift of the portfolio with all the history, in the window and in the window without increases; every
#      increase in play with the month of its cycle, the future months that still carry it and where its effect is
#      lost; the steps not confirmed yet
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. table sff_uplift_homogeneidad written and read back (only with the exact base)
#
# Output: the homogeneity (or None) · table sff_uplift_homogeneidad.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_3_2_LABEL = "3.2"
STEP_3_2_NAME = "PRICE DIAGNOSTICS"
STEP_3_2_PURPOSE = ("explain the price of the renewers for business, from the closed months: the homogeneity of the "
                "uplift cells, the contract rule, the revaluation of the renewers and the price increases; no "
                "prediction reads it")
STEP_3_2_ACTIONS = ["the homogeneity of every cell: the renewers' average due price against the cell's (exact base only)",
                "write the homogeneity (check 1)",
                "count the checks; stop if any failed",
                "the contract rule read for business: who meets it, the list price implied by every bucket",
                "the revaluation of the renewers: population, distribution, discount split, monthly path",
                "the price increases and the window of the uplift: the month of every cycle, where its effect is lost"]
STEP_3_2_OUTPUT = "the homogeneity of every uplift cell · table sff_uplift_homogeneidad"


def diagnose_the_price(fine_table: pd.DataFrame, basis: dict, contract: pd.DataFrame, configuration: Config):
    """The stage 3.2 of the pipeline (see the header). Returns the homogeneity (None without the exact base)."""
    configuration.log_step_start(STEP_3_2_LABEL, STEP_3_2_NAME, STEP_3_2_PURPOSE, STEP_3_2_ACTIONS, STEP_3_2_OUTPUT)
    check_log = []
    renewers = basis["renewers"]

    # [1] the homogeneity: where renewers were worth the cell's average, the approximation was exact
    homogeneity = homogeneity_table(fine_table, renewers, configuration)
    if homogeneity is not None:
        weighted_ratio = (homogeneity["ratio_seleccion"] * homogeneity["usd_renovadores"]).sum() / homogeneity["usd_renovadores"].sum()
        configuration.log_action(STEP_3_2_LABEL, 1, f"homogeneity of {len(homogeneity):,} cells with renewers: the renewers' due "
                                                f"price over the cell's averages {weighted_ratio:.3f} weighted by their USD "
                                                f"(1.000 = renewers worth the cell's average: the old approximation was exact)")
        if configuration.discount_bucket_column in homogeneity.columns:
            by_bucket = (homogeneity.assign(_pesado=homogeneity["ratio_seleccion"] * homogeneity["usd_renovadores"])
                         .groupby(configuration.discount_bucket_column)
                         .agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"), usd_renovadores=("usd_renovadores", "sum"),
                              _pesado=("_pesado", "sum")))
            by_bucket["ratio_seleccion"] = by_bucket.pop("_pesado") / by_bucket["usd_renovadores"]
            configuration.logger.doc(f"[{STEP_3_2_LABEL}] the homogeneity by discount bucket (ratio_seleccion 1 = renewers worth "
                                     f"the cell's average; above 1 = the dearer renew more):")
            configuration.show_table(by_bucket.reset_index())
    else:
        configuration.log_action(STEP_3_2_LABEL, 1, "no exact base declared: the homogeneity cannot be measured")

    # [2] the table
    configuration.log_action(STEP_3_2_LABEL, 2, "writing the homogeneity")
    if homogeneity is not None:
        configuration.write_table(STEP_3_2_LABEL, check_log, homogeneity, TABLE_UPLIFT_HOMOGENEITY)

    # [3] the count of the checks; stop if anything failed
    configuration.log_action(STEP_3_2_LABEL, 3, "counting the checks")
    configuration.log_check_summary(STEP_3_2_LABEL, STEP_3_2_NAME, check_log)

    # [4] the contract rule, read for business
    configuration.log_action(STEP_3_2_LABEL, 4, "the contract rule read for business")
    if len(contract):
        configuration.logger.doc(f"[{STEP_3_2_LABEL}] ══════════ LA REGLA DEL CONTRATO ══════════")
        configuration.logger.doc(f"[{STEP_3_2_LABEL}] the contract rule 1/(1 − d) against the past renewals with a known discount "
                                 f"(realizacion 1 = the rule is exact; neutras = no negative customer state, the rows the "
                                 f"path contrato_neutras applies it to):")
        configuration.show_table(contract.round(3))
        review_the_contract_rule(contract, fine_table, configuration)

    # [5] the revaluation of the renewers, read for business
    configuration.log_action(STEP_3_2_LABEL, 5, "the revaluation of the renewers (isolated renewals, no acquisition): "
                                            "population, distribution, discount split, monthly path")
    revaluation_report(renewers, configuration)

    # [6] the price increases and the window of the uplift, read for business
    configuration.log_action(STEP_3_2_LABEL, 6, "the price increases and the window of the uplift")
    show_price_increases(fine_table, renewers, basis["history"], basis["increases"], basis["price"], basis["current_month"],
                         configuration)
    return homogeneity


def homogeneity_table(fine_table: pd.DataFrame, renewers: pd.DataFrame, configuration: Config):
    """One row per uplift cell with renewers (exact base only): what the renewers were worth per unit
    against the cell's average due price, and the approximate uplift against the exact one.

    ratio_seleccion = (renewers' due USD / renewed units) / (cell's due USD / cell's due units). At 1 the
    renewers were worth the cell's average and the old approximation was exact; above 1 the dearer renew
    more and the approximation inflated the uplift by that ratio (uplift_aproximado = uplift × ratio).
    Where the ratio moves away from 1 the cell mixes prices: the answer is to segment better, not to add
    a factor to the forecast. Returns None when the exact base is not declared.
    """
    if not configuration.renewed_pipeline_usd_col:
        return None
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                        & (fine_table[configuration.pipeline_units_col] > 0)]
    cell_due = closed.groupby(UPLIFT_CELL_ID_COLUMN).agg(
        vencen_unidades=(configuration.pipeline_units_col, "sum"),
        vencen_usd=(configuration.pipeline_usd_col, "sum"))
    by_cell = renewers.groupby(UPLIFT_CELL_ID_COLUMN).agg(
        renovadores=(configuration.renewed_units_col, "sum"),
        usd_renovadores=("_denominador", "sum"),
        usd_renovado=("_numerador", "sum"),
        base_aproximada=("_denominador_aproximado", "sum"))
    bucket_of_cell = (renewers.drop_duplicates(UPLIFT_CELL_ID_COLUMN)
                      .set_index(UPLIFT_CELL_ID_COLUMN)[configuration.discount_bucket_column]
                      if configuration.discount_bucket_column in renewers.columns else None)
    homogeneity = by_cell.join(cell_due, how="left").reset_index()
    homogeneity["precio_medio_celda"] = homogeneity["vencen_usd"] / homogeneity["vencen_unidades"]
    homogeneity["precio_medio_renovadores"] = homogeneity["usd_renovadores"] / homogeneity["renovadores"]
    homogeneity["ratio_seleccion"] = homogeneity["precio_medio_renovadores"] / homogeneity["precio_medio_celda"]
    homogeneity["uplift_exacto"] = homogeneity["usd_renovado"] / homogeneity["usd_renovadores"]
    homogeneity["uplift_aproximado"] = homogeneity["usd_renovado"] / homogeneity["base_aproximada"]
    if bucket_of_cell is not None:
        homogeneity[configuration.discount_bucket_column] = homogeneity[UPLIFT_CELL_ID_COLUMN].map(bucket_of_cell)
    return homogeneity.drop(columns=["base_aproximada"])



def weighted_percentiles(values: np.ndarray, weights: np.ndarray, points: list) -> list:
    """The value below which each share of the total weight sits (weights all positive)."""
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    cumulative_share = np.cumsum(ordered_weights) / ordered_weights.sum()
    return [float(np.interp(point, cumulative_share, ordered_values)) for point in points]


def histogram_lines(series_table: pd.DataFrame, title: str, configuration: Config) -> list:
    """The uplift histogram of a set of series, one text line per bucket: how many series fall in it,
    how much renewed USD, and a bar proportional to the USD share (the money is what matters)."""
    lines = [f"histogram · {title} (bar = % of the renewed USD)",
             f"{'uplift':>15}  {'series':>6}  {'% series':>8}  {'USD renovado':>14}  {'% USD':>6}"]
    assigned = pd.cut(series_table["uplift"], configuration.revaluation_uplift_buckets, right=False)
    grouped = series_table.groupby(assigned, observed=False).agg(series=("uplift", "size"), usd=("usd_renovado", "sum"))
    usd_share = grouped["usd"] / max(grouped["usd"].sum(), 1e-9)
    series_share = grouped["series"] / max(grouped["series"].sum(), 1e-9)
    longest = max(usd_share.max(), 1e-9)
    for (interval, row), usd_part, series_part in zip(grouped.iterrows(), usd_share, series_share):
        label = (f"      < {interval.right:.3f}" if interval.left == 0.0
                 else f"      ≥ {interval.left:.3f}" if np.isinf(interval.right)
                 else f"{interval.left:.3f} – {interval.right:.3f}")
        bar = "█" * int(round(configuration.histogram_width * usd_part / longest)) if row["usd"] > 0 else ""
        lines.append(f"{label:>15}  {int(row['series']):>6}  {series_part:>8.1%}  {row['usd']:>14,.0f}  {usd_part:>6.1%}  {bar}")
    return lines


def report_population(rows: pd.DataFrame, configuration: Config) -> tuple:
    """The population of the revaluation report, with its own measures in _rep_* columns.

    The ISOLATED renewals when the Config declares them (isolated_renewed_usd_col, and its base:
    isolated_renewed_pipeline_usd_col, exact, or isolated_renewed_units_col × the row's due price,
    approximate); every renewer otherwise. Acquisitions out (acquisition_column / acquisition_values).
    Returns (population, funnel rows as (label, frame, usd column), population text, base text).
    """
    row_price = rows[configuration.pipeline_usd_col] / rows[configuration.pipeline_units_col]
    funnel_rows = [("renewer rows", rows, "_numerador")]
    if configuration.isolated_renewed_usd_col and (configuration.isolated_renewed_pipeline_usd_col
                                                   or configuration.isolated_renewed_units_col):
        population = rows.copy()
        population["_rep_numerador"] = population[configuration.isolated_renewed_usd_col]
        if configuration.isolated_renewed_pipeline_usd_col:
            population["_rep_denominador"] = population[configuration.isolated_renewed_pipeline_usd_col]
            base_text = "exact (what the isolated renewers were worth)"
        else:
            population["_rep_denominador"] = population[configuration.isolated_renewed_units_col] * row_price
            base_text = "APPROXIMATE (isolated renewed units × the row's due price)"
        population["_rep_renovadas"] = (population[configuration.isolated_renewed_units_col]
                                        if configuration.isolated_renewed_units_col
                                        else population[configuration.renewed_units_col])
        population["_rep_vencen_unidades"] = (population[configuration.isolated_pipeline_units_col]
                                              if configuration.isolated_pipeline_units_col
                                              else population[configuration.pipeline_units_col])
        population = population[population["_rep_denominador"] > 0].copy()
        add_licence_level_columns(population, configuration)
        funnel_rows.append(("· isolated renewals (normal process, no retention event)", population, "_rep_numerador"))
        population_text = "the ISOLATED renewals"
    else:
        population = rows.copy()
        population["_rep_numerador"] = population["_numerador"]
        population["_rep_denominador"] = population["_denominador"]
        population["_rep_renovadas"] = population[configuration.renewed_units_col]
        population["_rep_vencen_unidades"] = population[configuration.pipeline_units_col]
        base_text = ("exact (what the renewers were worth)" if configuration.renewed_pipeline_usd_col
                     else "APPROXIMATE (renewed units × the row's due price)")
        population_text = "EVERY renewer (the isolated renewals are not declared in the Config)"
        for helper_column in ["_rep_momento", "_rep_cerca_100", "_rep_mas_110"]:
            population[helper_column] = np.nan                     # the licence level only exists for the isolated
    if configuration.acquisition_column and configuration.acquisition_column in population.columns:
        population = population[~population[configuration.acquisition_column].isin(configuration.acquisition_values)]
        funnel_rows.append((f"· and not an acquisition ({configuration.acquisition_column})", population, "_rep_numerador"))
    return population, funnel_rows, population_text, base_text


def add_licence_level_columns(population: pd.DataFrame, configuration: Config) -> None:
    """The licence-level measures of the isolated renewals, in _rep_* columns (NaN when not declared):
    the second moment (for the std deviation), the value in [0.95, 1.05) and the value at or above 1.10."""
    moment_column = configuration.isolated_renewed_usd_sq_over_tr_col
    population["_rep_momento"] = population[moment_column] if moment_column else np.nan
    bands = configuration.isolated_ratio_bands
    near_columns = [column for column, _, low, high, _ in bands if low >= 0.95 and high <= 1.05]
    high_columns = [column for column, _, low, _, _ in bands if low >= 1.10]
    complete = len(bands) == len(ISOLATED_RATIO_BANDS)
    population["_rep_cerca_100"] = population[near_columns].sum(axis=1) if complete else np.nan
    population["_rep_mas_110"] = population[high_columns].sum(axis=1) if complete else np.nan


def licence_level_summary(population: pd.DataFrame, group_column: str, configuration: Config) -> pd.DataFrame:
    """Per group: the value renewed, the uplift, and (when the extract carries them) the std deviation of the
    ratio between licences and the share of the value near 1 and at or above 1.10."""
    summary = (population.groupby(group_column, dropna=False)
               .agg(usd_renovado=("_rep_numerador", "sum"), base=("_rep_denominador", "sum"),
                    momento=("_rep_momento", "sum"), cerca=("_rep_cerca_100", "sum"), alto=("_rep_mas_110", "sum")))
    summary["uplift"] = (summary["usd_renovado"] / summary["base"]).round(3)
    with_moment, with_bands = population["_rep_momento"].notna().any(), population["_rep_cerca_100"].notna().any()
    summary["desv"] = (weighted_ratio_spread(summary["base"], summary["usd_renovado"], summary["momento"]).round(3)
                       if with_moment else np.nan)
    summary["cerca_100"] = (summary["cerca"] / summary["base"]).round(3) if with_bands else np.nan
    summary["mas_110"] = (summary["alto"] / summary["base"]).round(3) if with_bands else np.nan
    summary["usd_renovado"] = summary["usd_renovado"].round(0)
    return summary.reset_index().drop(columns=["base", "momento", "cerca", "alto"])


def licence_level_report(population: pd.DataFrame, configuration: Config) -> None:
    """The isolated renewals licence by licence (from the sums the extract builds per licence): the weighted
    std deviation of the ratio, the histogram of the six ratio bands, and the verdict on the claim "renewing
    the same product with no retention event, a licence renews at about its value"."""
    doc = configuration.logger.doc
    base = population["_rep_denominador"].sum()
    renewed = population["_rep_numerador"].sum()
    mean_ratio = renewed / base
    spread_text = ""
    if population["_rep_momento"].notna().any():
        spread = float(weighted_ratio_spread(pd.Series([base]), pd.Series([renewed]),
                                             pd.Series([population["_rep_momento"].sum()])).iloc[0])
        spread_text = f" · std deviation between licences {spread:.3f} (two thirds of the value within ±{spread:.1%})"
    doc(f"[{STEP_3_2_LABEL}] LICENCE BY LICENCE, the isolated renewals: mean ratio {mean_ratio:.3f}{spread_text}")
    bands = configuration.isolated_ratio_bands
    if len(bands) != len(ISOLATED_RATIO_BANDS):
        return
    values = [population[column].sum() for column, _, _, _, _ in bands]
    total = max(sum(values), 1e-9)
    if abs(sum(values) / max(base, 1e-9) - 1) > 0.001:
        doc(f"[{STEP_3_2_LABEL}] ATENCIÓN: the six bands add up to {sum(values) / max(base, 1e-9):.3f} × the isolated base "
            f"(must be 1.000): the extract's bands count other licences, this histogram is NOT reliable (see step 1.2, "
            f"ISOLATED: COHERENCIA DEL EXTRACTO)")
    longest = max(max(values) / total, 1e-9)
    doc(f"[{STEP_3_2_LABEL}] histogram · the value of the isolated renewals by renewal ratio, licence by licence")
    doc(f"[{STEP_3_2_LABEL}] {'ratio':>13}  {'USD que valían':>16}  {'% valor':>7}")
    for (_, _, _, _, label), value in zip(bands, values):
        bar = "█" * int(round(configuration.histogram_width * (value / total) / longest)) if value > 0 else ""
        doc(f"[{STEP_3_2_LABEL}] {label:>13}  {value:>16,.0f}  {value / total:>7.1%}  {bar}")
    near = population["_rep_cerca_100"].sum() / total
    high = population["_rep_mas_110"].sum() / total
    if near >= 0.80:
        verdict = (f"the claim HOLDS: {near:.0%} of the value renews within ±5 % of what it was worth; the uplift of "
                   f"this population is price, not behaviour")
    elif high >= 0.25:
        verdict = (f"the claim does NOT hold: {high:.0%} of the value renews 10 % or more above what it was worth and "
                   f"only {near:.0%} within ±5 %. With the same product and no retention event, that is a price "
                   f"increase or a discount that ends; the monthly path below says which")
    else:
        verdict = (f"the claim holds only in part: {near:.0%} of the value within ±5 %, {high:.0%} at or above 1.10")
    doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: {verdict}")


def revaluation_report(renewers: pd.DataFrame, configuration: Config) -> None:
    """Action 9: the revaluation of the renewers, read for business, on screen only.

    INPUT:   the renewer rows of action 1 (closed months, renewed > 0, with the uplift base in
             _denominador: exact when the extract carries it, approximate otherwise).
    OUTPUT:  nothing returned, nothing written: the funnel of the population (isolated renewals, no
             acquisition), where the uplift of its heavy series sits (aggregate, weighted percentiles,
             histogram), the discount split, the 15 biggest series and the monthly path with its steps.
    RULES:   the population is the ISOLATED renewals (isolated_* of the Config: the normal renewal
             process, no retention event) when they are declared, every renewer otherwise (logged);
             the acquisition filter uses acquisition_column / acquisition_values. See report_population.
             A series enters the distribution when it averages ≥ configuration.revaluation_units_floor units due per month
             in the window, or is among the configuration.revaluation_top_series biggest by USD due.
    EDGE CASES: with no series past the floor, the report says so and stops; a single series still
             gets its percentiles (they are all its own uplift).
    """
    doc = configuration.logger.doc
    closed_periods = sorted(renewers[configuration.period_col].unique())
    window = closed_periods[-configuration.revaluation_window_months:]
    population, funnel_rows, population_text, base_text = report_population(
        renewers[renewers[configuration.period_col].isin(window)], configuration)
    doc(f"[{STEP_3_2_LABEL}] window {window[0]} to {window[-1]} · population: {population_text} · uplift base: {base_text}")
    configuration.show_table(pd.DataFrame([{"population": label, "rows": len(stage),
                                            "usd_renovado": round(float(stage[usd_column].sum()))}
                                           for label, stage, usd_column in funnel_rows]))

    # one row per forecast series over its months of the window; the heavy ones
    discount = configuration.discount_value_column
    per_series = (population.groupby(SERIES_ID_COLUMN)
                  .agg(meses=(configuration.period_col, "nunique"),
                       vencen_unidades=("_rep_vencen_unidades", "sum"),
                       vencen_usd=(configuration.pipeline_usd_col, "sum"),
                       renovadas=("_rep_renovadas", "sum"),
                       usd_renovado=("_rep_numerador", "sum"),
                       base=("_rep_denominador", "sum"),
                       momento=("_rep_momento", "sum"),
                       cerca_100=("_rep_cerca_100", "sum"),
                       mas_110=("_rep_mas_110", "sum"),
                       usd_desconocido=("_rep_numerador", lambda values:
                                        values[population.loc[values.index, discount].isna()].sum() if discount else 0.0))
                  .reset_index())
    per_series["unidades_mes"] = per_series["vencen_unidades"] / per_series["meses"]
    per_series["tasa_unidades"] = per_series["renovadas"] / per_series["vencen_unidades"]
    per_series["uplift"] = per_series["usd_renovado"] / per_series["base"]
    with_licence_level = population["_rep_momento"].notna().any()
    with_bands = population["_rep_cerca_100"].notna().any()
    per_series["desv"] = (weighted_ratio_spread(per_series["base"], per_series["usd_renovado"], per_series["momento"])
                          if with_licence_level else np.nan)
    per_series["cerca_100"] = per_series["cerca_100"] / per_series["base"] if with_bands else np.nan
    per_series["mas_110"] = per_series["mas_110"] / per_series["base"] if with_bands else np.nan
    biggest = per_series.nlargest(configuration.revaluation_top_series, "vencen_usd")[SERIES_ID_COLUMN]
    selected = per_series[(per_series["unidades_mes"] >= configuration.revaluation_units_floor)
                          | per_series[SERIES_ID_COLUMN].isin(set(biggest))].copy()
    if not len(selected):
        doc(f"[{STEP_3_2_LABEL}] no series past the floor ({configuration.revaluation_units_floor:,.0f} units due per month): nothing to show")
        return
    if with_licence_level or with_bands:
        licence_level_report(population, configuration)
    covered = selected["usd_renovado"].sum() / max(per_series["usd_renovado"].sum(), 1e-9)
    doc(f"[{STEP_3_2_LABEL}] {len(selected):,} series in the distribution (floor {configuration.revaluation_units_floor:,.0f} units/month or "
        f"top {configuration.revaluation_top_series} by USD due): {covered:.0%} of the renewed USD of the population")

    # the claim, measured: the aggregate, the weighted percentiles and the money in each zone
    weights = selected["usd_renovado"].to_numpy(dtype=float)
    uplifts = selected["uplift"].to_numpy(dtype=float)
    p10, p25, p50, p75, p90 = weighted_percentiles(uplifts, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    share_100_105 = weights[(uplifts >= 1.00) & (uplifts < 1.05)].sum() / weights.sum()
    share_095_105 = weights[(uplifts >= 0.95) & (uplifts < 1.05)].sum() / weights.sum()
    share_above_110 = weights[uplifts >= 1.10].sum() / weights.sum()
    doc(f"[{STEP_3_2_LABEL}] aggregate uplift {selected['usd_renovado'].sum() / selected['base'].sum():.3f} · weighted "
        f"percentiles p10 {p10:.3f} · p25 {p25:.3f} · MEDIAN {p50:.3f} · p75 {p75:.3f} · p90 {p90:.3f}")
    doc(f"[{STEP_3_2_LABEL}] between SERIES (each series at its mean): {share_100_105:.0%} of the renewed USD in series "
        f"whose uplift sits in [1.00, 1.05) · {share_095_105:.0%} in [0.95, 1.05) · {share_above_110:.0%} at or above 1.10")
    if not with_bands and share_above_110 > 0.15:      # with the bands, the licence level above already answered the claim
        doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: a heavy tail at or above 1.10. Look at its discount buckets below: a tail in "
            f"high KNOWN buckets is the contract losing its discount (legitimate); one in the unknown bucket is the "
            f"statistical path and deserves the homogeneity table")
    elif not with_bands and share_095_105 > 0.80:
        doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: the claim holds: most of the money renews within ±5 % of what it was worth")
    elif not with_bands:
        doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: the money is spread beyond ±5 %: the buckets below say whether the spread "
            f"follows the discount (expected) or not")

    if discount:
        known_label = np.where(population[discount].notna(), "discount known", "discount unknown")
        by_known = licence_level_summary(population.assign(_corte=known_label), "_corte", configuration)
        configuration.show_table(by_known)
        doc(f"[{STEP_3_2_LABEL}] by discount bucket (desv = the std deviation of the ratio between licences; cerca_100 = "
            f"share of the value in [0.95, 1.05); mas_110 = share at or above 1.10):")
        configuration.show_table(licence_level_summary(population, configuration.discount_bucket_column, configuration))

    for line in histogram_lines(selected, "every series in the distribution", configuration):
        doc(f"[{STEP_3_2_LABEL}] {line}")
    mostly_unknown = selected[selected["usd_desconocido"] / selected["usd_renovado"] > 0.5]
    if len(mostly_unknown):
        for line in histogram_lines(mostly_unknown, "series whose renewed USD is mostly of UNKNOWN discount", configuration):
            doc(f"[{STEP_3_2_LABEL}] {line}")

    top = selected.nlargest(15, "vencen_usd")[[SERIES_ID_COLUMN, "meses", "unidades_mes", "vencen_usd",
                                               "tasa_unidades", "uplift", "desv", "cerca_100", "mas_110"]].copy()
    top["unidades_mes"] = top["unidades_mes"].round(0)
    for rounded_column in ["tasa_unidades", "uplift", "desv", "cerca_100", "mas_110"]:
        top[rounded_column] = top[rounded_column].round(3)
    doc(f"[{STEP_3_2_LABEL}] the 15 series with the most USD due:")
    configuration.show_table(top)

    # the monthly path of the same population, over a longer window: steps betray a price increase
    monthly_window = closed_periods[-configuration.revaluation_monthly_months:]
    monthly_population = report_population(renewers[renewers[configuration.period_col].isin(monthly_window)],
                                           configuration)[0]
    monthly = (monthly_population.groupby(configuration.period_col)
               .agg(renovadas=("_rep_renovadas", "sum"), usd_renovado=("_rep_numerador", "sum"),
                    base=("_rep_denominador", "sum"), momento=("_rep_momento", "sum"),
                    cerca=("_rep_cerca_100", "sum"), alto=("_rep_mas_110", "sum")).reset_index())
    monthly["uplift"] = monthly["usd_renovado"] / monthly["base"]
    monthly["desv"] = (weighted_ratio_spread(monthly["base"], monthly["usd_renovado"], monthly["momento"])
                       if with_licence_level else np.nan)
    monthly["cerca_100"] = monthly["cerca"] / monthly["base"] if with_bands else np.nan
    monthly["mas_110"] = monthly["alto"] / monthly["base"] if with_bands else np.nan
    monthly["media_12_previos"] = monthly["uplift"].shift(1).rolling(window=12, min_periods=6).mean()
    monthly["escalon"] = monthly["uplift"] / monthly["media_12_previos"] - 1
    doc(f"[{STEP_3_2_LABEL}] month by month ({len(monthly)} closed, same population): escalon = the month over the average "
        f"of its previous 12")
    shown = monthly.drop(columns=["usd_renovado", "base", "momento", "cerca", "alto"]).copy()
    for rounded_column in ["uplift", "desv", "cerca_100", "mas_110"]:
        shown[rounded_column] = shown[rounded_column].round(3)
    shown["media_12_previos"] = shown["media_12_previos"].round(3)
    shown["escalon"] = (100 * shown["escalon"]).round(1)
    configuration.show_table(shown.rename(columns={"escalon": "escalon_pct"}))
    recent = monthly.tail(configuration.revaluation_window_months).dropna(subset=["escalon"]).reset_index(drop=True)
    above = (recent["escalon"] >= configuration.price_step_threshold).to_numpy()
    longest_run, current_run = 0, 0
    for is_above in above:
        current_run = current_run + 1 if is_above else 0
        longest_run = max(longest_run, current_run)
    listed = " · ".join(f"{row[configuration.period_col]} ({row['escalon']:+.1%})"
                        for _, row in recent[above].iterrows())
    if longest_run >= configuration.price_persistence_months:
        doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: a step of ≥ {configuration.price_step_threshold:.0%} held {longest_run} months in a row in the "
            f"last 12 closed ({listed}): the shape of a price increase (its effect lasts one renewal cycle)")
    elif above.any():
        doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: months above {configuration.price_step_threshold:.0%} but never {configuration.price_persistence_months} in a "
            f"row ({listed}): noise, not an increase")
    elif len(recent):
        doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: no step of ≥ {configuration.price_step_threshold:.0%} in the last 12 closed months: no sign "
            f"of a price increase in this population")

def show_price_increases(fine_table: pd.DataFrame, renewers: pd.DataFrame, history: pd.DataFrame, increases: pd.DataFrame,
                         price: dict, current_month, configuration: Config) -> None:
    """Action 10: the price increases and the window of the uplift, on screen only.

    INPUT:   the fine table (its future rows: what still falls due), every renewer row, the rows of the window
             (uplift_history), the increases in play (active_increases), the detection (price_increases_as_of)
             and the current month.
    OUTPUT:  nothing returned, nothing written: the uplift of the portfolio three ways (all the history, the
             window, the window without increases), every increase in play (its size, the month of its cycle
             today, the future months that still carry it and the USD that falls due in them, the month its
             effect is lost) and the steps not confirmed yet, with a VALORACIÓN line for each.
    """
    doc = configuration.logger.doc
    period = configuration.period_col
    cycle = configuration.price_cycle_months
    group_dims = price_group_columns(fine_table, configuration)
    doc(f"[{STEP_3_2_LABEL}] ══════════ SUBIDAS DE PRECIO Y VENTANA DEL UPLIFT ══════════")
    window_months = configuration.uplift_window_months
    doc(f"[{STEP_3_2_LABEL}] detector on {price['detector_series'] or '—'} per price group "
        f"({', '.join(group_dims) or 'cartera'}): a step ≥ {configuration.price_step_threshold:.0%} over its previous 12 "
        f"months held {configuration.price_persistence_months} months; its effect lasts {cycle} months · uplift window: "
        f"{f'the last {window_months} closed months' if window_months else 'all the history'} · increases "
        f"{'REMOVED from the window and applied again to the future months inside their cycle' if configuration.uplift_without_increases else 'NOT removed'}")

    # the uplift of the portfolio three ways: what the window and the removal change
    def portfolio_uplift(rows, numerator_column):
        return float(rows[numerator_column].sum() / rows["_denominador"].sum()) if len(rows) else np.nan
    three_ways = pd.DataFrame([
        {"forma": "toda la historia", "meses": renewers[period].nunique(), "usd_renovado": renewers["_numerador"].sum(),
         "uplift": portfolio_uplift(renewers, "_numerador")},
        {"forma": "ventana", "meses": history[period].nunique(), "usd_renovado": history["_numerador"].sum(),
         "uplift": portfolio_uplift(history, "_numerador")},
        {"forma": "ventana sin subidas (el que usa el forecast)", "meses": history[period].nunique(),
         "usd_renovado": history["_numerador_uplift"].sum(), "uplift": portfolio_uplift(history, "_numerador_uplift")}])
    three_ways["usd_renovado"] = three_ways["usd_renovado"].round(0)
    three_ways["uplift"] = three_ways["uplift"].round(4)
    doc(f"[{STEP_3_2_LABEL}] the uplift of the portfolio (Σ renewed USD / Σ base of the renewers), three ways:")
    configuration.show_table(three_ways)

    # every increase in play: the month of its cycle today, what still falls due inside it, where it is lost
    # only the columns it sums (not a copy of every column of the future rows: memory, 26-oct-2026)
    future = fine_table.loc[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION, [*group_dims, period, configuration.pipeline_usd_col]]
    future_due = (future.assign(_grupo=price_group_of(future, group_dims))
                  .groupby(["_grupo", period])[configuration.pipeline_usd_col].sum())
    if len(increases):
        shown = increases.copy()
        shown["usd_vence_con_efecto"] = [
            float(future_due.loc[group][(future_due.loc[group].index >= current_month)
                                        & (future_due.loc[group].index <= end)].sum())
            if group in future_due.index.get_level_values(0) else 0.0
            for group, end in zip(shown["price_group"], shown["fin_ciclo"])]
        for share_column in ("escalon", "tamano"):
            shown[share_column] = (100 * shown[share_column]).round(1)
        shown["factor"] = shown["factor"].round(4)
        for money_column in ("usd_renovado_ciclo", "usd_vence_con_efecto"):
            shown[money_column] = shown[money_column].round(0)
        doc(f"[{STEP_3_2_LABEL}] the increases in play ({len(shown)}; escalon = the step of its first month, tamano = of the "
            f"cycle seen, in %; mes_del_ciclo = the month of its cycle that {current_month} is; usd_vence_con_efecto = "
            f"what falls due from {current_month} inside its cycle):")
        configuration.show_table(shown.drop(columns=["price_group"]))
        for _, increase in shown.head(configuration.price_increases_shown).iterrows():
            name = increase["price_group"]
            if increase["meses_futuros_con_efecto"] > 0:
                doc(f"[{STEP_3_2_LABEL}] VALORACIÓN {name}: subida de {increase['tamano']:+.1f}% desde {increase['inicio']}; "
                    f"{current_month} es el mes {increase['mes_del_ciclo']} de {cycle}: el forecast la aplica a "
                    f"{current_month}..{increase['fin_ciclo']} (${increase['usd_vence_con_efecto']:,.0f} que vencen) y desde "
                    f"{increase['fin_ciclo'] + 1} el efecto se pierde (su uplift vuelve al de sin subida)")
            else:
                doc(f"[{STEP_3_2_LABEL}] VALORACIÓN {name}: subida de {increase['tamano']:+.1f}% desde {increase['inicio']}; su "
                    f"ciclo terminó en {increase['fin_ciclo']}: solo se quita de la ventana, el futuro ya no la lleva")
    else:
        doc(f"[{STEP_3_2_LABEL}] VALORACIÓN: no increase whose cycle touches the window: the uplift is the window's as it is")

    # the steps at the end of the history, not confirmed yet
    pending = price["pending"]
    if len(pending):
        shown_pending = pending.copy()
        shown_pending["escalon_medio"] = (100 * shown_pending["escalon_medio"]).round(1)
        shown_pending["usd_renovado"] = shown_pending["usd_renovado"].round(0)
        doc(f"[{STEP_3_2_LABEL}] steps at the end of the history NOT confirmed yet (fewer than "
            f"{configuration.price_persistence_months} months above {configuration.price_step_threshold:.0%}; escalon_medio in %):")
        configuration.show_table(shown_pending.drop(columns=["price_group"]))
        for _, step in shown_pending.head(configuration.price_increases_shown).iterrows():
            doc(f"[{STEP_3_2_LABEL}] VALORACIÓN {step['price_group']}: {step['meses_por_encima']} meses por encima del umbral "
                f"desde {step['inicio']} ({step['escalon_medio']:+.1f}%), hacen falta {configuration.price_persistence_months}: "
                f"si es una subida, el forecast NO la lleva en {current_month}..{step['inicio'] + (cycle - 1)} (lo que le "
                f"queda de ciclo) y sus meses están en la ventana sin deflactar (suben algo el uplift de su celda)")
    show_series_with_increases(fine_table, renewers, price["calendar"], configuration)
    doc(f"[{STEP_3_2_LABEL}] ══════════ FIN DE SUBIDAS DE PRECIO Y VENTANA DEL UPLIFT ══════════")


def show_series_with_increases(fine_table: pd.DataFrame, renewers: pd.DataFrame, calendar: pd.DataFrame,
                               configuration: Config) -> None:
    """The step of an increase, seen in the series that carry the most money: the price_series_shown forecast series
    with the most renewed USD among the price groups with an increase detected, month by month over their last
    price_series_months_shown closed months.

    INPUT:   the fine table, the renewer rows (with _numerador / _denominador) and the calendar of increases detected.
    OUTPUT:  nothing returned: one table per series on screen, with
               precio_vence      USD due / units due: the price the licence was bought at (it rises one cycle after)
               precio_renovado   renewed USD / renewed units: the price paid when renewing (it rises at the increase)
               uplift            renewed USD / base of the renewers: inside the cycle it jumps by about the step
               subida            ▲ +x % in the month an increase of its group starts; · in the other months of its cycle
    """
    doc = configuration.logger.doc
    period = configuration.period_col
    cycle = configuration.price_cycle_months
    if calendar is None or not len(calendar):
        doc(f"[{STEP_3_2_LABEL}] no increase detected: no series to show its step")
        return
    if SERIES_ID_COLUMN not in fine_table.columns or SERIES_ID_COLUMN not in renewers.columns:
        doc(f"[{STEP_3_2_LABEL}] the fine table has no {SERIES_ID_COLUMN}: no series to show the step of its increase")
        return
    group_dims = price_group_columns(fine_table, configuration)
    # the closed rows of the groups with an increase, with only the columns the tables below read. Until 26-oct-2026
    # this took a copy of every closed row with every column, twice (assign, then the filter): with the extract of
    # 1.1M rows that copy did not fit in memory (ArrowMemoryError in the real run)
    measures = [configuration.pipeline_units_col, configuration.pipeline_usd_col,
                configuration.renewed_units_col, configuration.renewed_usd_col]
    is_closed = (fine_table[CALENDAR_ROLE_COLUMN] != ROLE_PROJECTION).to_numpy()
    group_of_closed = price_group_of(fine_table.loc[is_closed, group_dims], group_dims)
    increased_groups = set(calendar["price_group"])
    in_increased_group = group_of_closed.isin(increased_groups)
    closed = fine_table.loc[group_of_closed.index[in_increased_group.to_numpy()], [SERIES_ID_COLUMN, period, *measures]]
    closed = closed.assign(_grupo=group_of_closed[in_increased_group].to_numpy())
    money = (closed.groupby([SERIES_ID_COLUMN, "_grupo"])[configuration.renewed_usd_col].sum()
             .sort_values(ascending=False))
    largest = money.head(configuration.price_series_shown)
    doc(f"[{STEP_3_2_LABEL}] the {len(largest)} forecast series with the most renewed USD in a price group with an increase, "
        f"month by month (the last {configuration.price_series_months_shown} closed months): precio_vence rises one cycle "
        f"after the increase (the licences are bought at the new price), precio_renovado and the uplift at the increase:")
    renewer_months = renewers.groupby([SERIES_ID_COLUMN, period])[["_numerador", "_denominador"]].sum()
    for (series_id, group), renewed_usd in largest.items():
        rows = closed[closed[SERIES_ID_COLUMN] == series_id]
        monthly = rows.groupby(period)[[configuration.pipeline_units_col, configuration.pipeline_usd_col,
                                        configuration.renewed_units_col, configuration.renewed_usd_col]].sum()
        monthly = monthly.tail(configuration.price_series_months_shown)
        table = pd.DataFrame(index=monthly.index)
        table["vencen_unid"] = monthly[configuration.pipeline_units_col].round(0)
        table["precio_vence"] = (monthly[configuration.pipeline_usd_col]
                                 / monthly[configuration.pipeline_units_col].where(monthly[configuration.pipeline_units_col] > 0)).round(2)
        table["precio_renovado"] = (monthly[configuration.renewed_usd_col]
                                    / monthly[configuration.renewed_units_col].where(monthly[configuration.renewed_units_col] > 0)).round(2)
        if series_id in renewer_months.index.get_level_values(0):
            ratio = renewer_months.loc[series_id]
            table["uplift"] = (ratio["_numerador"] / ratio["_denominador"]).reindex(table.index).round(3)
        states = pd.Series("", index=table.index)
        for _, increase in calendar[calendar["price_group"] == group].iterrows():
            start = increase["inicio"]
            in_cycle = (table.index >= start) & (table.index <= start + (cycle - 1))
            states[in_cycle & (states == "")] = "·"
            if start in states.index:
                states[start] = f"▲ {100 * increase['escalon']:+.1f}%"
        table["subida"] = states
        starts = ", ".join(f"{increase['inicio']} ({100 * increase['escalon']:+.1f}%)"
                           for _, increase in calendar[calendar["price_group"] == group].iterrows())
        doc(f"[{STEP_3_2_LABEL}] · {series_id} · grupo de precio {group} · ${renewed_usd:,.0f} renovados en total · subidas: {starts}")
        configuration.show_table(table.reset_index())


def review_the_contract_rule(check: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config) -> None:
    """The reading of the contract check and the test of the reference price; on screen only.

    INPUT:   the contract check (contract_check), the fine table.
    OUTPUT:  nothing returned: the VALORACIÓN of who meets the rule (weighted by the renewed USD), and B · the list
             price implied by every row, due USD / units / (1 − d), against the one of the lowest bucket of its same
             sku, cell and year: the MSRP is a list price, so ~1 in every bucket; growing with the bucket means the
             discount is measured over another reference. Whether the increases are counted twice is measured in the
             exam (step 3.3, rows with and without an increase in its cycle)."""
    discount = configuration.discount_value_column
    doc = configuration.logger.doc
    if not discount or check.empty:
        return
    period = configuration.period_col
    bucket = configuration.discount_bucket_column
    weights = check["usd_renovado"] / check["usd_renovado"].sum()
    def weighted(column: str) -> float:
        return float((check[column].fillna(0) * weights).sum()) if check[column].notna().any() else float("nan")
    doc(f"[{STEP_3_2_LABEL}] VALORACIÓN de la regla del contrato: weighted by the renewed USD, the rule is realized "
        f"{weighted('realizacion_neutras'):.0%} by the neutral renewals, {weighted('realizacion_con_estado'):.0%} by the "
        f"ones with a state, {weighted('realizacion_isolated'):.0%} by the isolated and {weighted('ratio_realizacion'):.0%} by "
        f"all. The exam of step 3.3 decides which path the forecast uses (contrato_neutras = the rule on the neutral rows)")
    # [B] the list price implied by every row, against the lowest bucket of its sku, cell and year
    sku = configuration.sku_column
    rows = fine_table[fine_table[discount].notna() & (fine_table[configuration.pipeline_units_col] > 0)].copy()
    keys = [column for column in [sku] + configuration.extract_mandatory_dims if column and column in rows.columns]
    if not keys:
        doc(f"[{STEP_3_2_LABEL}] B · no sku nor mandatory dims: the implied list price cannot be compared")
    else:
        rows["_ano"] = rows[period].map(lambda month: month.year)
        rows["_lista"] = rows[configuration.pipeline_usd_col] * contract_rule(rows[discount], configuration)
        by_cell = rows.groupby(keys + ["_ano", bucket])[["_lista", configuration.pipeline_units_col, configuration.pipeline_usd_col]].sum()
        by_cell["precio_lista"] = by_cell["_lista"] / by_cell[configuration.pipeline_units_col]
        lowest = by_cell["precio_lista"].groupby(level=list(range(len(keys) + 1))).transform("first")
        by_cell["ratio_al_tramo_bajo"] = by_cell["precio_lista"] / lowest
        weights = by_cell[configuration.pipeline_usd_col]
        implied = (by_cell.assign(_w=weights, _wr=weights * by_cell["ratio_al_tramo_bajo"])
                   .groupby(level=bucket)[["_w", "_wr"]].sum())
        implied["ratio_lista_implicita"] = implied["_wr"] / implied["_w"]
        implied["usd_vence"] = implied["_w"]
        doc(f"[{STEP_3_2_LABEL}] B · the list price implied by the rows, due USD / units / (1 − d), over the one of the lowest "
            f"bucket of the same cell ({len(keys)} columns: sku and the mandatory dims) and year (1 = the same MSRP in "
            f"every bucket, as it should; growing with the bucket = the discount is measured over another reference):")
        configuration.show_table(implied[["usd_vence", "ratio_lista_implicita"]].round(3).reset_index())
    doc(f"[{STEP_3_2_LABEL}] ══════════ FIN DE LA REGLA DEL CONTRATO ══════════")


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 3.3 · BACKTEST OF THE UPLIFT  (until 25-oct-2026: step_16_uplift_backtest.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# Which uplift predicts the price of the renewals better:
# the statistical one of the cell or the contract rule 1 / (1 − discount), on every row with a known discount
# or only on the neutral ones?
#
# HOW THE TEST WORKS:
#   · the statistical uplift of every cell is estimated ONLY with the renewals of the months
#     BEFORE the exam (no peeking), the way step 3.1 estimates it for the forecast: over the window
#     (uplift_window_months before the first exam month) and without the price increases detected
#     with those months (uplift_without_increases); in the exam, a row still inside the cycle of an
#     increase takes it again (× its factor), as the forecast does;
#   · in the exam months, for every row that renewed, the renewed money is predicted GIVEN
#     its real renewed units (so only the price is judged, not the rate), the SAME way the
#     forecast computes it (USD due × rate × uplift = renewed units × pipeline AUV × uplift once
#     the real rate is given):
#         statistical   renewed units × pipeline AUV × uplift of its cell
#         contract      renewed units × pipeline AUV × 1 / (1 − discount)   (known discount only)
#         contract on the neutral rows   the contract on the rows without a negative state (the typical
#                       renewal the rule speaks of), the statistical one on the rows with a state
#     The discount is measured against the MSRP of the contract at expiry: the rule already carries the
#     increases between the purchase and the expiry, so the contract paths take no increase factor.
#     The base is the row's average price, NOT the exact value of the renewers that step 3.1 uses
#     to ESTIMATE the uplift: the forecast cannot know who will renew, so it values them at the
#     row's average, and the test must judge that. Where renewers were worth more than the
#     average (sff_uplift_homogeneidad), the error shows up here, as it will in the forecast.
#   · both are compared on the SAME rows (the ones with a known discount): WAPE = Σ|predicted −
#     real| / Σ real, and the bias of the total = (Σ predicted − Σ real) / Σ real;
#   · VERDICT: the path with the lowest WAPE on the rows with a known discount is the one the forecast
#     uses there (the statistical one on a tie). The verdict governs. The same comparison is shown for the
#     neutral and the rows with a state, and for the rows with and without an increase in its cycle (an increase
#     counted twice shows up as a statistical path that is high only with an increase).
# The rows with an unknown discount are measured too (statistical only): they always take it.
#
# THE PRICE BY STATES: the uplift cell has no states, and its uplift comes mostly from the neutral renewers. A softcancel
# that renews does it with a retention offer, below its cell. The factor of every combination of negative states
# (step 3.1: what its renewers paid over their cell) is estimated with the SAME months before the exam, and the
# statistical path × the factor of the row's states (estadistica_estado) is judged on every renewer row of the exam
# (known discount or not), in total and per combination. SECOND VERDICT: the factor is used by the forecast when its
# WAPE on every renewer row is lower than the statistical path's.
#
# THE WAYS OF ESTIMATING THE STATISTICAL UPLIFT, side by side in the same exam (on screen): all the
# history (no window, no increases removed), the window as it is, and the window without increases;
# the one the Config says is the one judged above. Their bias in total and month by month says what
# the window and the removal of the increases change.
#
# Actions (logged as they are done):
#   1. the price increases known before the exam; the statistical uplift estimated with the months
#      before the exam, every way (all the history, the window, the window without increases)
#   2. the renewals of the exam months, predicted by both paths (and the statistical one every way)
#   3. the error of each path, the verdict; the price by states and its verdict
#   4. check the test                                                  checks 1-2
#   5. write the result                                                check 3
#   6. count the checks; stop if any failed
#   7. show the comparison, as a table; the ways of the statistical uplift, in total and month by month
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. the statistical uplift was estimated without the exam months
#    2. both paths were compared on the same rows
#    3. table sff_backtest_uplift written and read back
#
# Output: the comparison (per path and per discount bucket) and the verdicts (a dict: the path where the discount is
# known, and whether the price factor of the states is used) ·
# table sff_backtest_uplift.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_3_3_LABEL = "3.3"
STEP_3_3_NAME = "BACKTEST OF THE UPLIFT"
STEP_3_3_PURPOSE = ("decide, in the exam months, which uplift predicts the price of the renewals better where the discount "
                "is known: the statistical one of the cell (estimated before the exam), the contract rule 1 / (1 − d), or "
                "the contract on the neutral rows and the statistical one on the rows with a state; "
                "the verdict governs the forecast")
STEP_3_3_ACTIONS = ["the price increases known before the exam; the statistical uplift estimated with the months before "
                "the exam, every way (all the history, the window, the window without increases)",
                "the renewals of the exam months, predicted by both paths (and the statistical one every way)",
                "the error of each path, the verdict; the price by states and its verdict",
                "check the test (checks 1-2)",
                "write the result (check 3)",
                "count the checks; stop if any failed",
                "show the comparison, as a table; the ways of the statistical uplift, in total and month by month"]
STEP_3_3_OUTPUT = "the error of each path (total and per discount bucket) and the verdict · table sff_backtest_uplift"


def backtest_uplift(fine_table: pd.DataFrame, configuration: Config) -> tuple:
    """Statistical vs contract uplift in the exam months; the verdict."""
    configuration.log_step_start(STEP_3_3_LABEL, STEP_3_3_NAME, STEP_3_3_PURPOSE, STEP_3_3_ACTIONS, STEP_3_3_OUTPUT)
    check_log = []
    exam_start = configuration.calendar_boundaries()["test_start"]

    # [1] the price increases known before the exam, and the statistical uplift estimated with the months before
    #     it, every way; the one of the Config is the one judged
    before_exam = renewer_rows(fine_table, configuration, before_month=exam_start)
    price = price_increases_as_of(fine_table, exam_start, configuration)
    configured = (configuration.uplift_window_months, configuration.uplift_without_increases)
    ways = statistical_ways(configuration)
    estimates = {}
    for label, (window_months, without_increases) in ways.items():
        increases = active_increases(price["calendar"], exam_start, window_months, configuration)
        history = uplift_history(before_exam, exam_start, increases, window_months, without_increases, configuration)
        cells = estimate_cell_uplifts(fine_table, history, configuration, numerator_column="_numerador_uplift", with_band=False)
        estimates[label] = dict(history=history, cells=cells, increases=increases, without_increases=without_increases)
    configured_label = next(label for label, way in ways.items() if way == configured)
    history = estimates[configured_label]["history"]
    # the price by states, with the same months before the exam and the cells of the configured way
    state_factors = state_uplift_factors(history, estimates[configured_label]["cells"], configuration)
    configuration.log_action(STEP_3_3_LABEL, 1, f"statistical uplift estimated with the renewals before {exam_start}, "
                                            f"{len(ways)} ways ({' · '.join(ways)}); judged: {configured_label} "
                                            f"({len(history):,} renewer rows, "
                                            f"{len(estimates[configured_label]['increases'])} price increases in play)")

    # [2] the renewals of the exam, predicted by both paths
    exam_rows = renewer_rows(fine_table, configuration)
    exam_rows = exam_rows[exam_rows[CALENDAR_ROLE_COLUMN] == ROLE_TEST].copy()
    # the base the forecast will use: the renewed units at the row's average due price (not the exact base)
    forecast_base = exam_rows["_denominador_aproximado"]
    exam_groups = price_group_of(exam_rows, price_group_columns(exam_rows, configuration))
    for label, estimate in estimates.items():
        cell_uplift = exam_rows[UPLIFT_CELL_ID_COLUMN].map(estimate["cells"].set_index(UPLIFT_CELL_ID_COLUMN)["uplift"])
        prediction = forecast_base * cell_uplift
        if estimate["without_increases"] and len(estimate["increases"]):
            prediction = prediction * increase_factor(exam_groups, exam_rows[configuration.period_col], estimate["increases"])
        exam_rows["pred_forma_" + label] = prediction
    exam_rows["pred_" + PATH_STATISTICAL] = exam_rows["pred_forma_" + configured_label]
    exam_rows["pred_" + PATH_STATISTICAL_STATES] = (exam_rows["pred_" + PATH_STATISTICAL]
                                                   * state_factor_of_rows(exam_rows, state_factors, configuration))
    exam_rows["_combinacion"] = state_combination(exam_rows, negative_states(configuration))
    # which exam rows renew inside the cycle of an increase (to see whether the increase is counted twice: the
    # discount of the extract is measured against the MSRP at expiry, so it already carries the increase)
    configured_increases = estimates[configured_label]["increases"]
    exam_rows["_con_subida"] = (np.asarray(increase_factor(exam_groups, exam_rows[configuration.period_col], configured_increases)) > 1.0
                                if len(configured_increases) else False)
    discount = configuration.discount_value_column
    known = exam_rows[discount].notna() if discount else pd.Series(False, index=exam_rows.index)
    exam_rows["pred_" + PATH_CONTRACT] = (np.where(known, forecast_base * contract_rule(exam_rows[discount], configuration), np.nan)
                                          if discount else np.nan)
    # the contract on the neutral renewals (no negative state: the typical renewal the rule speaks of), the statistical
    # uplift on the ones with a state (softcancel, dormant…: offers and changes the rule does not see)
    exam_rows["_neutra"] = has_no_negative_state(exam_rows, configuration)
    exam_rows["pred_" + PATH_CONTRACT_NEUTRAL] = np.where(known & exam_rows["_neutra"], exam_rows["pred_" + PATH_CONTRACT],
                                                          np.where(known, exam_rows["pred_" + PATH_STATISTICAL], np.nan))
    configuration.log_action(STEP_3_3_LABEL, 2, f"{len(exam_rows):,} renewer rows in the exam; {int(known.sum()):,} with a known discount, "
                                            f"{int((known & exam_rows['_neutra']).sum()):,} of them neutral (no negative state)")

    # [3] the error of each path, and the verdict
    comparison = compare_paths(exam_rows, known, configuration)
    # the path with the lowest WAPE on the rows with a known discount; the statistical one on a tie
    known_rows = comparison[comparison["filas"] == "descuento conocido"].set_index("via")["wape"]
    chosen = min((path for path in (PATH_STATISTICAL, PATH_CONTRACT, PATH_CONTRACT_NEUTRAL) if path in known_rows.index),
                 key=lambda path: (known_rows[path], path != PATH_STATISTICAL), default=PATH_STATISTICAL)
    # the price by states: on every renewer row, the statistical path with and without the factor of its states
    all_rows = comparison[comparison["filas"] == "todas"].set_index("via")["wape"]
    state_factor_wins = bool(PATH_STATISTICAL_STATES in all_rows.index and PATH_STATISTICAL in all_rows.index
                            and all_rows[PATH_STATISTICAL_STATES] < all_rows[PATH_STATISTICAL])
    verdict = {"via_contrato_gana": chosen != PATH_STATISTICAL, "via_usada_con_descuento": chosen,
               "factor_estado": state_factor_wins}
    # the error of the price of the TOTAL of a month, with the paths the forecast will use: the band of the price
    verdict["error_precio_por_mes"] = price_error_of_the_monthly_total(exam_rows, known, verdict, configuration)
    monthly_errors = verdict["error_precio_por_mes"]
    verdict["error_precio_total"] = float(monthly_errors["error_precio"].abs().mean()) if len(monthly_errors) else 0.0
    configuration.log_action(STEP_3_3_LABEL, 3, f"verdict: where the discount is known the forecast uses the {chosen.upper()} path "
                                            f"(WAPE " + " · ".join(f"{path} {known_rows[path]:.1%}" for path in known_rows.index) + ")"
                                            + (f" · price by states {'USED' if state_factor_wins else 'NOT used'} (WAPE of every "
                                               f"renewer row: {PATH_STATISTICAL_STATES} {all_rows[PATH_STATISTICAL_STATES]:.1%} · "
                                               f"{PATH_STATISTICAL} {all_rows[PATH_STATISTICAL]:.1%})"
                                               if PATH_STATISTICAL_STATES in all_rows.index else "")
                                            + f" · error of the price of the monthly total with these paths: "
                                              f"{verdict['error_precio_total']:.1%} (mean over "
                                              f"{len(verdict['error_precio_por_mes'])} exam months; the band of the price)")

    # [4] the checks
    configuration.log_action(STEP_3_3_LABEL, 4, "checking the test")
    last_month_used = max((estimate["history"][configuration.period_col].max() for estimate in estimates.values()
                           if len(estimate["history"])), default=None)
    configuration.log_check(STEP_3_3_LABEL, check_log, "the statistical uplift was estimated without the exam months (every way)",
                            last_month_used < exam_start if last_month_used is not None else True,
                            failure_detail="the estimate saw the exam", context=f"last month used {last_month_used}")
    same_rows = comparison[comparison["filas"] == "descuento conocido"]["renovadas_filas"].nunique() <= 1
    configuration.log_check(STEP_3_3_LABEL, check_log, "both paths were compared on the same rows", bool(same_rows),
                            failure_detail="the two paths were measured on different rows")

    # [5] the result
    configuration.log_action(STEP_3_3_LABEL, 5, "writing the result")
    configuration.write_table(STEP_3_3_LABEL, check_log, comparison.assign(via_usada_con_descuento=verdict["via_usada_con_descuento"],
                                                                       factor_estado=int(verdict["factor_estado"])),
                              TABLE_UPLIFT_BACKTEST)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_3_3_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_3_3_LABEL, STEP_3_3_NAME, check_log)

    # [7] the comparison
    configuration.log_action(STEP_3_3_LABEL, 7, "the price of the exam renewals predicted by each path (wape: Σ|pred − real| / "
                                            "Σ real; sesgo: (Σ pred − Σ real) / Σ real):")
    configuration.show_table(comparison)
    show_the_double_count(comparison, configuration)
    show_the_price_by_states_in_the_exam(comparison, state_factors, verdict, configuration)
    # the statistical uplift every way, in the same exam: the verdict of block 3 says whether the way of the Config is the best
    verdict["formas_uplift"] = show_statistical_ways(exam_rows, known, list(estimates), configured_label, configuration)
    return comparison, verdict


def show_the_price_by_states_in_the_exam(comparison: pd.DataFrame, state_factors: pd.DataFrame, verdict: dict,
                            configuration: Config) -> None:
    """EL PRECIO POR ESTADO DEL CLIENTE EN EL EXAMEN on screen: the factors estimated before the exam, and per combination of states
    the bias and WAPE of the statistical path with and without them (and of the contract where the discount is known)."""
    by_state = comparison[comparison["filas"].str.startswith("estado · ") | (comparison["filas"] == "todas")]
    if by_state.empty:
        return
    doc = configuration.logger.doc
    doc(f"[{STEP_3_3_LABEL}] ══════════ EL PRECIO POR ESTADO DEL CLIENTE EN EL EXAMEN ══════════")
    doc(f"[{STEP_3_3_LABEL}] the factor of every combination of negative states, estimated before the exam (what its renewers "
        f"paid over their cell):")
    configuration.show_table(state_factors.round(3))
    doc(f"[{STEP_3_3_LABEL}] the exam per combination ({PATH_STATISTICAL_STATES} = {PATH_STATISTICAL} × the factor; "
        f"{PATH_CONTRACT}: only the rows with a known discount):")
    configuration.show_table(by_state.round(4))
    doc(f"[{STEP_3_3_LABEL}] VALORACIÓN: the price factor of the states is "
        + ("USED by the forecast: it lowers the WAPE of every renewer row" if verdict["factor_estado"]
           else "NOT used: it does not lower the WAPE of every renewer row"))
    doc(f"[{STEP_3_3_LABEL}] ══════════ FIN DE EL PRECIO POR ESTADO DEL CLIENTE EN EL EXAMEN ══════════")


def show_the_double_count(comparison: pd.DataFrame, configuration: Config) -> None:
    """VALORACIÓN of the split by increase: the bias of each path with an increase in its cycle minus the one without.
    The statistical path counts the increase twice when its gap is clearly positive and the contract one's is not."""
    by_label = comparison.set_index(["filas", "via"])["sesgo"]
    try:
        gaps = {path: float(by_label[("conocido · con subida en ciclo", path)] - by_label[("conocido · sin subida", path)])
                for path in (PATH_STATISTICAL, PATH_CONTRACT)}
    except KeyError:
        return
    doc = configuration.logger.doc
    doc(f"[{STEP_3_3_LABEL}] VALORACIÓN de las subidas: bias with an increase minus without — statistical "
        f"{gaps[PATH_STATISTICAL]:+.1%}, contract {gaps[PATH_CONTRACT]:+.1%}: "
        + ("the statistical path counts the increase twice on the rows with a known discount (the discount already "
           "carries it)" if gaps[PATH_STATISTICAL] > 0.02 and gaps[PATH_STATISTICAL] - gaps[PATH_CONTRACT] > 0.02
           else "no double count of the increases"))


def statistical_ways(configuration: Config) -> dict:
    """The ways of estimating the statistical uplift compared in the exam, label → (window months, without
    increases): all the history, the window as it is, the window without increases, and the one of the Config
    if it is none of them. The same way is kept once."""
    window_months = configuration.uplift_window_months
    candidates = [("toda_la_historia", (0, False)),
                  (f"ventana_{window_months}", (window_months, False)),
                  (f"ventana_{window_months}_sin_subidas", (window_months, True)),
                  ("configurada", (window_months, configuration.uplift_without_increases))]
    ways = {}
    for label, way in candidates:
        if way not in ways.values():
            ways[label] = way
    return ways


def show_statistical_ways(exam_rows: pd.DataFrame, known: pd.Series, labels: list, configured_label: str,
                          configuration: Config) -> pd.DataFrame:
    """The statistical uplift every way in the same exam, on screen: bias and WAPE by discount known or not and in
    total, and the bias month by month (a bias that shrinks with the window is a drift of the price over the
    years; one that stays is something else: the mix, the base, the borrowed cells)."""
    doc = configuration.logger.doc
    period = configuration.period_col
    rows = []
    for subset_label, subset in (("descuento conocido", exam_rows[known]), ("descuento desconocido", exam_rows[~known]),
                                 ("todas", exam_rows)):
        real = subset["_numerador"]
        for label in labels:
            predicted = subset["pred_forma_" + label]
            valid = predicted.notna()
            if not valid.any():
                continue
            rows.append({"filas": subset_label, "forma": label, "usada": "←" if label == configured_label else "",
                         "usd_real": round(float(real[valid].sum())), "usd_pred": round(float(predicted[valid].sum())),
                         "wape": round(float((predicted[valid] - real[valid]).abs().sum() / real[valid].sum()), 4),
                         "sesgo": round(float((predicted[valid].sum() - real[valid].sum()) / real[valid].sum()), 4)})
    doc(f"[{STEP_3_3_LABEL}] ══════════ EL UPLIFT ESTADÍSTICO DE CADA FORMA, EN EL MISMO EXAMEN ══════════")
    doc(f"[{STEP_3_3_LABEL}] estimated before the exam; ← = the way of the Config (the one the forecast uses); "
        f"sesgo = (Σ pred − Σ real) / Σ real:")
    configuration.show_table(pd.DataFrame(rows))
    by_month = exam_rows.groupby(period)
    monthly = pd.DataFrame({"usd_real": by_month["_numerador"].sum().round(0)})
    for label in labels:
        monthly["sesgo_" + label] = (by_month["pred_forma_" + label].sum() / by_month["_numerador"].sum() - 1).round(4)
    doc(f"[{STEP_3_3_LABEL}] the bias month by month (every renewer row of the exam):")
    configuration.show_table(monthly.reset_index())
    doc(f"[{STEP_3_3_LABEL}] ══════════ FIN DE EL UPLIFT ESTADÍSTICO DE CADA FORMA ══════════")
    return pd.DataFrame(rows)


def price_error_of_the_monthly_total(exam_rows: pd.DataFrame, known: pd.Series, verdict: dict,
                                     configuration: Config) -> pd.DataFrame:
    """THE BAND OF THE PRICE: per exam month, the error of the renewed USD of the TOTAL predicted the way the forecast
    will predict it (given the real renewed units, so only the price is judged): the path of the verdict where the
    discount is known, the statistical uplift elsewhere (× the factor of the states when the verdict uses it).
    error_precio = Σ predicted / Σ renewed − 1. Its mean |error| over the exam months is the half band of the price that
    step 7.2 adds to the band of the rate (step 7.1), in quadrature: the rate and the price are different errors."""
    statistical = exam_rows["pred_" + (PATH_STATISTICAL_STATES if verdict["factor_estado"] else PATH_STATISTICAL)]
    chosen = verdict["via_usada_con_descuento"]
    used = statistical
    if chosen != PATH_STATISTICAL and "pred_" + chosen in exam_rows.columns:
        used = pd.Series(np.where(known & exam_rows["pred_" + chosen].notna(), exam_rows["pred_" + chosen], statistical),
                         index=exam_rows.index)
    valid = used.notna() & exam_rows["_numerador"].notna()
    by_month = (exam_rows[valid].assign(_usado=used[valid])
                .groupby(configuration.period_col)[["_usado", "_numerador"]].sum())
    by_month = by_month[by_month["_numerador"] > 0]
    return pd.DataFrame({configuration.period_col: by_month.index, "usd_predicho": by_month["_usado"].to_numpy(),
                         "usd_real": by_month["_numerador"].to_numpy(),
                         "error_precio": (by_month["_usado"] / by_month["_numerador"] - 1).to_numpy()})


def compare_paths(exam_rows: pd.DataFrame, known: pd.Series, configuration: Config) -> pd.DataFrame:
    """WAPE and bias of each path: on the rows with a known discount (both paths) and on the
    rows with an unknown one (statistical only); and per discount bucket."""
    rows = []
    def measure(subset, path, label):
        predicted, real = subset["pred_" + path], subset["_numerador"]
        valid = predicted.notna()
        if not valid.any():
            return
        rows.append({"filas": label, "via": path, "renovadas_filas": int(valid.sum()), "usd_real": real[valid].sum(),
                     "usd_pred": predicted[valid].sum(),
                     "wape": float((predicted[valid] - real[valid]).abs().sum() / real[valid].sum()),
                     "sesgo": float((predicted[valid].sum() - real[valid].sum()) / real[valid].sum())})
    for path in (PATH_STATISTICAL, PATH_CONTRACT, PATH_CONTRACT_NEUTRAL):
        measure(exam_rows[known], path, "descuento conocido")
    measure(exam_rows[~known], PATH_STATISTICAL, "descuento desconocido")
    # the price by states: every renewer row, and every combination of negative states, with and without the factor
    if "pred_" + PATH_STATISTICAL_STATES in exam_rows.columns:
        for path in (PATH_STATISTICAL, PATH_STATISTICAL_STATES):
            measure(exam_rows, path, "todas")
        for combination in exam_rows["_combinacion"].value_counts().index:
            subset = exam_rows[exam_rows["_combinacion"] == combination]
            for path in (PATH_STATISTICAL, PATH_STATISTICAL_STATES):
                measure(subset, path, f"estado · {combination}")
            if known[subset.index].any():
                measure(subset[known[subset.index]], PATH_CONTRACT, f"estado · {combination}")
    # the known discount split by the states: does the rule hold on the neutral renewals and fail on the ones with a state?
    if "_neutra" in exam_rows.columns:
        for label, neutral in (("conocido · neutras", True), ("conocido · con estado", False)):
            subset = exam_rows[known & (exam_rows["_neutra"] == neutral)]
            for path in (PATH_STATISTICAL, PATH_CONTRACT):
                measure(subset, path, label)
    # the known discount split by an increase in its cycle: the contract path takes the increase from the discount,
    # the statistical one multiplies its cell by the factor; if the statistical path is high only with an increase,
    # the increase is counted twice (once in the bucket of the discount, once in the factor)
    if "_con_subida" in exam_rows.columns:
        for label, with_increase in (("conocido · con subida en ciclo", True), ("conocido · sin subida", False)):
            subset = exam_rows[known & (exam_rows["_con_subida"] == with_increase)]
            for path in (PATH_STATISTICAL, PATH_CONTRACT):
                measure(subset, path, label)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 3 · THE BLOCK: run the steps in order, keep the results
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B3"
BLOCK_NAME = "PRICE"
BLOCK_MISSION = "at what price the renewers renew (the uplift of every row), the path that predicts it best, and its band"
BLOCK_READS = ("fine_table",)
# uplift_basis (the renewer rows, the history of the uplift, the price monitor) is read by step 3.2 only: it stays
# inside the block and is released after 3.2, not given to the pipeline (until 26-oct-2026 it was given, and the
# renewer rows stayed in memory the whole run)
BLOCK_GIVES = ("uplift_cells", "uplift_state_factors", "contract_check", "price_increases",
               "uplift_homogeneity", "uplift_backtest", "uplift_verdict")


class PriceBlock:
    """BLOCK 3 · PRICE (see the header of this module): its results are attributes, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.uplift_cells = None
        self.uplift_state_factors = None
        self.contract_check = None
        self.price_increases = None
        self.uplift_basis = None
        self.uplift_homogeneity = None
        self.uplift_backtest = None
        self.uplift_verdict = None
        self.seconds = {}

    def run(self, fine_table: pd.DataFrame) -> "PriceBlock":
        """Every step of the block, in order."""
        configuration = self.configuration
        configuration.log_block_start(BLOCK_LABEL, 3, BLOCK_NAME, BLOCK_MISSION)

        started = time.time()
        (self.uplift_cells, self.uplift_state_factors, self.contract_check, self.price_increases,
         self.uplift_basis) = estimate_uplift(fine_table, configuration)
        self.seconds["3.1 uplift"] = time.time() - started

        started = time.time()
        self.uplift_homogeneity = diagnose_the_price(fine_table, self.uplift_basis, self.contract_check, configuration)
        self.uplift_basis = None          # read by 3.2 only: released before the exam of 3.3 (see BLOCK_GIVES)
        gc.collect()
        self.seconds["3.2 price_diagnostics"] = time.time() - started

        started = time.time()
        self.uplift_backtest, self.uplift_verdict = backtest_uplift(fine_table, configuration)
        self.seconds["3.3 uplift_backtest"] = time.time() - started
        return self

    def results(self) -> dict:
        """The results under the names the pipeline gives them."""
        return {name: getattr(self, name) for name in BLOCK_GIVES}
