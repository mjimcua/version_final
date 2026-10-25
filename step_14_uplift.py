"""
step_14_uplift.py — The revaluation: at what price does a renewer renew?

What falls due has a price (USD due / units due = the pipeline AUV of the row). A renewer
pays a price too (renewed USD / renewed units). The UPLIFT is the second over the first:
1.0 = renews at the same price; 1.43 = pays 43 % more. The forecast of a row is
USD due × renewal rate × uplift.

Two ways to know the uplift of a future row:
  · STATISTICAL (the default, and the only one where the discount is unknown): the uplift
    observed in the past renewals of its UPLIFT CELL (uplift mandatory dims, revaluation
    extras, discount bucket), as a ratio of sums over the renewers (a contract that did not
    renew has no price), weighted by money:
        Σ renewed USD / Σ (the USD that was falling due of the contracts that RENEWED)
    when the extract carries that base (renewed_pipeline_usd_col): the revaluation pure,
    conditional on renewing. Over the RECENT window only (uplift_window_months, 12 = one price
    cycle; 0 = all the history), and WITHOUT the price increases (uplift_without_increases): a
    month of the window inside the cycle of an increase detected in its price group
    (price_increases.py) has its renewed USD divided by (1 + the size of the increase). The uplift
    of the cell is then the one without increases; step 20 applies the increase again to the future
    months still inside its cycle, and after the cycle its effect is lost. Without it, the base is approximated as renewed units × the
    row's pipeline AUV, which is exact only if renewers were worth the row's average: the
    difference between the two bases is the HOMOGENEITY of the cell, measured below.
    Its n is the number of renewers. A cell with fewer than uplift_floor
    renewers takes its PARENT (same mandatory dims and discount bucket, the other extras
    set to '*'), then its mandatory cell, then the whole portfolio. The band of the
    uplift: bootstrap of the renewer rows (p5-p95).
  · CONTRACT (where the exact discount is known): a customer who paid list × (1 − d)
    renews at list, so the uplift is 1 / (1 − d). This step checks the rule against the past
    renewals with a known discount (how much of the renewed money it explains within ±2 %);
    step 16 decides, in the exam months, which path predicts better.

Actions (logged as they are done):
  1. the renewer rows of the closed months, with the base of their uplift
  2. the price increases of every price group (closed months), the window of the uplift and its
     months deflated; the statistical uplift of every uplift cell: own, parent, cell, global; the band;
     the price factor of every combination of negative customer states (what its renewers paid over their cell)
  3. the contract rule against the past renewals with a known discount
  4. check the uplifts                                               checks 1-4
  5. write the tables                                                checks 5-6
  6. count the checks; stop if any failed
  7. show the uplift by origin and the price by customer states
The diagnostics of the price for business (homogeneity, the contract rule read, the revaluation of the
renewers, the price increases on screen) are the next stage (step_15_price_diagnostics.py): no prediction
reads them.

Checks (logged as they are made, numbered, at the level of their status):
   1. every uplift cell of the fine table has an uplift (future cells included)
   2. every uplift is inside [0.01, uplift_cap]
   3. the band contains the uplift of every cell with its own or its parent's
   4. no price group ends its history with a step not confirmed yet (WARN: if it is an increase, the
      forecast does not carry it, and its months are in the window without being deflated)
   5-6. tables sff_uplift_celda and sff_uplift_contrato_check written and read back

Output: the uplift of every cell (with its origin and band), the price factor of every combination of customer
states (the forecast applies it when step 16 says it predicts better), the contract check, the price increases in
play (for the forecast) and the basis the diagnostics read · tables sff_uplift_celda, sff_uplift_contrato_check.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, Config, TABLE_CONTRACT_CHECK, TABLE_UPLIFT_CELLS, TRUTH_ROLES, UPLIFT_CELL,
                    UPLIFT_CELL_ID_COLUMN, UPLIFT_GLOBAL, UPLIFT_OWN, UPLIFT_PARENT, WILDCARD, has_no_negative_state,
                    join_columns, negative_states, state_combination)
from price_increases import (active_increases, increase_calendar, increase_factor, pending_steps, price_group_columns,
                             price_group_of, price_monitor)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "14"
STEP_NAME = "UPLIFT"
STEP_PURPOSE = ("estimate at what price a renewer renews relative to what fell due: the statistical uplift of every "
                "uplift cell from its past renewals (borrowing from its parent when it has few renewers), and the check "
                "of the contract rule 1 / (1 − discount) where the discount is known")
STEP_ACTIONS = ["the renewer rows of the closed months, with the base of their uplift",
                "the price increases, the window of the uplift and its months deflated; the statistical uplift of "
                "every uplift cell: own, parent, cell, global; the band; the price factor of every combination of "
                "customer states",
                "the contract rule against the past renewals with a known discount",
                "check the uplifts (checks 1-4)",
                "write the tables (checks 5-6)",
                "count the checks; stop if any failed",
                "show the uplift by origin and the price by customer states"]
STEP_OUTPUT = ("the uplift of every uplift cell (origin, band) · the price factor of every combination of customer states · "
               "the contract check · the price increases in play · tables sff_uplift_celda, sff_uplift_contrato_check")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RENEWERS AND THE WINDOW OF THE UPLIFT
# ═══════════════════════════════════════════════════════════════════════════════════

def renewer_rows(fine_table: pd.DataFrame, configuration: Config, before_month=None) -> pd.DataFrame:
    """The rows of closed truth months that renewed (optionally only before a month), with
    their pipeline AUV, the two sides of the ratio and their parent and cell ids."""
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                        & (fine_table[configuration.renewed_units_col].fillna(0) > 0)
                        & (fine_table[configuration.pipeline_units_col] > 0)].copy()
    if before_month is not None:
        closed = closed[closed[configuration.period_col] < before_month]
    closed["auv_pipeline"] = closed[configuration.pipeline_usd_col] / closed[configuration.pipeline_units_col]
    closed["_numerador"] = closed[configuration.renewed_usd_col]
    # the base of the uplift: what the RENEWERS were worth before renewing. Exact when the extract carries it;
    # otherwise the renewed units valued at the row's average due price (the whole row, renewers or not)
    closed["_denominador_aproximado"] = closed[configuration.renewed_units_col] * closed["auv_pipeline"]
    if configuration.renewed_pipeline_usd_col:
        closed["_denominador"] = closed[configuration.renewed_pipeline_usd_col]
        closed = closed[closed["_denominador"] > 0]        # a renewer row without its base cannot opine (warned in step 01)
    else:
        closed["_denominador"] = closed["_denominador_aproximado"]
    return add_parent_ids(closed, configuration)


def closed_view_of_fine_table(fine_table: pd.DataFrame, configuration: Config, before_month=None) -> pd.DataFrame:
    """The closed pipeline months of the fine table (optionally only before a month) with the columns the price
    monitor reads, named as in the core: period, forecast_to_renew_units / _USD, forecast_renewed_units / _USD,
    the closed-month measures by their core names (CORE_*) and the price dims.

    The core carries exactly these values for its closed pipeline rows (step 24 copies the fine table), so the
    monitor read here and the one of step 25 (from the core) are the same; steps 14 and 16 cannot wait for the
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


def add_parent_ids(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The parent of an uplift cell (mandatory dims and discount bucket kept, the other extras
    '*') and its mandatory cell (only the uplift mandatory dims)."""
    uplift_mandatory = configuration.uplift_mandatory_dims or configuration.business_mandatory_dims
    bucket = [configuration.discount_bucket_column] if configuration.discount_value_column else []
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
    with_band=False skips the bootstrap (the exam of step 16 judges the uplift, not its band)."""
    renewed_units = configuration.renewed_units_col
    cells = add_parent_ids(fine_table, configuration)[[UPLIFT_CELL_ID_COLUMN, "uplift_padre_id", "uplift_celda_mandatory_id"]]
    cells = cells.drop_duplicates(UPLIFT_CELL_ID_COLUMN).reset_index(drop=True)

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
    doc(f"[{STEP_LABEL}] ══════════ EL PRECIO POR ESTADO DEL CLIENTE ══════════")
    doc(f"[{STEP_LABEL}] what the renewers of every combination of negative states paid over what the uplift of their cell "
        f"predicts (factor 1 = they renew at the price of their cell; below 1 = cheaper, e.g. with a retention offer; "
        f"fewer than {configuration.uplift_floor:.0f} renewed units: factor 1):")
    configuration.show_table(state_factors.round(3))
    neutral = state_factors[state_factors["combinacion"] == "neutral"]
    in_state = state_factors[(state_factors["combinacion"] != "neutral") & (state_factors["factor"] != 1.0)]
    if len(neutral) and len(in_state):
        lowest = in_state.sort_values("factor").iloc[0]
        doc(f"[{STEP_LABEL}] VALORACIÓN: the neutral renewers pay {neutral['factor'].iloc[0]:.1%} of their cell's price; "
            f"{lowest['combinacion']} the least, {lowest['factor']:.1%}. Step 16 decides whether the factor predicts better.")
    doc(f"[{STEP_LABEL}] ══════════ FIN DE EL PRECIO POR ESTADO DEL CLIENTE ══════════")


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
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    current_month = configuration.calendar_boundaries()["current"]

    # [1] the renewers
    renewers = renewer_rows(fine_table, configuration)
    base_text = (f"exact base: {configuration.renewed_pipeline_usd_col}" if configuration.renewed_pipeline_usd_col
                 else "approximate base: renewed units × the row's pipeline AUV (renewed_pipeline_usd_col not declared)")
    configuration.log_action(STEP_LABEL, 1, f"{len(renewers):,} renewer rows in the closed months "
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
    # same window (step 20 applies it when step 16 finds it predicts better)
    state_factors = state_uplift_factors(history, cells, configuration)
    window_text = (f"window {history[configuration.period_col].min()}..{history[configuration.period_col].max()} "
                   f"({window_months} closed months)" if window_months and len(history) else "all the history")
    deflated = history[history["_factor_subida"] != 1.0]
    increases_text = (f"{len(increases)} price increases in play, removed from {len(deflated):,} renewer rows "
                      f"(${deflated['_numerador'].sum():,.0f} renewed USD deflated to ${deflated['_numerador_uplift'].sum():,.0f})"
                      if configuration.uplift_without_increases else
                      f"{len(increases)} price increases in play, NOT removed (uplift_without_increases off)")
    configuration.log_action(STEP_LABEL, 2, f"{window_text}: {len(history):,} renewer rows · {increases_text} · "
                                            f"{len(cells):,} uplift cells · origins {cells['uplift_origen'].value_counts().to_dict()} · "
                                            f"global uplift {cells.attrs['uplift_global']:.3f} · price factor by states "
                                            + " · ".join(f"{row.combinacion} {row.factor:.3f}" for row in state_factors.itertuples()))

    # [3] the contract rule
    check = contract_check(renewers, configuration)
    known_share = renewers[configuration.discount_value_column].notna().mean() if configuration.discount_value_column else 0.0
    configuration.log_action(STEP_LABEL, 3, f"contract rule checked on the {known_share:.0%} of renewer rows with a known "
                                            f"discount" if configuration.discount_value_column else "no exact discount declared")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the uplifts")
    missing = set(fine_table[UPLIFT_CELL_ID_COLUMN]) - set(cells[UPLIFT_CELL_ID_COLUMN])
    configuration.log_check(STEP_LABEL, check_log, "every uplift cell of the fine table has an uplift (future cells included)",
                            not missing and cells["uplift"].notna().all(),
                            failure_detail=f"{len(missing)} cells without uplift", context=f"{len(cells):,} cells")
    configuration.log_check(STEP_LABEL, check_log, f"every uplift is inside [{configuration.min_uplift}, {configuration.uplift_cap}]",
                            cells["uplift"].between(configuration.min_uplift, configuration.uplift_cap).all(),
                            failure_detail="uplifts outside the range",
                            context=f"from {cells['uplift'].min():.3f} to {cells['uplift'].max():.3f}; {int(cells['recortado'].sum())} clipped")
    with_band = cells[cells["uplift_bajo"].notna() & (cells["recortado"] == 0)]
    outside = with_band[(with_band["uplift"] < with_band["uplift_bajo"] - 1e-9) | (with_band["uplift"] > with_band["uplift_alto"] + 1e-9)]
    configuration.log_check(STEP_LABEL, check_log, "the band contains the uplift of every cell that has one", outside.empty,
                            failure_detail=f"{len(outside)} cells whose uplift is outside its band",
                            context=f"{len(with_band):,} cells with a band", examples=outside)
    pending = price["pending"]
    configuration.log_check(STEP_LABEL, check_log,
                            f"no price group ends its history with a step not confirmed yet (above "
                            f"{configuration.price_step_threshold:.0%} for fewer than {configuration.price_persistence_months} months)",
                            pending.empty,
                            failure_detail=f"{len(pending)} groups with a step still unconfirmed "
                                           f"(${pending['usd_renovado'].sum() if len(pending) else 0:,.0f} renewed in its months): "
                                           f"if it is an increase, the forecast does not carry it · see SUBIDAS DE PRECIO",
                            blocking=False, context=f"{price['monitor']['price_group'].nunique():,} price groups watched",
                            examples=pending if len(pending) else None)

    # [5] the tables
    configuration.log_action(STEP_LABEL, 5, "writing the tables")
    configuration.write_table(STEP_LABEL, check_log, cells, TABLE_UPLIFT_CELLS)
    configuration.write_table(STEP_LABEL, check_log, check, TABLE_CONTRACT_CHECK)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the uplift by origin and the price by customer states (the diagnostics of the price for business: the next
    #     stage)
    configuration.log_action(STEP_LABEL, 7, "uplift cells by origin (propia: ≥ floor renewers; padre: its mandatory dims and "
                                            "discount bucket; celda: its mandatory dims; global):")
    configuration.show_table(cells.groupby("uplift_origen").agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"),
                                                                uplift_medio=("uplift", "mean"),
                                                                renovadores=("renovadores", "sum")).reset_index())
    show_the_price_by_states(state_factors, configuration)
    basis = dict(renewers=renewers, history=history, price=price, increases=increases, current_month=current_month)
    return cells, state_factors, check, increases, basis
