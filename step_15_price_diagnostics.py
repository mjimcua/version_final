"""
step_15_price_diagnostics.py — THE DIAGNOSTICS OF THE PRICE, for business: how the renewers' price behaves, read
from the closed months. Nothing here changes a prediction: the uplift is estimated in step 14 and examined in step 16;
this stage only explains it (what kind of stage: knowledge of the patterns).

INPUT:   the fine table · the uplift estimation of step 14 (its renewer rows, the window of the uplift, the price
         monitor and the increases in play) · its contract check.
OUTPUT:  the homogeneity of every uplift cell (table sff_uplift_homogeneidad, exact base only) · on screen: the
         contract rule read for business, the revaluation of the renewers, the price increases and the window.

Actions (logged as they are done):
  1. the homogeneity of every cell (only with the exact base): renewers' average due price against the cell's, and
     the approximate uplift against the exact one; by discount bucket
  2. write the homogeneity                                           check 1
  3. count the checks; stop if any failed
  4. the contract rule read for business (LA REGLA DEL CONTRATO): who meets it, and the list price implied by
     every discount bucket
  5. the revaluation of the renewers, read for business: the population of the claim (isolated renewals, no
     acquisition), where its uplift sits (percentiles, histogram by series), the discount split and the monthly path
     with its steps
  6. the price increases and the window of the uplift, read for business (SUBIDAS DE PRECIO Y VENTANA DEL UPLIFT):
     the uplift of the portfolio with all the history, in the window and in the window without increases; every
     increase in play with the month of its cycle, the future months that still carry it and where its effect is
     lost; the steps not confirmed yet

Checks (logged as they are made, numbered, at the level of their status):
   1. table sff_uplift_homogeneidad written and read back (only with the exact base)

Output: the homogeneity (or None) · table sff_uplift_homogeneidad.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, Config, ISOLATED_RATIO_BANDS, ROLE_PROJECTION, SERIES_ID_COLUMN,
                    TABLE_UPLIFT_HOMOGENEITY, TRUTH_ROLES, UPLIFT_CELL_ID_COLUMN, weighted_ratio_spread)
from price_increases import price_group_columns, price_group_of
from step_14_uplift import contract_rule


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "15"
STEP_NAME = "PRICE DIAGNOSTICS"
STEP_PURPOSE = ("explain the price of the renewers for business, from the closed months: the homogeneity of the "
                "uplift cells, the contract rule, the revaluation of the renewers and the price increases; no "
                "prediction reads it")
STEP_ACTIONS = ["the homogeneity of every cell: the renewers' average due price against the cell's (exact base only)",
                "write the homogeneity (check 1)",
                "count the checks; stop if any failed",
                "the contract rule read for business: who meets it, the list price implied by every bucket",
                "the revaluation of the renewers: population, distribution, discount split, monthly path",
                "the price increases and the window of the uplift: the month of every cycle, where its effect is lost"]
STEP_OUTPUT = "the homogeneity of every uplift cell · table sff_uplift_homogeneidad"


def diagnose_the_price(fine_table: pd.DataFrame, basis: dict, contract: pd.DataFrame, configuration: Config):
    """The stage 15 of the pipeline (see the header). Returns the homogeneity (None without the exact base)."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    renewers = basis["renewers"]

    # [1] the homogeneity: where renewers were worth the cell's average, the approximation was exact
    homogeneity = homogeneity_table(fine_table, renewers, configuration)
    if homogeneity is not None:
        weighted_ratio = (homogeneity["ratio_seleccion"] * homogeneity["usd_renovadores"]).sum() / homogeneity["usd_renovadores"].sum()
        configuration.log_action(STEP_LABEL, 1, f"homogeneity of {len(homogeneity):,} cells with renewers: the renewers' due "
                                                f"price over the cell's averages {weighted_ratio:.3f} weighted by their USD "
                                                f"(1.000 = renewers worth the cell's average: the old approximation was exact)")
        if configuration.discount_bucket_column in homogeneity.columns:
            by_bucket = (homogeneity.assign(_pesado=homogeneity["ratio_seleccion"] * homogeneity["usd_renovadores"])
                         .groupby(configuration.discount_bucket_column)
                         .agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"), usd_renovadores=("usd_renovadores", "sum"),
                              _pesado=("_pesado", "sum")))
            by_bucket["ratio_seleccion"] = by_bucket.pop("_pesado") / by_bucket["usd_renovadores"]
            configuration.logger.doc(f"[{STEP_LABEL}] the homogeneity by discount bucket (ratio_seleccion 1 = renewers worth "
                                     f"the cell's average; above 1 = the dearer renew more):")
            configuration.show_table(by_bucket.reset_index())
    else:
        configuration.log_action(STEP_LABEL, 1, "no exact base declared: the homogeneity cannot be measured")

    # [2] the table
    configuration.log_action(STEP_LABEL, 2, "writing the homogeneity")
    if homogeneity is not None:
        configuration.write_table(STEP_LABEL, check_log, homogeneity, TABLE_UPLIFT_HOMOGENEITY)

    # [3] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 3, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [4] the contract rule, read for business
    configuration.log_action(STEP_LABEL, 4, "the contract rule read for business")
    if len(contract):
        configuration.logger.doc(f"[{STEP_LABEL}] ══════════ LA REGLA DEL CONTRATO ══════════")
        configuration.logger.doc(f"[{STEP_LABEL}] the contract rule 1/(1 − d) against the past renewals with a known discount "
                                 f"(realizacion 1 = the rule is exact; neutras = no negative customer state, the rows the "
                                 f"path contrato_neutras applies it to):")
        configuration.show_table(contract.round(3))
        review_the_contract_rule(contract, fine_table, configuration)

    # [5] the revaluation of the renewers, read for business
    configuration.log_action(STEP_LABEL, 5, "the revaluation of the renewers (isolated renewals, no acquisition): "
                                            "population, distribution, discount split, monthly path")
    revaluation_report(renewers, configuration)

    # [6] the price increases and the window of the uplift, read for business
    configuration.log_action(STEP_LABEL, 6, "the price increases and the window of the uplift")
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
    doc(f"[{STEP_LABEL}] LICENCE BY LICENCE, the isolated renewals: mean ratio {mean_ratio:.3f}{spread_text}")
    bands = configuration.isolated_ratio_bands
    if len(bands) != len(ISOLATED_RATIO_BANDS):
        return
    values = [population[column].sum() for column, _, _, _, _ in bands]
    total = max(sum(values), 1e-9)
    if abs(sum(values) / max(base, 1e-9) - 1) > 0.001:
        doc(f"[{STEP_LABEL}] ATENCIÓN: the six bands add up to {sum(values) / max(base, 1e-9):.3f} × the isolated base "
            f"(must be 1.000): the extract's bands count other licences, this histogram is NOT reliable (see step 01, "
            f"ISOLATED: COHERENCIA DEL EXTRACTO)")
    longest = max(max(values) / total, 1e-9)
    doc(f"[{STEP_LABEL}] histogram · the value of the isolated renewals by renewal ratio, licence by licence")
    doc(f"[{STEP_LABEL}] {'ratio':>13}  {'USD que valían':>16}  {'% valor':>7}")
    for (_, _, _, _, label), value in zip(bands, values):
        bar = "█" * int(round(configuration.histogram_width * (value / total) / longest)) if value > 0 else ""
        doc(f"[{STEP_LABEL}] {label:>13}  {value:>16,.0f}  {value / total:>7.1%}  {bar}")
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
    doc(f"[{STEP_LABEL}] VALORACIÓN: {verdict}")


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
    doc(f"[{STEP_LABEL}] window {window[0]} to {window[-1]} · population: {population_text} · uplift base: {base_text}")
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
        doc(f"[{STEP_LABEL}] no series past the floor ({configuration.revaluation_units_floor:,.0f} units due per month): nothing to show")
        return
    if with_licence_level or with_bands:
        licence_level_report(population, configuration)
    covered = selected["usd_renovado"].sum() / max(per_series["usd_renovado"].sum(), 1e-9)
    doc(f"[{STEP_LABEL}] {len(selected):,} series in the distribution (floor {configuration.revaluation_units_floor:,.0f} units/month or "
        f"top {configuration.revaluation_top_series} by USD due): {covered:.0%} of the renewed USD of the population")

    # the claim, measured: the aggregate, the weighted percentiles and the money in each zone
    weights = selected["usd_renovado"].to_numpy(dtype=float)
    uplifts = selected["uplift"].to_numpy(dtype=float)
    p10, p25, p50, p75, p90 = weighted_percentiles(uplifts, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    share_100_105 = weights[(uplifts >= 1.00) & (uplifts < 1.05)].sum() / weights.sum()
    share_095_105 = weights[(uplifts >= 0.95) & (uplifts < 1.05)].sum() / weights.sum()
    share_above_110 = weights[uplifts >= 1.10].sum() / weights.sum()
    doc(f"[{STEP_LABEL}] aggregate uplift {selected['usd_renovado'].sum() / selected['base'].sum():.3f} · weighted "
        f"percentiles p10 {p10:.3f} · p25 {p25:.3f} · MEDIAN {p50:.3f} · p75 {p75:.3f} · p90 {p90:.3f}")
    doc(f"[{STEP_LABEL}] between SERIES (each series at its mean): {share_100_105:.0%} of the renewed USD in series "
        f"whose uplift sits in [1.00, 1.05) · {share_095_105:.0%} in [0.95, 1.05) · {share_above_110:.0%} at or above 1.10")
    if not with_bands and share_above_110 > 0.15:      # with the bands, the licence level above already answered the claim
        doc(f"[{STEP_LABEL}] VALORACIÓN: a heavy tail at or above 1.10. Look at its discount buckets below: a tail in "
            f"high KNOWN buckets is the contract losing its discount (legitimate); one in the unknown bucket is the "
            f"statistical path and deserves the homogeneity table")
    elif not with_bands and share_095_105 > 0.80:
        doc(f"[{STEP_LABEL}] VALORACIÓN: the claim holds: most of the money renews within ±5 % of what it was worth")
    elif not with_bands:
        doc(f"[{STEP_LABEL}] VALORACIÓN: the money is spread beyond ±5 %: the buckets below say whether the spread "
            f"follows the discount (expected) or not")

    if discount:
        known_label = np.where(population[discount].notna(), "discount known", "discount unknown")
        by_known = licence_level_summary(population.assign(_corte=known_label), "_corte", configuration)
        configuration.show_table(by_known)
        doc(f"[{STEP_LABEL}] by discount bucket (desv = the std deviation of the ratio between licences; cerca_100 = "
            f"share of the value in [0.95, 1.05); mas_110 = share at or above 1.10):")
        configuration.show_table(licence_level_summary(population, configuration.discount_bucket_column, configuration))

    for line in histogram_lines(selected, "every series in the distribution", configuration):
        doc(f"[{STEP_LABEL}] {line}")
    mostly_unknown = selected[selected["usd_desconocido"] / selected["usd_renovado"] > 0.5]
    if len(mostly_unknown):
        for line in histogram_lines(mostly_unknown, "series whose renewed USD is mostly of UNKNOWN discount", configuration):
            doc(f"[{STEP_LABEL}] {line}")

    top = selected.nlargest(15, "vencen_usd")[[SERIES_ID_COLUMN, "meses", "unidades_mes", "vencen_usd",
                                               "tasa_unidades", "uplift", "desv", "cerca_100", "mas_110"]].copy()
    top["unidades_mes"] = top["unidades_mes"].round(0)
    for rounded_column in ["tasa_unidades", "uplift", "desv", "cerca_100", "mas_110"]:
        top[rounded_column] = top[rounded_column].round(3)
    doc(f"[{STEP_LABEL}] the 15 series with the most USD due:")
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
    doc(f"[{STEP_LABEL}] month by month ({len(monthly)} closed, same population): escalon = the month over the average "
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
        doc(f"[{STEP_LABEL}] VALORACIÓN: a step of ≥ {configuration.price_step_threshold:.0%} held {longest_run} months in a row in the "
            f"last 12 closed ({listed}): the shape of a price increase (its effect lasts one renewal cycle)")
    elif above.any():
        doc(f"[{STEP_LABEL}] VALORACIÓN: months above {configuration.price_step_threshold:.0%} but never {configuration.price_persistence_months} in a "
            f"row ({listed}): noise, not an increase")
    elif len(recent):
        doc(f"[{STEP_LABEL}] VALORACIÓN: no step of ≥ {configuration.price_step_threshold:.0%} in the last 12 closed months: no sign "
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
    doc(f"[{STEP_LABEL}] ══════════ SUBIDAS DE PRECIO Y VENTANA DEL UPLIFT ══════════")
    window_months = configuration.uplift_window_months
    doc(f"[{STEP_LABEL}] detector on {price['detector_series'] or '—'} per price group "
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
    doc(f"[{STEP_LABEL}] the uplift of the portfolio (Σ renewed USD / Σ base of the renewers), three ways:")
    configuration.show_table(three_ways)

    # every increase in play: the month of its cycle today, what still falls due inside it, where it is lost
    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION]
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
        doc(f"[{STEP_LABEL}] the increases in play ({len(shown)}; escalon = the step of its first month, tamano = of the "
            f"cycle seen, in %; mes_del_ciclo = the month of its cycle that {current_month} is; usd_vence_con_efecto = "
            f"what falls due from {current_month} inside its cycle):")
        configuration.show_table(shown.drop(columns=["price_group"]))
        for _, increase in shown.head(configuration.price_increases_shown).iterrows():
            name = increase["price_group"]
            if increase["meses_futuros_con_efecto"] > 0:
                doc(f"[{STEP_LABEL}] VALORACIÓN {name}: subida de {increase['tamano']:+.1f}% desde {increase['inicio']}; "
                    f"{current_month} es el mes {increase['mes_del_ciclo']} de {cycle}: el forecast la aplica a "
                    f"{current_month}..{increase['fin_ciclo']} (${increase['usd_vence_con_efecto']:,.0f} que vencen) y desde "
                    f"{increase['fin_ciclo'] + 1} el efecto se pierde (su uplift vuelve al de sin subida)")
            else:
                doc(f"[{STEP_LABEL}] VALORACIÓN {name}: subida de {increase['tamano']:+.1f}% desde {increase['inicio']}; su "
                    f"ciclo terminó en {increase['fin_ciclo']}: solo se quita de la ventana, el futuro ya no la lleva")
    else:
        doc(f"[{STEP_LABEL}] VALORACIÓN: no increase whose cycle touches the window: the uplift is the window's as it is")

    # the steps at the end of the history, not confirmed yet
    pending = price["pending"]
    if len(pending):
        shown_pending = pending.copy()
        shown_pending["escalon_medio"] = (100 * shown_pending["escalon_medio"]).round(1)
        shown_pending["usd_renovado"] = shown_pending["usd_renovado"].round(0)
        doc(f"[{STEP_LABEL}] steps at the end of the history NOT confirmed yet (fewer than "
            f"{configuration.price_persistence_months} months above {configuration.price_step_threshold:.0%}; escalon_medio in %):")
        configuration.show_table(shown_pending.drop(columns=["price_group"]))
        for _, step in shown_pending.head(configuration.price_increases_shown).iterrows():
            doc(f"[{STEP_LABEL}] VALORACIÓN {step['price_group']}: {step['meses_por_encima']} meses por encima del umbral "
                f"desde {step['inicio']} ({step['escalon_medio']:+.1f}%), hacen falta {configuration.price_persistence_months}: "
                f"si es una subida, el forecast NO la lleva en {current_month}..{step['inicio'] + (cycle - 1)} (lo que le "
                f"queda de ciclo) y sus meses están en la ventana sin deflactar (suben algo el uplift de su celda)")
    show_series_with_increases(fine_table, renewers, price["calendar"], configuration)
    doc(f"[{STEP_LABEL}] ══════════ FIN DE SUBIDAS DE PRECIO Y VENTANA DEL UPLIFT ══════════")


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
        doc(f"[{STEP_LABEL}] no increase detected: no series to show its step")
        return
    if SERIES_ID_COLUMN not in fine_table.columns or SERIES_ID_COLUMN not in renewers.columns:
        doc(f"[{STEP_LABEL}] the fine table has no {SERIES_ID_COLUMN}: no series to show the step of its increase")
        return
    group_dims = price_group_columns(fine_table, configuration)
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN] != ROLE_PROJECTION]
    closed = closed.assign(_grupo=price_group_of(closed, group_dims))
    increased_groups = set(calendar["price_group"])
    money = (closed[closed["_grupo"].isin(increased_groups)]
             .groupby([SERIES_ID_COLUMN, "_grupo"])[configuration.renewed_usd_col].sum().sort_values(ascending=False))
    largest = money.head(configuration.price_series_shown)
    doc(f"[{STEP_LABEL}] the {len(largest)} forecast series with the most renewed USD in a price group with an increase, "
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
        doc(f"[{STEP_LABEL}] · {series_id} · grupo de precio {group} · ${renewed_usd:,.0f} renovados en total · subidas: {starts}")
        configuration.show_table(table.reset_index())


def review_the_contract_rule(check: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config) -> None:
    """The reading of the contract check and the test of the reference price; on screen only.

    INPUT:   the contract check (contract_check), the fine table.
    OUTPUT:  nothing returned: the VALORACIÓN of who meets the rule (weighted by the renewed USD), and B · the list
             price implied by every row, due USD / units / (1 − d), against the one of the lowest bucket of its same
             sku, cell and year: the MSRP is a list price, so ~1 in every bucket; growing with the bucket means the
             discount is measured over another reference. Whether the increases are counted twice is measured in the
             exam (step 16, rows with and without an increase in its cycle)."""
    discount = configuration.discount_value_column
    doc = configuration.logger.doc
    if not discount or check.empty:
        return
    period = configuration.period_col
    bucket = configuration.discount_bucket_column
    weights = check["usd_renovado"] / check["usd_renovado"].sum()
    def weighted(column: str) -> float:
        return float((check[column].fillna(0) * weights).sum()) if check[column].notna().any() else float("nan")
    doc(f"[{STEP_LABEL}] VALORACIÓN de la regla del contrato: weighted by the renewed USD, the rule is realized "
        f"{weighted('realizacion_neutras'):.0%} by the neutral renewals, {weighted('realizacion_con_estado'):.0%} by the "
        f"ones with a state, {weighted('realizacion_isolated'):.0%} by the isolated and {weighted('ratio_realizacion'):.0%} by "
        f"all. The exam of step 16 decides which path the forecast uses (contrato_neutras = the rule on the neutral rows)")
    # [B] the list price implied by every row, against the lowest bucket of its sku, cell and year
    sku = configuration.sku_column
    rows = fine_table[fine_table[discount].notna() & (fine_table[configuration.pipeline_units_col] > 0)].copy()
    keys = [column for column in [sku] + configuration.extract_mandatory_dims if column and column in rows.columns]
    if not keys:
        doc(f"[{STEP_LABEL}] B · no sku nor mandatory dims: the implied list price cannot be compared")
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
        doc(f"[{STEP_LABEL}] B · the list price implied by the rows, due USD / units / (1 − d), over the one of the lowest "
            f"bucket of the same cell ({len(keys)} columns: sku and the mandatory dims) and year (1 = the same MSRP in "
            f"every bucket, as it should; growing with the bucket = the discount is measured over another reference):")
        configuration.show_table(implied[["usd_vence", "ratio_lista_implicita"]].round(3).reset_index())
    doc(f"[{STEP_LABEL}] ══════════ FIN DE LA REGLA DEL CONTRATO ══════════")
