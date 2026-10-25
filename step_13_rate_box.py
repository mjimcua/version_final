"""
step_13_rate_box.py — THE BOX OF THE RATE: the renewal rate of every forecast series h months ahead, learned with only
the closed months up to an origin. One box that the forecast, the exam of every forecast series and the audit call,
so what is examined is exactly what is used.

THE INTERFACE
    box = RateBox(rated_units, series_estimate, pool_series, backtest, reference_members, configuration)
    rates = box.predict(rows, as_of)                 the rate of every row, learned up to as_of
    rates = box.predict(rows, as_of, DEFAULT_MODE)   the same with the default technique for every composition
    low, high = box.band(rates, units_due)           the interval of the rate (the error bands of step 12)
  rows   future rows (or exam rows): the series (fs_id), the month, and the dims of the series (its mandatory dims and
         its customer states, for a series with no composition)
  rates  per row: h (month − as_of), the composition, the technique used, the composition's prediction
         (tasa_pool_h), its level and its reference's level at as_of, z, the credibility shift (logit), the rate of
         the series (tasa) and where it comes from (origen_tasa: pool · celda_mandatory · global)

WHAT IS INSIDE THE BOX (what can be improved here without touching anything else)
  · pool    the series has a composition with at least min_months closed months at as_of (1 for the forecast; the
            exam asks exam_min_months_to_predict, 3: at an early origin a composition of 1-2 months is noise): the
            technique chosen for that composition and horizon band (step 12; the default technique in DEFAULT_MODE or
            when the chosen one cannot predict) predicts the composition's rate at h with the months up to as_of;
            the series moves it toward its credibility reference by (1 − z) × (logit reference level − logit
            composition level), both levels with the months up to as_of (prediction.py)
  · celda   no composition (or too short a history): the level of its mandatory cell up to as_of; with
            rate_fallback_with_states, of its mandatory cell AND its combination of customer states (a softcancel series
            then never takes the level of a cell made mostly of neutral units; the exam says it predicts the total worse,
            so it is off by default)
  · global  not even that: the portfolio's level (with rate_fallback_with_states, first its combination's)

THE STAGE 27 THE PIPELINE (label 13, after the backtest): the box is built once and checked; the forecast, the exam
and the audit receive it.

Actions (logged as they are done):
  1. the box: the compositions and their closed months, the techniques chosen per band, the references
  2. the rate of every composition at h = 1 at the last closed month, as a check of the box
  3. check the box                                                    checks 1-2
  4. count the checks; stop if any failed

Checks (logged as they are made, numbered, at the level of their status):
   1. every composition with enough closed months predicts a rate in [0, 1] at h = 1
   2. the levels of the box at the last closed month are the ones of the ladder (step 09): the same composition
      and reference rates

Output: the box (RateBox) · no table.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, Config, RATE_COLUMN, RATE_FROM_CELL, RATE_FROM_GLOBAL,
                    RATE_FROM_POOL, SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TRUTH_ROLES, join_columns, negative_states,
                    state_combination)
from prediction import band_quantiles, index_history, months_known, predict_composition, rate_band, shifted_rate
from step_12_backtest import band_of_horizon
from techniques import logit


# ─── the stage ───────────────────────────────────────────────────────────────────
STEP_LABEL = "13"
STEP_NAME = "THE BOX OF THE RATE"
STEP_PURPOSE = ("put the prediction of the renewal rate in ONE box: the technique chosen for every composition and "
                "horizon, moved toward its credibility reference, learned with the closed months up to any origin; the "
                "forecast, the exam and the audit call it")
STEP_ACTIONS = ["the box: the compositions and their closed months, the techniques chosen per band, the references",
                "the rate of every composition at h = 1 at the last closed month, as a check of the box",
                "check the box (checks 1-2)",
                "count the checks; stop if any failed"]
STEP_OUTPUT = "the box (RateBox) for the forecast, the exam and the audit"

CHOSEN_MODE = "chosen"           # the technique chosen by the backtest for the composition and band
DEFAULT_MODE = "default"         # the default technique for every composition (the cascade's por_defecto)
NO_COMPOSITION_TECHNIQUE = "mandatory_cell"


class RateBox:
    """THE BOX OF THE RATE (see the header of this module)."""

    def __init__(self, rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                 backtest: dict, reference_members: pd.DataFrame, configuration: Config):
        self.configuration = configuration
        period, due, renewed = configuration.period_col, configuration.pipeline_units_col, configuration.renewed_units_col
        self.decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"].to_dict()
        self.bands = backtest["bands"]
        self.states = negative_states(configuration)
        # the compositions: their monthly rate (logit), renewed and due, indexed once
        composition_truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()
                                        & (pool_series["vencen"] > 0)]
        self.composition_index = index_history(composition_truth, COMPOSITION_ID_COLUMN, period, [RATE_COLUMN, "renovadas", "vencen"])
        for entry in self.composition_index.values():
            entry["rate_logit"] = logit(entry[RATE_COLUMN])
            entry["rate_logit"].flags.writeable = False
        # the closed months of every series (no gaps): the levels of the references, the cells and the portfolio
        closed = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                             & (rated_units[due] > 0)].copy()
        closed["_combinacion"] = state_combination(closed, self.states)
        closed["_celda"] = self.cell_key(closed)
        self.closed = closed
        self.reference_index = {}
        if reference_members is not None and len(reference_members):
            members = closed.merge(reference_members[[SERIES_ID_COLUMN, "credibility_ref_id"]], on=SERIES_ID_COLUMN)
            self.reference_index = index_history(members.groupby(["credibility_ref_id", period])[[renewed, due]].sum().reset_index(),
                                                 "credibility_ref_id", period, [renewed, due])
        self.estimate = series_estimate.drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN)[
            [COMPOSITION_ID_COLUMN, "z", "credibility_ref_id"]]
        self._composition_cache, self._fallback_cache = {}, {}

    # ─── what the box knows at an origin ───
    def composition_history(self, composition_id, as_of) -> tuple:
        """(logit of its monthly rate, calendar months, Σ renewed, Σ due) of a composition up to as_of; None if unknown."""
        entry = self.composition_index.get(composition_id)
        if entry is None:
            return None
        known = months_known(entry, as_of)
        return (entry["rate_logit"][:known], entry["month"][:known], float(entry["renovadas"][:known].sum()),
                float(entry["vencen"][:known].sum()))

    def composition_prediction(self, composition_id, as_of, horizon: int, mode: str = CHOSEN_MODE,
                               min_months: int = 1) -> tuple:
        """(technique used, predicted rate, level at as_of) of a composition h months after as_of; (None, nan, nan)
        with fewer than min_months closed months."""
        key = (composition_id, as_of, int(horizon), mode, int(min_months))
        if key in self._composition_cache:
            return self._composition_cache[key]
        configuration = self.configuration
        history = self.composition_history(composition_id, as_of)
        if history is None or len(history[0]) < max(int(min_months), 1):
            result = (None, np.nan, np.nan)
        else:
            rate_logit, months, renewed, due = history
            band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
            technique = (configuration.default_technique if mode == DEFAULT_MODE
                         else self.decision.get((composition_id, band_name), configuration.default_technique))
            technique, rate = predict_composition(rate_logit, months, technique, int(horizon), configuration.default_technique)
            result = (technique, rate, renewed / due if due > 0 else np.nan)
        self._composition_cache[key] = result
        return result

    def reference_level(self, reference_id, as_of) -> float:
        entry = self.reference_index.get(reference_id)
        if entry is None:
            return np.nan
        known = months_known(entry, as_of)
        due = entry[self.configuration.pipeline_units_col][:known].sum()
        return float(entry[self.configuration.renewed_units_col][:known].sum() / due) if due > 0 else np.nan

    def cell_key(self, rows: pd.DataFrame) -> pd.Series:
        """The fallback cell of every row: its mandatory dims, and its combination of customer states when
        rate_fallback_with_states (the exam of 25-oct on the like-real synthetic: with the states the error of the
        total at 6 months rises from 1.4 % to 1.5 %, and the honest one from 1.5 % to 1.7 %; off by default)."""
        cell = join_columns(rows, self.configuration.business_mandatory_dims)
        if self.configuration.rate_fallback_with_states:
            cell = cell + "|" + state_combination(rows, self.states)
        return cell

    def fallback_levels(self, as_of) -> tuple:
        """(level of every cell with its states, level of every combination of states, global level) up to as_of."""
        if as_of not in self._fallback_cache:
            renewed, due = self.configuration.renewed_units_col, self.configuration.pipeline_units_col
            known = self.closed[self.closed[self.configuration.period_col] <= as_of]
            by_cell = known.groupby("_celda")[[renewed, due]].sum()
            by_combination = known.groupby("_combinacion")[[renewed, due]].sum()
            self._fallback_cache[as_of] = (
                (by_cell[renewed] / by_cell[due].where(by_cell[due] > 0)).to_dict(),
                (by_combination[renewed] / by_combination[due].where(by_combination[due] > 0)).to_dict(),
                float(known[renewed].sum() / known[due].sum()) if known[due].sum() > 0 else np.nan)
        return self._fallback_cache[as_of]

    # ─── the prediction ───
    def predict(self, rows: pd.DataFrame, as_of, mode: str = CHOSEN_MODE, min_months: int = 1) -> pd.DataFrame:
        """The rate of every row h months after as_of (see the header). Same index as rows. A composition with fewer
        than min_months closed months at as_of does not predict (the row takes its cell with its states)."""
        configuration = self.configuration
        period = configuration.period_col
        result = pd.DataFrame(index=rows.index)
        result["h"] = rows[period].map(lambda month: (month - as_of).n).astype(int)
        estimate = self.estimate.reindex(rows[SERIES_ID_COLUMN].to_numpy())
        result[COMPOSITION_ID_COLUMN] = estimate[COMPOSITION_ID_COLUMN].to_numpy()
        result["z"] = estimate["z"].to_numpy()
        result["credibility_ref_id"] = estimate["credibility_ref_id"].to_numpy()
        predictions = {pair: self.composition_prediction(pair[0], as_of, pair[1], mode, min_months)
                       for pair in set(zip(result[COMPOSITION_ID_COLUMN].dropna(), result.loc[result[COMPOSITION_ID_COLUMN].notna(), "h"]))}
        pairs = list(zip(result[COMPOSITION_ID_COLUMN], result["h"]))
        result["tecnica"] = [predictions.get(pair, (None, np.nan, np.nan))[0] for pair in pairs]
        result["tasa_pool_h"] = np.array([predictions.get(pair, (None, np.nan, np.nan))[1] for pair in pairs], dtype=float)
        result["group_rate"] = np.array([predictions.get(pair, (None, np.nan, np.nan))[2] for pair in pairs], dtype=float)
        references = {reference: self.reference_level(reference, as_of) for reference in set(result["credibility_ref_id"].dropna())}
        result["ref_rate"] = result["credibility_ref_id"].map(references)
        rate, shift = shifted_rate(result["tasa_pool_h"], result["z"], result["ref_rate"], result["group_rate"],
                                   configuration.apply_credibility_shift)
        has_pool = result["tasa_pool_h"].notna().to_numpy()
        result["credibility_shift_logit"] = np.where(has_pool, shift, 0.0)
        result["tasa"] = np.where(has_pool, rate, np.nan)
        result["origen_tasa"] = np.where(has_pool, RATE_FROM_POOL, None)
        # no composition: its mandatory cell (with its states when rate_fallback_with_states), then the portfolio
        cell_level, combination_level, global_level = self.fallback_levels(as_of)
        combination = state_combination(rows, self.states)
        from_cell = self.cell_key(rows).map(cell_level)
        missing = result["tasa"].isna()
        result.loc[missing & from_cell.notna(), "origen_tasa"] = RATE_FROM_CELL
        result.loc[missing & from_cell.notna(), "tasa"] = from_cell
        still_missing = result["tasa"].isna()
        result.loc[still_missing, "origen_tasa"] = RATE_FROM_GLOBAL
        portfolio = (combination[still_missing].map(combination_level).fillna(global_level)
                     if configuration.rate_fallback_with_states else global_level)
        result.loc[still_missing, "tasa"] = portfolio
        result["tecnica"] = result["tecnica"].where(result["origen_tasa"] == RATE_FROM_POOL, NO_COMPOSITION_TECHNIQUE)
        return result

    def band(self, rates: pd.DataFrame, units_due) -> tuple:
        """(low, high) of the rate of every row: the error quantiles of its technique at its horizon (step 12; a row
        with no composition takes the default technique's) × the binomial error with its units due."""
        techniques = rates["tecnica"].where(rates["tecnica"] != NO_COMPOSITION_TECHNIQUE, self.configuration.default_technique)
        q_low, q_high = band_quantiles(self.bands, techniques, rates["h"], self.configuration)
        return rate_band(rates["tasa"], units_due, q_low, q_high)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STAGE: build the box and check it
# ═══════════════════════════════════════════════════════════════════════════════════

def build_the_rate_box(rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                       backtest: dict, ladder: dict, configuration: Config) -> RateBox:
    """The stage 13 of the pipeline: the box of the rate, checked."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    last_closed = configuration.calendar_boundaries()["current"] - 1

    # [1] the box
    box = RateBox(rated_units, series_estimate, pool_series, backtest, ladder.get("reference_members"), configuration)
    chosen = backtest["decision"].groupby("tramo_h")["tecnica"].value_counts()
    configuration.log_action(STEP_LABEL, 1, f"{len(box.composition_index):,} compositions with closed months · "
                                            f"{len(box.reference_index):,} credibility references · techniques per band: "
                                            + " · ".join(f"{band} {block.droplevel(0).head(3).to_dict()}"
                                                         for band, block in chosen.groupby(level=0)))

    # [2] every composition at h = 1 at the last closed month
    at_one = pd.DataFrame([(composition_id, *box.composition_prediction(composition_id, last_closed, 1))
                           for composition_id in box.composition_index],
                          columns=[COMPOSITION_ID_COLUMN, "tecnica", "tasa", "nivel"])
    predicted = at_one[at_one["tasa"].notna()]
    short = sum(len(entry["month"]) < configuration.exam_min_months_to_predict for entry in box.composition_index.values())
    configuration.log_action(STEP_LABEL, 2, f"{len(predicted):,} of {len(at_one):,} compositions predict at h = 1 for the "
                                            f"forecast · {short:,} have fewer than {configuration.exam_min_months_to_predict} "
                                            f"closed months: the exam does not predict them at an origin that short (their "
                                            f"series take the level of their cell and states there)")

    # [3] the checks
    configuration.log_action(STEP_LABEL, 3, "checking the box")
    configuration.log_check(STEP_LABEL, check_log, "every composition with enough closed months predicts a rate in [0, 1] at h = 1",
                            bool(predicted["tasa"].between(0, 1).all()),
                            failure_detail=f"{int((~predicted['tasa'].between(0, 1)).sum())} compositions outside [0, 1]",
                            context=f"{len(predicted):,} compositions")
    ladder_rates = series_estimate.dropna(subset=[COMPOSITION_ID_COLUMN, "group_rate"]).drop_duplicates(COMPOSITION_ID_COLUMN)
    levels = at_one.set_index(COMPOSITION_ID_COLUMN)["nivel"]
    compared = ladder_rates[ladder_rates[COMPOSITION_ID_COLUMN].isin(levels.dropna().index)]
    differences = (compared[COMPOSITION_ID_COLUMN].map(levels) - compared["group_rate"]).abs()
    configuration.log_check(STEP_LABEL, check_log,
                            "the levels of the box at the last closed month are the ones of the ladder (step 09)",
                            bool((differences <= configuration.rate_tolerance).all()), blocking=False,
                            failure_detail=f"{int((differences > configuration.rate_tolerance).sum())} compositions differ "
                                           f"(largest {differences.max():.4f})",
                            context=f"{len(compared):,} compositions compared")

    # [4] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 4, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
    return box
