"""
step_16_uplift_backtest.py — Which uplift predicts the price of the renewals better:
the statistical one of the cell or the contract rule 1 / (1 − discount), on every row with a known discount
or only on the neutral ones?

HOW THE TEST WORKS:
  · the statistical uplift of every cell is estimated ONLY with the renewals of the months
    BEFORE the exam (no peeking), the way step 14 estimates it for the forecast: over the window
    (uplift_window_months before the first exam month) and without the price increases detected
    with those months (uplift_without_increases); in the exam, a row still inside the cycle of an
    increase takes it again (× its factor), as the forecast does;
  · in the exam months, for every row that renewed, the renewed money is predicted GIVEN
    its real renewed units (so only the price is judged, not the rate), the SAME way the
    forecast computes it (USD due × rate × uplift = renewed units × pipeline AUV × uplift once
    the real rate is given):
        statistical   renewed units × pipeline AUV × uplift of its cell
        contract      renewed units × pipeline AUV × 1 / (1 − discount)   (known discount only)
        contract on the neutral rows   the contract on the rows without a negative state (the typical
                      renewal the rule speaks of), the statistical one on the rows with a state
    The discount is measured against the MSRP of the contract at expiry: the rule already carries the
    increases between the purchase and the expiry, so the contract paths take no increase factor.
    The base is the row's average price, NOT the exact value of the renewers that step 14 uses
    to ESTIMATE the uplift: the forecast cannot know who will renew, so it values them at the
    row's average, and the test must judge that. Where renewers were worth more than the
    average (sff_uplift_homogeneidad), the error shows up here, as it will in the forecast.
  · both are compared on the SAME rows (the ones with a known discount): WAPE = Σ|predicted −
    real| / Σ real, and the bias of the total = (Σ predicted − Σ real) / Σ real;
  · VERDICT: the path with the lowest WAPE on the rows with a known discount is the one the forecast
    uses there (the statistical one on a tie). The verdict governs. The same comparison is shown for the
    neutral and the rows with a state, and for the rows with and without an increase in its cycle (an increase
    counted twice shows up as a statistical path that is high only with an increase).
The rows with an unknown discount are measured too (statistical only): they always take it.

THE PRICE BY STATES: the uplift cell has no states, and its uplift comes mostly from the neutral renewers. A softcancel
that renews does it with a retention offer, below its cell. The factor of every combination of negative states
(step 14: what its renewers paid over their cell) is estimated with the SAME months before the exam, and the
statistical path × the factor of the row's states (estadistica_estado) is judged on every renewer row of the exam
(known discount or not), in total and per combination. SECOND VERDICT: the factor is used by the forecast when its
WAPE on every renewer row is lower than the statistical path's.

THE WAYS OF ESTIMATING THE STATISTICAL UPLIFT, side by side in the same exam (on screen): all the
history (no window, no increases removed), the window as it is, and the window without increases;
the one the Config says is the one judged above. Their bias in total and month by month says what
the window and the removal of the increases change.

Actions (logged as they are done):
  1. the price increases known before the exam; the statistical uplift estimated with the months
     before the exam, every way (all the history, the window, the window without increases)
  2. the renewals of the exam months, predicted by both paths (and the statistical one every way)
  3. the error of each path, the verdict; the price by states and its verdict
  4. check the test                                                  checks 1-2
  5. write the result                                                check 3
  6. count the checks; stop if any failed
  7. show the comparison, as a table; the ways of the statistical uplift, in total and month by month

Checks (logged as they are made, numbered, at the level of their status):
   1. the statistical uplift was estimated without the exam months
   2. both paths were compared on the same rows
   3. table sff_backtest_uplift written and read back

Output: the comparison (per path and per discount bucket) and the verdicts (a dict: the path where the discount is
known, and whether the price factor of the states is used) ·
table sff_backtest_uplift.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, Config, PATH_CONTRACT, PATH_CONTRACT_NEUTRAL, PATH_STATISTICAL,
                    PATH_STATISTICAL_STATES, ROLE_TEST, TABLE_UPLIFT_BACKTEST, UPLIFT_CELL_ID_COLUMN,
                    has_no_negative_state)
from price_increases import active_increases, increase_factor, price_group_columns, price_group_of
from config import state_combination, negative_states
from step_14_uplift import (contract_rule, estimate_cell_uplifts, state_factor_of_rows, state_uplift_factors,
                            price_increases_as_of, renewer_rows, uplift_history)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "16"
STEP_NAME = "BACKTEST OF THE UPLIFT"
STEP_PURPOSE = ("decide, in the exam months, which uplift predicts the price of the renewals better where the discount "
                "is known: the statistical one of the cell (estimated before the exam), the contract rule 1 / (1 − d), or "
                "the contract on the neutral rows and the statistical one on the rows with a state; "
                "the verdict governs the forecast")
STEP_ACTIONS = ["the price increases known before the exam; the statistical uplift estimated with the months before "
                "the exam, every way (all the history, the window, the window without increases)",
                "the renewals of the exam months, predicted by both paths (and the statistical one every way)",
                "the error of each path, the verdict; the price by states and its verdict",
                "check the test (checks 1-2)",
                "write the result (check 3)",
                "count the checks; stop if any failed",
                "show the comparison, as a table; the ways of the statistical uplift, in total and month by month"]
STEP_OUTPUT = "the error of each path (total and per discount bucket) and the verdict · table sff_backtest_uplift"


def backtest_uplift(fine_table: pd.DataFrame, configuration: Config) -> tuple:
    """Statistical vs contract uplift in the exam months; the verdict."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
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
    configuration.log_action(STEP_LABEL, 1, f"statistical uplift estimated with the renewals before {exam_start}, "
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
    configuration.log_action(STEP_LABEL, 2, f"{len(exam_rows):,} renewer rows in the exam; {int(known.sum()):,} with a known discount, "
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
    configuration.log_action(STEP_LABEL, 3, f"verdict: where the discount is known the forecast uses the {chosen.upper()} path "
                                            f"(WAPE " + " · ".join(f"{path} {known_rows[path]:.1%}" for path in known_rows.index) + ")"
                                            + (f" · price by states {'USED' if state_factor_wins else 'NOT used'} (WAPE of every "
                                               f"renewer row: {PATH_STATISTICAL_STATES} {all_rows[PATH_STATISTICAL_STATES]:.1%} · "
                                               f"{PATH_STATISTICAL} {all_rows[PATH_STATISTICAL]:.1%})"
                                               if PATH_STATISTICAL_STATES in all_rows.index else ""))

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the test")
    last_month_used = max((estimate["history"][configuration.period_col].max() for estimate in estimates.values()
                           if len(estimate["history"])), default=None)
    configuration.log_check(STEP_LABEL, check_log, "the statistical uplift was estimated without the exam months (every way)",
                            last_month_used < exam_start if last_month_used is not None else True,
                            failure_detail="the estimate saw the exam", context=f"last month used {last_month_used}")
    same_rows = comparison[comparison["filas"] == "descuento conocido"]["renovadas_filas"].nunique() <= 1
    configuration.log_check(STEP_LABEL, check_log, "both paths were compared on the same rows", bool(same_rows),
                            failure_detail="the two paths were measured on different rows")

    # [5] the result
    configuration.log_action(STEP_LABEL, 5, "writing the result")
    configuration.write_table(STEP_LABEL, check_log, comparison.assign(via_usada_con_descuento=verdict["via_usada_con_descuento"],
                                                                       factor_estado=int(verdict["factor_estado"])),
                              TABLE_UPLIFT_BACKTEST)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the comparison
    configuration.log_action(STEP_LABEL, 7, "the price of the exam renewals predicted by each path (wape: Σ|pred − real| / "
                                            "Σ real; sesgo: (Σ pred − Σ real) / Σ real):")
    configuration.show_table(comparison)
    show_the_double_count(comparison, configuration)
    show_the_price_by_states(comparison, state_factors, verdict, configuration)
    show_statistical_ways(exam_rows, known, list(estimates), configured_label, configuration)
    return comparison, verdict


def show_the_price_by_states(comparison: pd.DataFrame, state_factors: pd.DataFrame, verdict: dict,
                            configuration: Config) -> None:
    """EL PRECIO POR ESTADO DEL CLIENTE EN EL EXAMEN on screen: the factors estimated before the exam, and per combination of states
    the bias and WAPE of the statistical path with and without them (and of the contract where the discount is known)."""
    by_state = comparison[comparison["filas"].str.startswith("estado · ") | (comparison["filas"] == "todas")]
    if by_state.empty:
        return
    doc = configuration.logger.doc
    doc(f"[{STEP_LABEL}] ══════════ EL PRECIO POR ESTADO DEL CLIENTE EN EL EXAMEN ══════════")
    doc(f"[{STEP_LABEL}] the factor of every combination of negative states, estimated before the exam (what its renewers "
        f"paid over their cell):")
    configuration.show_table(state_factors.round(3))
    doc(f"[{STEP_LABEL}] the exam per combination ({PATH_STATISTICAL_STATES} = {PATH_STATISTICAL} × the factor; "
        f"{PATH_CONTRACT}: only the rows with a known discount):")
    configuration.show_table(by_state.round(4))
    doc(f"[{STEP_LABEL}] VALORACIÓN: the price factor of the states is "
        + ("USED by the forecast: it lowers the WAPE of every renewer row" if verdict["factor_estado"]
           else "NOT used: it does not lower the WAPE of every renewer row"))
    doc(f"[{STEP_LABEL}] ══════════ FIN DE EL PRECIO POR ESTADO DEL CLIENTE EN EL EXAMEN ══════════")


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
    doc(f"[{STEP_LABEL}] VALORACIÓN de las subidas: bias with an increase minus without — statistical "
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
                          configuration: Config) -> None:
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
    doc(f"[{STEP_LABEL}] ══════════ EL UPLIFT ESTADÍSTICO DE CADA FORMA, EN EL MISMO EXAMEN ══════════")
    doc(f"[{STEP_LABEL}] estimated before the exam; ← = the way of the Config (the one the forecast uses); "
        f"sesgo = (Σ pred − Σ real) / Σ real:")
    configuration.show_table(pd.DataFrame(rows))
    by_month = exam_rows.groupby(period)
    monthly = pd.DataFrame({"usd_real": by_month["_numerador"].sum().round(0)})
    for label in labels:
        monthly["sesgo_" + label] = (by_month["pred_forma_" + label].sum() / by_month["_numerador"].sum() - 1).round(4)
    doc(f"[{STEP_LABEL}] the bias month by month (every renewer row of the exam):")
    configuration.show_table(monthly.reset_index())
    doc(f"[{STEP_LABEL}] ══════════ FIN DE EL UPLIFT ESTADÍSTICO DE CADA FORMA ══════════")


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
