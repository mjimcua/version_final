"""
step_24_core.py — The core table: the whole story of every row in ONE wide table.

Decided on 27-sep-2026: the prediction lives in one wide table at the fine grain, history
and future, WITHOUT hash keys (in Power BI they cost dictionary and space and say nothing
to a reader). The columns go from left to right in the order of the steps, each with the
prefix of the step that creates it (s00_, s02_, s03_…), so a row reads as its own story.
The dimensions keep their names: they are the slicers.

It can be built after ANY step from 03 on: every block of columns is added only when the
step that produces it has run (its argument is not None), and the legend describes the
columns present. main.py builds it last, with everything. It grows with every step of the
framework: a new step adds its block here (today steps 00-19 and the forecast). The values of steps 11 and
12 belong to the ESTIMATION ID of the series (the pool it takes its rate from); a series
whose pool was not measured (no support, short history) has them empty.

Three levels live in the same table, and the legend says which is which:
  · fila    a value of the row: it adds up (SUM)
  · unidad  a value of the forecast unit (fu_id), repeated in its fine rows: do NOT add it up
  · serie   a value of the rate series (fs_id), repeated in all its rows: do NOT add it up;
            with one series selected, MAX (or MIN) returns it
A RATE is never stored per row: a rate belongs to an aggregation and is the ratio of sums
of the rows selected (in Power BI too). The estimates of the ladder are stored, as
attributes of the series.

GAPS are rows: a month with no expirations inside the history of a series (step 06) gets
a row with every measure at 0 (origen_fila = hueco), so the months of a series are
complete and the sums still reconcile with the extract.

Actions (logged as they are done):
  1. the rows of the extract (the fine table) with their raw and step-02 measures and their ids
  2. the gap rows of step 06
  3. the values of the forecast unit (step 04), if it has run
  4. the forecast rows (step 20), the final block, and the forecast series dimension (one row per series)
  5. the legend: every column, its step, its level and how to aggregate it
  6. check the core against the extract, the legend and the total      checks 1-7
  7. write the core, the dimension and the legend                      checks 8-10
  8. count the checks; stop if any failed
  9. show one series as it looks in the core, and how to read it in Power BI

Checks (logged as they are made, numbered, at the level of their status):
   1. the forecast series dimension has one row per forecast series of the core
   2. the exam sums of the dimension equal step 19's (each series once)
   3. every original record and every synthetic row, once
   4. the money reconciles with the extract (units and USD due, renewed as the raw had it)
   5. every row has the values of every block present
   6. the legend describes every column of the core
   7. the core summed by year and origin = sff_forecast_total
   8-10. tables sff_nucleo, sff_forecast_series and sff_nucleo_leyenda written and read back

Output: the core table, the forecast series dimension and the legend · tables sff_nucleo, sff_forecast_series,
sff_nucleo_leyenda.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import re

import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, Config,
                    DUE_UNITS_AT_DUE_COLUMN, DUE_USD_AT_DUE_COLUMN, FINE_ROWS_COLUMN, MATURATION_ROW_COLUMN,
                    PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, ROLE_TEST,
                    ROLE_TRAIN, ROUTE_COLUMN, ROW_FROM_GAP, ROW_FROM_MATURATION, ROW_FROM_RAW, ROW_ORIGIN_COLUMN,
                    S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
                    SERIES_ID_COLUMN, SIGN_COLUMN, SYNTHETIC_COLUMN, TABLE_CORE, TABLE_CORE_LEGEND,
                    TABLE_FORECAST_SERIES, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
                    TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL, TRUTH_ROLES as TRUTH_ROLES_OF_CORE, TS_PROJECTED,
                    TS_REAL, TS_REENTRY, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN)


STEP_COLUMN_PATTERN = re.compile(r"^s\d+_")      # a column of a step block of the core (s03_fs_id, s20_tasa…)

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "24"
STEP_NAME = "CORE TABLE"
STEP_PURPOSE = ("put the whole story of every row in one wide table, left to right in the order of the steps, "
                "with the gaps as explicit rows, so that selecting one forecast series shows everything the "
                "framework knows about it; a legend says how every column is aggregated")
STEP_ACTIONS = ["the rows of the extract (the fine table) with their raw and step-02 measures and their ids",
                "the gap rows of step 06",
                "the values of the forecast unit (steps 04, 07), if they have run",
                "the forecast rows (step 20), the final block, and the forecast series dimension (one row per series)",
                "the legend: every column, its step, its level and how to aggregate it",
                "check the core against the extract and the legend (checks 1-4)",
                "write the core and its legend (checks 5-6)",
                "count the checks; stop if any failed",
                "show one series as it looks in the core, and how to read it in Power BI"]
STEP_OUTPUT = "one row per fine row of the extract plus one per gap · tables sff_nucleo, sff_nucleo_leyenda"

# ─── named constants ─────────────────────────────────────────────────────────────
LEVEL_ROW, LEVEL_UNIT, LEVEL_SERIES = "fila", "unidad", "serie"
AGGREGATE_SUM = "SUM"
AGGREGATE_ATTRIBUTE = "no sumar: atributo repetido; con una serie o unidad seleccionada, MAX"
AGGREGATE_SLICER = "segmentar / filtrar"

# The measure columns of the core: (core name, step, description). The raw measures are as
# the extract had them; the step-02 renewals are the ones the framework learns from.
RAW_MEASURES = [("s00_vencen_unidades", "00", "units falling due, as in the extract"),
                ("s00_vencen_usd", "00", "USD falling due, as in the extract"),
                ("s00_renovadas_unidades", "00", "renewed units as the extract had them (early results of the future included)"),
                ("s00_renovado_usd", "00", "renewed USD as the extract had them (early results of the future included)")]
CALENDAR_MEASURES = [("s02_vencen_unidades", "02", "units due after the calendar: the 1-year licences sold or renewed from the current month on are 0 (not known yet, projected)"),
                     ("s02_vencen_usd", "02", "USD due after the calendar: the 1-year licences sold or renewed from the current month on are 0 (not known yet, projected)"),
                     ("s02_renovadas_unidades", "02", "renewed units after the calendar: null → 0 in closed months, wiped from the current month on"),
                     ("s02_renovado_usd", "02", "renewed USD after the calendar: null → 0 in closed months, wiped from the current month on")]

# steps 14-20, per FUTURE ROW (a value of the row: the money columns add up)
ROW_VALUES_STEP_20 = [(PIPELINE_ORIGIN_COLUMN, "s20_origen_pipeline", "20", "real (extract) · proyectada · simulada (a row of the "
                                                                             "maturation: the origin of the units it receives)"),
                      ("_vencen_unidades", "s20_vencen_unidades", "20", "units due of the future row TODAY: the extract's or the extended "
                                                                         "horizon's; 0 on a row of the maturation (SUM)"),
                      ("_vencen_usd", "s20_vencen_usd", "20", "USD due of the future row TODAY: the extract's or the extended horizon's; "
                                                              "0 on a row of the maturation (SUM)"),
                      (DUE_UNITS_AT_DUE_COLUMN, "s20_vencen_unidades_al_vencer", "20",
                       "units the row will have WHEN IT FALLS DUE, after the maturation of the states (stage 17): a neutral row "
                       "keeps what does not take a state, a row of the maturation has what it receives (SUM)"),
                      (DUE_USD_AT_DUE_COLUMN, "s20_vencen_usd_al_vencer", "20",
                       "USD due the row will have when it falls due (what leaves a neutral row leaves at its average price) (SUM)"),
                      ("maduracion_unidades", "s20_maduracion_unidades", "17",
                       "units at due date − units today: negative on a neutral row (they take a state), positive on a row of "
                       "the maturation (they arrive at its series with a state) (SUM)"),
                      ("h", "s20_h", "20", "months from the last closed month"),
                      ("origen_tasa", "s20_origen_tasa", "20", "pool (technique of its estimation id) · celda_mandatory · global"),
                      ("tecnica", "s20_tecnica", "20", "the technique that predicted the rate of this row"),
                      ("tasa", "s20_tasa", "20", "the predicted rate of the row (units)"),
                      ("tasa_baja", "s20_tasa_baja", "20", "low end of the rate band"),
                      ("tasa_alta", "s20_tasa_alta", "20", "high end of the rate band"),
                      ("confidence", "s20_confidence", "20", "high · medium · low: how sure the forecast is of the rate of the row"),
                      ("via_uplift", "s14_via_uplift", "14", "contrato (1/(1 − d)) · estadistica (its cell)"),
                      ("uplift", "s14_uplift", "14", "the uplift applied to the row"),
                      ("esperado_unidades", "s20_esperado_unidades", "20", "expected renewed units (SUM)"),
                      ("esperado_usd", "s20_esperado_usd", "20", "expected renewed USD (SUM)"),
                      ("esperado_usd_bajo", "s20_esperado_usd_bajo", "20", "low end of the row's USD band (SUM = the worst case of a total)"),
                      ("esperado_usd_alto", "s20_esperado_usd_alto", "20", "high end of the row's USD band (SUM = the worst case of a total)"),
                      ("esperado_unidades_hoy", "s20_esperado_unidades_hoy", "20",
                       "expected renewed units with TODAY's states: units due today × rate (SUM)"),
                      ("esperado_usd_hoy", "s20_esperado_usd_hoy", "20",
                       "expected USD with TODAY's states: USD due today × rate × uplift (SUM)"),
                      ("maduracion_delta_unidades", "s20_maduracion_delta_unidades", "17",
                       "the maturation of the renewed units: already IN the forecast (expected = today's + this) (SUM)"),
                      ("maduracion_delta_usd", "s20_maduracion_delta_usd", "17",
                       "the maturation of the renewed USD: already IN the forecast (expected = today's + this) (SUM)")]

# The blocks of values of the forecast unit, of the series and of its estimation id:
# (source column, core name, step, description). A block enters when its step has run.
UNIT_VALUES_STEP_04 = [(FINE_ROWS_COLUMN, "s04_filas_finas", "04", "fine rows added into the unit")]
SERIES_VALUES_STEP_05 = [(ROUTE_COLUMN, "s05_ruta", "05", "predecible · solo_historia · solo_futuro"),
                         (COVERAGE_COLUMN, "s05_cobertura", "05", "the roles the series has, in time order"),
                         ("usd_por_predecir", "s05_usd_por_predecir", "05", "USD due in the months to predict, whole series")]
SERIES_VALUES_STEP_06 = [("n_propio", "s06_n_propio", "06", "units due in a typical closed month of the series"),
                         ("tasa_propia", "s06_tasa_propia", "06", "own rate of the series: Σ renewed / Σ due over its closed months"),
                         ("error_binomial_pp", "s06_error_binomial_pp", "06", "Wilson half-width of one month at n_propio (pp)"),
                         (SIGN_COLUMN, "s06_signo", "06", "neutro · negativo · positivo · mixto"),
                         ("bajo_suelo", "s06_bajo_suelo", "06", "1 when n_propio is below support_floor")]
# step 08, per forecast series: the 4 stages of the ladder (grouping by any stage id adds up to the raw)
SERIES_VALUES_STEP_08 = [("stage0_id", "s08_stage0_id", "08", "stage 0 · raw: the forecast series itself"),
                         ("stage0_support", "s08_stage0_support", "08", "stage 0 · its own support (units due in a typical month)"),
                         ("stage1_id", "s08_stage1_id", "08", "stage 1 · sign: its group after the signs are merged (repeats stage 0 if nothing changed)"),
                         ("stage1_support", "s08_stage1_support", "08", "stage 1 · the support of that group"),
                         ("stage2_id", "s08_stage2_id", "08", "stage 2 · extras: its group after the extras are annulled"),
                         ("stage2_support", "s08_stage2_support", "08", "stage 2 · the support of that group"),
                         ("stage3_id", "s08_stage3_id", "08", "stage 3 · collapse: its COMPOSITION, the group that is predicted"),
                         ("stage3_support", "s08_stage3_support", "08", "stage 3 · the support of its composition")]
# step 09, per forecast series: the rate of its composition and the credibility (stage 4)
SERIES_VALUES_STEP_09 = [("group_series", "s09_composition_series", "09", "forecast series in its composition"),
                         ("group_rate", "s09_composition_rate", "09", "rate of its composition"),
                         ("credibility_ref_id", "s09_credibility_ref_id", "09", "the wider group its rate is blended with (below own_rate_floor)"),
                         ("ref_support", "s09_ref_support", "09", "support of the credibility reference"),
                         ("ref_rate", "s09_ref_rate", "09", "rate of the credibility reference"),
                         ("credibility_effect_pp", "s09_credibility_effect_pp", "09", "how much the credibility moves the rate of its composition (pp); 0 when it predicts alone"),
                         ("alcanzo_suelo", "s09_alcanzo_suelo", "09", "1 when its final group reaches the support floor"),
                         ("k", "s09_k", "09", "Bühlmann k of the reference"),
                         ("z", "s09_z", "09", "credibility of the group's own rate: n / (n + k); 1 without a reference"),
                         ("tasa_estimada", "s09_tasa_estimada", "09", "diagnostic: z·group + (1 − z)·reference over the whole history (the forecast moves the technique's prediction by the same credibility, step 20)"),
                         ("se_estimacion_pp", "s09_se_estimacion_pp", "09", "error of the estimate (pp)"),
                         ("se_prediccion_pp", "s09_se_prediccion_pp", "09", "error of next month's prediction (pp): never below the series' own noise"),
                         ("nivel_riesgo", "s09_nivel_riesgo", "09", "how the rate was obtained: A_propio … N_sin_impacto")]
# per forecast series, from the audit step: its own dynamics (to cross the precision with it)
SERIES_VALUES_DYNAMICS = [("phi", "s11_series_phi", "11", "volatility of its own rate: observed variation / binomial noise (≈ 1: only chance)"),
                          ("trend", "s11_series_trend", "11", "+1 / −1 significant trend of its own rate, 0 none"),
                          ("trend_pp_year", "s11_series_trend_pp_year", "11", "slope of its own rate, pp per year"),
                          ("seasonal", "s11_series_seasonal", "11", "1 when its own rate has a month effect (≥ 24 months)"),
                          ("amplitude_pp", "s11_series_amplitude_pp", "11", "size of its month effect (pp)"),
                          ("measurable", "s11_series_measurable", "11", "yes · low_support (< 30 a month: not conclusive) · short_history"),
                          ("differs_from_composition", "s11_series_differs", "11", "1 when its trend or season differs from its composition's")]
# per forecast series: the exam of its chosen technique (every exam month and horizon), as ratios repeated on its rows
SERIES_VALUES_EXAM = [("framework_mae_pp", "s19_exam_mae_pp", "19", "mean absolute error of its rate in the exam, framework (pp)"),
                      ("framework_bias_pp", "s19_exam_bias_pp", "19", "mean error of its rate in the exam, framework (pp): + over-predicts"),
                      ("framework_wape", "s19_exam_wape", "19", "Σ |predicted − real| / Σ real renewed units in the exam, framework"),
                      ("framework_coverage", "s19_exam_coverage", "19", "share of its framework predictions whose real rate fell inside the interval"),
                      ("raw_mae_pp", "s19_raw_mae_pp", "19", "the same error predicting the series alone with its own history (raw)"),
                      ("raw_coverage", "s19_raw_coverage", "19", "share of its raw predictions inside their interval"),
                      ("improvement_mae_pp", "s19_improvement_mae_pp", "19", "raw error − framework error (pp): + the framework predicts it better"),
                      ("framework_rmse_pp", "s19_exam_rmse_pp", "19", "root mean square error of its rate in the exam, framework (pp)"),
                      ("framework_noise_pp", "s19_exam_noise_pp", "19", "binomial noise of its monthly rate: the error of a perfect prediction (pp)"),
                      ("framework_error_over_noise", "s19_exam_error_over_noise", "19",
                       "framework error / noise: ≈ 1 at the limit · well above 1, something knowable is missing"),
                      ("raw_error_over_noise", "s19_raw_error_over_noise", "19", "the same for the series alone (raw)")]
# … and as counts and units: in the dimension a SUM over any filter counts every forecast series once
SERIES_SUMMABLE_EXAM = [("framework_predictions", "s19_exam_predictions", "19", "exam predictions (SUM)"),
                        ("framework_in_band", "s19_exam_in_band", "19", "framework predictions inside their interval (SUM)"),
                        ("framework_pred_units", "s19_exam_pred_units", "19", "renewed units predicted by the framework (SUM)"),
                        ("framework_real_units", "s19_exam_real_units", "19", "renewed units real in the exam (SUM)"),
                        ("framework_abs_err_units", "s19_exam_abs_err_units", "19", "Σ |predicted − real| units, framework (SUM)"),
                        ("raw_in_band", "s19_raw_in_band", "19", "raw predictions inside their interval (SUM)"),
                        ("raw_abs_err_units", "s19_raw_abs_err_units", "19", "Σ |predicted − real| units, raw (SUM)"),
                        ("raw_real_units", "s19_raw_real_units", "19", "renewed units real in the exam, raw rows (SUM)"),
                        ("raw_predictions", "s19_raw_predictions", "19", "raw exam predictions (SUM)")]
# step 11, per estimation id (the dynamics of the rate the series takes)
ESTIMATION_VALUES_STEP_11 = [("phi", "s11_phi", "11", "φ of the estimation id: observed variation of its rate / binomial noise (≈ 1: nothing to model)"),
                             ("tendencia", "s11_tendencia", "11", "+1 / −1 significant trend of the rate, 0 none"),
                             ("tendencia_pp_ano", "s11_tendencia_pp_ano", "11", "slope of the rate, pp per year"),
                             ("estacional", "s11_estacional", "11", "1 when the month of the year is significant (on the rate without trend)"),
                             ("amplitud_pp", "s11_amplitud_pp", "11", "highest month minus lowest month (pp)"),
                             ("meses_alto", "s11_meses_alto", "11", "calendar months above the others beyond twice their error"),
                             ("meses_bajo", "s11_meses_bajo", "11", "calendar months below the others beyond twice their error")]
# step 12, per estimation id and horizon band (pivoted: one column per band)
ESTIMATION_VALUES_STEP_12 = [("tecnica", "s12_tecnica", "12", "the technique chosen for the estimation id in this horizon band"),
                             ("tecnica_origen", "s12_origen", "12", "campeon (beat the default technique) · por_defecto · sin_soporte (not judged)"),
                             ("elegida_err_pp_medio", "s12_examen_err_pp", "12", "mean |error| of the chosen technique in the exam months (pp of rate)"),
                             ("por_defecto_err_pp_medio", "s12_examen_por_defecto_err_pp", "12", "the same for the default technique: the one to beat"),
                             ("dentro_banda", "s12_examen_dentro_banda", "12", "share of the exam errors inside the band (nominal 90 %)")]


def build_core_table(fine_table: pd.DataFrame, configuration: Config, forecast_units: pd.DataFrame = None,
                     rated_units: pd.DataFrame = None,
                     series_table: pd.DataFrame = None, series_rate: pd.DataFrame = None,
                     series_estimate: pd.DataFrame = None, pool_dynamics: pd.DataFrame = None,
                     technique_decision: pd.DataFrame = None, exam_by_pool: pd.DataFrame = None,
                     forecast: pd.DataFrame = None, time_series_rows: pd.DataFrame = None,
                     time_series_table: pd.DataFrame = None, forecast_total: pd.DataFrame = None,
                     ladder_stages: pd.DataFrame = None, series_dynamics: pd.DataFrame = None,
                     series_exam: pd.DataFrame = None) -> tuple:
    """The core table with every block whose step has run, and its legend; checked and written.

    INPUT:   the fine table and the results of every step that has run (None = not run).
    OUTPUT:  (core, legend). The core has EVERY row: the original records of the pipeline and of the
             time_series universe, and the synthetic ones (gaps, extended horizon, ts_proyectado,
             ts_reentrada); every decision as a column; and the FINAL block (forecast_*) that answers the
             business questions by summing it.
    RULES:   the forecast_* columns are the same for every row: pipeline due (forecast_to_renew_*) and renewed or
             revenue (forecast_renewed_*), real or expected, with its origin. A total of any year and origin
             is a SUM of the core; sff_forecast_total must equal it (check 5).
    EDGE CASES: a step not run leaves its block out; no time_series universe → no ts rows."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    dimension_columns = core_dimension_columns(configuration)
    blocks_present = []

    # [1] the rows of the extract
    raw_rows = rows_of_the_extract(fine_table, dimension_columns, configuration)
    configuration.log_action(STEP_LABEL, 1, f"{len(raw_rows):,} rows of the extract with {len(dimension_columns)} "
                                            f"dimension columns, their measures and their ids")

    # [2] the gap rows
    gap_rows = rows_of_the_gaps(rated_units, raw_rows.columns, configuration) if rated_units is not None else raw_rows.iloc[0:0]
    core = pd.concat([raw_rows, gap_rows], ignore_index=True) if len(gap_rows) else raw_rows
    extension_rows = rows_of_the_extension(forecast, core.columns, configuration) if forecast is not None else core.iloc[0:0]
    if len(extension_rows):
        core = pd.concat([core, extension_rows], ignore_index=True)
    time_series_core_rows = rows_of_the_time_series(time_series_rows, time_series_table, core.columns, configuration)
    if len(time_series_core_rows):
        core = pd.concat([core, time_series_core_rows], ignore_index=True)
    configuration.log_action(STEP_LABEL, 2, f"{len(gap_rows):,} gap rows · {len(extension_rows):,} extended-horizon rows · "
                                            f"{len(time_series_core_rows):,} time_series rows "
                                            f"({time_series_core_rows[ROW_ORIGIN_COLUMN].value_counts().to_dict() if len(time_series_core_rows) else {}})")

    # [3] the values of the forecast unit
    for block_frame, block_specs in ((forecast_units, UNIT_VALUES_STEP_04),):
        if block_frame is not None:
            core = add_block(core, block_frame, UNIT_ID_COLUMN, "s03_fu_id", block_specs)
            blocks_present.append(block_specs)
    configuration.log_action(STEP_LABEL, 3, f"unit blocks added: {[specs[0][2] for specs in blocks_present] or 'none'}")

    # [4] the forecast series dimension: one row per forecast series with everything of its level
    #     (route, own rate, the 4 stages, its composition and credibility, technique, dynamics, exam);
    #     the core keeps only what belongs to a row (or a unit) and joins it by s03_fs_id
    if forecast is not None:
        forecast_values = forecast.assign(_clave=forecast_keys(forecast), _vencen_unidades=forecast[configuration.pipeline_units_col],
                                          _vencen_usd=forecast[configuration.pipeline_usd_col])
        forecast_values = forecast_values[["_clave"] + [source for source, _, _, _ in ROW_VALUES_STEP_20]]
        forecast_values = forecast_values.rename(columns={source: name for source, name, _, _ in ROW_VALUES_STEP_20})
        core = core.merge(forecast_values, on="_clave", how="left")
        blocks_present.append(ROW_VALUES_STEP_20)
    core = core.drop(columns=["_fila", "_clave"], errors="ignore")
    core = add_final_block(core, configuration)
    blocks_present.append(FINAL_VALUES)
    if "s04_filas_finas" in core.columns:
        core.loc[core[ROW_ORIGIN_COLUMN] == ROW_FROM_GAP, "s04_filas_finas"] = 0     # a gap adds no fine row
    core = core.sort_values(["s03_fs_id", configuration.period_col, ROW_ORIGIN_COLUMN]).reset_index(drop=True)
    dimension = build_series_dimension(core, series_table, series_rate, series_estimate, ladder_stages, pool_dynamics,
                                       technique_decision, exam_by_pool, series_dynamics, series_exam, configuration)
    configuration.log_action(STEP_LABEL, 4, f"the core has {len(core):,} rows × {len(core.columns)} columns · the forecast series "
                                            f"dimension {len(dimension):,} rows × {len(dimension.columns)} columns, joined by s03_fs_id")

    # [5] the legend
    legend = core_legend(core, dimension_columns, blocks_present, configuration)
    legend = pd.concat([legend.assign(tabla="sff_" + TABLE_CORE), dimension_legend(dimension)], ignore_index=True)
    configuration.log_action(STEP_LABEL, 5, f"legend of {len(legend)} columns: {legend['nivel'].value_counts().to_dict()}")

    # [6] the checks
    configuration.log_action(STEP_LABEL, 6, "checking the core against the extract and the legend")
    check_core(core, fine_table, gap_rows, legend[legend["tabla"] == "sff_" + TABLE_CORE], blocks_present, forecast_total,
               configuration, check_log)
    check_dimension(core, dimension, series_exam, configuration, check_log)

    # [7] the tables, written
    configuration.log_action(STEP_LABEL, 7, "writing the core and its legend")
    configuration.write_table(STEP_LABEL, check_log, core, TABLE_CORE)
    configuration.write_table(STEP_LABEL, check_log, dimension, TABLE_FORECAST_SERIES)
    configuration.write_table(STEP_LABEL, check_log, legend, TABLE_CORE_LEGEND)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] one series as it looks in the core, how to read it in Power BI, and the questions answered from it
    log_core_report(core, dimension, configuration)
    log_core_answers(core, configuration)
    return core, legend, dimension


DIMENSION_SUMMARY_VALUES = [("s20_esperado_usd_total", "20", "USD expected to renew in the months to predict (sum of its rows)"),
                            ("s20_pct_usd_high", "20", "share of its expected USD with confidence high"),
                            ("s20_pct_usd_medium", "20", "share of its expected USD with confidence medium"),
                            ("s20_pct_usd_low", "20", "share of its expected USD with confidence low"),
                            ("s20_confidence", "20", "the confidence of most of its expected USD")]


def build_series_dimension(core: pd.DataFrame, series_table, series_rate, series_estimate, ladder_stages, pool_dynamics,
                           technique_decision, exam_by_pool, series_dynamics, series_exam, configuration: Config) -> pd.DataFrame:
    """sff_forecast_series: one row per forecast series of the core, with every value of its level."""
    dimension = pd.DataFrame({"s03_fs_id": sorted(core["s03_fs_id"].dropna().unique())})
    ladder_stages = stages_of_every_series(ladder_stages, series_rate)
    for block_frame, block_specs in ((series_table, SERIES_VALUES_STEP_05), (series_rate, SERIES_VALUES_STEP_06),
                                     (ladder_stages, SERIES_VALUES_STEP_08), (series_estimate, SERIES_VALUES_STEP_09),
                                     (series_dynamics, SERIES_VALUES_DYNAMICS),
                                     (series_exam, SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM)):
        if block_frame is not None:
            dimension = add_block(dimension, block_frame, SERIES_ID_COLUMN, "s03_fs_id", block_specs)
    if pool_dynamics is not None and len(pool_dynamics) and "s08_stage3_id" in dimension.columns:
        dynamics_values = pool_dynamics[[COMPOSITION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_11]]
        dynamics_values = dynamics_values.rename(columns={source: name for source, name, _, _ in ESTIMATION_VALUES_STEP_11})
        dimension = dimension.merge(dynamics_values, left_on="s08_stage3_id", right_on=COMPOSITION_ID_COLUMN,
                                    how="left").drop(columns=COMPOSITION_ID_COLUMN)
    if technique_decision is not None and "s08_stage3_id" in dimension.columns:
        dimension, _ = add_backtest_block(dimension, technique_decision, exam_by_pool, configuration)
    # the money to predict and its confidence, from the rows of the core
    if "s20_confidence" in core.columns:
        future = core[core["s20_esperado_usd"].notna()]
        by_label = future.pivot_table(index="s03_fs_id", columns="s20_confidence", values="s20_esperado_usd", aggfunc="sum").fillna(0.0)
        total = by_label.sum(axis=1)
        summary = pd.DataFrame({"s20_esperado_usd_total": total})
        for label in ("high", "medium", "low"):
            summary[f"s20_pct_usd_{label}"] = (by_label[label] / total.where(total > 0)) if label in by_label.columns else 0.0
        summary["s20_confidence"] = by_label.idxmax(axis=1)
        dimension = dimension.merge(summary, left_on="s03_fs_id", right_index=True, how="left")
    return dimension


def dimension_legend(dimension: pd.DataFrame) -> pd.DataFrame:
    """The legend of sff_forecast_series: every column with its step and how to aggregate it."""
    described = {name: (step, description) for _, name, step, description in
                 SERIES_VALUES_STEP_05 + SERIES_VALUES_STEP_06 + SERIES_VALUES_STEP_08 + SERIES_VALUES_STEP_09
                 + SERIES_VALUES_DYNAMICS + SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM + ESTIMATION_VALUES_STEP_11}
    described.update({name: (step, description) for name, step, description in DIMENSION_SUMMARY_VALUES})
    summable = {name for _, name, _, _ in SERIES_SUMMABLE_EXAM} | {"s05_usd_por_predecir", "s20_esperado_usd_total"}
    rows = [("s03_fs_id", "03", LEVEL_SERIES, "clave: relación 1 → n con sff_nucleo[s03_fs_id]", "the forecast series")]
    for column_name in dimension.columns[1:]:
        base = next((name for _, name, _, _ in ESTIMATION_VALUES_STEP_12 if column_name.startswith(name + "_")), None)
        step, description = described.get(column_name, ("12", next((d for _, n, _, d in ESTIMATION_VALUES_STEP_12 if n == base), ""))
                                          if base else ("", ""))
        rows.append((column_name, step, LEVEL_SERIES, AGGREGATE_SUM if column_name in summable else AGGREGATE_SLICER, description))
    legend = pd.DataFrame(rows, columns=["columna", "paso", "nivel", "como_agregar", "descripcion"])
    return legend.assign(tabla="sff_" + TABLE_FORECAST_SERIES)


def check_dimension(core: pd.DataFrame, dimension: pd.DataFrame, series_exam, configuration: Config, check_log: list) -> None:
    """Check 6 and 7: one row per forecast series of the core; its exam sums equal step 19's."""
    configuration.log_check(STEP_LABEL, check_log, "the forecast series dimension has one row per forecast series of the core",
                            dimension["s03_fs_id"].is_unique and set(dimension["s03_fs_id"]) == set(core["s03_fs_id"].dropna()),
                            failure_detail="the dimension and the core do not have the same forecast series",
                            context=f"{len(dimension):,} forecast series")
    if series_exam is None or "s19_exam_real_units" not in dimension.columns:
        configuration.log_not_evaluated(STEP_LABEL, check_log, "the exam sums of the dimension equal step 19's", "no exam")
        return
    same = (abs(dimension["s19_exam_real_units"].sum() - series_exam["framework_real_units"].sum()) < 1e-6
            and int(dimension["s19_exam_predictions"].sum()) == int(series_exam["framework_predictions"].sum()))
    configuration.log_check(STEP_LABEL, check_log, "the exam sums of the dimension equal step 19's (each series once)", same,
                            failure_detail="the dimension lost or duplicated exam predictions",
                            context=f"{int(dimension['s19_exam_predictions'].sum()):,} exam predictions")


def add_block(core: pd.DataFrame, block_frame: pd.DataFrame, key_column: str, core_key: str, block_specs: list) -> pd.DataFrame:
    """The values of one block, joined to every row of the core by its unit or its series."""
    source_columns = [source for source, _, _, _ in block_specs]
    values = block_frame[[key_column] + source_columns].drop_duplicates(key_column)
    values = values.rename(columns={source: name for source, name, _, _ in block_specs})
    return core.merge(values, left_on=core_key, right_on=key_column, how="left").drop(columns=key_column)


def add_backtest_block(core: pd.DataFrame, technique_decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
                       configuration: Config) -> tuple:
    """The chosen technique and the exam error of the row's estimation id, one column per horizon band."""
    per_band = technique_decision[[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]]
    if exam_by_pool is not None:
        per_band = per_band.merge(exam_by_pool[[COMPOSITION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "por_defecto_err_pp_medio",
                                                "dentro_banda"]], on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
    wide = per_band.pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h")
    specs = []
    renamed = {}
    for source, name, step, description in ESTIMATION_VALUES_STEP_12:
        for band_name in configuration.horizon_bands:
            if (source, band_name) in wide.columns:
                renamed[(source, band_name)] = f"{name}_{band_name}"
                specs.append((source, f"{name}_{band_name}", step, f"{description} (band {band_name})"))
    wide = wide[list(renamed)]
    wide.columns = [renamed[column] for column in wide.columns]
    core = core.merge(wide, left_on="s08_stage3_id", right_index=True, how="left")
    return core, specs


# ─── the final block: the same columns for every row, the ones the business questions sum ───
FINAL_UNIVERSE_COLUMN = "forecast_universe"
FINAL_ORIGIN_COLUMN = "forecast_pipeline_source"
FINAL_STATE_COLUMN = "forecast_status"
FINAL_YEAR_COLUMN = "forecast_year"
FINAL_UNIVERSE_PIPELINE, FINAL_UNIVERSE_TIME_SERIES = "pipeline", "time_series"
STATE_REAL, STATE_EXPECTED = "actual", "forecast"
TS_WITHOUT_RESULT = "ts_sin_resultado"       # an original time_series row of a month not closed yet (its result is projected)
FINAL_VALUES = [
    ("forecast_universe", "forecast_universe", "FIN", "pipeline (renewals) · time_series (retail to subscription)"),
    ("forecast_pipeline_source", "forecast_pipeline_source", "FIN", "pipeline_renovado_real · pipeline_real_esperado · pipeline_proyectada · pipeline_simulada · "
                                        "ts_real · ts_proyectado · ts_reentrada · hueco · ts_sin_resultado"),
    ("forecast_status", "forecast_status", "FIN", "actual (already happened) · forecast (predicted)"),
    ("forecast_year", "forecast_year", "FIN", "the year of the month"),
    ("forecast_to_renew_units", "forecast_to_renew_units", "FIN", "PIPELINE: units falling due (extract, extended horizon, ts re-entry) · SUM"),
    ("forecast_to_renew_USD", "forecast_to_renew_USD", "FIN", "PIPELINE: USD falling due (acquisition and ts re-entry: at their discount) · SUM"),
    ("forecast_renewed_units", "forecast_renewed_units", "FIN", "RENEWED units: real in closed months, expected in the future; ts: converted units · SUM"),
    ("forecast_renewed_USD", "forecast_renewed_USD", "FIN", "RENEWED USD (or time_series revenue): real in closed months, expected in the future · SUM"),
    ("forecast_renewed_USD_low", "forecast_renewed_USD_low", "FIN", "low end of the row's band (SUM = the worst case of a total; real rows: the real value)"),
    ("forecast_renewed_USD_high", "forecast_renewed_USD_high", "FIN", "high end of the row's band (SUM = the worst case of a total; real rows: the real value)"),
    ("forecast_to_renew_units_at_due", "forecast_to_renew_units_at_due", "FIN",
     "PIPELINE WHEN IT FALLS DUE: the units of every series after the maturation of the states (stage 17); the same total "
     "as forecast_to_renew_units, moved from the neutral series to the ones with a state · SUM"),
    ("forecast_to_renew_USD_at_due", "forecast_to_renew_USD_at_due", "FIN",
     "PIPELINE WHEN IT FALLS DUE, in USD (what moves, moves at the average price of its neutral row) · SUM"),
    ("forecast_maturation_units", "forecast_maturation_units", "FIN",
     "units at due date − units today: − on a neutral series (they take a state), + on a in the state one · SUM"),
    ("forecast_renewed_units_maturation_adj", "forecast_renewed_units_maturation_adj", "FIN",
     "the part of forecast_renewed_units due to the states still to come (already IN it; 0 where nothing moved) · SUM"),
    ("forecast_renewed_USD_maturation_adj", "forecast_renewed_USD_maturation_adj", "FIN",
     "the part of forecast_renewed_USD due to the states still to come (already IN it): forecast − this = the "
     "forecast with today's states · SUM")]
FINAL_SUMMABLE = {"forecast_to_renew_units", "forecast_to_renew_USD", "forecast_renewed_units", "forecast_renewed_USD", "forecast_renewed_USD_low",
                  "forecast_renewed_USD_high", "forecast_to_renew_units_at_due", "forecast_to_renew_USD_at_due",
                  "forecast_maturation_units", "forecast_renewed_units_maturation_adj", "forecast_renewed_USD_maturation_adj"}


def rows_of_the_time_series(time_series_rows, time_series_table, core_columns, configuration: Config) -> pd.DataFrame:
    """The time_series universe in the core: its ORIGINAL records (ts_real in the closed months,
    ts_sin_resultado in the months not closed yet) and its SYNTHETIC rows (ts_proyectado, ts_reentrada).
    Helper columns _ts_* carry their final values until the final block reads them."""
    frames = []
    helper = ["_ts_vence_unidades", "_ts_vence_usd", "_ts_renovadas", "_ts_usd"]
    current = configuration.calendar_boundaries()["current"]
    if time_series_rows is not None and len(time_series_rows):
        original = pd.DataFrame({configuration.period_col: time_series_rows[configuration.period_col].to_numpy()})
        for column_name in core_columns:
            if column_name in time_series_rows.columns and column_name not in original.columns:
                original[column_name] = time_series_rows[column_name].to_numpy()
        closed = original[configuration.period_col] < current
        original[ROW_ORIGIN_COLUMN] = np.where(closed, TS_REAL, TS_WITHOUT_RESULT)
        original["s00_vencen_unidades"] = time_series_rows[configuration.pipeline_units_col].fillna(0).to_numpy()
        original["s00_vencen_usd"] = time_series_rows[configuration.pipeline_usd_col].fillna(0).to_numpy()
        original["s00_renovadas_unidades"] = time_series_rows[configuration.renewed_units_col].fillna(0).to_numpy()
        original["s00_renovado_usd"] = time_series_rows[configuration.renewed_usd_col].fillna(0).to_numpy()
        original["s02_rol"] = configuration.role_of_months(original[configuration.period_col])
        original["_ts_vence_unidades"], original["_ts_vence_usd"] = 0.0, 0.0
        original["_ts_renovadas"] = np.where(closed, original["s00_renovadas_unidades"], 0.0)
        original["_ts_usd"] = np.where(closed, original["s00_renovado_usd"], 0.0)
        frames.append(original)
    if time_series_table is not None and len(time_series_table):
        synthetic = time_series_table[time_series_table["origen"].isin([TS_PROJECTED, TS_REENTRY])]
        rows = pd.DataFrame({ROW_ORIGIN_COLUMN: synthetic["origen"].to_numpy(),
                             configuration.period_col: synthetic[configuration.period_col].to_numpy()})
        for column_name in core_columns:
            if column_name in synthetic.columns and column_name not in rows.columns:
                rows[column_name] = synthetic[column_name].to_numpy()
        for measure_name, _, _ in RAW_MEASURES:
            rows[measure_name] = 0.0
        rows["s02_rol"] = configuration.role_of_months(rows[configuration.period_col])
        reentry = (synthetic["origen"] == TS_REENTRY).to_numpy()
        rows["_ts_vence_unidades"] = np.where(reentry, synthetic["unidades"].to_numpy(), 0.0)
        rows["_ts_vence_usd"] = np.where(reentry, synthetic["valor"].to_numpy(), 0.0)      # the pipeline: value at 40 % off
        rows["_ts_renovadas"] = np.where(reentry, synthetic.get("unidades_renovadas", pd.Series(0.0, index=synthetic.index)).fillna(0).to_numpy(),
                                         synthetic["unidades"].to_numpy())
        rows["_ts_usd"] = synthetic["revenue"].to_numpy()
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=list(core_columns))
    return pd.concat(frames, ignore_index=True).reindex(columns=list(core_columns) + helper)


def add_final_block(core: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The final block: universe, origin, state, year, pipeline due and renewed (or revenue), row by row."""
    origin = core[ROW_ORIGIN_COLUMN]
    closed = core["s02_rol"].isin(TRUTH_ROLES_OF_CORE)
    is_time_series = origin.astype(str).str.startswith("ts_")
    zeros = pd.Series(0.0, index=core.index)
    column = lambda name: core[name].fillna(0.0) if name in core.columns else zeros

    raw_closed, raw_future = (origin == ROW_FROM_RAW) & closed, (origin == ROW_FROM_RAW) & ~closed
    extended = origin.isin([PIPELINE_PROJECTED, PIPELINE_SIMULATED])
    # a row of the maturation: a prediction (no data); its money belongs to the origin of the units it receives
    matured = origin == ROW_FROM_MATURATION
    pipeline_of_row = core["s20_origen_pipeline"] if "s20_origen_pipeline" in core.columns else pd.Series(None, index=core.index)
    core[FINAL_UNIVERSE_COLUMN] = np.where(is_time_series, FINAL_UNIVERSE_TIME_SERIES, FINAL_UNIVERSE_PIPELINE)
    core[FINAL_ORIGIN_COLUMN] = np.select(
        [raw_closed, raw_future, origin == PIPELINE_PROJECTED, origin == PIPELINE_SIMULATED, origin == ROW_FROM_GAP,
         matured & (pipeline_of_row == PIPELINE_REAL), matured & (pipeline_of_row == PIPELINE_PROJECTED),
         matured & (pipeline_of_row == PIPELINE_SIMULATED)],
        [TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_SIMULATED, ROW_FROM_GAP,
         TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_SIMULATED],
        default=origin.astype(str))
    extended = extended | matured
    core[FINAL_STATE_COLUMN] = np.where(raw_closed | (origin == ROW_FROM_GAP) | origin.isin([TS_REAL, TS_WITHOUT_RESULT]),
                                        STATE_REAL, STATE_EXPECTED)
    core[FINAL_YEAR_COLUMN] = core[configuration.period_col].map(lambda month: month.year)

    extract_rows = origin.isin([ROW_FROM_RAW, ROW_FROM_GAP])            # the pipeline after the calendar (s02): not what is not known yet
    core["forecast_to_renew_units"] = np.select([extract_rows, extended, is_time_series],
                                           [column("s02_vencen_unidades"), column("s20_vencen_unidades"), column("_ts_vence_unidades")], 0.0)
    core["forecast_to_renew_USD"] = np.select([extract_rows, extended, is_time_series],
                                      [column("s02_vencen_usd"), column("s20_vencen_usd"), column("_ts_vence_usd")], 0.0)
    core["forecast_renewed_units"] = np.select([raw_closed, raw_future | extended, is_time_series],
                                               [column("s02_renovadas_unidades"), column("s20_esperado_unidades"), column("_ts_renovadas")], 0.0)
    core["forecast_renewed_USD"] = np.select([raw_closed, raw_future | extended, is_time_series],
                                         [column("s02_renovado_usd"), column("s20_esperado_usd"), column("_ts_usd")], 0.0)
    with_band = raw_future | extended
    core["forecast_renewed_USD_low"] = np.where(with_band, column("s20_esperado_usd_bajo"), core["forecast_renewed_USD"])
    core["forecast_renewed_USD_high"] = np.where(with_band, column("s20_esperado_usd_alto"), core["forecast_renewed_USD"])
    # the pipeline when it falls due and the maturation: next to the forecast (closed rows: the pipeline as it was)
    core["forecast_to_renew_units_at_due"] = np.where(raw_future | extended, column("s20_vencen_unidades_al_vencer"),
                                                      core["forecast_to_renew_units"])
    core["forecast_to_renew_USD_at_due"] = np.where(raw_future | extended, column("s20_vencen_usd_al_vencer"),
                                                    core["forecast_to_renew_USD"])
    core["forecast_maturation_units"] = np.where(raw_future | extended, column("s20_maduracion_unidades"), 0.0)
    core["forecast_renewed_units_maturation_adj"] = np.where(raw_future | extended, column("s20_maduracion_delta_unidades"), 0.0)
    core["forecast_renewed_USD_maturation_adj"] = np.where(raw_future | extended, column("s20_maduracion_delta_usd"), 0.0)
    # the closed-month measures: observed, not predicted: only in the closed months of the extract
    for measure in configuration.closed_month_measures:
        core[measure.core_column] = np.where(raw_closed, column(measure.s02_column), np.nan)
    return core.drop(columns=[name for name in core.columns if name.startswith("_ts_")])


def stages_of_every_series(ladder_stages, series_rate):
    """The 4 stages of every forecast series: the estimable ones as the ladder left them; the others
    (solo_historia, solo_futuro: they do not climb) are themselves in every stage, with their own support."""
    if ladder_stages is None or series_rate is None:
        return ladder_stages
    missing = series_rate[~series_rate[SERIES_ID_COLUMN].isin(ladder_stages[SERIES_ID_COLUMN])]
    if missing.empty:
        return ladder_stages
    themselves = pd.DataFrame({SERIES_ID_COLUMN: missing[SERIES_ID_COLUMN].to_numpy()})
    for stage in range(4):
        themselves[f"stage{stage}_id"] = missing[SERIES_ID_COLUMN].to_numpy()
        themselves[f"stage{stage}_support"] = missing["n_propio"].to_numpy()
    return pd.concat([ladder_stages, themselves], ignore_index=True)


def core_dimension_columns(configuration: Config) -> list:
    """The dimension columns of the core, as the extract names them (they are the slicers):
    the rate series columns, the revaluation extras, the exact discount and its bucket."""
    discount = [configuration.discount_value_column, configuration.discount_bucket_column] if configuration.discount_value_column else []
    return list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion + discount))


def rows_of_the_extract(fine_table: pd.DataFrame, dimension_columns: list, configuration: Config) -> pd.DataFrame:
    """One row per fine row: its month, its dimensions, its measures (raw and after the
    calendar), its role and its ids in text (no hash keys)."""
    raw_rows = pd.DataFrame({ROW_ORIGIN_COLUMN: ROW_FROM_RAW, configuration.period_col: fine_table[configuration.period_col]})
    for column_name in dimension_columns:
        raw_rows[column_name] = fine_table[column_name]
    raw_rows["s00_vencen_unidades"] = fine_table[S0_PIPELINE_UNITS_COLUMN]
    raw_rows["s00_vencen_usd"] = fine_table[S0_PIPELINE_USD_COLUMN]
    raw_rows["s02_vencen_unidades"] = fine_table[configuration.pipeline_units_col]
    raw_rows["s02_vencen_usd"] = fine_table[configuration.pipeline_usd_col]
    raw_rows["s00_renovadas_unidades"] = fine_table[S0_RENEWED_UNITS_COLUMN]
    raw_rows["s00_renovado_usd"] = fine_table[S0_RENEWED_USD_COLUMN]
    raw_rows["s02_rol"] = fine_table[CALENDAR_ROLE_COLUMN]
    raw_rows["s02_es_mes_en_curso"] = fine_table[CURRENT_MONTH_COLUMN]
    raw_rows["s02_renovadas_unidades"] = fine_table[configuration.renewed_units_col]
    raw_rows["s02_renovado_usd"] = fine_table[configuration.renewed_usd_col]
    for measure in configuration.closed_month_measures:
        raw_rows[measure.s02_column] = fine_table[measure.raw_column]
    raw_rows["s03_fs_id"] = fine_table[SERIES_ID_COLUMN]
    raw_rows["s03_fu_id"] = fine_table[UNIT_ID_COLUMN]
    raw_rows["s03_uplift_cell_id"] = fine_table[UPLIFT_CELL_ID_COLUMN]
    raw_rows["_fila"] = fine_table.index      # the position of the fine row: the forecast joins on it, then it is dropped
    raw_rows["_clave"] = [f"f{position}" for position in fine_table.index]
    return raw_rows.reset_index(drop=True)


def forecast_keys(forecast: pd.DataFrame) -> list:
    """The join key of every forecast row: 'f<position>' for a row of the extract, 'e<n>' for an extended row."""
    return [f"f{int(position)}" if pd.notna(position) else f"e{number}"
            for number, position in enumerate(forecast["_fila"])]


def rows_of_the_extension(forecast: pd.DataFrame, core_columns, configuration: Config) -> pd.DataFrame:
    """One row per future row that is not in the extract: the extended horizon (proyectada, simulada) and the
    prediction rows of the maturation (maduracion): its month, dims and ids; the measures of the extract at 0 (it is
    not in the extract: its pipeline is in s20_vencen_*, at due date in s20_vencen_*_al_vencer)."""
    keys = forecast_keys(forecast)
    extension = forecast[forecast["_fila"].isna()]
    if extension.empty:
        return pd.DataFrame(columns=core_columns)
    is_maturation = (extension[MATURATION_ROW_COLUMN] == 1).to_numpy() if MATURATION_ROW_COLUMN in extension.columns \
        else np.zeros(len(extension), dtype=bool)
    rows = pd.DataFrame({ROW_ORIGIN_COLUMN: np.where(is_maturation, ROW_FROM_MATURATION, extension[PIPELINE_ORIGIN_COLUMN].to_numpy()),
                         configuration.period_col: extension[configuration.period_col].to_numpy()})
    for column_name in core_columns:
        # the dimensions of the extract (softcancel, sku… too); the step blocks (s00_, s03_, s20_…) are filled below
        if column_name in extension.columns and column_name not in rows.columns and not STEP_COLUMN_PATTERN.match(column_name):
            rows[column_name] = extension[column_name].to_numpy()
    for measure_name, _, _ in RAW_MEASURES:
        rows[measure_name] = 0.0
    rows["s02_rol"] = extension[CALENDAR_ROLE_COLUMN].to_numpy()
    rows["s02_es_mes_en_curso"] = 0
    rows["s03_fs_id"] = extension[SERIES_ID_COLUMN].to_numpy()
    rows["s03_fu_id"] = extension[UNIT_ID_COLUMN].to_numpy()
    rows["s03_uplift_cell_id"] = extension[UPLIFT_CELL_ID_COLUMN].to_numpy()
    rows["_clave"] = [key for key, position in zip(keys, forecast["_fila"]) if pd.isna(position)]
    return rows.reindex(columns=list(core_columns))


def rows_of_the_gaps(rated_units: pd.DataFrame, core_columns, configuration: Config) -> pd.DataFrame:
    """One row per gap of step 06: its month, its series columns, every measure at 0, its
    role; the revaluation columns are empty (a gap has no contract)."""
    gaps = rated_units[rated_units[SYNTHETIC_COLUMN] == 1]
    if gaps.empty:
        return pd.DataFrame(columns=core_columns)
    gap_rows = pd.DataFrame({ROW_ORIGIN_COLUMN: ROW_FROM_GAP, configuration.period_col: gaps[configuration.period_col].to_numpy()})
    for column_name in configuration.rate_series_columns:
        gap_rows[column_name] = gaps[column_name].to_numpy()
    for measure_name, _, _ in RAW_MEASURES + CALENDAR_MEASURES:
        gap_rows[measure_name] = 0.0
    gap_rows["s02_rol"] = gaps[CALENDAR_ROLE_COLUMN].to_numpy()
    gap_rows["s02_es_mes_en_curso"] = 0
    gap_rows["s03_fs_id"] = gaps[SERIES_ID_COLUMN].to_numpy()
    gap_rows["s03_fu_id"] = gaps[UNIT_ID_COLUMN].to_numpy()
    return gap_rows.reindex(columns=core_columns)


def core_legend(core: pd.DataFrame, dimension_columns: list, blocks_present: list, configuration: Config) -> pd.DataFrame:
    """Every column of the core with its step, its level and how to aggregate it."""
    legend_rows = [(ROW_ORIGIN_COLUMN, "24", LEVEL_ROW, AGGREGATE_SLICER, "raw = a row of the extract · hueco = a month with nothing "
                    "due inside a history · proyectada / simulada = a row of the extended horizon (renewal falling due again / "
                    "last year's acquisition) · maduracion = a prediction row of the stage 17: the units of a neutral series "
                    "that will carry a state when they fall due, in their series with a state (no data: its s00/s02 measures are 0) · "
                    "ts_real / ts_sin_resultado = an original row of the time_series universe · "
                    "ts_proyectado / ts_reentrada = its synthetic rows"),
                   (configuration.period_col, "00", LEVEL_ROW, AGGREGATE_SLICER, "the month")]
    for column_name in dimension_columns:
        description = ("the exact discount (share)" if column_name == configuration.discount_value_column
                       else "the discount bucket, derived in step 03" if column_name == configuration.discount_bucket_column
                       else "a dimension of the extract")
        legend_rows.append((column_name, "03" if column_name == configuration.discount_bucket_column else "00",
                            LEVEL_ROW, AGGREGATE_SLICER, description))
    for name, step, description in RAW_MEASURES:
        legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
    legend_rows += [("s02_rol", "02", LEVEL_ROW, AGGREGATE_SLICER, "entrenamiento · examen · proyeccion"),
                    ("s02_es_mes_en_curso", "02", LEVEL_ROW, AGGREGATE_SLICER, "1 in the current month")]
    for name, step, description in CALENDAR_MEASURES:
        legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
    legend_rows += [("s03_fs_id", "03", LEVEL_ROW, AGGREGATE_SLICER, "the rate series: select one to see its story"),
                    ("s03_fu_id", "03", LEVEL_ROW, AGGREGATE_SLICER, "the forecast unit: the series in its month"),
                    ("s03_uplift_cell_id", "03", LEVEL_ROW, AGGREGATE_SLICER, "the price context (uplift cell)")]
    unit_names = {name for _, name, _, _ in UNIT_VALUES_STEP_04}
    row_names = {name for _, name, _, _ in ROW_VALUES_STEP_20}
    summable = {"s20_esperado_unidades", "s20_esperado_usd", "s20_esperado_usd_bajo", "s20_esperado_usd_alto",
                "s20_vencen_unidades", "s20_vencen_usd", "s20_vencen_unidades_al_vencer", "s20_vencen_usd_al_vencer",
                "s20_maduracion_unidades", "s20_maduracion_delta_unidades", "s20_esperado_unidades_hoy", "s20_esperado_usd_hoy",
                "s20_maduracion_delta_usd"}
    for block_specs in blocks_present:
        for _, name, step, description in block_specs:
            if name in {spec_name for _, spec_name, _, _ in FINAL_VALUES}:
                legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM if name in FINAL_SUMMABLE else AGGREGATE_SLICER,
                                    description))
                continue
            if name in row_names:
                legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM if name in summable else AGGREGATE_SLICER
                                    if name in ("s20_origen_tasa", "s20_tecnica", "s14_via_uplift", "s20_h")
                                    else "no sumar: valor de la fila (una tasa o un uplift)", description))
                continue
            if name in {spec_name for _, spec_name, _, _ in SERIES_SUMMABLE_EXAM}:
                legend_rows.append((name, step, LEVEL_SERIES, AGGREGATE_SUM, description))
                continue
            level = LEVEL_UNIT if name in unit_names else LEVEL_SERIES
            legend_rows.append((name, step, level, AGGREGATE_ATTRIBUTE, description))
    for measure in configuration.closed_month_measures:
        legend_rows.append((measure.s02_column, "02", LEVEL_ROW, AGGREGATE_SUM,
                            f"{measure.config_field} ({measure.raw_column}) after the calendar (closed months; wiped "
                            f"from the current month on)"))
        legend_rows.append((measure.core_column, "FIN", LEVEL_ROW, AGGREGATE_SUM,
                            f"{measure.config_field} ({measure.raw_column}): observed in the closed months of the "
                            f"extract, empty elsewhere (it is only known once the month closes) · SUM"))
    legend = pd.DataFrame(legend_rows, columns=["columna", "paso", "nivel", "como_agregar", "descripcion"])
    return legend[legend["columna"].isin(core.columns)].drop_duplicates("columna").reset_index(drop=True)


def check_core(core: pd.DataFrame, fine_table: pd.DataFrame, gap_rows: pd.DataFrame, legend: pd.DataFrame,
               blocks_present: list, forecast_total, configuration: Config, check_log: list) -> None:
    """Checks 1 to 5."""
    # [1] rows: the extract, the gaps, the extended horizon and the time_series universe
    origin_counts = core[ROW_ORIGIN_COLUMN].value_counts().to_dict()
    configuration.log_check(STEP_LABEL, check_log, "every original record and every synthetic row, once",
                            int((core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW).sum()) == len(fine_table)
                            and int((core[ROW_ORIGIN_COLUMN] == ROW_FROM_GAP).sum()) == len(gap_rows),
                            failure_detail=f"rows by origin {origin_counts} for {len(fine_table):,} fine rows and {len(gap_rows):,} gaps",
                            context=" · ".join(f"{origin} {count:,}" for origin, count in origin_counts.items()))

    # [2] the money of the pipeline reconciles with the extract (the time_series rows are apart)
    pipeline_rows = core[FINAL_UNIVERSE_COLUMN] == FINAL_UNIVERSE_PIPELINE
    reconciliation = {"s00_vencen_unidades": fine_table[S0_PIPELINE_UNITS_COLUMN].sum(),
                      "s00_vencen_usd": fine_table[S0_PIPELINE_USD_COLUMN].sum(),
                      "s02_vencen_unidades": fine_table[configuration.pipeline_units_col].sum(),
                      "s02_vencen_usd": fine_table[configuration.pipeline_usd_col].sum(),
                      "s00_renovadas_unidades": fine_table[S0_RENEWED_UNITS_COLUMN].sum(),
                      "s00_renovado_usd": fine_table[S0_RENEWED_USD_COLUMN].sum()}
    differences = {name: float(core.loc[pipeline_rows, name].sum() - total) for name, total in reconciliation.items()
                   if abs(core.loc[pipeline_rows, name].sum() - total) > configuration.money_tolerance}
    configuration.log_check(STEP_LABEL, check_log, "the money reconciles with the extract", not differences,
                            failure_detail=f"differences core − extract: {differences}",
                            context=f"${reconciliation['s00_vencen_usd']:,.0f} due · ${reconciliation['s00_renovado_usd']:,.0f} renewed")

    # [3] every row has the values of every block present (the first column of each block);
    #     a gap is not a unit of step 04, so the unit blocks are checked on the rows of the extract
    unit_names = {name for _, name, _, _ in UNIT_VALUES_STEP_04}
    first_columns = [block_specs[0][1] for block_specs in blocks_present if block_specs and block_specs[0][1] in core.columns
                     and not block_specs[0][1].startswith(("s11_", "s12_", "s20_", "s19_", "forecast_"))]
    series_columns = [name for name in first_columns if name not in unit_names]
    unit_columns = [name for name in first_columns if name in unit_names]
    raw_rows = core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW
    history_rows = core[ROW_ORIGIN_COLUMN].isin([ROW_FROM_RAW, ROW_FROM_GAP])       # an extended row may be a new series
    without_values = (int(core.loc[history_rows, series_columns].isna().any(axis=1).sum()) if series_columns else 0) + \
                     (int(core.loc[raw_rows, unit_columns].isna().any(axis=1).sum()) if unit_columns else 0)
    configuration.log_check(STEP_LABEL, check_log, "every row has the values of every block present", without_values == 0,
                            failure_detail=f"{without_values:,} rows without some block's values",
                            context=f"{len(blocks_present)} blocks")

    # [4] the legend covers the core
    undescribed = [column_name for column_name in core.columns if column_name not in set(legend["columna"])]
    configuration.log_check(STEP_LABEL, check_log, "the legend describes every column of the core", not undescribed,
                            failure_detail=f"columns without legend: {undescribed}",
                            context=f"{len(core.columns)} columns")

    # [5] the core answers the totals: summed by year and origin it equals sff_forecast_total
    if forecast_total is None:
        configuration.log_not_evaluated(STEP_LABEL, check_log, "the core summed by year and origin = sff_forecast_total",
                                        "step 21 has not run")
        return
    parts = forecast_total[forecast_total["origen"] != TOTAL_ORIGIN_TOTAL].set_index(["ano", "origen"])
    summed = core.groupby([FINAL_YEAR_COLUMN, FINAL_ORIGIN_COLUMN])[["forecast_to_renew_USD", "forecast_renewed_USD"]].sum()
    summed.index = summed.index.set_names(["ano", "origen"])
    compared = parts.join(summed, how="left").fillna(0.0)
    mismatched = compared[((compared["usd_vence"] - compared["forecast_to_renew_USD"]).abs() > configuration.money_tolerance)
                          | ((compared["usd_renovado"] - compared["forecast_renewed_USD"]).abs() > configuration.money_tolerance)]
    configuration.log_check(STEP_LABEL, check_log, "the core summed by year and origin = sff_forecast_total (the total comes from the core)",
                            mismatched.empty, failure_detail=f"{len(mismatched)} year × origin differ",
                            context=f"{len(compared)} year × origin compared",
                            examples=mismatched.reset_index()[["ano", "origen", "usd_vence", "forecast_to_renew_USD", "usd_renovado",
                                                               "forecast_renewed_USD"]])


def log_core_answers(core: pd.DataFrame, configuration: Config) -> None:
    """The business questions answered by SUMMING the core: renewed and pipeline by year, state and origin."""
    current_year = configuration.calendar_boundaries()["current"].year
    answers = (core[core[FINAL_YEAR_COLUMN] >= current_year]
               .groupby([FINAL_YEAR_COLUMN, FINAL_STATE_COLUMN, FINAL_ORIGIN_COLUMN])[["forecast_to_renew_USD", "forecast_renewed_USD"]].sum()
               .reset_index())
    answers = answers[(answers["forecast_to_renew_USD"] != 0) | (answers["forecast_renewed_USD"] != 0)]
    configuration.logger.doc(f"[{STEP_LABEL}] the questions answered by SUMMING the core (forecast_renewed_USD: renewed or revenue, "
                             f"real or expected · forecast_to_renew_USD: pipeline), by year, state and origin:")
    configuration.show_table(answers)


def log_core_report(core: pd.DataFrame, dimension: pd.DataFrame, configuration: Config) -> None:
    """Action 9: one series as it looks in the two tables, and how to read them in Power BI."""
    largest_series = core.groupby("s03_fs_id")["s00_vencen_usd"].sum().idxmax()
    configuration.log_action(STEP_LABEL, 9, f"the series with the most money, '{largest_series}': its row of sff_forecast_series "
                                            f"and its last {configuration.core_months_shown} closed months in sff_nucleo:")
    shown = ["s03_fs_id", "s05_ruta", "s06_n_propio", "s06_tasa_propia", "s08_stage3_id", "s08_stage3_support", "s09_tasa_estimada",
             "s09_nivel_riesgo", "s19_exam_mae_pp", "s19_raw_mae_pp", "s19_exam_coverage", "s20_confidence"]
    configuration.show_table(dimension[dimension["s03_fs_id"] == largest_series][[column for column in shown if column in dimension.columns]])
    series_rows = core[(core["s03_fs_id"] == largest_series) & core["s02_rol"].isin([ROLE_TRAIN, ROLE_TEST])]
    shown_columns = [configuration.period_col, ROW_ORIGIN_COLUMN, "s02_rol", "s00_vencen_unidades", "s02_renovadas_unidades"]
    configuration.show_table(series_rows.tail(configuration.core_months_shown)[[column for column in shown_columns if column in core.columns]])
    configuration.logger.doc(f"[{STEP_LABEL}] in Power BI: relate sff_forecast_series[s03_fs_id] 1 → n sff_nucleo[s03_fs_id]; a "
                             f"slicer on the dimension selects forecast series (by id, confidence, risk level, stage, exam…) and "
                             f"filters their rows. A rate is a measure (a ratio of sums), never a column. Measures to create:")
    configuration.show_table(pd.DataFrame([
        ("Tasa renovación (meses cerrados)",
         "DIVIDE(CALCULATE(SUM(sff_nucleo[s02_renovadas_unidades]), sff_nucleo[s02_rol] IN {\"entrenamiento\", \"examen\"}), "
         "CALCULATE(SUM(sff_nucleo[s00_vencen_unidades]), sff_nucleo[s02_rol] IN {\"entrenamiento\", \"examen\"}))"),
        ("Vence USD", "SUM(sff_nucleo[s00_vencen_usd])"),
        ("Renovado (actual + forecast)", "SUM(sff_nucleo[forecast_renewed_USD])  — segmentar por forecast_year, forecast_status, forecast_pipeline_source"),
        ("Pipeline", "SUM(sff_nucleo[forecast_to_renew_USD])  — segmentar por forecast_year, forecast_pipeline_source"),
        ("Examen: dentro del intervalo", "DIVIDE(SUM(sff_forecast_series[s19_exam_in_band]), SUM(sff_forecast_series[s19_exam_predictions]))"),
        ("Examen: WAPE framework / raw", "DIVIDE(SUM(sff_forecast_series[s19_exam_abs_err_units]), SUM(sff_forecast_series[s19_exam_real_units]))"
                                        "  ·  the same with s19_raw_abs_err_units")], columns=["medida", "DAX"]))


