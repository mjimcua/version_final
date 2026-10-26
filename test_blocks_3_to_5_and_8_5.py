"""
test_blocks_3_to_5_and_8_5.py — Steps 3.1 to 8.5 on the synthetic: the uplift, its backtest, the customer states, the future
pipeline and its exam, the forecast in money (with its prediction rows aggregated) and the final validation.

    python test_blocks_3_to_5_and_8_5.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from prediction import judged_horizon
from config import (ACTIVE_FLAG_VALUES, CALENDAR_ROLE_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_REAL, ROLE_PROJECTION, TRUTH_ROLES,
                    UNITS_AT_DUE_COLUMN, UNITS_TODAY_COLUMN, WITH_DATES_COLUMN, region_columns_of)
from block_4_customer_states import StateMaturation, maturation_by_region, states_above_final
from test_helpers import check, console_of, finish, synthetic_with, step_section


def test_acquisition_simulation() -> None:
    print("B · the acquisition of the window, simulated per acquisition value (hand-made fine table)")
    from block_5_future_pipeline import FuturePipeline, simulate_acquisitions
    configuration = synthetic_with(acquisition_column="newcust", acquisition_values=["pure", "not_pure"],
                                   acquisition_discount=0.4)
    rows = []
    for due_month in pd.period_range("2025-01", "2027-08", freq="M"):
        acquired_in = due_month - 12
        for kind, units in (("pure", 100.0), ("not_pure", 40.0)):
            growth = 1.2 if acquired_in.year == 2026 else 1.0             # acquisitions of 2026 are 20 % above 2025
            rows.append({"period": due_month, "region": "EU", "product": "A", "dormant": False, "softcancel": False,
                         "no_instalado": False, "autorenew": False, "channel": "web", "newcust": kind,
                         "total_tr_units": units * growth, "total_tr_usd": units * growth * 10.0})
    fine = pd.DataFrame(rows)
    fine["rol"] = "entrenamiento"
    simulated = simulate_acquisitions(FuturePipeline(fine, configuration), list(pd.period_range("2026-09", "2026-12", freq="M")))
    check(len(simulated) == 8 and set(simulated["newcust"]) == {"pure", "not_pure"},
          "one line per acquisition value and month: 2 lines × 4 months")
    pure_september = simulated[(simulated["newcust"] == "pure") & (simulated["period"].astype(str) == "2027-09")].iloc[0]
    check(abs(pure_september["total_tr_units"] - 120.0) < 1e-9 and abs(pure_september["total_tr_usd"] - 1200.0) < 1e-9,
          "pure, acquired 2026-09: 100 units of 2025-09 × level 1.2 = 120 units × $10 = $1,200, due 2027-09")
    check((simulated["discount_pct"] == 0.4).all() and (~simulated[["dormant", "softcancel", "no_instalado", "autorenew"]].astype(bool)).all().all(),
          "an acquisition carries the acquisition discount (0.4) and every timevarying flag off")


def test_steps_15_to_18() -> None:
    print("A · uplift, its backtest, the forecast and the validation")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    for label, count in (("3.1", 6), ("3.2", 1), ("3.3", 3), ("4.1", 2), ("5.1", 3), ("7.2", 10)):
        check(f"{count} checks: {count} ok" in step_section(console, label), f"the checks of step {label} pass")
    check("7 checks:" in step_section(console, "8.5") and "0 failed" in step_section(console, "8.5"),
          "the final validation passes (warnings allowed)")

    cells = results["uplift_cells"]
    fine = results["fine_table"]
    renewers = fine[fine["rol"].isin(["entrenamiento", "examen"]) & (fine["total_renewed_units"] > 0)]
    # the window of the uplift: the last uplift_window_months closed months (the synthetic has no price increase)
    current_month = configuration.calendar_boundaries()["current"]
    renewers = renewers[renewers["period"] >= current_month - configuration.uplift_window_months]
    one_cell = cells[cells["uplift_origen"] == "propia"].iloc[0]
    rows = renewers[renewers["uplift_cell_id"] == one_cell["uplift_cell_id"]]
    expected = rows["total_renewed_usd"].sum() / rows["total_tr_usd_renewed"].sum()
    check(abs(one_cell["uplift"] - expected) < 1e-9,
          "the uplift of a cell = Σ renewed USD / Σ (the USD that was falling due of the contracts that renewed), "
          "over the window of the last uplift_window_months closed months")
    homogeneity = results["uplift_homogeneity"]
    same_cell = homogeneity[homogeneity["uplift_cell_id"] == one_cell["uplift_cell_id"]].iloc[0]
    check(abs(same_cell["uplift_aproximado"] - same_cell["uplift_exacto"] * same_cell["ratio_seleccion"]) < 1e-9,
          "homogeneity: the approximate uplift = the exact one × the selection ratio (the old base hid that factor)")
    check("LICENCE BY LICENCE, the isolated renewals: mean ratio" in console and "std deviation between licences" in console
          and "histogram · the value of the isolated renewals by renewal ratio" in console,
          "the revaluation report reads the isolated renewals licence by licence: std deviation and ratio bands")
    check("the maturation of every state by" in step_section(console, "4.1"),
          "step 4.1 shows the maturation of every state by region, as a table and a chart")
    check("LO QUE RENUEVA CADA COMBINACIÓN DE ESTADOS DEL CLIENTE" in console and "REPARTO DE ESTADOS DEL CLIENTE" in console,
          "step 7.2 prints what every combination of states renews, and the distribution of the states (today, final, adjustment)")
    check("between SERIES" in console and "histogram · every series in the distribution" in console
          and "█" in console and "month by month (24 closed" in console,
          "the revaluation report prints the claim, the histogram and the monthly path on screen (action 9)")
    check(results["uplift_verdict"]["via_usada_con_descuento"] == "estadistica",
          "in the synthetic the contract rule loses (observed 1.50 vs rule 1.67 at 40 %): the statistical path is used")
    comparison = results["uplift_backtest"]
    check({"estadistica", "contrato", "contrato_neutras"} <= set(comparison[comparison["filas"] == "descuento conocido"]["via"])
          and {"conocido · neutras", "conocido · con estado", "conocido · sin subida"} <= set(comparison["filas"]),
          "the exam of the uplift judges three paths on the known discount, split by states and by increase "
          "(this synthetic has no increase: only the rows without one)")
    # the contract on the neutral rows: the rule where the discount is known and no negative state is on, the
    # statistical uplift on the rows with a state and where the discount is unknown
    from block_7_exam_and_assembly import uplift_of_rows
    from config import has_no_negative_state
    future = fine[fine["rol"] == "proyeccion"].copy()
    neutral_path = uplift_of_rows(future.copy(), cells, {"via_usada_con_descuento": "contrato_neutras"}, configuration)
    statistical_path = uplift_of_rows(future.copy(), cells, {"via_usada_con_descuento": "estadistica"}, configuration)
    known = future[configuration.discount_value_column].notna()
    neutral = has_no_negative_state(future, configuration)
    rule = 1 / (1 - future[configuration.discount_value_column].clip(upper=configuration.contract_discount_cap))
    check((neutral_path.loc[known & neutral, "via_uplift"] == "contrato").all()
          and np.allclose(neutral_path.loc[known & neutral, "uplift"], rule[known & neutral])
          and (neutral_path.loc[~(known & neutral), "via_uplift"] == "estadistica").all()
          and np.allclose(neutral_path.loc[~(known & neutral), "uplift"], statistical_path.loc[~(known & neutral), "uplift"]),
          "contrato_neutras: the rule on the neutral rows with a known discount, the statistical uplift on every other row")
    # where the uplift of a row comes from: its cell; a cell never seen takes its mandatory cell's (or the global one)
    unseen = future.head(3).copy()
    unseen["uplift_cell_id"] = "a cell never seen"
    with_origin = uplift_of_rows(pd.concat([future.head(3), unseen], ignore_index=True), cells,
                                 {"via_usada_con_descuento": "estadistica"}, configuration)
    check((with_origin["origen_uplift"].iloc[:3] == "celda").all()
          and with_origin["origen_uplift"].iloc[3:].isin(["celda_mandatory", "global"]).all()
          and (neutral_path.loc[known & neutral, "origen_uplift"] == "contrato").all(),
          "origen_uplift: celda where the row's cell has an uplift, celda_mandatory / global for a cell never seen, "
          "contrato where the rule applies (the column s3_1_origen_uplift of the core)")

    forecast = results["forecast"]["forecast"]
    extended = forecast[forecast["origen_pipeline"].isin(["proyectada", "simulada"])]
    check(set(extended["period"].astype(str)) == {"2027-09", "2027-10", "2027-11", "2027-12"},
          "the simulation window (2026-09..2026-12, the current month included) falls due in 2027-09..2027-12")
    september_2026 = forecast[(forecast["origen_pipeline"] == "real") & (forecast["period"].astype(str) == "2026-09")]
    projected_september = extended[(extended["origen_pipeline"] == "proyectada") & (extended["period"].astype(str) == "2027-09")]
    check(abs(projected_september["total_tr_units"].sum() - september_2026["esperado_unidades"].sum()) < 1e-6
          and abs(projected_september["total_tr_usd"].sum() - september_2026["esperado_usd"].sum()) < 1e-6,
          "what is expected to renew in 2026-09 is the pipeline of 2027-09: units = renewed units, value = renewed USD")
    check((projected_september.loc[projected_september["fila_maduracion"] == 0, "discount_pct"] == 0).all(),
          "a renewal falls due again with discount 0 and the same dims")
    check(np.allclose(forecast["esperado_usd"], forecast["vencen_usd_al_vencer"] * forecast["tasa"] * forecast["uplift"])
          and np.allclose(forecast["esperado_usd_hoy"], forecast["total_tr_usd"] * forecast["tasa"] * forecast["uplift"]),
          "expected USD of a row = USD due at due date × rate × uplift; with today's states = USD due today × rate × uplift")
    matured = forecast[forecast["fila_maduracion"] == 1]
    check(len(matured) > 0 and (matured["total_tr_units"] == 0).all() and (matured["vencen_unidades_al_vencer"] > 0).all()
          and abs(forecast["vencen_unidades_al_vencer"].sum() - forecast["total_tr_units"].sum()) < 1e-6,
          "the maturation creates prediction rows with no data (units due 0) that receive units at due date; the units "
          "at due date of the whole future add up to its units due")
    check((forecast["esperado_usd_bajo"] <= forecast["esperado_usd"] + 1e-9).all()
          and (forecast["esperado_usd"] <= forecast["esperado_usd_alto"] + 1e-9).all(), "the band of every row contains its expected value")
    by_month = results["forecast"]["by_month"]
    check((by_month["banda_lineal_baja"] <= by_month["banda_cuadratura_baja"] + 1e-6).all()
          and (by_month["banda_cuadratura_alta"] <= by_month["banda_lineal_alta"] + 1e-6).all(),
          "by month, the quadrature band is inside the linear one (independent errors partly cancel)")
    check(judged_horizon(1, [1, 6]) == 1 and judged_horizon(3, [1, 6]) == 6 and judged_horizon(9, [1, 6]) == 6,
          "a horizon takes the band of the judged horizon above it, or the last")
    core = results["core"]
    check(abs(core["s7_2_esperado_usd"].sum() - forecast["esperado_usd"].sum()) < 0.01,
          "the core carries the forecast of every future row: Σ s7_2_esperado_usd = Σ of the forecast")
    report = open(f"{configuration.output_folder}/informe_sff.md", encoding="utf-8").read()
    check("## 7 · B7 · El forecast en dinero" in report and "La validación final de la cadena" in report,
          "the report tells the forecast (B7) and the validation (B8)")


def test_the_level_of_the_acquisitions_is_aggregate() -> None:
    print("L · the level of the acquisitions is aggregate: a small group that grew does not multiply its month")
    from block_5_future_pipeline import FuturePipeline, acquisition_events, acquisition_levels, simulate_acquisitions
    configuration = synthetic_with(acquisition_column="newcust", acquisition_values=["pure"], acquisition_discount=0.4)
    rows = []
    for due_month in pd.period_range("2025-01", "2027-08", freq="M"):
        sold_in = due_month - 12
        # a big group, stable: 1,000 a month · a small one: 1 a month in 2025, 100 in September 2025, 50 a month in 2026
        small = 100.0 if str(sold_in) == "2025-09" else (50.0 if sold_in.year == 2026 else 1.0)
        for channel, units in (("web", 1000.0), ("tele", small)):
            rows.append({"period": due_month, "region": "EU", "product": "A", "dormant": False, "softcancel": False,
                         "no_instalado": False, "autorenew": False, "channel": channel, "newcust": "pure",
                         "total_tr_units": units, "total_tr_usd": units * 10.0})
    fine = pd.DataFrame(rows)
    fine["rol"] = "entrenamiento"
    simulated = simulate_acquisitions(FuturePipeline(fine, configuration), [pd.Period("2026-09", freq="M")])
    level = (3000.0 + 150.0) / (3000.0 + 3.0)
    check(abs(simulated["total_tr_units"].sum() - (1000.0 + 100.0) * level) < 1e-6,
          "September: (1,000 + 100 sold a year before) × the level of the portfolio (3,150 / 3,003) = 1,154 units, not "
          "1,000 + 100 × 50 = 6,000 (the level of the small group alone)")
    events, dims, _ = acquisition_events(fine, configuration)
    by_region = acquisition_levels(events, dims, synthetic_with(acquisition_column="newcust", acquisition_values=["pure"],
                                                                acquisition_level_dims=["channel"], acquisition_level_min_units=100))
    origins = by_region.set_index("clave_nivel")["origen"].to_dict()
    check(origins == {"web": "propio", "tele": "cartera"},
          "per channel: web has its own level (3,000 units a year before ≥ 100); tele (3) takes the level of the portfolio")


def test_a_projected_renewal_is_a_retention() -> None:
    print("P · a projected renewal: no longer an acquisition, its states neutral")
    configuration = synthetic_with()
    results = {}
    console_of(lambda: results.update(run(configuration)))
    forecast = results["forecast"]["forecast"]
    # the projected rows the window creates (the prediction rows of their maturation carry states: they are the
    # units that will take one before falling due)
    projected = forecast[(forecast["origen_pipeline"] == "proyectada") & (forecast["fila_maduracion"] == 0)]
    check(len(projected) > 0 and not projected[configuration.acquisition_column].isin(configuration.acquisition_values).any()
          and (projected[configuration.acquisition_column] == configuration.renewed_acquisition_value).sum() > 0,
          "the renewal of an acquisition falls due next year as a retention (renewed_acquisition_value): not counted "
          "again as an acquisition")
    check(all((~projected[column_name].isin(ACTIVE_FLAG_VALUES)).all()
              for column_name in configuration.structural_timevarying_dims if column_name in projected.columns),
          "the timevarying states of a projected renewal are neutral (nothing is known yet of them)")


def hand_made_configuration(**overrides):
    """The synthetic Config for the hand-made tests of the box."""
    return synthetic_with(**overrides)


def hand_made_history(configuration, combinations: list, months: int = 12) -> pd.DataFrame:
    """Closed rows of one group: every month, every (states, units, renewed) of `combinations`."""
    base_row = {column_name: "x" for column_name in configuration.rate_series_columns + configuration.business_mandatory_dims}
    base_row.update({state: 0 for state in configuration.structural_timevarying_dims})
    last_closed = configuration.calendar_boundaries()["current"] - 1
    rows = []
    for month in pd.period_range(last_closed - (months - 1), last_closed, freq="M"):
        for states_on, units, renewed in combinations:
            rows.append({**base_row, **states_on, configuration.period_col: month, CALENDAR_ROLE_COLUMN: TRUTH_ROLES[0],
                         configuration.pipeline_units_col: units, configuration.pipeline_usd_col: units * 10,
                         configuration.renewed_units_col: renewed, configuration.renewed_usd_col: renewed * 10})
    return pd.DataFrame(rows)


def hand_made_units(configuration, target, rows: list, with_dates: bool = True) -> pd.DataFrame:
    """The input of the box: one row per (states, units) in the target month."""
    base_row = {column_name: "x" for column_name in configuration.rate_series_columns + configuration.business_mandatory_dims}
    base_row.update({state: 0 for state in configuration.structural_timevarying_dims})
    return pd.DataFrame([{**base_row, **states_on, configuration.period_col: target, UNITS_TODAY_COLUMN: units,
                          WITH_DATES_COLUMN: with_dates} for states_on, units in rows])


def at_due(table: pd.DataFrame, combination: str) -> float:
    return float(table.loc[table["combinacion"] == combination, UNITS_AT_DUE_COLUMN].sum())


def test_the_box_of_the_states() -> None:
    print("M · the box of stage 4.1: hand-made, two states, every number by hand (units only)")
    configuration = hand_made_configuration()
    softcancel, dormant = "softcancel", "dormant"
    last_closed = configuration.calendar_boundaries()["current"] - 1
    target = last_closed + 10                                           # far away: today 10 % softcancel, no dormant yet
    # the final mix: 80 % neutral (renews 70 %), 15 % softcancel (10 %), 5 % dormant (50 %)
    history = hand_made_history(configuration, [({}, 80.0, 56.0), ({softcancel: 1}, 15.0, 1.5), ({dormant: 1}, 5.0, 2.5)])
    box = StateMaturation(history, last_closed, configuration)
    table = box.mature(hand_made_units(configuration, target, [({}, 90.0), ({softcancel: 1}, 10.0)]))
    check(abs(at_due(table, "neutral") - 80.0) < 1e-9 and abs(at_due(table, softcancel) - 15.0) < 1e-9
          and abs(at_due(table, dormant) - 5.0) < 1e-9,
          "gaps: softcancel 15 % − 10 % = 5 points, dormant 5 % − 0 = 5 points: 10 of the 90 neutral units move "
          "(neutral 90 → 80, softcancel 10 → 15, dormant 0 → 5)")
    check(abs(table[UNITS_AT_DUE_COLUMN].sum() - table[UNITS_TODAY_COLUMN].sum()) < 1e-9,
          "the box moves units, it does not create them: Σ at due date = Σ today")
    identity = StateMaturation(history, last_closed, hand_made_configuration(distribute_marks=False)).mature(
        hand_made_units(configuration, target, [({}, 90.0), ({softcancel: 1}, 10.0)]))
    check(np.allclose(identity[UNITS_AT_DUE_COLUMN], identity[UNITS_TODAY_COLUMN]),
          "the identity box (distribute_marks off) moves nothing: the maturation is 0")
    above = box.mature(hand_made_units(configuration, target, [({}, 90.0), ({softcancel: 1}, 30.0)]))
    check(abs(at_due(above, softcancel) - 30.0) < 1e-9 and abs(at_due(above, dormant) - 120 * 0.05) < 1e-9,
          "a state only grows: softcancel already above its final share (25 % vs 15 %) moves nothing back; only "
          "dormant's gap moves (5 % of 120 = 6 units)")
    saturated_history = hand_made_history(configuration, [({}, 10.0, 7.0), ({softcancel: 1}, 50.0, 5.0), ({dormant: 1}, 40.0, 20.0)])
    saturated = StateMaturation(saturated_history, last_closed, configuration).mature(
        hand_made_units(configuration, target, [({}, 7.0), ({softcancel: 1}, 93.0)]))
    check(at_due(saturated, "neutral") >= -1e-12,
          "when the states lack more than all that is neutral, every neutral unit moves and no more (neutral ≥ 0)")
    # a fresh cohort: dormant today 40 % (final 15 % with dormant+softcancel), softcancel below its final share
    fresh_history = hand_made_history(configuration, [({}, 75.0, 37.5), ({softcancel: 1}, 10.0, 5.0), ({dormant: 1}, 5.0, 2.5),
                                                      ({dormant: 1, softcancel: 1}, 10.0, 5.0)])
    fresh = StateMaturation(fresh_history, last_closed, configuration).mature(
        hand_made_units(configuration, last_closed + 11, [({}, 60.0), ({dormant: 1}, 40.0)]))
    check(abs(at_due(fresh, dormant) + at_due(fresh, "dormant+softcancel") - 40.0) < 1e-9
          and abs(at_due(fresh, softcancel) - 10.0) < 1e-9,
          "a state above its final share (dormant 40 % vs 15 %) does not grow through dormant+softcancel; softcancel "
          "alone still reaches its 10 %")
    check(abs(fresh.drop_duplicates("_grupo")["cuota_final_dormant"].iloc[0] - 0.15) < 1e-12,
          "the box says the final share it aimed at: dormant 15 % (dormant 5 % + dormant+softcancel 10 %)")


def test_the_maturation_by_region() -> None:
    print("M2 · the maturation by region of step 4.1: hand-made, an undated state, every number by hand")
    configuration = hand_made_configuration()
    region = region_columns_of(configuration)[0]
    dormant = "dormant"
    last_closed = configuration.calendar_boundaries()["current"] - 1
    # the past: region A ends 10 % dormant, region B 20 %
    history_a = hand_made_history(configuration, [({}, 90.0, 63.0), ({dormant: 1}, 10.0, 5.0)]).assign(**{region: "A"})
    history_b = hand_made_history(configuration, [({}, 80.0, 56.0), ({dormant: 1}, 20.0, 10.0)]).assign(**{region: "B"})
    # today, 3 months ahead: region A 5 % dormant (half its final share), region B 20 % (all of it)
    future = hand_made_history(configuration, [({}, 95.0, 0.0), ({dormant: 1}, 5.0, 0.0)], months=1).assign(**{region: "A"})
    future = pd.concat([future, hand_made_history(configuration, [({}, 80.0, 0.0), ({dormant: 1}, 20.0, 0.0)], months=1)
                        .assign(**{region: "B"})], ignore_index=True)
    future[configuration.period_col] = last_closed + 3
    future[CALENDAR_ROLE_COLUMN] = ROLE_PROJECTION
    fine = pd.concat([history_a, history_b, future], ignore_index=True)
    box = StateMaturation(fine, last_closed, configuration)
    table = maturation_by_region(fine, box, configuration)
    today = table[(table["estado"] == dormant) & (table["vista"] == "hoy") & (table["meses_antes"] == 3)].set_index("region")
    check(abs(today.loc["A", "cuota_final"] - 0.10) < 1e-12 and abs(today.loc["B", "cuota_final"] - 0.20) < 1e-12,
          "the final share of every region comes from its closed months (A 10 %, B 20 %)")
    check(abs(today.loc["A", "visible_del_final"] - 0.5) < 1e-12 and abs(today.loc["B", "visible_del_final"] - 1.0) < 1e-12,
          "an undated state is seen TODAY: 3 months ahead, A shows half its final dormant (5 % of 10 %), B all of it")
    check(abs(today.loc["cartera", "visible_del_final"] - (25 / 200) / (30 / 200)) < 1e-12,
          "the portfolio line is the same reading on every region together")
    check(set(table["vista"]) == {"hoy"}, "with no maturation columns there is no past curve, only the view of today")


def test_the_maturation_is_valued_at_the_average_price() -> None:
    print("N · the maturation valued (step 7.2): what leaves a neutral row leaves at its average price")
    from block_7_exam_and_assembly import apply_the_maturation
    configuration = hand_made_configuration()
    softcancel, dormant = "softcancel", "dormant"
    last_closed = configuration.calendar_boundaries()["current"] - 1
    target = last_closed + 10
    history = hand_made_history(configuration, [({}, 80.0, 56.0), ({softcancel: 1}, 15.0, 1.5), ({dormant: 1}, 5.0, 2.5)])
    box = StateMaturation(history, last_closed, configuration)
    base_row = {column_name: "x" for column_name in configuration.rate_series_columns + configuration.business_mandatory_dims}
    base_row.update({state: 0 for state in configuration.structural_timevarying_dims})
    future = pd.DataFrame([{**base_row, **states_on, configuration.period_col: target, PIPELINE_ORIGIN_COLUMN: PIPELINE_REAL,
                            configuration.pipeline_units_col: units, configuration.pipeline_usd_col: usd, "uplift_cell_id": "c"}
                           for states_on, units, usd in (({}, 90.0, 900.0), ({softcancel: 1}, 10.0, 100.0))])
    table = box.mature(future.assign(**{UNITS_TODAY_COLUMN: future[configuration.pipeline_units_col], WITH_DATES_COLUMN: True}))
    rows, arriving = apply_the_maturation(future, table, box, configuration)
    neutral = rows[rows[softcancel] == 0].iloc[0]
    check(abs(neutral["vencen_unidades_al_vencer"] - 80.0) < 1e-9 and abs(neutral["vencen_usd_al_vencer"] - 800.0) < 1e-9
          and neutral[configuration.pipeline_units_col] == 90.0,
          "the neutral row keeps 80 units and $800 at due date; its data (90 units due) does not change")
    by_state = arriving.set_index(arriving[softcancel].astype(str) + arriving[dormant].astype(str))
    check(len(arriving) == 2 and np.allclose(arriving["vencen_unidades_al_vencer"], 5.0)
          and np.allclose(arriving["vencen_usd_al_vencer"], 50.0) and (arriving[configuration.pipeline_units_col] == 0).all()
          and (arriving["fila_maduracion"] == 1).all(),
          "two prediction rows (softcancel, dormant) receive 5 units each at the neutral row's price ($10): $50 each; "
          "their data (units due) is 0")
    check(abs(rows["vencen_usd_al_vencer"].sum() + arriving["vencen_usd_al_vencer"].sum() - 1000.0) < 1e-9,
          "the USD due is conserved: $1,000 today, $1,000 at due date")
    check(len(by_state) == 2, "every prediction row carries the states of its combination")
    # the extended rows (no maturation columns): the historical final share, never the curve's p(h)
    curve_configuration = hand_made_configuration(maturation_dims=["softcancel"])
    columns = curve_configuration.maturation_columns_of(softcancel)
    dated_history = hand_made_history(curve_configuration, [({}, 80.0, 56.0), ({softcancel: 1}, 20.0, 2.0)])
    for column_name in columns:
        dated_history[column_name] = 0.0
    dated_history.loc[dated_history[softcancel] == 1, columns[-1]] = 10.0     # half in the state 12 months before or more
    dated_history.loc[dated_history[softcancel] == 1, columns[0]] = 10.0      # half in the due month
    dated_box = StateMaturation(dated_history, last_closed, curve_configuration)
    probability = (20.0 - 10.0) / (100.0 - 10.0)
    far = last_closed + 13
    real_only = dated_box.mature(hand_made_units(curve_configuration, far, [({}, 100.0)], with_dates=True))
    projected_only = dated_box.mature(hand_made_units(curve_configuration, far, [({}, 100.0)], with_dates=False))
    check(abs(at_due(real_only, softcancel) / 100.0 - probability) < 1e-9,
          "a row of the extract with nothing seen 13 months before: the curve, p(12) = (20 − 10) / (100 − 10) = 11.1 %")
    check(abs(at_due(projected_only, softcancel) / 100.0 - 0.20) < 1e-9,
          "a projected row (no columns, nothing can be seen): the historical final share, 20 %, not p(12) = 11.1 %")


def test_the_detector() -> None:
    print("O · the detector of states above their final share, by hand")
    configuration = hand_made_configuration()
    softcancel, dormant = "softcancel", "dormant"
    last_closed = configuration.calendar_boundaries()["current"] - 1
    history = hand_made_history(configuration, [({}, 80.0, 56.0), ({softcancel: 1}, 15.0, 1.5), ({dormant: 1}, 5.0, 2.5)])
    box = StateMaturation(history, last_closed, configuration)
    month = pd.DataFrame([{configuration.period_col: last_closed + 10, "vencen_unidades": 120.0, "vencen_usd": 1200.0,
                           **{f"{state}_{moment}": 0.0 for state in box.states for moment in ("hoy", "final")},
                           f"{softcancel}_hoy": 0.25, f"{softcancel}_final": 0.15, f"{dormant}_hoy": 0.0, f"{dormant}_final": 0.05}])
    detector = states_above_final(month, box, configuration)
    softcancel_row = detector[detector["estado"] == softcancel].iloc[0]
    check(abs(softcancel_row["exceso_pp"] - 10.0) < 1e-9 and abs(softcancel_row["unidades_exceso"] - 12.0) < 1e-9
          and abs(softcancel_row["impacto_usd_aprox"] - 12 * 10.0 * (0.7 - 0.1)) < 1e-9,
          "softcancel today 25 % vs final 15 % = +10 pp, 12 units; if they renew as neutral (70 % instead of 10 %, $10 "
          "a unit) the forecast lacks 12 × $10 × 0.6 = $72")
    check(detector[detector["estado"] == dormant]["exceso_pp"].iloc[0] < 0,
          "dormant below its final share (0 % today vs 5 %) is no excess: the states still to come")


def test_the_price_by_states() -> None:
    print("O · the price by states: what the renewers of every combination paid over their cell, by hand")
    from block_3_price import state_factor_of_rows, state_uplift_factors
    from config import UPLIFT_CELL_ID_COLUMN
    configuration = synthetic_with()
    base_row = {state: 0 for state in configuration.structural_timevarying_dims}
    history = pd.DataFrame([
        {**base_row, UPLIFT_CELL_ID_COLUMN: "c", configuration.renewed_units_col: 50.0, "_denominador": 1000.0,
         "_numerador_uplift": 1100.0},                                            # neutral: pays 1.10 its base
        {**base_row, "softcancel": 1, UPLIFT_CELL_ID_COLUMN: "c", configuration.renewed_units_col: 50.0,
         "_denominador": 100.0, "_numerador_uplift": 80.0},                        # softcancel: 0.80
        {**base_row, "dormant": 1, UPLIFT_CELL_ID_COLUMN: "c", configuration.renewed_units_col: 10.0,
         "_denominador": 100.0, "_numerador_uplift": 50.0}])                      # dormant: few renewers
    cell_uplift = 1230.0 / 1200.0                                                 # the cell, all of them together
    cells = pd.DataFrame({UPLIFT_CELL_ID_COLUMN: ["c"], "uplift": [cell_uplift]})
    factors = state_uplift_factors(history, cells, configuration).set_index("combinacion")
    check(abs(factors.loc["neutral", "factor"] - 1100.0 / (1000.0 * cell_uplift)) < 1e-12
          and abs(factors.loc["softcancel", "factor"] - 80.0 / (100.0 * cell_uplift)) < 1e-12,
          "neutral 1100 / (1000 × 1.025) = 1.073, softcancel 80 / (100 × 1.025) = 0.780: what they paid over their cell")
    check(factors.loc["dormant", "factor"] == 1.0
          and abs(factors.loc["dormant", "factor_medido"] - 50.0 / (100.0 * cell_uplift)) < 1e-12,
          f"dormant with 10 renewed units (< uplift_floor {configuration.uplift_floor:.0f}): measured but not applied, factor 1")
    future = pd.DataFrame([{**base_row}, {**base_row, "softcancel": 1}, {**base_row, "softcancel": 1, "dormant": 1}])
    applied = state_factor_of_rows(future, factors.reset_index(), configuration).to_numpy()
    check(abs(applied[1] - factors.loc["softcancel", "factor"]) < 1e-12 and applied[2] == 1.0,
          "every row takes the factor of its combination; a combination never measured (dormant+softcancel) takes 1")


def test_the_exam_of_the_acquisitions() -> None:
    print("X · the exam of the acquisitions: the box rebuilt at every origin, against what was sold")
    from block_5_future_pipeline import FuturePipeline, acquisition_exam
    configuration = synthetic_with(acquisition_column="newcust", acquisition_values=["pure"], acquisition_discount=0.4,
                                   acquisition_exam_origins=6)
    rows = []
    for due_month in pd.period_range("2025-01", "2027-08", freq="M"):
        sold_in = due_month - 12
        units = 120.0 if sold_in.year == 2026 else 100.0              # the sales of 2026 are 20 % above 2025
        rows.append({"period": due_month, "region": "EU", "product": "A", "dormant": False, "softcancel": False,
                     "no_instalado": False, "autorenew": False, "channel": "web", "newcust": "pure",
                     "total_tr_units": units, "total_tr_usd": units * 10.0})
    fine = pd.DataFrame(rows)
    fine["rol"] = "entrenamiento"
    exam = acquisition_exam(FuturePipeline(fine, configuration))
    months = exam["months"]
    check(sorted(months["origen"].astype(str).unique()) == [f"2026-0{n}" for n in range(3, 9)],
          "six origins, the last closed months whose sales the extract reaches: 2026-03 … 2026-08")
    late = months[months["origen"] >= pd.Period("2026-04", freq="M")]
    check(np.allclose(late["unidades_simuladas"], late["unidades_real"]),
          "from origin 2026-04 the level is read on months of 2026 only (1.2): the simulation is exact")
    first = months[(months["origen"] == pd.Period("2026-03", freq="M")) & (months["h"] == 0)].iloc[0]
    check(abs(first["unidades_simuladas"] - 100.0 * (100.0 + 120.0 + 120.0) / 300.0) < 1e-9,
          "origin 2026-03: the level of 2025-12 … 2026-02 over a year before (340 / 300) × the 100 sold a year before")
    check(np.allclose((months["unidades_ano_anterior"] - months["unidades_real"]).abs() / months["unidades_real"], 20.0 / 120.0),
          "the baseline (a year before) errs 20 / 120 = 16.7 % in every month of 2026")
    check(exam["read_after_origin"] == 0, "no month at or after an origin is read to simulate it")


def test_the_prediction_rows_are_aggregated() -> None:
    print("G · the prediction rows of the customer states: one per series, month and origin, exactly")
    from block_7_exam_and_assembly import aggregate_the_prediction_rows, expected_renewals
    from config import (DUE_UNITS_AT_DUE_COLUMN, DUE_USD_AT_DUE_COLUMN, MATURATION_ROW_COLUMN, SERIES_ID_COLUMN,
                             UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN)
    configuration = synthetic_with()

    class FixedBand:
        def band(self, rates, units_due):
            return rates["tasa"] - 0.1, rates["tasa"] + 0.1

    month = pd.Period("2027-01", freq="M")
    fine = pd.DataFrame({SERIES_ID_COLUMN: ["EU|A|soft", "EU|A|soft", "US|A|soft"], "period": [month] * 3,
                         PIPELINE_ORIGIN_COLUMN: [PIPELINE_REAL] * 3, UNIT_ID_COLUMN: ["EU|A|soft|2027-01"] * 2 + ["US|A|soft|2027-01"],
                         UPLIFT_CELL_ID_COLUMN: ["EU|0-10", "EU|10-20", "US|0-10"], "h": [4, 4, 4], "tecnica": ["T3_ma3"] * 3,
                         "tasa": [0.8, 0.8, 0.6], "uplift": [1.0, 1.5, 1.2], "uplift_bajo": [0.9, 1.4, 1.1],
                         "uplift_alto": [1.1, 1.6, 1.3], "discount": [np.nan, np.nan, np.nan],
                         configuration.pipeline_units_col: [0.0, 0.0, 0.0], configuration.pipeline_usd_col: [0.0, 0.0, 0.0],
                         DUE_UNITS_AT_DUE_COLUMN: [10.0, 30.0, 5.0], DUE_USD_AT_DUE_COLUMN: [100.0, 300.0, 50.0],
                         MATURATION_ROW_COLUMN: [1, 1, 1]})
    context = dict(forecast_units=pd.DataFrame({UNIT_ID_COLUMN: ["EU|A|soft|2027-01"], configuration.pipeline_units_col: [500.0]}),
                   rate_box=FixedBand())
    fine_valued = expected_renewals(fine.assign(tasa_baja=0.7, tasa_alta=0.9), configuration)
    aggregated = expected_renewals(aggregate_the_prediction_rows(fine, context, configuration), configuration)
    eu = aggregated[aggregated[SERIES_ID_COLUMN] == "EU|A|soft"].iloc[0]
    check(len(aggregated) == 2 and eu[DUE_USD_AT_DUE_COLUMN] == 400.0 and eu[DUE_UNITS_AT_DUE_COLUMN] == 40.0,
          "two series → two rows; EU sums its units and USD at due date (40 units, $400)")
    check(abs(eu["uplift"] - (100.0 * 1.0 + 300.0 * 1.5) / 400.0) < 1e-12 and eu[UPLIFT_CELL_ID_COLUMN] == "varias"
          and eu["tecnica"] == "T3_ma3" and eu["h"] == 4,
          "the uplift is the USD-weighted mean (1.375); the uplift cell, mixed, reads 'varias'; what agrees is kept")
    check(abs(aggregated["esperado_usd"].sum() - fine_valued["esperado_usd"].sum()) < 1e-9
          and abs(aggregated["esperado_unidades"].sum() - fine_valued["esperado_unidades"].sum()) < 1e-9,
          "the expected renewals are exactly the ones of the fine rows ($370 + $36): the rate depends on the series only")


def test_the_error_of_the_simulated_acquisitions() -> None:
    print("Q · the band of the simulated acquisitions: the error of their exam at their horizon of sale, row by row")
    from block_7_exam_and_assembly import acquisition_error_of_rows
    from config import PIPELINE_SIMULATED
    configuration = synthetic_with(current_month="2026-09", acquisition_column="adquisicion",
                                   acquisition_values=["Pure-New", "Not-New"])
    summary = pd.DataFrame({"adquisicion": ["Pure-New", "Pure-New", "Not-New"], "h": [0, 2, 0],
                            "error_usd": [0.10, 0.30, 0.50]})
    future = pd.DataFrame({"period": pd.PeriodIndex(["2027-09", "2027-10", "2027-12", "2027-09", "2027-09"], freq="M"),
                           PIPELINE_ORIGIN_COLUMN: [PIPELINE_SIMULATED] * 4 + [PIPELINE_REAL],
                           "adquisicion": ["Pure-New", "Pure-New", "Pure-New", "Retention", "Pure-New"]})
    error = acquisition_error_of_rows(future, {"summary": summary}, configuration)
    # sold in 2026-09 (h 0), 2026-10 (h 1: between 0.10 and 0.30), 2026-12 (h 3: held at the last, 0.30); a value the
    # exam did not see: the mean over the values (h 0: (0.10 + 0.50) / 2); a real row: 0
    check(np.allclose(error, [0.10, 0.20, 0.30, 0.30, 0.0]),
          "a simulated row takes the error of its acquisition value at the month it is sold (interpolated, held beyond the "
          "last); a value not examined takes the mean of the values; a row of the extract takes none")
    check(np.allclose(acquisition_error_of_rows(future, None, configuration), 0.0),
          "without an exam of the acquisitions no row takes an error")


if __name__ == "__main__":
    test_steps_15_to_18()
    test_acquisition_simulation()
    test_the_level_of_the_acquisitions_is_aggregate()
    test_the_box_of_the_states()
    test_the_maturation_by_region()
    test_the_maturation_is_valued_at_the_average_price()
    test_the_detector()
    test_a_projected_renewal_is_a_retention()
    test_the_price_by_states()
    test_the_exam_of_the_acquisitions()
    test_the_prediction_rows_are_aggregated()
    test_the_error_of_the_simulated_acquisitions()
    finish()
