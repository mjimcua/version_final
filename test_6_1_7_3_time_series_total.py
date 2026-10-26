"""
test_6_1_7_3_time_series_total.py — The time_series universe (retail to subscription: its box, built in step 6.1) and the total (step 7.3).
The rules and the exam on a history built by hand; the box, the forecast that adds it and the total on the synthetic.

    python test_6_1_7_3_time_series_total.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from block_6_time_series import REGION_KEY, examine_the_simulation, project_months, reentry_rows, region_parameters
from test_helpers import check, console_of, finish, synthetic_with, step_section


def hand_made_history():
    """Region A: 2025 and 2026, 100 units a month in 2025, 110 in 2026, $10 a unit.
    Region B: only 2026 (no same month a year before): it must fall back to its share of the total."""
    rows = []
    for month in pd.period_range("2025-01", "2026-08", freq="M"):
        units = 100.0 if month.year == 2025 else 110.0
        rows.append({"period": month, REGION_KEY: "A", "region": "A", "unidades": units, "valor": units * 10.0})
    for month in pd.period_range("2026-01", "2026-08", freq="M"):
        rows.append({"period": month, REGION_KEY: "B", "region": "B", "unidades": 50.0, "valor": 50.0 * 20.0})
    return pd.DataFrame(rows)


def test_the_rules() -> None:
    print("A · the rules on a history built by hand (current month 2026-09)")
    configuration = synthetic_with(ts_region_columns=["region"])
    current = pd.Period("2026-09", freq="M")
    history = hand_made_history()
    parameters = region_parameters(history, current, configuration)
    check(abs(parameters.loc["A", "nivel"] - 1.10) < 1e-12 and abs(parameters.loc["A", "valor_medio"] - 10.0) < 1e-12,
          "level of A = Σ units Jun-Aug 2026 / Σ same months 2025 = 1.10; value per unit = Σ value / Σ units = $10")
    months = list(pd.period_range("2026-09", "2026-12", freq="M"))
    projected, fallback = project_months(history, parameters, months, configuration)
    a_september = projected[(projected[REGION_KEY] == "A") & (projected["period"] == pd.Period("2026-09", freq="M"))].iloc[0]
    check(abs(a_september["unidades"] - 110.0) < 1e-9 and abs(a_september["valor"] - 1100.0) < 1e-9,
          "A, September 2026 = 100 units of September 2025 × 1.10 = 110 units × $10 = $1,100")
    b_rows = projected[projected[REGION_KEY] == "B"]
    check(fallback == {"B"} and (b_rows["por_cuota"] == 1).all() and len(b_rows) == 4,
          "B has no same month in 2025: it takes its share of the projected total (and it is flagged)")

    # next year, on this year's months: real where closed, projected where not
    this_year = pd.concat([history[history["period"].map(lambda month: month.year) == 2026], projected], ignore_index=True)
    next_year, _ = project_months(this_year, parameters, list(pd.period_range("2027-01", "2027-12", freq="M")), configuration)
    a_next = next_year[next_year[REGION_KEY] == "A"].set_index("period")["unidades"]
    check(len(a_next) == 12 and abs(a_next[pd.Period("2027-01", freq="M")] - 121.0) < 1e-9
          and abs(a_next[pd.Period("2027-11", freq="M")] - 121.0) < 1e-9,
          "the whole of 2027: January = 110 real units of January 2026 × 1.10 = 121; November = 110 PROJECTED units "
          "of November 2026 × 1.10 = 121")
    b_next = next_year[next_year[REGION_KEY] == "B"]
    check(len(b_next) == 12 and (b_next["por_cuota"] == 1).all(),
          "B in 2027 has a same month a year before (2026) but no level (nothing in 2025): its share of the total")

    rates = {1: pd.Series({"A": 0.8})}
    reentry = reentry_rows(projected.assign(region=projected[REGION_KEY]), rates, 0.6, configuration)
    a_reentry = reentry[(reentry["region"] == "A") & (reentry["period"] == pd.Period("2027-09", freq="M"))].iloc[0]
    check(abs(a_reentry["valor"] - 1100.0) < 1e-9 and a_reentry["descuento"] == 0.4 and abs(a_reentry["revenue"] - 1100.0 / 0.6 * 0.8) < 1e-9,
          "September 2026 falls due in September 2027 as pipeline: $1,100 at 40 % off; renewed at 100 %: $1,100 / 0.6 × rate 0.8")
    check((reentry.loc[reentry["region"] == "B", "tasa_global"] == 1).all(), "a region with no rate takes the global one, flagged")

    two_levels = synthetic_with(ts_region_columns=["region", "product"])
    projected_two = projected.assign(region=projected[REGION_KEY], product="A1")
    level_rates = {1: pd.Series({"A": 0.8, "B": 0.5}), 2: pd.Series({"A|A1": 0.9})}
    reentry_two = reentry_rows(projected_two, level_rates, 0.6, two_levels)
    check((reentry_two.loc[reentry_two["region"] == "A", "nivel_tasa"] == "product").all()
          and (reentry_two.loc[reentry_two["region"] == "B", "nivel_tasa"] == "region").all()
          and abs(reentry_two.loc[reentry_two["region"] == "B", "tasa"].iloc[0] - 0.5) < 1e-12,
          "the rate climbs the levels: A has its finest-level rate (0.9); B has none there and takes its region's (0.5)")


def test_the_exam() -> None:
    print("B · the exam of the simulation on the history built by hand (region A, origins March to August 2026)")
    configuration = synthetic_with(ts_region_columns=["region"], ts_exam_origins=6, ts_level_rule="region")
    history = hand_made_history()
    exam = examine_the_simulation(history[history[REGION_KEY] == "A"], configuration)
    by_horizon = exam["by_horizon"].set_index("h")["error_usd"]
    # origin March 2026: December 2024 is before the history, so the level reads January - February 2026 against the
    # same months of 2025 only (220 / 200 = 1.10): exact. Origin February: the level reads January 2026 only (1.10).
    # Every origin is exact: the error is 0 at every horizon
    march_error = 0.0
    check(exam["read_after_origin"] == 0 and sorted(by_horizon.index) == [1, 2, 3, 4, 5, 6],
          "6 origins, horizons 1 to 6; no month at or after an origin is read")
    check(np.allclose(by_horizon.to_numpy(), march_error),
          "the level is 1.10 at every origin (a month a year before the history is not read as 0 sold): exact, error 0")
    shifted = history[history[REGION_KEY] == "A"].copy()
    shifted.loc[shifted["period"] == pd.Period("2026-05", freq="M"), ["unidades", "valor"]] = [132.0, 1320.0]
    exam = examine_the_simulation(shifted, configuration)
    by_horizon = exam["by_horizon"].set_index("h")["error_usd"]
    # May 2026 sells 132 (not 110): the origins March to May simulate it at 110 (error 22 / 132 at h = 3, 2, 1); the
    # origins June to August read it in their level, (110 + 110 + 132) / 300 = 1.1733, and simulate 117.33 for months
    # that sold 110 (error 7.33 / 110)
    may_error, level_error = 22 / 132, (100 * 352 / 300 - 110) / 110
    check(abs(by_horizon[1] - (may_error + 3 * level_error) / 6) < 1e-9
          and abs(by_horizon[3] - (may_error + level_error) / 4) < 1e-9,
          "May sells 132: h = 1 is (May at its origin + June, July, August read with May in the level) / 6 origins; "
          "h = 3 is (May from March + August from June) / 4 origins")


def test_the_level_rules() -> None:
    print("B2 · the level rules and the choice of the exam (TS_LEVEL_RULES)")
    configuration = synthetic_with(ts_region_columns=["region"], ts_exam_origins=6)
    current = pd.Period("2026-09", freq="M")
    history = hand_made_history()
    # region C: 2025 and 2026, 300 units a month in 2025 and 2026 (level 1.00); with A (level 1.10) the total's level is
    # (330 + 900) / (300 + 900) = 1.025 over June-August
    rows_c = [{"period": month, REGION_KEY: "C", "region": "C", "unidades": 300.0, "valor": 3000.0}
              for month in pd.period_range("2025-01", "2026-08", freq="M")]
    two = pd.concat([history[history[REGION_KEY] == "A"], pd.DataFrame(rows_c)], ignore_index=True)
    total_level = (3 * 110 + 3 * 300) / (3 * 100 + 3 * 300)
    by_rule = {rule: region_parameters(two, current, configuration, rule) for rule in ("region", "credibilidad", "total", "sin_nivel")}
    z_a = 300 / (300 + (300 + 900) / 2)
    check(abs(by_rule["region"].loc["A", "nivel"] - 1.10) < 1e-12
          and abs(by_rule["total"].loc["A", "nivel"] - total_level) < 1e-12
          and abs(by_rule["credibilidad"].loc["A", "nivel"] - (z_a * 1.10 + (1 - z_a) * total_level)) < 1e-12
          and (by_rule["sin_nivel"]["nivel"] == 1.0).all(),
          "region = its own 1.10 · total = 1.025 for every region · credibilidad = z·1.10 + (1−z)·1.025, "
          "z = 300 / (300 + the mean of the regions, 600) · sin_nivel = 1")
    exam = examine_the_simulation(two, configuration)
    table = exam["by_rule"]
    # the value per unit is 10 in every month: every window of it gives the same, so the rule decides
    mean_error = table[table["ventana_valor_medio"] == 12].groupby("regla_nivel")["error_medio_regla"].first()
    check(set(table["regla_nivel"]) == {"region", "credibilidad", "total", "sin_nivel"}
          and set(table["ventana_valor_medio"]) == {1, 3, 12}
          and table.groupby("candidato")["elegida"].first().sum() == 1
          and exam["rule"] == "region" and abs(mean_error["region"]) < 1e-12 and abs(mean_error["total"]) < 1e-12
          and mean_error["credibilidad"] > 0 and mean_error["sin_nivel"] > 0,
          "every rule is examined on the same origins; the exam judges the TOTAL of a month: A grows 10 % and C 0 %, so "
          "the region's own levels and the total's (1.025 on 400 units = 410) are both exact; credibilidad and sin_nivel "
          "are not; the tie goes to the first rule (region); every rule with every window (1, 3, 12), one candidate chosen")
    check(exam["by_horizon"]["error_usd"].abs().max() < 1e-12,
          "the band is the error of the rule chosen (0 here)")


def test_the_window_of_the_value_per_unit() -> None:
    print("B3 · the window of the value per unit, chosen by the exam with the level rule")
    configuration = synthetic_with(ts_region_columns=["region"], ts_exam_origins=6)
    current = pd.Period("2026-09", freq="M")
    # region A: 100 units every month, the value per unit falls 2 % a month (30 $ in 2025-01): the level is 1 for every
    # rule, the value per unit decides; the longer its window, the further behind the falling price it is
    months = list(pd.period_range("2025-01", "2026-08", freq="M"))
    rows = [{"period": month, REGION_KEY: "A", "region": "A", "unidades": 100.0, "valor": 100.0 * 30.0 * 0.98 ** step}
            for step, month in enumerate(months)]
    falling = pd.DataFrame(rows)
    exam = examine_the_simulation(falling, configuration)
    table = exam["by_rule"]
    by_window = table[table["regla_nivel"] == exam["rule"]].groupby("ventana_valor_medio")["error_medio_regla"].first()
    check(exam["auv_window"] == 1 and by_window[1] < by_window[3] < by_window[12],
          "a value per unit falling every month: the last month errs least, then 3, then 12; the exam takes 1")
    last_price = 30.0 * 0.98 ** (len(months) - 1)
    parameters = region_parameters(falling, current, configuration, "sin_nivel", 1)
    year_mean = falling[falling["period"] >= current - 12]["valor"].sum() / falling[falling["period"] >= current - 12]["unidades"].sum()
    check(abs(parameters.loc["A", "valor_medio"] - last_price) < 1e-9
          and abs(region_parameters(falling, current, configuration, "sin_nivel", 12).loc["A", "valor_medio"] - year_mean) < 1e-9,
          f"window 1 = the price of the last month ({last_price:.2f} $); window 12 = the mean of the year ({year_mean:.2f} $)")
    fixed = examine_the_simulation(falling, synthetic_with(ts_region_columns=["region"], ts_exam_origins=6, ts_auv_window_months=12))
    check(set(fixed["by_rule"]["ventana_valor_medio"]) == {12} and fixed["auv_window"] == 12,
          "ts_auv_window_months fixes the window: only it is examined")


def test_the_step() -> None:
    print("C · the box (step 5.1), the forecast that adds it (step 7.2) and the total (step 7.3) on the synthetic")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    check("every time_series row has every region level" in console.split("STEP 1.2 ·")[1].split("STEP 1.3 ")[0]
          and "5 checks: 5 ok" in step_section(console, "6.1") and "3 checks: 3 ok" in step_section(console, "7.3"),
          "the separation (step 1.2), the checks of the universe (step 6.1) and of the total (step 7.3) pass")
    check("[1.2] chart: the time_series universe month by month" in console
          and "[6.1] chart: the time_series universe with its prediction" in console,
          "the universe is drawn twice: its history in step 1.2 and the same chart with its prediction in step 6.1")
    time_series_rows = results["time_series_rows"]
    check(len(time_series_rows) == 48 and not (results["fine_table"]["flag_time_series"] == 1).any(),
          "the flagged rows leave the raw in step 1.2: none reaches the fine table")
    universe = results["time_series_universe"]
    table, total = results["time_series"], results["forecast_total"]
    check(table is universe.table and set(table["origen"]) == {"ts_real", "ts_proyectado", "ts_simulado", "ts_reentrada"},
          "the four origins of the universe, from its box")
    months_of = lambda origin: set(table.loc[table["origen"] == origin, "period"].astype(str))
    check(months_of("ts_proyectado") == {"2026-09", "2026-10", "2026-11", "2026-12"},
          "projected: from the current month to December")
    check(months_of("ts_simulado") == {f"2027-{month:02d}" for month in range(1, 13)}, "simulated: the WHOLE of 2027")
    check(months_of("ts_reentrada") == {"2027-09", "2027-10", "2027-11", "2027-12"},
          "re-entry: only the projected months, a year later (the closed months are in the extract already)")
    future = table[table["origen"] != "ts_real"]
    check(bool((future["revenue_bajo"] <= future["revenue"] + 1e-9).all() and (future["revenue"] <= future["revenue_alto"] + 1e-9).all())
          and (future["revenue_alto"] > future["revenue"]).any()
          and np.allclose(table.loc[table["origen"] == "ts_real", "revenue_alto"], table.loc[table["origen"] == "ts_real", "revenue"]),
          "every simulated row has a band from the exam; the real months have none")

    by_month, by_year = results["forecast"]["by_month"], results["forecast"]["by_year"]
    simulated_revenue = table.loc[table["origen"].isin(["ts_proyectado", "ts_simulado"]), "revenue"].sum()
    reentry = table[table["origen"] == "ts_reentrada"]
    check(abs(by_month["ingresos_ts_usd"].sum() - simulated_revenue) < 0.01
          and abs(by_month["vence_ts_reentrada_usd"].sum() - reentry["valor"].sum()) < 0.01
          and abs(by_month["esperado_ts_reentrada"].sum() - reentry["revenue"].sum()) < 0.01,
          "the forecast by month adds the revenue of the universe, the pipeline of its re-entry and its expected renewals")
    check(np.allclose(by_month["total_usd"], by_month["esperado_usd"] + by_month["ingresos_ts_usd"] + by_month["esperado_ts_reentrada"])
          and np.allclose(by_month["usd_vence_total"], by_month["usd_vence"] + by_month["vence_ts_reentrada_usd"]),
          "total_usd of a month = the renewals of the pipeline + the revenue and the re-entry of the universe")
    pipeline_error = np.sqrt(by_month["error_tasa_usd"] ** 2 + by_month["error_precio_usd"] ** 2)
    check(np.allclose(by_month["error_examen_usd"], pipeline_error + by_month["error_captacion_usd"] + by_month["error_ts_usd"])
          and (by_month["error_ts_usd"] > 0).any()
          and np.allclose(by_month["banda_examen_alta"] - by_month["banda_examen_baja"], 2 * by_month["error_examen_usd"]),
          "the official band of a month: √(rate² + price²) of the pipeline + the acquisitions + the universe's exam, "
          "centred on total_usd")
    current_months = by_month["period"] < pd.Period("2027-01", freq="M")
    check(np.allclose(by_month.loc[current_months, "error_tasa_usd"],
                      by_month.loc[current_months, "error_examen_pct"] * by_month.loc[current_months, "esperado_usd"])
          and (by_month["error_precio_pct"] > 0).all()
          and abs(by_month["error_precio_pct"].iloc[0] - results["uplift_verdict"]["error_precio_total"]) < 1e-12,
          "the rate part is the exam's error × the expected renewals (a projected renewal adds the error of its volume); "
          "the price part is the error of the price of the monthly total measured in the exam of the uplift (3.3)")
    totals = total[total["origen"] == "TOTAL"].set_index("ano")["usd_renovado"]
    answered = by_year.set_index("ano")["total_usd"]
    check(all(abs(totals[year] - answered[year]) < 0.01 for year in totals.index),
          "the TOTAL of every year (step 7.3) = total_usd of the forecast by year (step 7.2)")
    total_2026 = total[total["ano"] == 2026].set_index("origen")["usd_renovado"]
    check(abs(total_2026["TOTAL"] - (total_2026["pipeline_renovado_real"] + total_2026["pipeline_real_esperado"]
                                     + total_2026["ts_real"] + total_2026["ts_proyectado"])) < 0.01,
          "total 2026 = renewals of the pipeline (booked + expected) + ts_real + ts_proyectado")
    total_2027 = total[total["ano"] == 2027].set_index("origen")
    projected_value = table.loc[table["origen"] == "ts_proyectado", "valor"].sum()
    check(abs(total_2027.loc["ts_reentrada", "usd_vence"] - projected_value) < 0.01
          and total_2027.loc["ts_simulado", "usd_renovado"] > 0,
          "2027: the pipeline of the re-entry is the projected value (at 40 % off), and the simulated year is in the total")

    this_year = table[table["origen"].isin(["ts_real", "ts_proyectado"])].set_index("period")["unidades"]
    next_year = table[table["origen"] == "ts_simulado"].set_index("period")["unidades"]
    check(all(abs(next_year[month] - this_year[month - 12]) < 1e-9 for month in next_year.index),
          "by default (ts_next_year_level = 1.0) next year repeats this year month by month")
    growing = synthetic_with(ts_next_year_level=None)
    growing_results = {}
    console_of(lambda: growing_results.update(run(growing)))
    universe = growing_results["time_series_universe"]
    growing_table = growing_results["time_series"]
    this_year = growing_table[growing_table["origen"].isin(["ts_real", "ts_proyectado"])].set_index("period")["unidades"]
    next_year = growing_table[growing_table["origen"] == "ts_simulado"].set_index("period")["unidades"]
    level = float(universe.parameters["nivel"].iloc[0])
    check(all(abs(next_year[month] - this_year[month - 12] * level) < 1e-6 for month in next_year.index),
          f"ts_next_year_level = None: next year = this year × the level of the region ({level:.3f}) again")


if __name__ == "__main__":
    test_the_rules()
    test_the_exam()
    test_the_level_rules()
    test_the_window_of_the_value_per_unit()
    test_the_step()
    finish()
