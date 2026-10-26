"""
test_price_increases.py — The price increases in the uplift: the window, the removal of the increases,
the month of the cycle and where their effect is lost (price_increases.py, steps 3.1, 3.3 and 7.2).

A · a hand-made fine table, three price groups: EU with a +5 % increase in 2026-01 (still in its cycle at the
    current month 2026-09), US flat, LA with a step in its last two closed months (not confirmed yet)
B · the uplift of the window without the increase, and the increase applied again to the future rows inside
    its cycle; after the cycle its effect is lost
C · the monitor read from the fine table (steps 3.1 and 3.3) is the one of the core (step 8.4)
D · with uplift_window_months = 0 and uplift_without_increases off, the uplift is the one of all the history
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from price_increases import active_increases, increase_factor, price_group_of
from block_3_price import (closed_view_of_fine_table, estimate_cell_uplifts, price_increases_as_of, renewer_rows,
                            uplift_history)
from block_3_price import show_price_increases
from block_7_exam_and_assembly import uplift_of_rows
from block_8_output import price_increase_monitor
from test_helpers import check, console_of, finish, synthetic_with
from config import CALENDAR_ROLE_COLUMN, PATH_STATISTICAL, ROLE_PROJECTION, TRUTH_ROLES, UPLIFT_CELL_ID_COLUMN


CURRENT_MONTH = pd.Period("2026-09", freq="M")
EU_INCREASE_STARTS = pd.Period("2026-01", freq="M")
LA_STEP_STARTS = pd.Period("2026-07", freq="M")
BASE_UPLIFT = 1.10
EU_INCREASE = 0.05


def hand_made_configuration():
    """The synthetic Config with only the exact base of the uplift among the closed-month measures, and no
    discount: the fine table of the test carries nothing else."""
    configuration = synthetic_with()
    overrides = {measure.config_field: None for measure in configuration.closed_month_measures
                 if measure.config_field != "renewed_pipeline_usd_col"}
    overrides.update(discount_value_column=None, price_group_min_renewed_units=0)
    return synthetic_with(**overrides)


def uplift_of_group_and_month(region: str, month) -> float:
    """The uplift each group really had: EU steps +5 % in 2026-01; LA steps +6 % in its last two months."""
    if region == "EU" and month >= EU_INCREASE_STARTS:
        return BASE_UPLIFT * (1 + EU_INCREASE)
    if region == "LA" and month >= LA_STEP_STARTS:
        return BASE_UPLIFT * 1.06
    return BASE_UPLIFT


def hand_made_fine_table(configuration) -> pd.DataFrame:
    """2024-01..2026-08 closed, 2026-09..2027-04 future; three price groups, one row per group and month."""
    rows = []
    for month in pd.period_range("2024-01", "2027-04", freq="M"):
        for region in ("EU", "US", "LA"):
            closed = month < CURRENT_MONTH
            renewed_units = 700.0 if closed else np.nan
            rows.append({configuration.period_col: month, "region": region, "product": "A", "newcust": 0,
                         CALENDAR_ROLE_COLUMN: TRUTH_ROLES[0] if closed else ROLE_PROJECTION,
                         UPLIFT_CELL_ID_COLUMN: region,
                         configuration.pipeline_units_col: 1000.0, configuration.pipeline_usd_col: 10000.0,
                         configuration.renewed_units_col: renewed_units,
                         configuration.renewed_usd_col: renewed_units * 10.0 * uplift_of_group_and_month(region, month),
                         configuration.renewed_pipeline_usd_col: renewed_units * 10.0})
    return pd.DataFrame(rows)


def test_the_increases_known_today() -> None:
    print("A · the increases known at the current month: EU in its cycle, LA not confirmed yet, US nothing")
    configuration = hand_made_configuration()
    fine_table = hand_made_fine_table(configuration)
    price = price_increases_as_of(fine_table, CURRENT_MONTH, configuration)
    calendar = price["calendar"]
    check(list(calendar["price_group"]) == ["EU|A"] and calendar["inicio"].iloc[0] == EU_INCREASE_STARTS,
          "one increase detected: EU in 2026-01 (US is flat, LA has only two months above the threshold)")
    pending = price["pending"]
    check(list(pending["price_group"]) == ["LA|A"] and pending["meses_por_encima"].iloc[0] == 2
          and pending["inicio"].iloc[0] == LA_STEP_STARTS,
          "LA: two months above the threshold at the end of the history, three needed: a step not confirmed yet")
    increases = active_increases(calendar, CURRENT_MONTH, 12, configuration)
    eu = increases.iloc[0]
    check(abs(eu["tamano"] - EU_INCREASE) < 1e-12 and abs(eu["factor"] - (1 + EU_INCREASE)) < 1e-12,
          "its size: the uplift of the cycle months seen over the uplift before − 1 = +5 %")
    check(eu["fin_ciclo"] == pd.Period("2026-12", freq="M") and eu["mes_del_ciclo"] == 9
          and eu["meses_futuros_con_efecto"] == 4,
          "its cycle: 2026-01..2026-12; 2026-09 is its 9th month; 4 future months still carry it (2026-09..12)")
    old_calendar = calendar.assign(inicio=pd.Period("2024-06", freq="M"))
    check(active_increases(old_calendar, CURRENT_MONTH, 12, configuration).empty
          and len(active_increases(old_calendar, CURRENT_MONTH, 0, configuration)) == 1,
          "an increase whose cycle ended before the window is out of play; with all the history (window 0) it is in")


def test_the_uplift_without_the_increase() -> None:
    print("B · the window without the increase, and the increase applied again where its cycle is not over")
    configuration = hand_made_configuration()
    fine_table = hand_made_fine_table(configuration)
    renewers = renewer_rows(fine_table, configuration)
    price = price_increases_as_of(fine_table, CURRENT_MONTH, configuration)
    increases = active_increases(price["calendar"], CURRENT_MONTH, 12, configuration)
    history = uplift_history(renewers, CURRENT_MONTH, increases, 12, True, configuration)
    check(history[configuration.period_col].min() == pd.Period("2025-09", freq="M")
          and history[configuration.period_col].max() == pd.Period("2026-08", freq="M"),
          "the window: the last 12 closed months, 2025-09..2026-08")
    eu_rows = history[history["region"] == "EU"]
    in_cycle = eu_rows[eu_rows[configuration.period_col] >= EU_INCREASE_STARTS]
    check((in_cycle["_factor_subida"] == 1 + EU_INCREASE).all()
          and (eu_rows[eu_rows[configuration.period_col] < EU_INCREASE_STARTS]["_factor_subida"] == 1.0).all()
          and (history[history["region"] != "EU"]["_factor_subida"] == 1.0).all(),
          "only the EU months inside its cycle are deflated, by 1.05; every other row keeps its renewed USD")

    cells = estimate_cell_uplifts(fine_table, history, configuration, numerator_column="_numerador_uplift", with_band=False)
    uplift_of_cell = cells.set_index(UPLIFT_CELL_ID_COLUMN)["uplift"]
    check(abs(uplift_of_cell["EU"] - BASE_UPLIFT) < 1e-12 and abs(uplift_of_cell["US"] - BASE_UPLIFT) < 1e-12,
          "the uplift of EU without the increase is its base 1.10, like US")
    as_it_is = uplift_history(renewers, CURRENT_MONTH, increases, 12, False, configuration)
    raw_cells = estimate_cell_uplifts(fine_table, as_it_is, configuration, numerator_column="_numerador_uplift", with_band=False)
    raw_eu = float(raw_cells.set_index(UPLIFT_CELL_ID_COLUMN).loc["EU", "uplift"])
    check(abs(raw_eu - (4 * BASE_UPLIFT + 8 * BASE_UPLIFT * (1 + EU_INCREASE)) / 12) < 1e-12,
          "the window as it is would mix 4 months at 1.10 and 8 at 1.155: 1.1367, an increase the future only half keeps")

    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION].copy()
    forecast_rows = uplift_of_rows(future, cells, {"via_usada_con_descuento": PATH_STATISTICAL}, configuration,
                                   price_increases=increases)
    eu_future = forecast_rows[forecast_rows["region"] == "EU"].set_index(configuration.period_col)["uplift"]
    check(all(abs(eu_future[pd.Period(month, freq="M")] - BASE_UPLIFT * (1 + EU_INCREASE)) < 1e-12
              for month in ("2026-09", "2026-10", "2026-11", "2026-12")),
          "EU 2026-09..12, still inside its cycle: the uplift takes the increase again (1.10 × 1.05)")
    check(all(abs(eu_future[pd.Period(month, freq="M")] - BASE_UPLIFT) < 1e-12 for month in ("2027-01", "2027-04")),
          "EU from 2027-01: the cycle is over, the effect is lost: back to 1.10")
    check((forecast_rows[forecast_rows["region"] != "EU"]["factor_subida"] == 1.0).all(),
          "US and LA (not confirmed) take no factor")
    off = uplift_of_rows(future, cells, {"via_usada_con_descuento": PATH_STATISTICAL},
                         synthetic_with(price_group_min_renewed_units=0, uplift_without_increases=False),
                         price_increases=increases)
    check((off["factor_subida"] == 1.0).all(), "with uplift_without_increases off, no factor is applied")

    console = console_of(lambda: show_price_increases(fine_table, renewers, history, increases, price, CURRENT_MONTH,
                                                      configuration))
    check("VALORACIÓN EU|A: subida de +5.0% desde 2026-01; 2026-09 es el mes 9 de 12" in console
          and "2026-09..2026-12 ($40,000 que vencen) y desde 2027-01 el efecto se pierde" in console,
          "on screen: the month of the cycle today, the future months that carry it and their USD, where it is lost")
    check("VALORACIÓN LA|A: 2 meses por encima del umbral desde 2026-07" in console
          and "el forecast NO la lleva en 2026-09..2027-06" in console,
          "on screen: the step of LA not confirmed yet, and what it means for the forecast")
    check("ventana sin subidas (el que usa el forecast)" in console, "on screen: the uplift of the portfolio three ways")

    groups = price_group_of(future, ["region", "product"])
    factor = increase_factor(groups, future[configuration.period_col], increases.iloc[0:0])
    check((factor == 1.0).all(), "no increase in play: every factor is 1")


def test_the_monitor_of_the_fine_table_is_the_one_of_the_core() -> None:
    print("C · the monitor of steps 3.1 and 3.3 (fine table) is the one of step 8.4 (core)")
    configuration = synthetic_with()
    results = {}
    console_of(lambda: results.update(run(configuration)))
    from price_increases import price_monitor
    from_fine = price_monitor(closed_view_of_fine_table(results["fine_table"], configuration), configuration)
    from_core = price_increase_monitor(results["core"], configuration)
    common = [column_name for column_name in from_core.columns if column_name in from_fine.columns]
    ordered_fine = from_fine.sort_values(["price_group", "period"]).reset_index(drop=True)[common]
    ordered_core = from_core.sort_values(["price_group", "period"]).reset_index(drop=True)[common]
    numeric = [column_name for column_name in common if pd.api.types.is_numeric_dtype(ordered_core[column_name])]
    same_numbers = np.allclose(ordered_fine[numeric].to_numpy(dtype=float), ordered_core[numeric].to_numpy(dtype=float),
                               rtol=1e-12, atol=1e-9, equal_nan=True)
    check(len(ordered_fine) == len(ordered_core) and ordered_fine["price_group"].equals(ordered_core["price_group"])
          and same_numbers, f"the same {len(ordered_core):,} price group × month rows, the same numbers")
    console = console_of(lambda: run(configuration))
    check("SUBIDAS DE PRECIO Y VENTANA DEL UPLIFT" in console and "EL UPLIFT ESTADÍSTICO DE CADA FORMA" in console
          and "ESTADOS DEL CLIENTE POR ENCIMA DE SU PROPORCIÓN FINAL" in console,
          "the three sections are on screen: the increases and the window (15), the ways in the exam (16), the detector (17)")


def test_all_the_history_is_the_old_uplift() -> None:
    print("D · window 0 and the increases not removed: the uplift of all the history")
    configuration = synthetic_with(uplift_window_months=0, uplift_without_increases=False)
    results = {}
    console_of(lambda: results.update(run(configuration)))
    cells = results["uplift_cells"]
    fine = results["fine_table"]
    renewers = fine[fine[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (fine["total_renewed_units"] > 0)]
    one_cell = cells[cells["uplift_origen"] == "propia"].iloc[0]
    rows = renewers[renewers[UPLIFT_CELL_ID_COLUMN] == one_cell[UPLIFT_CELL_ID_COLUMN]]
    check(abs(one_cell["uplift"] - rows["total_renewed_usd"].sum() / rows["total_tr_usd_renewed"].sum()) < 1e-9,
          "the uplift of a cell = Σ renewed USD / Σ exact base over every closed month")
    check((results["forecast"]["forecast"]["factor_subida"] == 1.0).all(), "no factor in the forecast")


if __name__ == "__main__":
    test_the_increases_known_today()
    test_the_uplift_without_the_increase()
    test_the_monitor_of_the_fine_table_is_the_one_of_the_core()
    test_all_the_history_is_the_old_uplift()
    finish()
