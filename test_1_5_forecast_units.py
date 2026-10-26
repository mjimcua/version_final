"""
test_1_5_forecast_units.py — Step 1.5 on the synthetic raw: one row per forecast unit, the money
conserved, the renewals null only in the future, and the units table written.

    python test_1_5_forecast_units.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from block_1_preparation import validate_raw
from block_1_preparation import apply_calendar
from block_1_preparation import build_fine_table
from block_1_preparation import build_forecast_units
from test_helpers import check, check_stops, console_of, count_status, finish, synthetic_with


def fine_synthetic(**overrides):
    """The synthetic raw after steps 1.1, 1.3 and 1.4, and its Config."""
    configuration = synthetic_with(**overrides)
    frames = {}
    def run_steps():
        calendared = apply_calendar(validate_raw(configuration.read_raw(), configuration), configuration)
        frames["fine"] = build_fine_table(calendared, configuration)
    console_of(run_steps)
    return frames["fine"], configuration


def units_synthetic():
    """The fine table and its forecast units, on the synthetic raw: the input of steps 1.6 to 2.5."""
    fine, configuration = fine_synthetic()
    frames = {}
    console_of(lambda: frames.setdefault("units", build_forecast_units(fine, configuration)))
    return fine, frames["units"], configuration


def run_step(fine, configuration):
    frames = {}
    console = console_of(lambda: frames.setdefault("units", build_forecast_units(fine, configuration)))
    return frames["units"], console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_units() -> None:
    print("A · one row per forecast unit, the money conserved")
    fine, configuration = fine_synthetic()
    units, console = run_step(fine, configuration)
    check(len(units) == fine["fu_id"].nunique() and units["fu_id"].is_unique, "one row per fu_id")
    two_combinations = units[units["n_filas_finas"] == 2].iloc[0]
    fine_rows_of_unit = fine[fine["fu_id"] == two_combinations["fu_id"]]
    check(two_combinations["total_tr_usd"] == fine_rows_of_unit["total_tr_usd"].sum(),
          "a unit with two combinations carries the sum of its two fine rows")
    check(units["total_renewed_units"].sum() == fine["total_renewed_units"].sum(), "renewals conserved")
    projection = units["rol"] == "proyeccion"
    check(units.loc[projection, "total_renewed_units"].isna().all() and units.loc[~projection, "total_renewed_units"].notna().all(),
          "renewals null in the projection, numbers in closed months")
    check(count_status(console, "ok") == 3 and "3 checks: 3 ok" in console, "the 3 checks pass and are logged")
    check("sff_fact_fu" not in pd.read_sql("SELECT name FROM sqlite_master", configuration.sql_engine)["name"].tolist(),
          "the units stay in memory: no SQL table")


def test_the_failures() -> None:
    print("B · what stops and what warns")
    fine, configuration = fine_synthetic()
    single_row_unit = fine["fu_id"].iloc[0]
    wiped = fine.copy()
    wiped.loc[wiped["fu_id"] == single_row_unit, "total_tr_units"] = 0          # the calendar wiped it, the extract had it
    _, console = run_step(wiped, configuration)
    check("WARN  every unit has something falling due in the extract" not in console
          and "1 wiped on purpose by the calendar" in console,
          "a unit the calendar wiped on purpose (due in the extract, 0 after the calendar) is counted apart, not warned")
    zero_in_extract = wiped.copy()
    zero_in_extract.loc[zero_in_extract["fu_id"] == single_row_unit, "s0_vencen_unidades"] = 0
    _, console = run_step(zero_in_extract, configuration)
    check("WARN  every unit has something falling due in the extract" in console,
          "a unit at 0 in the extract itself is a warning")


if __name__ == "__main__":
    test_the_units()
    test_the_failures()
    finish()
