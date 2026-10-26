"""
test_8_1_new_rows.py — The rows the framework adds or wipes, month by month in one format: the gaps of step 1.7,
what step 1.3 wiped, what step 7.2 created; the raw = what stays + what was wiped. A report, not a table:
it is not written to SQL, and its SQL over sff_nucleo is printed.

    python test_8_1_new_rows.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
from main import run
from test_helpers import check, console_of, finish, synthetic_with, step_section


def test_the_new_rows() -> None:
    print("A · the new rows on the synthetic")
    results = {}
    console = console_of(lambda: results.update(run(synthetic_with())))
    check("1 checks: 1 ok" in step_section(console, "8.1"), "the check of step 8.1 passes")
    check("sff_filas_nuevas" not in console and "the SQL that reproduces it" in step_section(console, "8.1"),
          "step 8.1 is a report: it writes no table and prints the SQL that reproduces it from sff_nucleo")
    table = results["new_rows"]
    check(list(table.columns) == ["mes", "rol", "origen", "filas", "series", "unidades", "usd", "esperado_usd"],
          "one format for every origin: month, role, origin, rows, series, units, USD, USD expected")
    rated = results["rated_units"]
    gaps = table[table["origen"] == "hueco"]
    check(int(gaps["filas"].sum()) == int((rated["sintetica"] == 1).sum()) and gaps["unidades"].sum() == 0 and gaps["usd"].sum() == 0,
          "every gap of step 1.7 is counted, with no units and no money (a month, not money)")
    forecast = results["forecast"]["forecast"]
    projected = table[table["origen"] == "proyectada"]
    created = (forecast["origen_pipeline"] == "proyectada") & (forecast["fila_maduracion"] == 0)
    check(abs(projected["usd"].sum() - forecast.loc[created, "total_tr_usd"].sum()) < 1e-6
          and abs(projected["esperado_usd"].sum() - forecast.loc[created, "esperado_usd"].sum()) < 1e-6,
          "the projected rows are the forecast's, with their USD due and expected")
    matured = table[table["origen"] == "maduracion"]
    is_matured = forecast["fila_maduracion"] == 1
    check(len(matured) > 0 and abs(matured["unidades"].sum() - forecast.loc[is_matured, "vencen_unidades_al_vencer"].sum()) < 1e-6
          and abs(matured["esperado_usd"].sum() - forecast.loc[is_matured, "esperado_usd"].sum()) < 1e-6,
          "the prediction rows of the maturation are counted apart, with their units at due date and their expected USD")
    fine = results["fine_table"]
    wiped_results = table[table["origen"] == "resultado_adelantado_borrado"]
    check(abs(wiped_results["unidades"].sum() - (fine["s0_renovados_unidades"].fillna(0) - fine["total_renewed_units"].fillna(0)).sum()) < 1e-6,
          "the wiped early results are the difference between the raw and what stays")
    check("raw" in console and "= kept" in console and "+ wiped" in console, "the reconciliation with the raw is on screen")


if __name__ == "__main__":
    test_the_new_rows()
    finish()
