"""
test_8_3_core.py — The core table on the synthetic: one row per fine row plus the gaps,
the money reconciles, the series values repeat on every row, the legend covers everything.

    python test_8_3_core.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from main import run
from test_helpers import check, console_of, count_status, finish, synthetic_with, step_section


def test_the_core() -> None:
    print("A · the core (fact table) and the forecast series dimension")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    core, legend, dimension = results["core"], results["core_legend"], results["forecast_series"]
    check("10 checks: 10 ok" in step_section(console, "8.3"), "the 10 checks of the core pass")
    extended = int(results["forecast"]["forecast"]["_fila"].isna().sum())
    time_series_rows = len(results["time_series_rows"])
    synthetic_time_series = int(results["time_series"]["origen"].isin(["ts_proyectado", "ts_simulado", "ts_reentrada"]).sum())
    check(len(core) == len(results["fine_table"]) + int((results["rated_units"]["sintetica"] == 1).sum()) + extended
          + time_series_rows + synthetic_time_series,
          "every original record (pipeline and time_series) and every synthetic row (gaps, extended, ts), once")
    check({"raw", "hueco", "proyectada", "maduracion", "ts_real", "ts_sin_resultado", "ts_proyectado", "ts_simulado",
           "ts_reentrada"}
          <= set(core["origen_fila"]) <= {"raw", "hueco", "proyectada", "simulada", "maduracion", "ts_real", "ts_sin_resultado",
                                          "ts_proyectado", "ts_simulado", "ts_reentrada"},
          "the origins of the rows: original and synthetic (gaps, extended, the prediction rows of the maturation), of both universes")
    matured = core[core["origen_fila"] == "maduracion"]
    pipeline_forecast = core[(core["forecast_universe"] == "pipeline") & (core["forecast_status"] == "forecast")]
    check((matured["s1_1_vencen_unidades"] == 0).all() and (matured["forecast_to_renew_units"] == 0).all()
          and matured["forecast_to_renew_units_at_due"].gt(0).all()
          and abs(core["forecast_to_renew_units_at_due"].sum() - core["forecast_to_renew_units"].sum()) < 1e-6
          and abs(pipeline_forecast["forecast_renewed_USD"].sum() - pipeline_forecast["forecast_renewed_USD_maturation_adj"].sum()
                  - pipeline_forecast["s7_2_esperado_usd_hoy"].sum()) < 1.0,
          "a prediction row of the maturation has no data (s00 and the pipeline at 0) and its units at due date; the "
          "pipeline at due date adds up to the pipeline; the forecast = today's states + the maturation")
    total = results["forecast_total"]
    renewed_2026 = core.loc[core["forecast_year"] == 2026, "forecast_renewed_USD"].sum()
    pipeline_2027 = core.loc[core["forecast_year"] == 2027, "forecast_to_renew_USD"].sum()
    check(abs(renewed_2026 - total.loc[(total["ano"] == 2026) & (total["origen"] == "TOTAL"), "usd_renovado"].item()) < 0.01
          and abs(pipeline_2027 - total.loc[(total["ano"] == 2027) & (total["origen"] == "TOTAL"), "usd_vence"].item()) < 0.01,
          "Q1 (renewed 2026) and Q2 (pipeline 2027) are a SUM of the core: equal to sff_forecast_total")
    real_2026 = core[(core["forecast_year"] == 2026) & (core["forecast_status"] == "actual")]["forecast_renewed_USD"].sum()
    expected_2026 = core[(core["forecast_year"] == 2026) & (core["forecast_status"] == "forecast")]["forecast_renewed_USD"].sum()
    check(abs(real_2026 + expected_2026 - renewed_2026) < 0.01, "2026 splits into actual (done) + forecast (to renew)")
    gaps = core[core["origen_fila"] == "hueco"]
    check(len(gaps) > 0 and (gaps["s1_1_vencen_unidades"] == 0).all() and gaps["s1_4_fs_id"].isin(dimension["s1_4_fs_id"]).all(),
          "a gap row has every measure at 0 and its forecast series is in the dimension")
    raw = configuration.read_raw()
    check(abs(core.loc[core["forecast_universe"] == "pipeline", "s1_1_vencen_usd"].sum()
              - raw.loc[raw["flag_time_series"] != 1, "total_tr_usd"].sum()) < 0.01,
          "Σ USD due in the core = Σ in the extract without the time_series universe (it is simulated apart, step 6.1)")
    check(dimension["s1_4_fs_id"].is_unique and set(dimension["s1_4_fs_id"]) == set(core["s1_4_fs_id"].dropna())
          and not any(column.startswith(("s1_6_", "s1_7_", "s2_2_", "s2_3_", "s7_1_")) for column in core.columns),
          "every value of a forecast series lives once, in sff_forecast_series; the core keeps the rows")
    one_series = dimension[dimension["s1_4_fs_id"] == "EU|A|0|0|0|0|web"]
    check(not any(column.endswith("_key") for column in core.columns), "no hash keys in the core")
    core_legend_rows = legend[legend["tabla"] == "sff_nucleo"]
    dimension_legend_rows = legend[legend["tabla"] == "sff_forecast_series"]
    check(set(core_legend_rows["columna"]) == set(core.columns) and set(dimension_legend_rows["columna"]) == set(dimension.columns),
          "the legend describes every column of both tables")
    closed = core[core["s1_3_rol"].isin(["entrenamiento", "examen"]) & (core["s1_4_fs_id"] == "EU|A|0|0|0|0|web")]
    check(abs(closed["s1_3_renovadas_unidades"].sum() / closed["s1_1_vencen_unidades"].sum()
              - one_series["s1_7_tasa_propia"].iloc[0]) < 1e-12,
          "the ratio of sums over the closed rows of a series = its own rate (what the Power BI measure computes)")
    written = pd.read_sql("SELECT COUNT(*) AS n FROM sff_nucleo", configuration.sql_engine)["n"].item()
    written_dimension = pd.read_sql("SELECT COUNT(*) AS n FROM sff_forecast_series", configuration.sql_engine)["n"].item()
    check(written == len(core) and written_dimension == len(dimension), "sff_nucleo and sff_forecast_series written")
    summed = dimension["s7_1_exam_real_units"].sum()
    check(abs(summed - results["series_exam"]["per_series"]["framework_real_units"].sum()) < 1e-6,
          "a SUM over the dimension counts every forecast series' exam once")


if __name__ == "__main__":
    test_the_core()
    finish()
