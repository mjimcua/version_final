"""
test_2_5_8_6_dynamics_report.py — Step 2.5 (dynamics) on series built by hand, and the report on the synthetic.

    python test_2_5_8_6_dynamics_report.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os

import numpy as np
import pandas as pd

from main import run
from block_2_rate import dynamics_of_one_series
from test_helpers import check, console_of, finish, synthetic_with, step_section


def monthly_series(rates, units=2000, start="2023-01"):
    months = pd.period_range(start, periods=len(rates), freq="M")
    renewed = np.round(np.asarray(rates) * units)
    return pd.DataFrame({"period": months, "renovadas": renewed, "vencen": float(units)})


def test_the_dynamics() -> None:
    print("A · φ, trend and seasonality on series built by hand")
    configuration = synthetic_with()
    generator = np.random.default_rng(3)
    flat = monthly_series(generator.binomial(2000, 0.7, 36) / 2000)
    result = dynamics_of_one_series(flat, "period", configuration)
    check(0.4 < result["phi"] < 2.0 and result["estacional"] == 0 and result["tendencia"] == 0,
          "a constant rate with binomial noise: φ ≈ 1, no season, no trend")
    december_peak = np.array([0.70] * 11 + [0.85])
    seasonal = monthly_series(np.tile(december_peak, 3) + generator.normal(0, 0.005, 36))
    result = dynamics_of_one_series(seasonal, "period", configuration)
    check(result["estacional"] == 1 and result["meses_alto"] == "12" and result["phi"] > 5,
          "a December peak every year: seasonal, December is a high month, φ well above 1")
    trending = monthly_series(np.linspace(0.60, 0.75, 36))
    result = dynamics_of_one_series(trending, "period", configuration)
    check(result["tendencia"] == 1 and 4 < result["tendencia_pp_ano"] < 6, "a rate rising 5 pp a year: trend +1, ≈ +5 pp/year")


def test_the_report() -> None:
    print("B · the report on the synthetic")
    configuration = synthetic_with(output_folder=os.path.join(os.path.dirname(configuration_path()), "salida_test"))
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    report_path = os.path.join(configuration.output_folder, "informe_sff.md")
    report = open(report_path, encoding="utf-8").read()
    check(report.count("\n## ") == 9 and "## 0 · El veredicto" in report and "### Qué tal se predice la tasa" in report
          and "## 7 · B7 · El forecast en dinero" in report and "## 4 · B4 · Los estados del cliente" in report,
          "the report has its nine chapters: the verdict and one per block (the rate inside B2, the states in B4)")
    check("**Los huecos:** 11 en 1 series" in report and "Resultados adelantados borrados" in report
          and "| resultado_adelantado_borrado |" in report and "| proyectada |" in report,
          "the report tells, in one format, the rows the framework adds or wipes (gaps, wiped, created)")
    check("El error frente al ruido" in report and "parte_del_error_que_es_ruido" in report,
          "the report tells how much of the exam error is noise, by size of series and for the total")
    check(results["report"] == report and "sff_ficha_serie" not in report and "### Dónde está cada serie" in report,
          "the step returns the text of the report; every series is in sff_forecast_series (no card table)")
    check("s2_5_phi" in results["forecast_series"].columns, "the forecast series dimension carries the dynamics of its composition")
    check("4 checks: 4 ok" in step_section(console, "8.6"),
          "the checks of the report pass: its chapters, sff_veredicto_bloques and sff_perfil_datos written and read back, "
          "and the year by region adds up to the total of the year")
    section = step_section(console, "8.6")
    check("THE YEAR 2026 BY" in section and "usd_real_hasta_hoy" in section and "usd_esperado_hasta_fin_de_ano" in section
          and "UNION ALL" in section,
          "the final table: the year by region, real so far and expected to December, with the SQL that gives it")
    from report_queries import year_by_region_query
    by_region = pd.read_sql(year_by_region_query(configuration), configuration.sql_engine)
    subtotals = by_region[by_region["orden_region"] == 1]
    detail = by_region[(by_region["orden_total"] == 0) & (by_region["orden_region"] == 0)]
    detail = detail if len(detail) else subtotals          # one region level declared: the regions are the subtotals
    total = by_region[by_region["orden_total"] == 1].iloc[0]
    check(abs(detail["usd_total_ano"].sum() - total["usd_total_ano"]) < 1e-6
          and abs(subtotals["usd_total_ano"].sum() - total["usd_total_ano"]) < 1e-6
          and abs(total["usd_real_hasta_hoy"] + total["usd_esperado_hasta_fin_de_ano"] - total["usd_total_ano"]) < 1e-6,
          "every region adds up to its subtotal and to the TOTAL; real + expected = the year")
    verdicts = configuration.block_verdicts
    check(sorted(verdicts) == ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"] and all(len(frame) for frame in verdicts.values())
          and "VEREDICTO DEL BLOQUE 3 · PRECIO" in console and "VEREDICTO DEL BLOQUE 7 · EXAMEN Y MONTAJE" in console,
          "every block closes with its verdict on screen, and every verdict has rows")
    profile = configuration.data_profile
    check({"cartera", "mes", "dimension", "concentracion", "soporte", "historia"} <= set(profile["seccion"])
          and abs(profile.loc[profile["seccion"] == "soporte", "cuota_usd_futuro"].sum() - 1) < 1e-9,
          "the profile of the pipeline has its six sections; the support buckets share out the money to predict")


def test_the_verdict_rules() -> None:
    print("C · the rules of the verdicts (verdicts.py)")
    from verdicts import gain_verdict, paired_comparison
    from config import VERDICT_FAILS, VERDICT_HOLDS, VERDICT_INCONCLUSIVE
    significance = synthetic_with().verdict_significance
    always_better = paired_comparison([0.01] * 6, [0.02] * 6)
    check(always_better["gana"] == 6 and abs(always_better["p_signos"] - 2 / 64) < 1e-12
          and gain_verdict(always_better, significance) == VERDICT_HOLDS,
          "6 of 6 months better: sign test p = 2 / 2^6 = 0.031 → se cumple")
    five_of_six = paired_comparison([0.010, 0.010, 0.010, 0.010, 0.010, 0.030], [0.011, 0.012, 0.009 + 0.004, 0.011, 0.012, 0.020])
    check(five_of_six["gana"] == 5 and abs(five_of_six["p_signos"] - 14 / 64) < 1e-12,
          "5 of 6 months better: sign test p = 14 / 64 = 0.22, not significant on its own")
    worse = paired_comparison([0.02] * 3, [0.01] * 3)
    check(gain_verdict(worse, significance) == VERDICT_FAILS and gain_verdict(paired_comparison([], []), significance) == VERDICT_INCONCLUSIVE,
          "worse on average → no se cumple; no months → no concluyente")


def configuration_path():
    import tempfile
    return tempfile.mkdtemp() + "/x"


def test_a_constant_series() -> None:
    print("C · series that never vary (0 % every month; 1-2 units that always renew): no numpy warning")
    import warnings
    import numpy as np
    from block_2_rate import dynamics_of_one_series
    tiny = np.r_[np.full(30, 1.0), np.full(6, 2.0)]
    for name, due, renewed in (("0 % every month", 10.0, 0.0), ("1-2 units that always renew", tiny, tiny)):
        months = pd.DataFrame({"period": pd.period_range("2023-01", periods=36, freq="M"), "vencen": due, "renovadas": renewed})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                dynamics_of_one_series(months, "period", synthetic_with())
                clean = True
            except RuntimeWarning:
                clean = False
        check(clean, f"{name}: measured without a RuntimeWarning (a profile that does not move has no correlation)")


if __name__ == "__main__":
    test_the_dynamics()
    test_the_report()
    test_the_verdict_rules()
    test_a_constant_series()
    finish()
