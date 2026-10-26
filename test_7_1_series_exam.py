"""
test_7_1_series_exam.py — The exam of every forecast series: three methods on the same rows (the series
alone, the framework, the spreadsheet), without peeking, every prediction traced and with its interval.

    python test_7_1_series_exam.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from test_helpers import check, console_of, finish, synthetic_with, step_section


def test_the_series_exam() -> None:
    print("A · the exam of every forecast series on the synthetic")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    check("8 checks: 8 ok" in step_section(console, "7.1"), "the 8 checks of step 7.1 pass")
    exam = results["series_exam"]
    detail, per_series, summary = exam["detail"], exam["per_series"], exam["summary"]
    rated = results["rated_units"]
    exam_real = rated[(rated["sintetica"] == 0) & (rated["rol"] == "examen")]
    check(abs(exam["by_month"][exam["by_month"]["h"] == 1]["renovadas_reales"].sum() - exam_real["total_renewed_units"].sum()) < 1e-6,
          "the real renewals of the exam are the renewals of every series in the exam months (nothing counted twice)")
    check(set(summary["metodo"]) == {"baseline", "raw", "por_defecto", "framework"} and set(summary["h"]) == {1, 6},
          "four ways on the same rows: the baseline, the series alone, the ladder with the default technique, the framework")
    check((detail.groupby("method").size().nunique() == 1), "every method predicts exactly the same rows")
    check((detail["origin"] == detail["period"] - detail["h"]).all(), "every prediction knows only up to its origin, T − h")
    framework = detail[detail["method"] == "framework"]
    traced = framework.dropna(subset=["composition_rate"])
    no_shift = traced[traced["credibility_shift_logit"] == 0]
    check(len(traced) > 0 and np.allclose(no_shift["pred_rate"], no_shift["composition_rate"]),
          "every framework prediction is traced: without credibility, the series' rate is its composition's prediction")
    with_interval = detail[detail["method"].isin(["raw", "por_defecto", "framework"])]
    check(with_interval["band_low"].notna().all() and (with_interval["band_low"] <= with_interval["pred_rate"] + 1e-12).all()
          and (with_interval["pred_rate"] <= with_interval["band_high"] + 1e-12).all()
          and detail.loc[detail["method"] == "baseline", "band_low"].isna().all(),
          "raw, por_defecto and framework predictions carry their interval; the baseline has none")
    check({"raw_mae_pp", "framework_mae_pp", "improvement_mae_pp", "raw_coverage", "framework_coverage"} <= set(per_series.columns)
          and per_series["fs_id"].is_unique,
          "per forecast series, raw and framework side by side: the improvement is one subtraction")
    expected_noise = 100 * np.sqrt(framework["pred_rate"] * (1 - framework["pred_rate"]) / framework["due_units"].clip(lower=1))
    check(np.allclose(framework["noise_pp"], expected_noise), "every prediction carries the binomial noise of the rate it tries to hit")
    check({"framework_rmse_pp", "framework_noise_pp", "framework_error_over_noise", "raw_error_over_noise"} <= set(per_series.columns)
          and "error_vs_ruido" in summary.columns and "s7_1_exam_error_over_noise" in results["forecast_series"].columns,
          "the error in units of noise: per series, per method and horizon, and in the forecast series dimension")
    by_method = summary.groupby("metodo")["wape_series"].mean()
    check(by_method["framework"] < by_method["baseline"],
          "series by series, the framework beats the baseline (its mandatory cell, last months) in the synthetic")
    real_usd_per_method = detail.groupby("method")["real_usd"].sum()
    check({"pred_usd", "real_usd", "err_usd"} <= set(detail.columns)
          and np.allclose(real_usd_per_method, real_usd_per_method.iloc[0]),
          "every prediction is also in USD, against the same real USD for every method")
    framework_row = summary[(summary["metodo"] == "framework") & (summary["h"] == 1)].iloc[0]
    baseline_row = summary[(summary["metodo"] == "baseline") & (summary["h"] == 1)].iloc[0]
    check(abs(framework_row["ganancia_usd_mes"] - (baseline_row["error_usd_medio"] - framework_row["error_usd_medio"])) < 1e-6
          and 0 <= framework_row["meses_mejor_que_baseline"] <= 1 and pd.isna(baseline_row["meses_mejor_que_baseline"]),
          "the gain over the baseline: its mean |error| in USD minus the method's, and in how many months it errs less")
    check("LA CASCADA: DE LA BASELINE AL FRAMEWORK" in console and "VALORACIÓN h = 1: la baseline yerra" in console,
          "on screen, the cascade from the baseline with its gain in USD and its consistency")
    report = open(f"{configuration.output_folder}/informe_sff.md", encoding="utf-8").read()
    check("framework vs baseline vs serie sola" in report, "the report puts the baseline in its headline")


if __name__ == "__main__":
    test_the_series_exam()
    finish()
