"""
test_8_2_audit.py — The 4 stages of the ladder and the audit tables: every stage adds up, an id
only changes when its group changes, and every satellite adds up to the table it comes from.

    python test_8_2_audit.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np

from main import run
from test_helpers import check, console_of, finish, synthetic_with, step_section


def run_synthetic():
    results = {}
    console = console_of(lambda: results.update(run(synthetic_with())))
    return results, console


def test_the_stages() -> None:
    print("A · the 4 stages of every forecast series")
    results, console = run_synthetic()
    stages, steps = results["ladder"]["stages"], results["ladder"]["steps"]
    tele = steps[steps["fs_id"] == "NA|A|0|0|0|0|tele"].sort_values("ladder_step")
    check(list(tele["group_id"][:2]) == ["NA|A|0|0|0|0|tele"] * 2 and tele["group_id"].iloc[2] == "NA|A|SIG=neutro|*"
          and tele["group_id"].iloc[-1] == "NA|A|SIG=neutro|*",
          "an id only changes when the group really changes: NA|A|tele keeps its own id until it joins its sibling web")
    negative = stages[stages["fs_id"] == "EU|A|0|0|1|0|web"].iloc[0]
    check(negative["stage0_id"] == "EU|A|0|0|1|0|web" and negative["stage0_support"] == 10
          and negative["stage1_id"] == "EU|A|SIG=negativo|web" and negative["stage1_support"] == 42
          and negative["stage3_id"] == negative["stage2_id"],
          "a negative series: own support 10 at stage 0, 42 after the sign; nothing more in stages 2 and 3")
    core, dimension = results["core"], results["forecast_series"]
    raw = core[core["origen_fila"] == "raw"].merge(dimension, on="s1_4_fs_id")
    check(all(raw.groupby(f"s2_2_stage{stage}_id")["s1_1_vencen_usd"].sum().sum() == raw["s1_1_vencen_usd"].sum() for stage in range(4)),
          "grouping the core (joined to its dimension) by the id of any stage, the USD due adds up to the same total")
    check(raw["s2_2_stage3_id"].nunique() <= raw["s2_2_stage0_id"].nunique(), "fewer groups at stage 3 than at stage 0")
    check({"s7_2_confidence"} <= set(core.columns) and {"s2_3_credibility_effect_pp", "s2_2_stage3_support", "s7_2_confidence"}
          <= set(dimension.columns),
          "the core carries the confidence of every row; the dimension the stages, the credibility effect and the confidence")


def test_the_audit_tables() -> None:
    print("B · the audit tables add up to their sources")
    results, console = run_synthetic()
    check("12 checks: 12 ok" in step_section(console, "8.2"), "the 12 checks of the audit step pass")
    audit = results["audit"]
    techniques = audit["composition_techniques"]
    check(set(techniques["status"]) <= {"tested", "not_enough_history", "composition_below_floor"}
          and (techniques.groupby(["composition_id", "tramo_h"])["is_chosen"].sum() == 1).all(),
          "every composition and band: one chosen technique; every technique with its status")
    members = audit["credibility_members"]
    check(set(members["role"]) <= {"borrower", "lender"} and len(audit["credibility"]) == members["credibility_ref_id"].nunique(),
          "every reference with its members, borrowers and lenders")
    check(not {"series_backtest", "series_technique_summary"} & set(audit),
          "the backtest of every series is not recomputed here: step 7.1 examines every forecast series")
    dynamics = audit["series_dynamics"]
    check(set(dynamics["measurable"]) <= {"yes", "low_support", "short_history"}
          and dynamics.loc[dynamics["measurable"] == "short_history", "phi"].isna().all(),
          "every forecast series has its dynamics, with whether it can be trusted")
    forecast_all = audit["composition_forecast_all"]
    check(forecast_all.groupby(["composition_id", "h"])["is_chosen"].sum().max() == 1 and forecast_all["rate"].between(0, 1).all()
          if forecast_all["rate"].notna().any() else False,
          "the future rate of every technique, one chosen per composition and horizon")


def test_the_exam_in_the_dimension() -> None:
    print("C · the exam of every forecast series: its interval, raw against framework, summed from the dimension")
    results, console = run_synthetic()
    detail = results["series_exam"]["detail"]
    row = detail[detail["method"] == "framework"].iloc[0]
    check(row["band_low"] <= row["pred_rate"] <= row["band_high"]
          and row["in_band"] == int(row["band_low"] - 1e-12 <= row["real_rate"] <= row["band_high"] + 1e-12),
          "every test has its interval around the prediction, and says whether the real rate fell inside")
    dimension, per_series = results["forecast_series"], results["series_exam"]["per_series"]
    check(dimension["s7_1_exam_predictions"].sum() == per_series["framework_predictions"].sum()
          and dimension["s7_1_exam_in_band"].sum() == per_series["framework_in_band"].sum()
          and abs(dimension["s7_1_exam_real_units"].sum() - per_series["framework_real_units"].sum()) < 1e-6,
          "SUM over the dimension counts every exam prediction once")
    check({"s7_1_raw_mae_pp", "s7_1_exam_mae_pp", "s7_1_improvement_mae_pp", "s2_2_stage0_support", "s2_2_stage3_support",
           "s1_7_error_binomial_pp", "s2_3_se_prediccion_pp"} <= set(dimension.columns),
          "the dimension shows the improvement over the raw: support, binomial error and exam error, raw against framework")
    check({"s2_5_series_phi", "s2_5_series_trend", "s2_5_series_seasonal", "s2_5_series_measurable"} <= set(dimension.columns),
          "the dimension carries each series' own dynamics, to cross it with the precision")
    report = open(f"{synthetic_with().output_folder}/informe_sff.md", encoding="utf-8").read()
    check("Cuánto acertamos" in report and "frente a la serie sola" in report,
          "the report says how much we erred, how many predictions were in their interval, and the framework against the raw")


if __name__ == "__main__":
    test_the_stages()
    test_the_audit_tables()
    test_the_exam_in_the_dimension()
    finish()
