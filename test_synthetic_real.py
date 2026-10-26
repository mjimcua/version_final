"""
test_synthetic_real.py — The synthetic that looks like the real extract (synthetic_real.py).

A · the extract: the production columns, one row per combination, the calendar and what is booked
B · the shape: the rates of the states and the terms, the discount unknown, the acquisitions
C · the framework reads it with the Config of the real run (main.like_real_configuration): steps 1.1-1.4
D · the maturation of softcancel: its columns pass steps 1.1-1.2, the leak check sees a wrong current month,
    and the curve learned from the closed months is the one the synthetic was built with
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import contextlib
import io
import os
import tempfile

import numpy as np
from sqlalchemy import create_engine

from synthetic_real import build_raw_like_real
from test_helpers import check, finish

SMALL = dict(seed=3, number_of_series=150, units_due_per_month=20000.0)


def test_the_extract() -> None:
    print("A · the extract: columns, grain, calendar, what is booked")
    raw = build_raw_like_real(**SMALL)
    expected_columns = {"period", "tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3", "tr_product_level_1",
                        "tr_product_level_2", "tr_renewal_type", "tr_term", "tr_band", "tr_master_partner_code", "dormant",
                        "softcancel", "not_installed", "net_new", "prev_OperationGroup", "discount", "sku", "flag_time_series",
                        "total_tr_units", "total_tr_usd", "total_renewed_units", "total_renewed_usd", "total_tr_usd_renewed",
                        "total_renewed_units_without_softcancel", "total_renewed_usd_without_softcancel",
                        "total_tr_usd_renewed_without_softcancel", "total_renewed_usd_sq_over_tr_isolated",
                        "total_reacquired_units", "total_reacquired_usd"}
    check(expected_columns <= set(raw.columns), "the columns of the production extract are there")
    keys = [column for column in raw.columns if not column.startswith("total_")]
    check(not raw.duplicated(keys).any(), "one row per combination of every column (the grain of the real extract)")
    check(raw["period"].min() == "2022-12" and raw["period"].max() == "2027-12", "the extract runs from 2022-12 to 2027-12")
    by_month = raw.groupby("period")[["total_tr_units", "total_renewed_units"]].sum()
    check(by_month.loc["2026-08", "total_renewed_units"] / by_month.loc["2026-08", "total_tr_units"] > 0.5
          and 0.3 < by_month.loc["2026-10", "total_renewed_units"] / by_month.loc["2026-09", "total_renewed_units"] < 0.7
          and by_month.loc["2027-03", "total_renewed_units"] == 0,
          "renewals: complete up to 2026-08, September booked, October about half, nothing later")
    check(by_month.loc["2027-10", "total_tr_units"] < 0.8 * by_month.loc["2027-08", "total_tr_units"]
          and by_month.loc["2027-11", "total_tr_units"] < 0.5 * by_month.loc["2027-08", "total_tr_units"],
          "the pipeline of 2027-10..12 is only partly built (the licences of 2026-10.. are not bought yet)")
    dormant_share = raw.groupby("period").apply(lambda month: (month["total_tr_units"] * month["dormant"]).sum()
                                                / month["total_tr_units"].sum())
    check(dormant_share.loc["2027-08"] < 0.6 * dormant_share.loc["2026-08"],
          "the states of the future months are only partly visible (the extract keeps their last state)")
    check((raw["total_tr_usd_renewed"] <= raw["total_tr_usd"] + 1e-6).all()
          and (raw["total_renewed_usd_without_softcancel"] <= raw["total_renewed_usd"] + 1e-6).all(),
          "the renewers' base never exceeds what fell due, the isolated renewed never exceeds the renewed")


def test_the_shape() -> None:
    print("B · the shape of the portfolio")
    raw = build_raw_like_real(**SMALL)
    closed = raw[raw["period"] <= "2026-08"]
    rate = closed["total_renewed_units"].sum() / closed["total_tr_units"].sum()
    check(0.55 < rate < 0.72, f"the closed months renew {rate:.0%} of the units (real 64 %)")
    neutral = closed[(closed[["dormant", "softcancel", "not_installed"]].sum(axis=1) == 0)]
    softcancel = closed[(closed["softcancel"] == 1) & (closed[["dormant", "not_installed"]].sum(axis=1) == 0)]
    check(neutral["total_renewed_units"].sum() / neutral["total_tr_units"].sum()
          > softcancel["total_renewed_units"].sum() / softcancel["total_tr_units"].sum() + 0.25,
          "a softcancel renews far less than a neutral licence (real 28 % vs 71 %)")
    unknown_usd = raw.loc[raw["discount"].isna(), "total_tr_usd"].sum() / raw["total_tr_usd"].sum()
    check(0.4 < unknown_usd < 0.8, f"the discount is unknown in {unknown_usd:.0%} of the USD due (real 59 %)")
    acquisitions = closed[closed["net_new"] != "Retention_Not-New"]
    check(0.1 < acquisitions["total_tr_units"].sum() / closed["total_tr_units"].sum() < 0.45,
          "the acquisitions are a minority of what falls due")
    river = raw[raw["tr_master_partner_code"] == "DigitalRiver"].groupby("period")["total_tr_units"].sum()
    check(len(river) == 0 or river.reindex(["2026-06"]).fillna(0).iloc[0] < 0.2 * river.max(),
          "the DigitalRiver series keep a history and almost nothing due at the end (the migration to NEXWAY)")


def test_the_framework_reads_it() -> None:
    print("C · the framework reads it with the Config of the real run (steps 1.1-1.4)")
    from main import like_real_configuration
    from block_1_preparation import validate_raw
    configuration = like_real_configuration("2026-10")
    fields = {name: getattr(configuration, name) for name in configuration.__dataclass_fields__}
    fields["sql_engine"] = create_engine(f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'test.db')}")
    fields["log_colors"] = False
    configuration = type(configuration)(**fields)
    raw = build_raw_like_real(**SMALL)
    with contextlib.redirect_stdout(io.StringIO()):
        validated = validate_raw(raw, configuration)
    check(len(validated) == len(raw), "step 1.1 accepts the extract: every column has its role in the Config of the real run")


def test_the_maturation() -> None:
    print("D · the maturation of softcancel: columns, leak check, curve")
    from main import like_real_configuration
    from maturation import MaturationCurve
    from block_1_preparation import validate_raw
    from maturation import closed_rows_of
    from block_1_preparation import validate_values_and_build_levels
    from block_1_preparation import apply_calendar
    from block_1_preparation import build_fine_table
    from synthetic_real import SOFTCANCEL_SEEN_MONTHS_BEFORE
    raw = build_raw_like_real(**SMALL)
    consoles = {}
    for month in ("2026-09", "2026-10"):
        configuration = like_real_configuration(month)
        fields = {name: getattr(configuration, name) for name in configuration.__dataclass_fields__}
        fields["sql_engine"] = create_engine(f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'test.db')}")
        fields["levels_path"] = os.path.join(tempfile.mkdtemp(), "levels.json")
        fields["log_colors"] = False
        configuration = type(configuration)(**fields)
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            validated = validate_raw(raw, configuration)
            leveled, _ = validate_values_and_build_levels(validated, configuration)
            fine_table = build_fine_table(apply_calendar(leveled, configuration), configuration)
        consoles[month] = (console.getvalue(), fine_table, configuration)
    october, fine_table, configuration = consoles["2026-10"]
    check("ok    on the rows with softcancel = 1 the maturation columns add up" in october
          and "ok    no unit of a future month has softcancel dated after the current month" in october,
          "with the right current month the softcancel dates add up and none is in the future")
    check("WARN  no unit of a future month has softcancel dated after the current month" in consoles["2026-09"][0],
          "with the current month one month early, the dates of October are in its future: the leak check warns")
    without = raw.drop(columns=["softcancel_month_0"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            validate_raw(without, configuration)
        stopped = False
    except ValueError as error:
        stopped = "softcancel_month_0" in str(error)
    check(stopped, "step 1.1 stops when a declared maturation column is missing")
    curve = MaturationCurve(closed_rows_of(fine_table, configuration), "softcancel", configuration).portfolio_table()
    check(np.allclose(curve["visible_del_final"].to_numpy()[:11], SOFTCANCEL_SEEN_MONTHS_BEFORE[:11], atol=0.03),
          "the curve learned from the closed months is the one the synthetic was built with (share seen k months before)")

    # the curve is used only if its exam passes; a threshold of 0 pp makes it fail: the state takes its historical share
    from block_4_customer_states import learn_the_maturation
    with contextlib.redirect_stdout(io.StringIO()):
        box = learn_the_maturation(fine_table, configuration)
    check(configuration.rejected_maturation_dims == [] and "softcancel" in box.curves
          and configuration.maturation_dims_in_use == ["softcancel"],
          "the softcancel curve passes its exam (it beats the states seen then): it is used")

    # the chart that explains the curve: when the state is set (a distribution), today's pipeline, the error by horizon
    from block_4_customer_states import maturation_explained
    from block_4_customer_states import MATURATION_EXPLAINED_CHART
    from time_series_charts import chart_path
    explained = maturation_explained(box, fine_table, "softcancel", configuration)
    when, today, exam = explained["when"], explained["today"], explained["exam"]
    check(abs(when["f"].sum() - 1) < 1e-9 and abs(when["F"].iloc[0] - 1) < 1e-9 and when["F"].is_monotonic_decreasing,
          "the explanation: f(k) is a distribution (it adds up to 1) and F(h) falls from 1 at the due month")
    final_share = today["visto_hoy"] + today["anade_la_curva"]
    check(len(today) > 0 and ((today["visto_hoy"] >= 0) & (final_share <= 1)).all() and explained["used"]
          and (exam["error_curva_abs_pp"] < exam["error_vista_abs_pp"]).all(),
          "the explanation: today's pipeline, seen + what the curve adds, and the curve beats the states seen at every horizon")
    check(os.path.exists(chart_path(configuration, MATURATION_EXPLAINED_CHART)),
          "step 4.1 draws the chart that explains the curve")
    strict_fields = {name: getattr(configuration, name) for name in configuration.__dataclass_fields__}
    strict_fields["maturation_curve_max_error_pp"] = 0.0
    strict = type(configuration)(**strict_fields)
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        strict_box = learn_the_maturation(fine_table, strict)
    check(strict.rejected_maturation_dims == ["softcancel"] and "softcancel" not in strict_box.curves
          and strict.maturation_dims_in_use == [] and "DECISIÓN softcancel: su curva NO se usa" in console.getvalue()
          and "WARN  every dated state's curve beats the states seen then" in console.getvalue(),
          "a curve that fails its exam is not used: the state takes its historical share, with a warning and the decision")
    check(not maturation_explained(strict_box, fine_table, "softcancel", strict)["used"],
          "the chart that explains the curve says a rejected curve is not used")


if __name__ == "__main__":
    test_the_extract()
    test_the_shape()
    test_the_framework_reads_it()
    test_the_maturation()
    finish()
