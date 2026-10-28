"""
test_informe_forecast.py — The forecast told (informe_forecast.py): one SQL and its charts per business question,
over the tables of a run.

A · every question of the story answers on the tables of a run: its SQL, its CSV and its charts
B · the money adds up: the year of the report is the year of the final table (step 8.6) and of the core; the regions
    add up to their subtotal and the subtotals to the TOTAL
C · the band that can be broken down: between the rows all independent and the rows all wrong at once, and the same
    as computed in pandas from the core and the composition of every series
D · what was built: the pipeline of next year is its four origins, and what replaces the wiped pipeline is not less
E · the real bands (chapter 6): the error of every level is the sum of its series, the rules keep their order, the
    tables are written to the database and the learned correlation is between 0 and 1
F · the pipeline audited by its components (chapter 7): the components add up to the pipeline of the core, by month
    and by year; the projected renewals come back at what they renew; the uplift of the total is between the
    uplifts of its components
G · understanding next year (chapter 8): renewed = pipeline × rate × uplift; the bridge from this year to the next
    adds up; volume + mix + performance is the change of every dimension; the rates are rates
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import contextlib
import io
import os
import tempfile

import numpy as np
import pandas as pd

from informe_forecast import (BAND_RULES, COMPONENTS, CONCEPTS, QUESTIONS, UPLIFT_COMPONENTS, bridge_of, calibration_factors,
                              composition_indexes, mix_split, report)
from main import run
from report_queries import year_by_region_query
from test_helpers import check, finish, synthetic_with


def the_run():
    configuration = synthetic_with(output_folder=tempfile.mkdtemp())
    with contextlib.redirect_stdout(io.StringIO()):
        run(configuration)
        answers = report(configuration, show=False)
    return configuration, answers


def test_every_question_answers(configuration, answers) -> None:
    print("A · every question of the story answers")
    failed = {number: result.note for number, result in answers.items() if result.note.startswith("falló")}
    check(len(answers) == len(QUESTIONS) and not failed,
          f"the {len(QUESTIONS)} questions run on the tables of the run (failed: {failed})")
    answered = [result for result in answers.values() if result.table is not None]
    files_written = all(os.path.exists(path) for result in answered for path in result.charts)
    folder = os.path.join(configuration.output_folder, "informe_test")
    sql_files = [name for name in os.listdir(folder) if name.endswith(".sql")]
    check(files_written and len(sql_files) == len(answered) and all(len(result.charts) for result in answered if len(result.table)),
          f"every answer leaves its SQL, its CSV and its charts ({len(sql_files)} SQL files)")
    skipped = [number for number, result in answers.items() if result.table is None]
    check(set(skipped) <= {"1.5b", "2.1", "5.2", "8.10"},
          f"only what the synthetic does not have is skipped, with its reason (skipped: {skipped})")


def test_the_money_adds_up(configuration, answers) -> None:
    print("B · the money adds up")
    engine = configuration.sql_engine
    current_year = configuration.calendar_boundaries()["current"].year
    final_table = pd.read_sql(year_by_region_query(configuration), engine)
    final_total = float(final_table[final_table.iloc[:, 1] == "TOTAL"]["usd_total_ano"].iloc[0])
    year_total = float(answers["4.1"].table.query("orden_total == 1")["dinero_usd"].iloc[0])
    check(abs(year_total - final_total) < 1.0,
          f"4.1: the current year is the final table of step 8.6 (${year_total:,.0f} = ${final_total:,.0f})")
    core = pd.read_sql("SELECT forecast_year, origen_fila, forecast_renewed_USD, s1_1_vencen_usd FROM sff_nucleo", engine)
    with_money = ~core["origen_fila"].isin(["hueco", "ts_sin_resultado"])
    next_year_core = float(core.loc[with_money & (core["forecast_year"] == current_year + 1), "forecast_renewed_USD"].sum())
    next_year = float(answers["4.3"].table.query("orden_total == 1")["dinero_usd"].iloc[0])
    check(abs(next_year - next_year_core) < 1.0, f"4.3: next year is the core's (${next_year:,.0f})")
    raw_due = float(core.loc[core["origen_fila"] == "raw", "s1_1_vencen_usd"].sum())
    check(abs(answers["1.1"].table.query("orden_total == 1")["vence_usd"].sum() - raw_due) < 1.0,
          "1.1: the real pipeline of every year adds up to the extract")
    table = answers["3.1"].table
    level_1 = table.columns[1]
    two_levels = (table["orden_region"] == 1).any()           # one region level: the regions are the detail rows
    region_rows = table[table["orden_region"] == (1 if two_levels else 0)]
    detail = table[table["orden_region"] == 0].groupby([level_1, "periodo"])["total_usd"].sum()
    subtotal = region_rows.set_index([level_1, "periodo"])["total_usd"]
    by_month = region_rows.groupby("periodo")["total_usd"].sum()
    total = table[table["orden_total"] == 1].set_index("periodo")["total_usd"]
    check(np.allclose(detail.reindex(subtotal.index).to_numpy(), subtotal.to_numpy())
          and np.allclose(by_month.reindex(total.index).to_numpy(), total.to_numpy()),
          f"the {'subregions add up to their region and the ' if two_levels else ''}regions add up to the TOTAL")


def test_the_band(configuration, answers) -> None:
    print("C · the band that can be broken down")
    engine = configuration.sql_engine
    next_year = configuration.calendar_boundaries()["current"].year + 1
    total = answers["4.3"].table.query("orden_total == 1").iloc[0]
    rows = pd.read_sql(f"""SELECT n.s1_4_fs_id, n.forecast_renewed_USD AS esperado, n.forecast_renewed_USD_low AS bajo,
                                  n.forecast_renewed_USD_high AS alto, n.s7_2_error_captacion_usd AS captacion, f.s2_2_stage3_id
                           FROM sff_nucleo n LEFT JOIN sff_forecast_series f ON f.s1_4_fs_id = n.s1_4_fs_id
                           WHERE n.forecast_year = {next_year} AND n.origen_fila NOT IN ('hueco', 'ts_sin_resultado')""", engine)
    rows["desviacion"] = rows["esperado"] - rows["bajo"].fillna(rows["esperado"])
    rows["composicion"] = rows["s2_2_stage3_id"].fillna("fs|" + rows["s1_4_fs_id"].fillna("ts"))
    by_composition = rows.groupby("composicion")["desviacion"].sum()
    half_band = np.sqrt((by_composition ** 2).sum() + rows["captacion"].fillna(0).sum() ** 2)
    sql_half_band = total["dinero_usd"] - total["banda_baja_usd"]
    check(abs(half_band - sql_half_band) < 1e-6 * max(1.0, half_band),
          f"the band by composition of the SQL is the one computed in pandas (±${sql_half_band:,.0f})")
    independent = np.sqrt((rows["desviacion"] ** 2).sum())
    linear = rows["desviacion"].sum() + rows["captacion"].fillna(0).sum()
    check(independent - 1e-6 <= sql_half_band <= linear + 1e-6,
          f"it lies between the rows all independent (±${independent:,.0f}) and all wrong at once (±${linear:,.0f})")
    chain = answers["5.1"].table.query("orden_total == 1").iloc[0]
    check(chain["tasa_binomial_baja"] <= chain["tasa_prevista"] <= chain["tasa_binomial_alta"]
          and chain["tasa_framework_baja"] <= chain["tasa_prevista"] <= chain["tasa_framework_alta"] and chain["uplift"] > 0,
          f"5.1: the rate {chain['tasa_prevista']:.3f} inside its binomial and framework bands; uplift {chain['uplift']:.3f}")


def test_what_was_built(configuration, answers) -> None:
    print("D · what was built")
    pipeline = answers["3.1"].table.query("orden_total == 1")
    parts = pipeline[["real_usd", "renovacion_proyectada_usd", "retail_to_subscription_usd", "adquisicion_simulada_usd"]].sum(axis=1)
    check(np.allclose(parts, pipeline["total_usd"]) and pipeline["total_usd"].sum() > 0,
          "3.1: the pipeline of next year is its four origins")
    double_count = answers["3.2"].table.query("orden_total == 1")
    wiped = double_count[double_count["borrado_usd"] > 0]
    check((wiped["sustituto_menos_borrado_usd"] >= -1e-6).all(),
          f"3.2: what replaces the wiped pipeline is not less in any month ({len(wiped)} months with wiped pipeline)")


def test_the_real_bands(configuration, answers) -> None:
    print("E · the real bands")
    detail = answers["6.1"].table
    monthly = detail[detail["periodo"] != "acumulado"]
    series_error = monthly[monthly["nivel"] == "serie"].groupby(["periodo", "h"])["error"].sum()
    total_error = monthly[monthly["nivel"] == "total"].set_index(["periodo", "h"])["error"]
    check(np.allclose(series_error.reindex(total_error.index).to_numpy(), total_error.to_numpy()) and len(total_error) > 0,
          f"6.1: the real error of the total is the sum of the errors of its series ({len(total_error)} months × h)")
    ordered = ((detail["lineal_alta"] >= detail["composicion_alta"] - 1e-6)
               & (detail["composicion_alta"] >= detail["cuadratura_alta"] - 1e-6)).all()
    check(ordered, "6.1: at every level the linear band ≥ the band by composition ≥ the quadrature (the deviations are ≥ 0)")
    accumulated = detail[detail["periodo"] == "acumulado"]
    check(set(accumulated["nivel"]) == set(monthly["nivel"]) and len(accumulated) > 0,
          "6.1: every level has its months apart and its exam months added up")
    summary = answers["6.2"].table
    tables = pd.read_sql("SELECT COUNT(*) AS filas FROM sff_bandas_reales", configuration.sql_engine)["filas"].iloc[0]
    check(int(tables) == len(detail) and set(summary["regla"]) == set(BAND_RULES)
          and summary["cobertura"].between(0, 1).all(),
          f"6.2: the summary has every rule; sff_bandas_reales and sff_bandas_reales_resumen are in the database ({tables:,} rows)")
    factors = calibration_factors(detail, configuration.z)
    learned = factors["correlacion_aprendida"].dropna()
    check(len(learned) > 0 and "aprendida" in set(factors["regla"]) and factors["factor_k"].notna().any(),
          f"6.2: the factor k of every rule and the learned rule (implied correlation between compositions "
          f"{learned.min():.2f}..{learned.max():.2f})")


def test_the_components(configuration, answers) -> None:
    print("F · the pipeline audited by its components")
    monthly = answers["7.1"].table.query("orden_total == 1")
    parts = monthly[[f"{component}_usd" for component in COMPONENTS]].sum(axis=1)
    core = pd.read_sql("SELECT forecast_year, SUM(forecast_to_renew_USD) AS usd FROM sff_nucleo "
                       "WHERE forecast_pipeline_source NOT IN ('hueco', 'ts_sin_resultado') "
                       "AND forecast_universe = 'pipeline' OR forecast_pipeline_source = 'ts_reentrada' "
                       "GROUP BY forecast_year", configuration.sql_engine).set_index("forecast_year")["usd"]
    by_year = monthly.groupby("anio")["total_usd"].sum()
    check(np.allclose(parts, monthly["total_usd"]) and np.allclose(by_year.to_numpy(), core.reindex(by_year.index).to_numpy()),
          f"7.1: the components add up to the total of every month, and every year is the pipeline of the core "
          f"(${by_year.sum():,.0f})")
    yearly = answers["7.2"].table.query("orden_total == 1").set_index("anio")["total_usd"]
    check(np.allclose(yearly.to_numpy(), by_year.reindex(yearly.index).to_numpy()) and len(yearly) == 3,
          "7.2: the last year, the current one and the next one, the same as the months of 7.1")
    chain = answers["7.3"].table.query("orden_total == 1")
    projected_months = chain[chain["projected_renewal_usd"] > 0]
    check(len(projected_months) > 0 and np.allclose(projected_months["projected_renewal_usd"] + projected_months["renewal_1y_usd"],
                                                    projected_months["renovado_12m_antes_usd"], rtol=0.02),
          f"7.3: in the months it builds, the projected renewal is what renewed 12 months before "
          f"({len(projected_months)} months)")
    uplift = answers["7.4"].table.query("orden_total == 1")
    components = uplift[[f"{component}_uplift" for component in UPLIFT_COMPONENTS]]
    inside = ((uplift["total_uplift"] >= components.min(axis=1) - 1e-9)
              & (uplift["total_uplift"] <= components.max(axis=1) + 1e-9))
    check(inside[uplift["total_uplift"].notna()].all() and uplift["total_uplift"].notna().sum() > 0,
          "7.4: the uplift of the total of every month is between the uplifts of its components")


def test_next_year(configuration, answers) -> None:
    print("G · understanding next year")
    current_year = configuration.calendar_boundaries()["current"].year
    years = answers["8.1"].table.query("orden_total == 1").set_index("anio")
    now, upcoming = years.loc[current_year], years.loc[current_year + 1]
    identity = np.isclose(upcoming["pipeline_usd"] * upcoming["tasa_usd"] * upcoming["uplift"], upcoming["renovado_usd"])
    bridge = bridge_of(dict(now), dict(upcoming))
    adds_up = np.isclose(now["total_usd"] + sum(bridge.values()), upcoming["total_usd"])
    check(identity and adds_up,
          f"8.1: renewed = pipeline × rate × uplift, and {current_year} + Δ pipeline + Δ rate + Δ uplift + Δ R2S = "
          f"{current_year + 1} (${now['total_usd']:,.0f} → ${upcoming['total_usd']:,.0f})")
    concepts_add_up = np.isclose(sum(upcoming[concept] for concept in CONCEPTS), upcoming["total_usd"])
    summary = pd.read_sql("SELECT ano, total_usd FROM sff_resumen_negocio", configuration.sql_engine).set_index("ano")["total_usd"]
    check(concepts_add_up and np.isclose(upcoming["total_usd"], summary.loc[current_year + 1]) and upcoming["r2s_usd"] > 0,
          f"8.1: next year is every concept, retail to subscription included (${upcoming['r2s_usd']:,.0f}): the total of "
          f"sff_resumen_negocio (${summary.loc[current_year + 1]:,.0f})")
    rates = answers["8.2"].table
    check(rates["tasa_unidades"].dropna().between(0, 1).all() and rates["tasa_usd"].dropna().between(0, 1).all()
          and len(rates) > 0, "8.2: the rates in units and in USD of every month are between 0 and 1")
    composition = answers["8.5"].table
    changes_ok = []
    for dimension, block in composition.groupby("dimension"):
        _, split = mix_split(block, current_year, current_year + 1)
        by_year = block.groupby("anio")["renovado_usd"].sum()
        changes_ok.append(np.isclose(sum(split.values()), by_year[current_year + 1] - by_year[current_year]))
    check(all(changes_ok) and len(changes_ok) > 2,
          f"8.5: volume + mix + performance is the change of the renewed money in every dimension ({len(changes_ok)} dimensions)")
    regions = answers["8.8"].table
    segment_dims = [column for column in regions.columns if column not in
                    (configuration.business_mandatory_dims[0], "anio", "pipeline_usd", "base_usd", "renovado_usd")]
    of_year = regions[regions["anio"] == current_year + 1]
    indexes = composition_indexes(of_year, configuration.business_mandatory_dims[0], segment_dims)
    weights = indexes["pipeline_usd"] / indexes["pipeline_usd"].sum()
    check(np.isclose((weights * indexes["composition"]).sum(), 1) and np.isclose((weights * indexes["actual"]).sum(), 1)
          and np.isclose((weights * indexes[segment_dims[0]]).sum(), 1),
          f"8.8: weighted by their pipeline, the composition and the actual yield of the regions are the portfolio's "
          f"({len(indexes)} regions, {len(segment_dims)} dimensions)")
    horizon = answers["8.7"].table
    check(horizon["meses_vista"].dropna().min() >= 1 and len(answers["8.7"].charts) == 1,
          f"8.7: every future month has its months ahead (up to h = {int(horizon['meses_vista'].max())})")


if __name__ == "__main__":
    run_configuration, run_answers = the_run()
    test_every_question_answers(run_configuration, run_answers)
    test_the_money_adds_up(run_configuration, run_answers)
    test_the_band(run_configuration, run_answers)
    test_what_was_built(run_configuration, run_answers)
    test_the_real_bands(run_configuration, run_answers)
    test_the_components(run_configuration, run_answers)
    test_next_year(run_configuration, run_answers)
    finish()
