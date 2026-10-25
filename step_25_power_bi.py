"""
step_25_power_bi.py — The adaptation to Power BI: the auxiliary tables the report needs, so that
Power BI only relates and sums.

The core (sff_nucleo) and the forecast series dimension (sff_forecast_series) carry the numbers. What
Power BI still needs is the context to show them to business: a name for every code, an order, a block.
That context is built here, once, in Python, instead of as calculated tables or DAX inside the report.

Tables written (one per auxiliary table; this step will grow with the report):

  sff_price_increase_calendar    one row per increase detected: price group, start month, size of the
                                 step, uplift before and during the cycle, the renewed USD it touched
  sff_price_increase_monitor     one row per PRICE GROUP (price_group_dims: a tariff changes by region and
                                 product) and closed pipeline month: the uplift of everyone, the uplift
                                 of the ISOLATED renewals (the detector: the normal renewal process, no
                                 retention discounts in it), the step against the average of the previous 12
                                 months, and the increase cycle (a price increase affects the renewals
                                 of ONE cycle: who renews pays the new price, but a year later everyone
                                 already bought at it, so the step lasts configuration.price_cycle_months and then fades)
  sff_parametros                 one row per field of the Config: its value in this run, its default,
                                 whether it was changed, the steps whose code reads it and the steps
                                 its comment documents, and the comment (what it decides)
  sff_forecast_pipeline_source   one row per value of forecast_pipeline_source (the origin of every
                                 row of the core): its business label, the block it belongs to and its
                                 order in the report. Relation in Power BI:
                                     sff_forecast_pipeline_source[forecast_pipeline_source]
                                         1 → n  sff_nucleo[forecast_pipeline_source]

Actions (logged as they are done):
  1. the table of pipeline sources: code, label, block, order
  2. the price increase monitor and its calendar: the uplift of the closed months and its steps
  3. the parameters of this run: every field of the Config, its value and the steps that read it
  4. check the tables against the core                                   checks 1-4
  5. write them                                                          checks 5-8
  6. count the checks; stop if any failed
  7. show the tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every forecast_pipeline_source of the core has its row (no row of the core left without a label)
   2. one row per source (the key is unique: the relation in Power BI is 1 → n)
   3. every closed pipeline month with renewals has its uplift in the monitor
   4. an increase is only flagged against a reference of enough previous months
   5-8. tables sff_forecast_pipeline_source, sff_price_increase_monitor, sff_price_increase_calendar and
        sff_parametros written and read back
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import (Config, ROW_FROM_GAP, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
                    TOTAL_ORIGIN_SIMULATED, TS_PROJECTED, TS_REAL, TS_REENTRY, parameter_table)
from price_increases import increase_calendar, price_monitor
from step_24_core import FINAL_ORIGIN_COLUMN, TS_WITHOUT_RESULT


STEP_LABEL = "25"
STEP_NAME = "ADAPTATION TO POWER BI"
STEP_PURPOSE = ("build the auxiliary tables of the Power BI report (names, orders and blocks of the codes of the core), "
                "so that Power BI only relates and sums: no logic inside the report")
STEP_ACTIONS = ["the table of pipeline sources: code, label, block, order",
                "the price increase monitor: the uplift of the closed months and its steps",
                "the parameters of this run: every field of the Config, its value and the steps that read it",
                "check the tables against the core (checks 1-4)",
                "write them (checks 5-8)",
                "count the checks; stop if any failed",
                "show the tables"]
STEP_OUTPUT = ("tables sff_forecast_pipeline_source (one row per forecast_pipeline_source), sff_price_increase_monitor "
               "(price group × closed pipeline month), sff_price_increase_calendar (one row per increase) and "
               "sff_parametros (one row per field of the Config)")

PIPELINE_SOURCE_TABLE = "forecast_pipeline_source"
PRICE_MONITOR_TABLE = "price_increase_monitor"
PARAMETERS_TABLE = "parametros"
PRICE_CALENDAR_TABLE = "price_increase_calendar"

# the detector of price increases, over the uplift of the isolated renewals (no retention event):
                               # (one noisy month is not an increase; the price of it: a step is confirmed 2 months late)

# the blocks of the report: where each source sits when the pipeline of a year is broken down
BLOCK_EXTRACT_PIPELINE = "Extract pipeline"        # due dates already in the extract
BLOCK_EXTENDED_HORIZON = "Extended horizon"        # pipeline built by the framework (simulation window, +12 months)
BLOCK_TIME_SERIES = "Time series"                  # the retail to subscription universe
BLOCK_CONTROL = "Control"                          # rows with no money: they keep the sums equal to the reporting

# one row per source: (code in the core, business label, block, order in the report)
PIPELINE_SOURCES = [
    (TOTAL_ORIGIN_RENEWED,   "Pipeline already due (actual renewal)",    BLOCK_EXTRACT_PIPELINE, 1),
    (TOTAL_ORIGIN_EXPECTED,  "Known pipeline still to fall due",         BLOCK_EXTRACT_PIPELINE, 2),
    (TOTAL_ORIGIN_PROJECTED, "Re-renewal of 1-year licences",            BLOCK_EXTENDED_HORIZON, 3),
    (TOTAL_ORIGIN_SIMULATED, "Simulated acquisition",                    BLOCK_EXTENDED_HORIZON, 4),
    (TS_REAL,                "Retail to subscription (actual)",          BLOCK_TIME_SERIES,      5),
    (TS_PROJECTED,           "Retail to subscription (projected)",       BLOCK_TIME_SERIES,      6),
    (TS_REENTRY,             "Retail to subscription (re-entry)",        BLOCK_TIME_SERIES,      7),
    (ROW_FROM_GAP,           "Months with nothing due",                  BLOCK_CONTROL,          8),
    (TS_WITHOUT_RESULT,      "Time series month not closed yet",         BLOCK_CONTROL,          9),
]


def build_power_bi_tables(core_table: pd.DataFrame, configuration: Config) -> dict:
    """The auxiliary tables of the Power BI report.

    INPUT:   the core (step 24), with its column forecast_pipeline_source.
    OUTPUT:  {"pipeline_source": the table of pipeline sources}.
    RULES:   every code of the core has its row; the table lists every source the framework can produce,
             also those absent in this run (a run without acquisition has no simulated acquisition, and the
             report keeps the same rows).
    EDGE CASES: a code in the core that the table does not know stops the step (check 1): a new origin
             was added to the core and this table was not updated.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the table of pipeline sources
    pipeline_source_table = build_pipeline_source_table()
    configuration.log_action(STEP_LABEL, 1, f"{len(pipeline_source_table)} pipeline sources in "
                                            f"{pipeline_source_table['source_block'].nunique()} blocks")

    # [2] the price increase monitor, per price group, and the calendar of the increases it detects
    price_monitor = price_increase_monitor(core_table, configuration)
    calendar = increase_calendar(price_monitor, configuration)
    groups = price_monitor["price_group"].nunique()
    configuration.log_action(STEP_LABEL, 2, f"{groups:,} price groups ({', '.join(configuration.price_group_dims) or 'the portfolio'}) "
                                            f"× {price_monitor['period'].nunique()} closed months · detector on "
                                            f"{price_monitor.attrs['detector_series']} · {len(calendar)} increases detected in "
                                            f"{calendar['price_group'].nunique() if len(calendar) else 0} groups")

    # [3] the parameters of this run
    parameters = parameter_table(configuration)
    changed = parameters[parameters["cambiado"] == 1]
    configuration.log_action(STEP_LABEL, 3, f"{len(parameters)} parameters in the Config · {len(changed)} with a value of "
                                            f"this run (not its default) · "
                                            f"{int((parameters['pasos'] == '').sum())} read by no step")

    # [4] the checks against the core
    configuration.log_action(STEP_LABEL, 4, "checking the tables against the core")
    sources_in_core = set(core_table[FINAL_ORIGIN_COLUMN].dropna().unique())
    sources_in_table = set(pipeline_source_table[FINAL_ORIGIN_COLUMN])
    sources_without_row = sorted(sources_in_core - sources_in_table)
    configuration.log_check(STEP_LABEL, check_log,
                            "every forecast_pipeline_source of the core has its row",
                            not sources_without_row,
                            failure_detail=f"sources of the core without a row: {sources_without_row}",
                            context=f"{len(sources_in_core)} sources in the core, "
                                    f"{len(sources_in_table - sources_in_core)} with no rows in this run")

    repeated_sources = pipeline_source_table[pipeline_source_table[FINAL_ORIGIN_COLUMN].duplicated()][FINAL_ORIGIN_COLUMN].tolist()
    configuration.log_check(STEP_LABEL, check_log,
                            "one row per source (the relation in Power BI is 1 → n)",
                            not repeated_sources,
                            failure_detail=f"repeated sources: {repeated_sources}")

    with_renewals = price_monitor[price_monitor["renewed_units"] > 0]
    configuration.log_check(STEP_LABEL, check_log,
                            "every closed pipeline month with renewals has its uplift in the monitor",
                            with_renewals["uplift"].notna().all(),
                            failure_detail=f"{int(with_renewals['uplift'].isna().sum())} months with renewals and no uplift",
                            context=f"{len(price_monitor):,} months")
    flags_without_reference = price_monitor[(price_monitor["price_increase_flag"] == 1)
                                            & price_monitor["uplift_previous_12"].isna()]
    configuration.log_check(STEP_LABEL, check_log,
                            f"an increase is only flagged against a reference of ≥ {configuration.price_min_reference_months} previous months",
                            flags_without_reference.empty,
                            failure_detail=f"{len(flags_without_reference)} months flagged without a reference")

    # [5] the writes
    configuration.log_action(STEP_LABEL, 5, "writing the tables")
    configuration.write_table(STEP_LABEL, check_log, pipeline_source_table, PIPELINE_SOURCE_TABLE)
    configuration.write_table(STEP_LABEL, check_log, price_monitor.drop(columns=["renewed_units"]), PRICE_MONITOR_TABLE)
    configuration.write_table(STEP_LABEL, check_log, calendar, PRICE_CALENDAR_TABLE)
    configuration.write_table(STEP_LABEL, check_log, parameters, PARAMETERS_TABLE)

    # [6] the count
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the tables on screen
    configuration.log_action(STEP_LABEL, 7, "the pipeline sources, as Power BI will show them, the last 13 months of the "
                                            "price increase monitor, and the parameters of this run by step:")
    configuration.show_table(pipeline_source_table)
    configuration.logger.doc(f"[{STEP_LABEL}] ══════════ CALENDARIO DE SUBIDAS DETECTADAS ══════════")
    configuration.logger.doc(f"[{STEP_LABEL}] por grupo de precio ({', '.join(configuration.price_group_dims) or 'cartera'}): un "
                             f"escalón ≥ {configuration.price_step_threshold:.0%} frente a sus 12 meses previos, mantenido "
                             f"{configuration.price_persistence_months} meses, en meses con ≥ "
                             f"{configuration.price_group_min_renewed_units:,.0f} renovaciones; las 30 de más valor:")
    if len(calendar):
        shown_calendar = calendar.head(30).copy()
        for column_name in ["escalon", "uplift_antes", "uplift_ciclo"]:
            shown_calendar[column_name] = shown_calendar[column_name].round(3)
        shown_calendar["usd_renovado_ciclo"] = shown_calendar["usd_renovado_ciclo"].round(0)
        configuration.show_table(shown_calendar.drop(columns=["price_group"]))
    else:
        configuration.logger.doc(f"[{STEP_LABEL}] no increase detected in any group")
    configuration.logger.doc(f"[{STEP_LABEL}] ══════════ FIN DEL CALENDARIO DE SUBIDAS ══════════")
    configuration.show_table(parameters[["parametro", "valor", "por_defecto", "cambiado", "pasos"]])

    return {"pipeline_source": pipeline_source_table, "price_monitor": price_monitor.drop(columns=["renewed_units"]),
            "price_calendar": calendar, "parameters": parameters}



def price_increase_monitor(core_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per price group and closed pipeline month of the core: the uplifts and the detection of the
    price increases (price_increases.price_monitor, the same that steps 14 and 16 read from the fine table)."""
    closed = core_table[(core_table["forecast_status"] == "actual") & (core_table["forecast_universe"] == "pipeline")
                        & (core_table["forecast_to_renew_units"] > 0)]
    return price_monitor(closed, configuration)


def build_pipeline_source_table() -> pd.DataFrame:
    """One row per pipeline source: code, business label, block and order."""
    pipeline_source_table = pd.DataFrame(PIPELINE_SOURCES,
                                         columns=[FINAL_ORIGIN_COLUMN, "source_label", "source_block", "source_order"])
    return pipeline_source_table
