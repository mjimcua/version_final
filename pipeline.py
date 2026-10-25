"""
pipeline.py — The steps of the SFF with formal interfaces, run over a common context.

Four pieces (the pattern of a data pipeline: contract + node + data catalogue + pipeline):

  TableContract     a table declared once: its grain, its key (unique) and its required columns.
                    Checked when a step writes it: a missing column or a repeated key stops the run
                    at the step that produced it, with the table's name.
  Step              the common interface of a step: its name, the tables it READS, the tables it
                    WRITES, and run(context) → {table: value}. Every step of the SFF is one Step; the
                    step functions are unchanged (each Step adapts the context to its arguments).
  PipelineContext   the catalogue of the run: every table produced so far, by name, with the step
                    that produced it, plus the Config.
  Orchestrator      orders the steps by their dependencies (what each reads must be written before)
                    and runs them all, in that order, refusing a step whose inputs are missing.

In a notebook:
    pipeline = Orchestrator(configuration)
    results = pipeline.run()                 # every step, in the order of their dependencies
    pipeline.context["forecast"]             # any table of the run
    pipeline.order                           # the steps, in the order they run
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import sqlalchemy

import prediction
import techniques
from block_1_preparation import PreparationBlock
from config import COMPOSITION_ID_COLUMN, Config, RETIRED_TABLES, SERIES_ID_COLUMN, UNIT_ID_COLUMN, parameter_table
from step_07_dimensions import analyse_dimensions
from step_08_ladder_groups import build_ladder_groups
from step_09_ladder import climb_the_ladder
from step_10_pool_series import build_pool_series
from step_11_dynamics import measure_dynamics
from step_12_backtest import run_backtest
from step_14_uplift import estimate_uplift
from step_16_uplift_backtest import backtest_uplift
from step_17_customer_states import learn_the_maturation
from step_15_price_diagnostics import diagnose_the_price
from step_18_future_pipeline import build_the_future_pipeline
from step_13_rate_box import build_the_rate_box
from step_20_forecast import assemble_forecast
from step_26_validation import validate_chain
from step_19_series_exam import examine_series
from step_21_time_series import build_time_series_and_total
from step_22_new_rows import count_the_new_rows
from step_25_power_bi import build_power_bi_tables
from step_23_audit import build_audit_tables
from step_27_report import build_report
from step_24_core import build_core_table


class ContractError(ValueError):
    """A table that does not keep its contract, or a step run without its inputs."""


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CONTRACTS OF THE TABLES THAT CROSS STEPS
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TableContract:
    """A table declared once: grain, unique key, required columns. `part` reads a table inside a dict."""
    name: str
    grain: str
    key: tuple = ()
    required: tuple = ()
    part: str = None

    def validate(self, value, producer: str) -> None:
        frame = value[self.part] if self.part else value
        if not isinstance(frame, pd.DataFrame):
            raise ContractError(f"{self.name} (written by {producer}) is not a table")
        missing = [column for column in self.required + self.key if column not in frame.columns]
        if missing:
            raise ContractError(f"{self.name} (written by {producer}) lacks the columns {missing}; grain: {self.grain}")
        if self.key and frame.duplicated(list(self.key)).any():
            repeated = int(frame.duplicated(list(self.key)).sum())
            raise ContractError(f"{self.name} (written by {producer}): {repeated:,} repeated keys {list(self.key)}; grain: {self.grain}")


def contracts(configuration: Config) -> dict:
    """The contract of every table that crosses steps (by context name; several per name for a dict)."""
    period = configuration.period_col
    return {
        "fine_table": [TableContract("fine_table", "one row per fine row of the extract", (),
                                     (SERIES_ID_COLUMN, UNIT_ID_COLUMN, period))],
        "forecast_units": [TableContract("forecast_units", "forecast series × month", (UNIT_ID_COLUMN,), (SERIES_ID_COLUMN, period))],
        "series_table": [TableContract("series_table", "forecast series", (SERIES_ID_COLUMN,), ("ruta",))],
        "series_rate": [TableContract("series_rate", "forecast series", (SERIES_ID_COLUMN,),
                                      ("ruta", "signo", "n_propio", "tasa_propia", "usd_por_predecir"))],
        "rated_units": [TableContract("rated_units", "forecast series × month (gaps included)", (SERIES_ID_COLUMN, period), ("tasa",))],
        "ladder": [TableContract("ladder.groups", "estimable forecast series", (SERIES_ID_COLUMN,),
                                 (COMPOSITION_ID_COLUMN, "final_step", "group_support", "group_rate", "credibility_ref_id"), part="groups"),
                   TableContract("ladder.composition_members", "composition × series in its rate",
                                 (COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN), ("role",), part="composition_members"),
                   TableContract("ladder.steps", "forecast series × pass", (SERIES_ID_COLUMN, "ladder_step"), ("group_id",), part="steps")],
        "series_estimate": [TableContract("series_estimate", "forecast series", (SERIES_ID_COLUMN,),
                                          (COMPOSITION_ID_COLUMN, "z", "credibility_ref_id", "group_rate", "ref_rate", "tasa_estimada"))],
        "pool_series": [TableContract("pool_series", "composition × month", (COMPOSITION_ID_COLUMN, period), ("vencen", "renovadas", "tasa"))],
        "pool_reference": [TableContract("pool_reference", "composition", (COMPOSITION_ID_COLUMN,), ("gate", "n_pool"))],
        "price_increases": [TableContract("price_increases", "price increase in play at the current month (its cycle "
                                          "touches the uplift window or the future)", ("price_group", "inicio"),
                                          ("fin_ciclo", "factor"))],
        "forecast": [TableContract("forecast.forecast", "future fine row", (), ("tasa", "tasa_baja", "tasa_alta", "esperado_usd"),
                                   part="forecast")],
        "series_exam": [TableContract("series_exam.per_series", "forecast series", (SERIES_ID_COLUMN,),
                                      ("framework_predictions", "raw_predictions"), part="per_series")],
        "core": [TableContract("core", "fine row of the extract, gap, synthetic or time_series row", (), ("s03_fs_id",))],
        "forecast_series": [TableContract("forecast_series", "forecast series", ("s03_fs_id",), ())],
        "power_bi": [TableContract("power_bi.pipeline_source", "forecast pipeline source", ("forecast_pipeline_source",),
                                   ("source_label", "source_block", "source_order"), part="pipeline_source"),
                     TableContract("power_bi.price_monitor", "price group and closed pipeline month", ("price_group", "period"),
                                   ("uplift", "uplift_step", "price_increase_flag", "in_increase_cycle"),
                                   part="price_monitor"),
                     TableContract("power_bi.price_calendar", "increase detected", ("price_group", "inicio"),
                                   ("escalon", "uplift_antes", "usd_renovado_ciclo"), part="price_calendar"),
                     TableContract("power_bi.parameters", "parameter of the Config", ("parametro",),
                                   ("valor", "por_defecto", "cambiado", "pasos"), part="parameters")],
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP INTERFACE, THE CONTEXT, THE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Step:
    """A step: the tables it reads and writes, and how it runs on the context. A step never changes the Config."""
    name: str
    label: str
    reads: tuple
    writes: tuple
    run: Callable


@dataclass
class PipelineContext:
    """The catalogue of the run: every table produced so far, with the step that produced it."""
    configuration: Config
    tables: dict = field(default_factory=dict)
    producer: dict = field(default_factory=dict)
    block_seconds: dict = field(default_factory=dict)     # how long every step inside a block took

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def __getitem__(self, name: str):
        if name not in self.tables:
            raise ContractError(f"'{name}' has not been produced in this run")
        return self.tables[name]

    def put(self, name: str, value, step_name: str, table_contracts: dict) -> None:
        for contract in table_contracts.get(name, []):
            contract.validate(value, step_name)
        self.tables[name] = value
        self.producer[name] = step_name

    def results(self) -> dict:
        """The tables under the names the report, the validation and the tests read."""
        renamed = {"calendared_raw": "raw", "series_table": "series"}
        results = {renamed.get(name, name): value for name, value in self.tables.items()}
        if "series_exam" in self.tables:
            results["portfolio_exam"] = self.tables["series_exam"]["by_month"]
            results["portfolio_exam_summary"] = self.tables["series_exam"]["summary"]
        return results


class Orchestrator:
    """Orders the steps by their dependencies and runs them, all or one at a time."""

    def __init__(self, configuration: Config, steps: list = None):
        self.configuration = configuration
        # the parameters of the techniques and of the prediction are read from this run's Config
        techniques.apply_configuration(configuration)
        prediction.apply_configuration(configuration)
        self.steps = {step.name: step for step in (steps or sff_steps())}
        self.order = dependency_order(list(self.steps.values()))
        self.table_contracts = contracts(configuration)
        self.context = PipelineContext(configuration)
        self.producer_of = {table: step.name for step in self.steps.values() for table in step.writes}
        self.seconds = {}                                  # how long every step took in this run

    def log_changed_parameters(self) -> None:
        """At the start of a run: the parameters with a value of this run (not the Config's default), each with
        the steps that read it. The whole table, with every parameter, is written by step 25 (sff_parametros)."""
        parameters = parameter_table(self.configuration)
        changed = parameters[parameters["cambiado"] == 1]
        self.configuration.logger.doc(f"[ORQ] {len(parameters)} parameters in the Config · {len(changed)} with a value of this "
                                      f"run (the rest keep their default; all of them in step 25, sff_parametros):")
        for _, row in changed.iterrows():
            self.configuration.logger.doc(f"[ORQ]   {row['parametro']} = {row['valor']}   · steps {row['pasos'] or '—'}")

    def warn_retired_tables(self) -> None:
        """At the start of a run: the tables an earlier version wrote and this one no longer writes, if they are still
        there (they would be read as if they were of this run). It does not drop them: it prints the SQL to do it."""
        configuration = self.configuration
        names = [f"{configuration.table_prefix}{table_name}" for table_name in RETIRED_TABLES]
        if configuration.sql_engine is not None:
            existing = set(sqlalchemy.inspect(configuration.sql_engine).get_table_names(schema=configuration.sql_schema))
            stale = [name for name in names if name in existing]
            qualified = [f"{configuration.sql_schema}.{name}" if configuration.sql_schema else name for name in stale]
            how = "; ".join(f"DROP TABLE {name}" for name in qualified)
        else:
            folder = configuration.output_folder or "."
            stale = [name for name in names if os.path.exists(os.path.join(folder, f"{name}.csv"))]
            how = "delete " + ", ".join(f"{name}.csv" for name in stale) + f" in {folder}"
        if stale:
            configuration.logger.doc(f"[ORQ] OJO: {len(stale)} tables of an earlier version are still there and this run will "
                                     f"not update them ({', '.join(stale)}). To remove them: {how}")

    def run(self, preparation: PreparationBlock = None) -> dict:
        """Every step, in the order of their dependencies; the results under their usual names.

        preparation: a block 1 already run or resumed from SQL (PreparationBlock). Its results go into the context
        and the step 'preparation' is not run again: the pipeline goes on from step 07."""
        self.log_changed_parameters()
        self.warn_retired_tables()
        if preparation is not None:
            self.use_this_preparation_block(preparation)
        for step_name in self.order:
            if step_name == "preparation" and preparation is not None:
                continue
            self.run_step(step_name)
        self.show_timings()
        return self.context.results()

    def use_this_preparation_block(self, preparation: PreparationBlock) -> None:
        """The results of a block 1 run before (in this session or resumed from SQL), into the context, each one
        checked against its contract."""
        if preparation.fine_table is None:
            raise ContractError("the preparation block has no results: run() it or resume_block_from_sql() first")
        for table, value in preparation.results().items():
            self.context.put(table, value, "preparation", self.table_contracts)
        self.seconds.update(preparation.seconds)
        self.configuration.logger.doc(f"[ORQ] block 1 (preparation) given: {preparation.origin}; the run goes on from "
                                      f"step 07")

    def show_timings(self) -> None:
        """How long every step took in this run, the slowest first."""
        if not self.seconds:
            return
        timings = pd.DataFrame({"step": list(self.seconds), "seconds": [round(value, 1) for value in self.seconds.values()]})
        timings["share"] = (timings["seconds"] / max(timings["seconds"].sum(), 1e-9)).map("{:.0%}".format)
        self.configuration.logger.doc(f"[pipeline] time per step ({sum(self.seconds.values()):.0f}s in total):")
        self.configuration.show_table(timings.sort_values("seconds", ascending=False).head(12).reset_index(drop=True))

    def run_step(self, step_name: str):
        """One step: its inputs must exist."""
        step = self.steps[step_name]
        missing = [table for table in step.reads if table not in self.context]
        if missing:
            needed = sorted({self.producer_of.get(table, "?") for table in missing}, key=self.order.index)
            raise ContractError(f"step '{step_name}' needs {missing}: run first {needed}")
        started = time.time()
        outputs = step.run(self.context)
        if set(outputs) != set(step.writes):
            raise ContractError(f"step '{step_name}' wrote {sorted(outputs)} but declares {sorted(step.writes)}")
        for table, value in outputs.items():
            self.context.put(table, value, step_name, self.table_contracts)
        self.seconds[step_name] = time.time() - started
        if self.context.block_seconds:
            # a block: the time of every step inside it, instead of the block as a whole
            del self.seconds[step_name]
            self.seconds.update(self.context.block_seconds)
            self.context.block_seconds = {}
        self.configuration.logger.debug(f"[pipeline] {step_name} ({step.label}) in {time.time() - started:.1f}s")
        return outputs


def dependency_order(steps: list) -> list:
    """The steps ordered so that every table is written before it is read (the declared order breaks ties).
    Stops on a table nobody writes or on a cycle."""
    writer = {}
    for step in steps:
        for table in step.writes:
            if table in writer:
                raise ContractError(f"'{table}' is written by both '{writer[table]}' and '{step.name}'")
            writer[table] = step.name
    unknown = sorted({table for step in steps for table in step.reads if table not in writer})
    if unknown:
        raise ContractError(f"tables read but written by no step: {unknown}")
    order, done = [], set()
    while len(order) < len(steps):
        ready = [step for step in steps if step.name not in done and all(writer[table] in done for table in step.reads)]
        if not ready:
            raise ContractError("the steps have a cycle: " + ", ".join(step.name for step in steps if step.name not in done))
        order.append(ready[0].name)
        done.add(ready[0].name)
    return order


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEPS OF THE SFF (each adapts the context to its step function, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════════

# what block 1 (preparation, steps 00 to 06) leaves for the steps after it
PREPARATION_RESULTS = ("raw_extract", "validated_extract", "leveled_raw", "time_series_rows", "calendared_raw",
                       "fine_table", "forecast_units", "series_table", "rated_units", "series_rate")


def sff_steps() -> list:
    """Every step of the SFF with what it reads and writes."""
    def preparation(context):
        block = PreparationBlock(context.configuration).run()
        context.block_seconds.update(block.seconds)
        return block.results()

    def dimensions(context):
        return {"dimension_decision": analyse_dimensions(context["series_rate"], context["series_table"], context.configuration)}

    def credibility(context):
        estimate, money = climb_the_ladder(context["series_rate"], context["ladder"], context.configuration)
        return {"series_estimate": estimate, "money_by_level": money}

    def compositions(context):
        series, reference = build_pool_series(context["rated_units"], context["ladder"], context["series_estimate"], context.configuration)
        return {"pool_series": series, "pool_reference": reference}

    def dynamics(context):
        pool_dynamics, profile, portfolio = measure_dynamics(context["pool_series"], context["pool_reference"], context.configuration,
                                                             rated_units=context["rated_units"])
        return {"pool_dynamics": pool_dynamics, "portfolio_profile": profile, "portfolio_dynamics": portfolio}

    def uplift(context):
        cells, state_factors, contract_check, increases, basis = estimate_uplift(context["fine_table"], context.configuration)
        return {"uplift_cells": cells, "uplift_state_factors": state_factors, "contract_check": contract_check,
                "price_increases": increases, "uplift_basis": basis}

    def price_diagnostics(context):
        return {"uplift_homogeneity": diagnose_the_price(context["fine_table"], context["uplift_basis"],
                                                         context["contract_check"], context.configuration)}

    def future_pipeline(context):
        return {"future_pipeline": build_the_future_pipeline(context["fine_table"], context.configuration)}

    def uplift_backtest(context):
        backtest, verdict = backtest_uplift(context["fine_table"], context.configuration)
        return {"uplift_backtest": backtest, "uplift_verdict": verdict}

    def forecast(context):
        return {"forecast": assemble_forecast(context["fine_table"], context["forecast_units"], context["series_estimate"],
                                              context["pool_series"], context["pool_reference"], context["backtest"],
                                              context["uplift_cells"], context["uplift_verdict"], context["rated_units"],
                                              context["price_increases"], context.configuration,
                                              uplift_state_factors=context["uplift_state_factors"],
                                              state_maturation=context["state_maturation"], rate_box=context["rate_box"],
                                              series_exam=context["series_exam"],
                                              future_pipeline=context["future_pipeline"]["box"])}

    def series_exam(context):
        return {"series_exam": examine_series(context["rated_units"], context["series_estimate"], context["pool_series"],
                                              context["backtest"], context.configuration,
                                              reference_members=context["ladder"]["reference_members"],
                                              fine_table=context["fine_table"], rate_box=context["rate_box"])}

    def rate_box(context):
        return {"rate_box": build_the_rate_box(context["rated_units"], context["series_estimate"], context["pool_series"],
                                               context["backtest"], context["ladder"], context.configuration)}

    def time_series(context):
        table, total = build_time_series_and_total(context["time_series_rows"], context["fine_table"],
                                                   context["forecast"]["forecast"], context.configuration)
        return {"time_series": table, "forecast_total": total}

    def audit(context):
        return {"audit": build_audit_tables(context["ladder"], context["series_rate"], context["series_estimate"],
                                            context["rated_units"], context["pool_series"], context["pool_reference"],
                                            context["pool_dynamics"], context["backtest"], context["forecast"]["forecast"],
                                            context.configuration, rate_box=context["rate_box"])}

    def core(context):
        core_table, legend, dimension = build_core_table(
            context["fine_table"], context.configuration, forecast_units=context["forecast_units"],
            rated_units=context["rated_units"], series_table=context["series_table"],
            series_rate=context["series_rate"], series_estimate=context["series_estimate"], pool_dynamics=context["pool_dynamics"],
            technique_decision=context["backtest"]["decision"], exam_by_pool=context["backtest"]["exam_by_pool"],
            forecast=context["forecast"]["forecast"], time_series_rows=context["time_series_rows"],
            time_series_table=context["time_series"], forecast_total=context["forecast_total"],
            ladder_stages=context["ladder"]["stages"], series_dynamics=context["audit"]["series_dynamics"],
            series_exam=context["series_exam"]["per_series"])
        return {"core": core_table, "core_legend": legend, "forecast_series": dimension}

    report_reads = ("raw_extract", "new_rows", "audit", "backtest", "contract_check", "fine_table", "forecast", "forecast_total", "ladder",
                    "money_by_level", "pool_dynamics", "pool_reference", "portfolio_dynamics", "portfolio_profile", "rated_units",
                    "series_table", "series_estimate", "series_exam", "series_rate", "time_series", "uplift_backtest",
                    "uplift_cells", "uplift_verdict", "validation", "core", "future_pipeline")
    return [
        Step("preparation", "B1", (), PREPARATION_RESULTS, preparation),
        Step("dimensions", "07", ("series_rate", "series_table"), ("dimension_decision",), dimensions),
        Step("ladder", "08", ("rated_units", "series_rate", "series_table", "dimension_decision"), ("ladder",),
             lambda context: {"ladder": build_ladder_groups(context["rated_units"], context["series_rate"],
                                                            context["series_table"], context["dimension_decision"],
                                                            context.configuration)}),
        Step("credibility", "09", ("series_rate", "ladder"), ("series_estimate", "money_by_level"), credibility),
        Step("compositions", "10", ("rated_units", "ladder", "series_estimate"), ("pool_series", "pool_reference"), compositions),
        Step("dynamics", "11", ("pool_series", "pool_reference", "rated_units"), ("pool_dynamics", "portfolio_profile", "portfolio_dynamics"), dynamics),
        Step("backtest", "12", ("pool_series", "pool_reference"), ("backtest",),
             lambda context: {"backtest": run_backtest(context["pool_series"], context["pool_reference"], context.configuration)}),
        Step("rate_box", "13", ("rated_units", "series_estimate", "pool_series", "backtest", "ladder"), ("rate_box",), rate_box),
        Step("uplift", "14", ("fine_table",), ("uplift_cells", "uplift_state_factors", "contract_check", "price_increases",
                                               "uplift_basis"), uplift),
        Step("price_diagnostics", "15", ("fine_table", "uplift_basis", "contract_check"), ("uplift_homogeneity",),
             price_diagnostics),
        Step("uplift_backtest", "16", ("fine_table",), ("uplift_backtest", "uplift_verdict"), uplift_backtest),
        Step("customer_states", "17", ("fine_table",), ("state_maturation",),
             lambda context: {"state_maturation": learn_the_maturation(context["fine_table"], context.configuration)}),
        Step("future_pipeline", "18", ("fine_table",), ("future_pipeline",), future_pipeline),
        Step("series_exam", "19", ("rated_units", "series_estimate", "pool_series", "backtest", "ladder", "fine_table", "rate_box"),
             ("series_exam",), series_exam),
        Step("forecast", "20", ("fine_table", "forecast_units", "series_estimate", "pool_series", "pool_reference", "backtest",
                                "uplift_cells", "uplift_state_factors", "uplift_verdict", "rated_units", "price_increases",
                                "state_maturation", "rate_box", "series_exam", "future_pipeline"),
             ("forecast",), forecast),
        Step("time_series", "21", ("time_series_rows", "fine_table", "forecast"), ("time_series", "forecast_total"), time_series),
        Step("new_rows", "22", ("fine_table", "rated_units", "forecast"), ("new_rows",),
             lambda context: {"new_rows": count_the_new_rows(context["fine_table"], context["rated_units"],
                                                             context["forecast"]["forecast"], context.configuration)}),
        Step("audit", "23", ("ladder", "series_rate", "series_estimate", "rated_units", "pool_series", "pool_reference",
                              "pool_dynamics", "backtest", "forecast", "rate_box"), ("audit",), audit),
        Step("core", "24", ("fine_table", "forecast_units", "rated_units", "series_table", "series_rate",
                            "series_estimate", "pool_dynamics", "backtest", "forecast", "time_series_rows", "time_series",
                            "forecast_total", "ladder", "audit", "series_exam"), ("core", "core_legend", "forecast_series"), core),
        Step("power_bi", "25", ("core",), ("power_bi",),
             lambda context: {"power_bi": build_power_bi_tables(context["core"], context.configuration)}),
        Step("validation", "26", ("raw_extract", "backtest", "fine_table", "forecast", "forecast_units", "pool_reference",
                                  "series_exam", "series_estimate", "time_series_rows", "core"), ("validation",),
             lambda context: {"validation": validate_chain(context["raw_extract"], context.results(), context.configuration)}),
        Step("report", "27", report_reads, ("report",),
             lambda context: {"report": build_report(context["raw_extract"], context.results(), context.configuration)}),
    ]
