"""
pipeline.py — The blocks of the SFF with formal interfaces, run over a common context.

THE EIGHT BLOCKS (each one a module block_<n>_<name>.py with its steps <n>.1, <n>.2 … and its class):
  B1  preparation          1.1-1.7  the extract verified, in the tables every prediction starts from
  B2  rate                 2.1-2.7  the renewal rate of any row, at any horizon, with its band
  B3  price                3.1-3.3  the uplift (revaluation) of the renewers, the path that predicts it best
  B4  customer states      4.1      the customer states of the units when they fall due
  B5  future pipeline      5.1      the rows of pipeline the extract cannot have yet (renewals, acquisitions)
  B6  time series          6.1      the retail-to-subscription universe, by region
  B7  exam and assembly    7.1-7.3  the exam (the band), the forecast in money, the total
  B8  output and control   8.1-8.6  core, audit, Power BI, validation, report: no number changes

Four pieces (the pattern of a data pipeline: contract + node + data catalogue + pipeline):

  TableContract     a table declared once: its grain, its key (unique) and its required columns.
                    Checked when a block gives it: a missing column or a repeated key stops the run
                    at the block that produced it, with the table's name.
  Step              the common interface of a block: its name, the results it READS, the results it
                    GIVES, and run(context) → {name: value}. The block runs its own steps in order.
  PipelineContext   the catalogue of the run: every result produced so far, by name, with the block
                    that produced it, plus the Config.
  Orchestrator      orders the blocks by their dependencies (what each reads must be given before)
                    and runs them all, in that order, refusing a block whose inputs are missing.

In a notebook:
    pipeline = Orchestrator(configuration)
    results = pipeline.run()                 # every block, in the order of their dependencies
    pipeline.context["forecast"]             # any result of the run
    pipeline.order                           # the blocks, in the order they run
    pipeline.seconds                         # how long every step took
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
import block_2_rate
import block_3_price
import block_4_customer_states
import block_5_future_pipeline
import block_6_time_series
import block_7_exam_and_assembly
import block_8_output
from block_1_preparation import PreparationBlock
from block_2_rate import RateBlock
from block_3_price import PriceBlock
from block_4_customer_states import CustomerStatesBlock
from block_5_future_pipeline import FuturePipelineBlock
from block_6_time_series import TimeSeriesBlock
from block_7_exam_and_assembly import ExamAndAssemblyBlock
from block_8_output import OutputBlock, results_of_the_run
from config import COMPOSITION_ID_COLUMN, Config, RETIRED_TABLES, SERIES_ID_COLUMN, UNIT_ID_COLUMN, parameter_table
from verdicts import record_the_verdict


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
        "core": [TableContract("core", "fine row of the extract, gap, synthetic or time_series row", (), ("s1_4_fs_id",))],
        "forecast_series": [TableContract("forecast_series", "forecast series", ("s1_4_fs_id",), ())],
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
        return results_of_the_run(self.tables)


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
        the steps that read it. The whole table, with every parameter, is written by step 8.4 (sff_parametros)."""
        parameters = parameter_table(self.configuration)
        changed = parameters[parameters["cambiado"] == 1]
        self.configuration.logger.doc(f"[ORQ] {len(parameters)} parameters in the Config · {len(changed)} with a value of this "
                                      f"run (the rest keep their default; all of them in step 8.4, sff_parametros):")
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
        and the step 'preparation' is not run again: the pipeline goes on from block 2."""
        self.log_changed_parameters()
        self.warn_retired_tables()
        # the verdicts of this run start empty (a Config may have run before)
        self.configuration.block_verdicts = {}
        self.configuration.data_profile = None
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
                                      f"block 2")
        record_the_verdict("B1", self.context.tables, self.configuration)

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
        # the verdict of the block: the hypotheses it rests on, contrasted with this run's data (verdicts.py); block 8
        # records its own before step 8.6 writes every verdict
        if step.label not in self.configuration.block_verdicts:
            record_the_verdict(step.label, self.context.tables, self.configuration)
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

# what block 1 (preparation, steps 1.1 to 1.7) leaves for the blocks after it
PREPARATION_RESULTS = ("extract_summary", "time_series_rows", "fine_table", "forecast_units", "series_table", "rated_units", "series_rate")


def sff_steps() -> list:
    """Every block of the SFF, with the results it reads and the ones it gives. Every block runs its steps in order
    (block.seconds: how long each one took) and gives its results under the names the pipeline knows them by."""
    def preparation(context):
        block = PreparationBlock(context.configuration).run()
        context.block_seconds.update(block.seconds)
        return block.results()

    def rate(context):
        block = RateBlock(context.configuration).run(context["rated_units"], context["series_rate"], context["series_table"])
        context.block_seconds.update(block.seconds)
        return block.results()

    def price(context):
        block = PriceBlock(context.configuration).run(context["fine_table"])
        context.block_seconds.update(block.seconds)
        return block.results()

    def customer_states(context):
        block = CustomerStatesBlock(context.configuration).run(context["fine_table"])
        context.block_seconds.update(block.seconds)
        return block.results()

    def future_pipeline(context):
        block = FuturePipelineBlock(context.configuration).run(context["fine_table"])
        context.block_seconds.update(block.seconds)
        return block.results()

    def time_series(context):
        block = TimeSeriesBlock(context.configuration).run(context["time_series_rows"], context["fine_table"])
        context.block_seconds.update(block.seconds)
        return block.results()

    def exam_and_assembly(context):
        block = ExamAndAssemblyBlock(context.configuration).run(dict(context.tables))
        context.block_seconds.update(block.seconds)
        return block.results()

    def output(context):
        block = OutputBlock(context.configuration).run(dict(context.tables))
        context.block_seconds.update(block.seconds)
        return block.results()

    return [
        Step("preparation", "B1", (), PREPARATION_RESULTS, preparation),
        Step("rate", "B2", block_2_rate.BLOCK_READS, block_2_rate.BLOCK_GIVES, rate),
        Step("price", "B3", block_3_price.BLOCK_READS, block_3_price.BLOCK_GIVES, price),
        Step("customer_states", "B4", block_4_customer_states.BLOCK_READS, block_4_customer_states.BLOCK_GIVES,
             customer_states),
        Step("future_pipeline", "B5", block_5_future_pipeline.BLOCK_READS, block_5_future_pipeline.BLOCK_GIVES,
             future_pipeline),
        Step("time_series", "B6", block_6_time_series.BLOCK_READS, block_6_time_series.BLOCK_GIVES, time_series),
        Step("exam_and_assembly", "B7", block_7_exam_and_assembly.BLOCK_READS, block_7_exam_and_assembly.BLOCK_GIVES,
             exam_and_assembly),
        Step("output", "B8", block_8_output.BLOCK_READS, block_8_output.BLOCK_GIVES, output),
    ]
