"""
test_pipeline.py — The orchestrator: steps ordered by their dependencies, a step refused without its
inputs, a whole run that fills the catalogue, and a table that breaks its contract stopped at once.

    python test_pipeline.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────

import pandas as pd

from config import what_the_configuration_implies
from pipeline import ContractError, Orchestrator, Step, TableContract, dependency_order, sff_steps
from test_helpers import check, check_stops, console_of, finish, synthetic_with


def test_the_order() -> None:
    print("A · the order comes from the dependencies")
    steps = sff_steps()
    order = dependency_order(list(reversed(steps)))
    position = {name: index for index, name in enumerate(order)}
    writer = {table: step.name for step in steps for table in step.writes}
    check(all(position[writer[table]] < position[step.name] for step in steps for table in step.reads),
          "every table is written before it is read, even with the steps declared backwards")
    check_stops(lambda: dependency_order([Step("a", "A", ("x",), ("y",), None), Step("b", "B", ("y",), ("x",), None)]),
                "cycle", "a cycle is refused")
    check_stops(lambda: dependency_order([Step("a", "A", ("nobody",), ("y",), None)]),
                "written by no step", "a table nobody writes is refused")


def test_the_run() -> None:
    print("B · a step refused without its inputs; a whole run")
    pipeline = Orchestrator(synthetic_with())
    check_stops(lambda: pipeline.run_step("exam_and_assembly"), "run first", "a block without its inputs is refused, naming what to run first")
    console_of(pipeline.run)
    check("forecast_series" in pipeline.context and "report" in pipeline.context, "the whole run fills the catalogue")


def test_the_contracts() -> None:
    print("C · a table that breaks its contract stops at the step that wrote it")
    contract = TableContract("series_rate", "forecast series", ("fs_id",), ("ruta",))
    check_stops(lambda: contract.validate(pd.DataFrame({"fs_id": ["a", "a"], "ruta": ["x", "y"]}), "rate_series"),
                "repeated keys", "a repeated key is refused, naming the table and the step")
    check_stops(lambda: contract.validate(pd.DataFrame({"fs_id": ["a"]}), "rate_series"),
                "lacks the columns", "a missing column is refused, naming it")


def test_what_the_configuration_implies() -> None:
    print("D · at the start: what the Config implies, not the list of its values")
    configuration = synthetic_with(current_month="2026-10", test_months=3)
    lines = dict.fromkeys(line for _, line in what_the_configuration_implies(configuration))
    text = " | ".join(lines)
    check("exam 2026-07..2026-09 (3 months)" in text and "the last closed month is 2026-09" in text,
          "current month 2026-10 with 3 exam months: the exam is 2026-07, 2026-08 and 2026-09")
    check("h = 6: 2026-09 is predicted with the months up to 2026-03" in text,
          "every horizon of the exam says which months the prediction can see")
    check("due from 2027-10 on" in text, "the 1-year pipeline not known yet: due 12 months after the current month or later")
    without_curve = synthetic_with(current_month="2026-10", maturation_dims=[])
    states_lines = [line for section, line in what_the_configuration_implies(without_curve) if section == "customer states"]
    check(any("no dates, so no curve" in line for line in states_lines),
          "a state without dates is said to take its historical share, with no curve and no exam")
    console = console_of(lambda: Orchestrator(configuration).log_what_the_configuration_implies())
    check("WHAT THIS CONFIG IMPLIES" in console and "sff_parametros" in console and " = [" not in console,
          "the console shows the consequences and points to sff_parametros for the values, without listing them")


if __name__ == "__main__":
    test_the_order()
    test_the_run()
    test_the_contracts()
    test_what_the_configuration_implies()
    finish()
