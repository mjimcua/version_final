"""
test_pipeline.py — The orchestrator: steps ordered by their dependencies, a step refused without its
inputs, a whole run that fills the catalogue, and a table that breaks its contract stopped at once.

    python test_pipeline.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────

import pandas as pd

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
    check_stops(lambda: pipeline.run_step("forecast"), "run first", "a step without its inputs is refused, naming what to run first")
    console_of(pipeline.run)
    check("forecast_series" in pipeline.context and "report" in pipeline.context, "the whole run fills the catalogue")


def test_the_contracts() -> None:
    print("C · a table that breaks its contract stops at the step that wrote it")
    contract = TableContract("series_rate", "forecast series", ("fs_id",), ("ruta",))
    check_stops(lambda: contract.validate(pd.DataFrame({"fs_id": ["a", "a"], "ruta": ["x", "y"]}), "rate_series"),
                "repeated keys", "a repeated key is refused, naming the table and the step")
    check_stops(lambda: contract.validate(pd.DataFrame({"fs_id": ["a"]}), "rate_series"),
                "lacks the columns", "a missing column is refused, naming it")


if __name__ == "__main__":
    test_the_order()
    test_the_run()
    test_the_contracts()
    finish()
