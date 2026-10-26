"""
test_block_1.py — Block 1 (preparation, steps 1.1 to 1.7) in one module: run, persisted to SQL, resumed from SQL,
and the rest of the pipeline from a resumed block gives the same forecast as a whole run.

    python test_block_1.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from block_1_preparation import PERSISTED_RESULTS, PreparationBlock, frame_for_sql, frame_from_sql
from pipeline import Orchestrator
from test_helpers import check, check_stops, console_of, finish, synthetic_with


def same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """The same columns, types, index and values (a missing value equals a missing value)."""
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=True, check_index_type=True)
        return True
    except AssertionError as difference:
        print(f"       {str(difference)[:300]}")
        return False


def test_a_table_goes_to_sql_and_comes_back_the_same() -> None:
    print("A · a table to SQL and back: every column with its type, the index with its gaps")
    configuration = synthetic_with()
    frame = pd.DataFrame({"period": pd.PeriodIndex(["2026-01", "2026-02", "2026-03"], freq="M"),
                          "flag_bool": [True, False, True], "units": [1.5, np.nan, 3.0], "count": [1, 2, 3],
                          "name": ["a", None, "c"], "mixed_flag": [0, "0", 1], "object_bools": [True, False, None]},
                         index=pd.Index([4, 9, 17], name=None))
    sql_frame, manifest_rows = frame_for_sql(frame, "prueba")
    sql_frame.to_sql("prueba", configuration.sql_engine, index=False)
    back = frame_from_sql(pd.read_sql("SELECT * FROM prueba", configuration.sql_engine),
                          pd.DataFrame(manifest_rows, columns=["tabla", "posicion", "columna", "dtype", "tipo"]))
    check(same_frame(frame, back), "Period, bool, float with NaN, int, text with None, a mixed 0 / '0' flag and object bools: "
                                   "all back with their types; the index keeps its gaps (4, 9, 17)")


def test_persist_and_resume() -> None:
    print("B · run, persist, resume: the same tables; then the same forecast")
    configuration = synthetic_with()
    block = PreparationBlock(configuration)
    console = console_of(block.run)
    check("STEP 1.1 · VALIDATE RAW" in console and "STEP 1.7 · RATE SERIES" in console and block.origin == "run",
          "run() runs the seven steps in order and keeps every result in memory")
    console_of(block.persist_block)
    resumed = PreparationBlock(synthetic_with(sql_engine=configuration.sql_engine))
    console = console_of(resumed.resume_block_from_sql)
    check(resumed.origin == "sql" and "STEP 1.1" not in console, "resume_block_from_sql() runs no step")
    check(all(same_frame(getattr(block, attribute), getattr(resumed, attribute)) for attribute in PERSISTED_RESULTS),
          "every persisted result comes back the same: " + ", ".join(PERSISTED_RESULTS))
    check(block.extract_summary == resumed.extract_summary and block.extract_summary["filas"] > 0,
          "the summary of the extract (rows, columns, units and USD due) is the same after resuming")
    check(not any(hasattr(block, name) for name in ("raw_extract", "validated_extract", "leveled_raw", "calendared_raw")),
          "the block keeps no copy of the extract: only the fine table and its summary stay in memory")

    whole = {}
    console_of(lambda: whole.update(Orchestrator(synthetic_with()).run()))
    from_resumed = {}
    resumed_configuration = synthetic_with(sql_engine=configuration.sql_engine)
    console = console_of(lambda: from_resumed.update(Orchestrator(resumed_configuration).run(
        preparation=PreparationBlock(resumed_configuration).resume_block_from_sql())))
    check("STEP 1.2 ·" not in console and "STEP 2.1 ·" in console, "with a resumed block the run goes on from step 2.1")
    check(same_frame(whole["forecast"]["by_year"], from_resumed["forecast"]["by_year"])
          and same_frame(whole["forecast"]["by_month"], from_resumed["forecast"]["by_month"]),
          "the forecast by month and by year of a resumed block = the one of a whole run")
    check(same_frame(whole["series_exam"]["summary"], from_resumed["series_exam"]["summary"]),
          "the exam of a resumed block = the one of a whole run")


def test_resume_refuses_a_block_of_another_run() -> None:
    print("C · a block persisted with another month or another Config is not resumed")
    configuration = synthetic_with()
    block = PreparationBlock(configuration)
    console_of(block.run)
    console_of(block.persist_block)
    other_month = synthetic_with(sql_engine=configuration.sql_engine, current_month="2026-08")
    check_stops(lambda: PreparationBlock(other_month).resume_block_from_sql(), "current_month",
                "another current month: refused, naming it")
    other_support = synthetic_with(sql_engine=configuration.sql_engine, support_floor=configuration.support_floor + 1)
    check_stops(lambda: PreparationBlock(other_support).resume_block_from_sql(), "config.support_floor",
                "a field of the Config the block reads changed: refused, naming it")


def test_without_an_engine_it_goes_to_csv() -> None:
    print("D · without an SQL engine the block is persisted to CSV and resumed from it")
    import tempfile
    folder = tempfile.mkdtemp()
    configuration = synthetic_with(sql_engine=None, output_folder=folder)
    block = PreparationBlock(configuration)
    console_of(block.run)
    console_of(block.persist_block)
    resumed = PreparationBlock(synthetic_with(sql_engine=None, output_folder=folder))
    console_of(resumed.resume_block_from_sql)
    check(all(same_frame(getattr(block, attribute), getattr(resumed, attribute)) for attribute in PERSISTED_RESULTS),
          "every persisted result comes back the same from the CSV files")


if __name__ == "__main__":
    test_a_table_goes_to_sql_and_comes_back_the_same()
    test_persist_and_resume()
    test_resume_refuses_a_block_of_another_run()
    test_without_an_engine_it_goes_to_csv()
    finish()
