"""
test_1_1_validate_raw.py — Step 1.1 on the synthetic raw: it passes when the raw fits, and each
problem of the column contract or of the months stops the run naming itself.

    python test_1_1_validate_raw.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from block_1_preparation import READ_TOTAL_LABEL, READ_UNIVERSE_RENEWALS, READ_UNIVERSE_TIME_SERIES, read_summary, validate_raw
from test_helpers import check, check_stops, console_of, console_of_failure, count_status, finish, synthetic_with
from config import CALENDAR_ROLE_COLUMN, ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_synthetic_raw_fits() -> None:
    print("A · the synthetic raw fits the configuration")
    configuration = synthetic_with()
    raw = configuration.read_raw()
    validated = None
    def run_step():
        nonlocal validated
        validated = validate_raw(raw, configuration)
    console_of(run_step)
    check(len(validated) == len(raw) and list(validated.columns) == list(raw.columns),
          "no row and no column added or removed")
    check(isinstance(validated["period"].dtype, pd.PeriodDtype), "the period is a monthly Period")
    check(raw["period"].dtype != validated["period"].dtype, "the input raw is not modified (a copy is returned)")
    console = console_of(lambda: validate_raw(raw, configuration))
    check(count_status(console, "ok") == 10 and "10 checks: 10 ok · 0 warnings · 0 failed" in console,
          "the 10 checks are logged, numbered, each with its status, and counted")
    check("purpose:" in console and "STEP 1.1 · VALIDATE RAW" in console, "the step logs its name and its purpose")
    failing = console_of_failure(lambda: validate_raw(raw.assign(extra_column=1), configuration))
    check(count_status(failing, "FAIL") == 1 and count_status(failing, "ok") == 9, "when a check fails, every check is still logged before stopping")


def test_the_column_contract() -> None:
    print("B · the column contract")
    configuration = synthetic_with()
    raw = configuration.read_raw()
    check_stops(lambda: validate_raw(raw.iloc[0:0], configuration), "no rows", "an empty raw stops")
    check_stops(lambda: validate_raw(raw.assign(extra_column=1), configuration), "no role",
                "a raw column with no role stops, naming it")
    check_stops(lambda: validate_raw(raw.drop(columns=["channel"]), configuration), "missing from the raw",
                "a declared column missing from the raw stops, naming it")
    check_stops(lambda: validate_raw(pd.concat([raw, raw[["region"]]], axis=1), configuration), "duplicated",
                "a column name that appears twice stops")
    console_of(lambda: validate_raw(raw.drop(columns=["dataset_role", "is_current_month"]), configuration))
    check(True, "an ignored column may be absent from the raw")
    check_stops(lambda: synthetic_with(extra_renovacion=["channel", "region"]), "more than one role",
                "a column declared in two roles stops when the Config is built")
    shared_extra = synthetic_with(extra_renovacion=["channel", "newcust"])
    check(shared_extra.column_roles()["newcust"] == "extra_renovacion+extra_revalorizacion",
          "the two extra groups may share a column")
    check_stops(lambda: synthetic_with(structural_timevarying_dims={"dormant": "neg"}), "signs",
                "a timevarying column with an invalid sign stops")
    two_problems = raw.assign(extra_column=1).drop(columns=["channel"])
    check_stops(lambda: validate_raw(two_problems, configuration), "2. ",
                "every problem is reported at once, not just the first")


def test_the_log() -> None:
    print("D · the log is configured in the Config, once")
    import logging, os, tempfile
    log_file = os.path.join(tempfile.mkdtemp(), "sff.log")
    configuration = synthetic_with(log_level="WARNING", log_file=log_file)
    check(configuration.logger is logging.getLogger("sff") and configuration.logger.level == logging.WARNING,
          "configuration.logger is the framework's logger, at the level of the Config")
    console = console_of(lambda: validate_raw(configuration.read_raw(), configuration))
    check("STEP 1.1" not in console and count_status(console, "ok") == 0,
          "at level WARNING the ok lines are not logged")
    console_of_failure(lambda: validate_raw(configuration.read_raw().drop(columns=["channel"]), configuration))
    check("FAIL" in open(log_file, encoding="utf-8").read(), "a failed check is written to the log file")
    coloured = synthetic_with(log_colors=True)
    console = console_of(lambda: validate_raw(coloured.read_raw(), coloured))
    blue, green = "\x1b[34m", "\x1b[32m"
    title_line = next(line for line in console.splitlines() if "STEP 1.1" in line)
    purpose_line = next(line for line in console.splitlines() if "purpose:" in line)
    ok_line = next(line for line in console.splitlines() if " 1. ok" in line)
    check(title_line.startswith(blue) and purpose_line.startswith(blue) and ok_line.startswith(green),
          "colours: title and purpose blue (DOC), a passed check green (INFO)")
    quiet_file = os.path.join(tempfile.mkdtemp(), "sff_quiet.log")
    quiet = synthetic_with(log_file=quiet_file, console_log_level="WARNING")
    console = console_of(lambda: validate_raw(quiet.read_raw(), quiet))
    written = open(quiet_file, encoding="utf-8").read()
    check("STEP 1.1" not in console and "entrenamiento" not in console
          and "STEP 1.1" in written and "1. ok" in written and "unidades_vencen" in written,
          "console_log_level WARNING: the console shows nothing of a clean step; the file keeps every line and every table")
    announced = console_of(lambda: (quiet.show_figure("4.1", "un gráfico de detalle", "detalle.png"),
                                    quiet.show_figure("4.1", "la curva explicada", "curva.png", always=True)))
    check("detalle.png" not in announced and "curva.png" in announced,
          "with a quiet console, only the charts that explain the forecast (always=True) are announced on screen")
    synthetic_with()    # leave the log as the other tests expect it


def test_the_months() -> None:
    print("C · the months and the calendar")
    configuration = synthetic_with()
    raw = configuration.read_raw()
    bad_month = raw.copy()
    bad_month.loc[0, "period"] = "septiembre"
    check_stops(lambda: validate_raw(bad_month, configuration), "not a month", "a value that is not a month stops")
    no_month = raw.copy()
    no_month.loc[0, "period"] = None
    check_stops(lambda: validate_raw(no_month, configuration), "no period", "a row with no month stops")
    check_stops(lambda: validate_raw(raw, synthetic_with(current_month=None)), "not declared",
                "an undeclared current_month stops")
    check_stops(lambda: validate_raw(raw, synthetic_with(current_month="2027-03")), "outside the months",
                "a current month outside the raw stops")
    check_stops(lambda: validate_raw(raw, synthetic_with(test_months=60)), "no month to train",
                "an exam that leaves no month to train stops")
    check_stops(lambda: synthetic_with(current_month="09-2026"), "is not a month",
                "a malformed current_month stops when the Config is built")

    for spelling in ("2026-09", "2026-09-01", "01/09/2026"):
        months = pd.Series(pd.period_range("2026-01", "2026-12", freq="M"))
        roles = dict(zip(months.astype(str), synthetic_with(current_month=spelling).role_of_months(months)))
        check(roles["2026-05"] == ROLE_TRAIN and roles["2026-06"] == ROLE_TEST and roles["2026-08"] == ROLE_TEST
              and roles["2026-09"] == ROLE_PROJECTION,
              f"current '{spelling}', 3 exam months: entrenamiento ≤ 05 · examen 06..08 · proyeccion ≥ 09")

    with_gap = raw[raw["period"] != "2025-03"]
    console = console_of(lambda: validate_raw(with_gap, configuration))
    check("2025-03" in console and "WARN" in console,
          "a month with no row is reported as a warning, not a stop")


def test_the_read_summary() -> None:
    print("E · the summary of what was read (action 7): rows, units and dollars of pipeline")
    configuration = synthetic_with()
    raw = configuration.read_raw()
    console = console_of(lambda: validate_raw(raw, configuration))
    check("▸ 7. read:" in console and "units and $" in console and "falling due" in console,
          "the step says how many rows, units and dollars of pipeline it read")
    validated = validate_raw(raw, configuration)
    summary = read_summary(validated, configuration)
    flag = configuration.flag_time_series_col
    renewal_rows = raw[raw[flag] != 1] if flag in raw.columns else raw
    renewals = summary[summary["universo"] == READ_UNIVERSE_RENEWALS].set_index(CALENDAR_ROLE_COLUMN)
    total = renewals.loc[READ_TOTAL_LABEL]
    check(int(total["filas"]) == len(renewal_rows)
          and abs(total["unidades_vencen"] - renewal_rows[configuration.pipeline_units_col].fillna(0).sum()) < 1e-6
          and abs(total["usd_vence"] - renewal_rows[configuration.pipeline_usd_col].fillna(0).sum()) < 1e-3,
          "the total of the renewal universe is the rows, units and USD due of the raw")
    by_role = renewals.drop(index=READ_TOTAL_LABEL)
    check(set(by_role.index) <= {ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION}
          and int(by_role["filas"].sum()) == int(total["filas"])
          and abs(by_role["usd_vence"].sum() - total["usd_vence"]) < 1e-3,
          "the calendar roles split the total: their rows and USD due add up to it")
    time_series_raw = raw.assign(**{flag: 0}).head(50).assign(**{flag: 1})
    time_series_raw[configuration.pipeline_units_col] = None
    time_series_raw[configuration.pipeline_usd_col] = None
    with_time_series = validate_raw(pd.concat([raw.assign(**{flag: 0}), time_series_raw], ignore_index=True), configuration)
    universes = read_summary(with_time_series, configuration)
    time_series_total = universes[(universes["universo"] == READ_UNIVERSE_TIME_SERIES)
                                  & (universes[CALENDAR_ROLE_COLUMN] == READ_TOTAL_LABEL)].iloc[0]
    check(int(time_series_total["filas"]) == 50 and time_series_total["usd_vence"] == 0,
          "the time_series rows go apart, with no pipeline (a null counts as 0)")


if __name__ == "__main__":
    test_the_synthetic_raw_fits()
    test_the_column_contract()
    test_the_months()
    test_the_log()
    test_the_read_summary()
    finish()
