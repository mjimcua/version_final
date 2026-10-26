"""
test_1_4_fine_table.py — Step 1.4 on the synthetic raw: every row gets its ids and keys, the
grain is checked, nothing else changes, and the fine table is written.

    python test_1_4_fine_table.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from block_1_preparation import validate_raw
from block_1_preparation import apply_calendar
from block_1_preparation import build_fine_table
from test_helpers import check, check_stops, console_of, console_of_failure, count_status, finish, synthetic_with


def calendared_synthetic(**overrides):
    """The synthetic raw after steps 1.1 and 1.3, and its Config."""
    configuration = synthetic_with(**overrides)
    frames = {}
    def run_steps():
        frames["raw"] = apply_calendar(validate_raw(configuration.read_raw(), configuration), configuration)
    console_of(run_steps)
    return frames["raw"], configuration


def run_step(calendared, configuration):
    """Step 1.4, returning (fine table, console)."""
    frames = {}
    console = console_of(lambda: frames.setdefault("fine", build_fine_table(calendared, configuration)))
    return frames["fine"], console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_ids() -> None:
    print("A · the ids of every row")
    calendared, configuration = calendared_synthetic()
    fine, console = run_step(calendared, configuration)
    first = fine.iloc[0]
    expected_series = "|".join(str(first[column]) for column in configuration.rate_series_columns)
    check(first["fs_id"] == expected_series, "fs_id = mandatory | timevarying | extra_renovacion, in that order")
    check(first["fu_id"] == f"{expected_series}|{first['period']}", "fu_id = fs_id | month")
    check(first["uplift_cell_id"] == f"{first['region']}|{first['product']}|{first['newcust']}|{first['tramo_descuento']}",
          "uplift_cell_id = mandatory | extra_revalorizacion | discount bucket (no timevarying, no extra_renovacion, no month)")
    check(not {"fs_key", "fu_key", "uplift_cell_key", "fila_key"} & set(fine.columns),
          "no hash keys: the ids are text, the core and the BI join on them")
    bucket_of_forty = fine.loc[fine["discount_pct"] == 0.4, "tramo_descuento"].unique().tolist()
    bucket_of_null = fine.loc[fine["discount_pct"].isna(), "tramo_descuento"].unique().tolist()
    check(bucket_of_forty == ["05 _ 40-50%"] and bucket_of_null == ["sin_dato"],
          "the bucket is derived like the extract's discount_interval: 0.40 → '05 _ 40-50%', null → 'sin_dato'")
    check(fine["uplift_cell_id"].nunique() < fine["fu_id"].nunique(),
          "many fine rows share an uplift cell: every month and signal state of the same price context")
    subset_cells = {}
    console_of(lambda: subset_cells.setdefault("fine", build_fine_table(calendared, synthetic_with(uplift_mandatory_dims=["region"]))))
    check(subset_cells["fine"]["uplift_cell_id"].iloc[0] == f"{first['region']}|{first['newcust']}|{first['tramo_descuento']}",
          "uplift_mandatory_dims narrows the cell to the declared mandatory dims")
    check(fine[list(calendared.columns)].equals(calendared) and len(fine.columns) == len(calendared.columns) + 4,
          "same rows, same order, four columns added (bucket + three ids), nothing else changed")
    check(count_status(console, "ok") == 3 and "3 checks: 3 ok" in console, "the 3 checks pass and are logged")

    calendared_no_extras, configuration_no_extras = calendared_synthetic(
        extra_revalorizacion=[], discount_value_column=None,
        ignore_cols=["dataset_role", "is_current_month", "discount", "newcust", "discount_pct"])
    one_row_per_unit = calendared_no_extras.drop_duplicates(configuration_no_extras.rate_series_columns + ["period"])
    no_extras, _ = run_step(one_row_per_unit, configuration_no_extras)
    check(len(no_extras) == len(one_row_per_unit) and no_extras["fu_id"].is_unique,
          "with no extra_revalorizacion, a fine row is its unit: one row per fu_id passes the grain")
    finer_console = console_of_failure(lambda: build_fine_table(calendared_no_extras, configuration_no_extras))
    check(count_status(finer_console, "FAIL") == 1,
          "a raw finer than the declared dimensions fails the grain, and only the grain (no false key collision)")
    check("differ inside the repeated rows" in finer_console and "discount_pct (" in finer_console and "newcust (" in finer_console,
          "the failure names the columns that split the grain (the ones left out of the taxonomy)")
    check("together they explain" in finer_console and "HAVING COUNT(*) > 1" in finer_console
          and "COUNT(DISTINCT discount_pct)" in finer_console and "GROUP BY period" in finer_console,
          "the failure says how many groups the columns explain and writes the SQL that reproduces it")


def test_one_row_per_exact_discount() -> None:
    print("D · the real extract: one row per exact discount inside the same bucket")
    calendared, configuration = calendared_synthetic()
    with_forty = calendared.index[calendared["discount_pct"] == 0.4][0]
    twin = calendared.loc[[with_forty]].copy()
    twin["discount_pct"] = 0.43                     # same unit, same bucket (40-50 %), another exact discount
    two_discounts = pd.concat([calendared, twin], ignore_index=True)
    fine, console = run_step(two_discounts, configuration)
    pair = fine[(fine["fu_id"] == fine.loc[with_forty, "fu_id"]) & fine["discount_pct"].isin([0.4, 0.43])]
    check("3 checks: 3 ok" in console and len(pair) == 2 and pair["discount_pct"].nunique() == 2,
          "the grain passes: two fine rows, one per exact discount")
    check(pair["tramo_descuento"].nunique() == 1 and pair["uplift_cell_id"].nunique() == 1,
          "both fall in the same bucket and the same uplift cell (the cell does not split by exact discount)")
    from block_1_preparation import build_forecast_units
    frames = {}
    console_of(lambda: frames.setdefault("units", build_forecast_units(fine, configuration)))
    unit = frames["units"].set_index("fu_id").loc[pair["fu_id"].iloc[0]]
    check(unit["total_tr_usd"] == fine.loc[fine["fu_id"] == pair["fu_id"].iloc[0], "total_tr_usd"].sum(),
          "step 1.5 adds both rows into their forecast unit: the rate does not depend on the discount")


def test_the_discount_buckets() -> None:
    print("C · the discount buckets, like the extract's discount_interval")
    from config import discount_bucket_labels
    edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = discount_bucket_labels(pd.Series([0.0, 0.05, 0.1, 0.65, 0.6, 0.7, 0.999, 1.0, None]), edges).tolist()
    check(labels == ["01 _ 0-10%", "01 _ 0-10%", "02 _ 10-20%", "07 _ 60-70%", "07 _ 60-70%", "08 _ 70-80%",
                     "10 _ 90-100%", "10 _ 90-100%", "sin_dato"],
          "[low, high) in %, the last bucket closed at 100 %, null → sin_dato; 0.65 → '07 _ 60-70%'")
    check_stops(lambda: synthetic_with(discount_bucket_edges=[0, 50, 40, 100]), "discount_bucket_edges",
                "edges that do not go up from 0 to 100 stop when the Config is built")


def test_the_grain_and_the_table() -> None:
    print("B · the grain")
    calendared, configuration = calendared_synthetic()
    doubled = pd.concat([calendared, calendared.iloc[[0]]], ignore_index=True)
    check_stops(lambda: build_fine_table(doubled, configuration), "one row per forecast unit, revaluation values and exact discount",
                "two rows with the same unit and revaluation values stop the step")
    failing_console = console_of_failure(lambda: build_fine_table(doubled, configuration))
    check("example rows below" in failing_console, "the first repeated group is shown as a table under the check")
    check("no column but the money differs" in failing_console,
          "a row repeated whole is diagnosed as below the declared grain (only the money could differ)")

    fine, _ = run_step(calendared, configuration)
    check("sff_fact_fine" not in pd.read_sql("SELECT name FROM sqlite_master", configuration.sql_engine)["name"].tolist(),
          "the fine table is not written to SQL (sff_nucleo carries every fine row)")


if __name__ == "__main__":
    test_the_ids()
    test_the_grain_and_the_table()
    test_the_discount_buckets()
    test_one_row_per_exact_discount()
    finish()
