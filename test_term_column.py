"""
test_term_column.py — The term of a licence when it is a leveled dimension: the column keeps its name and
its raw value in every step (step 1.2 only adds term_level_1), so step 1.3 wipes ONLY the pipeline of the
1-year licences that is not known yet and step 7.2 projects only them; a term_column that is not a
mandatory dim stops when the Config is built.

    python test_term_column.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import tempfile

import pandas as pd

from main import SyntheticConfig
from pipeline import Orchestrator
from test_helpers import check, check_stops, console_of, finish, synthetic_with


class TermSyntheticConfig(SyntheticConfig):
    """The synthetic extract with a term column ("1 year" for product A, "3 year" for B) and its pipeline
    of 2027 (the months 2026-09..2026-12 again, 12 months later, nothing renewed yet)."""
    def read_raw(self) -> pd.DataFrame:
        raw = super().read_raw()
        months = pd.PeriodIndex(pd.to_datetime(raw["period"]), freq="M")
        future = raw[months >= pd.Period("2026-09", "M")].copy()
        future["period"] = (pd.PeriodIndex(pd.to_datetime(future["period"]), freq="M") + 12).to_timestamp()
        for column in ("total_renewed_units", "total_renewed_usd"):
            future[column] = float("nan")
        raw = pd.concat([raw, future], ignore_index=True)
        raw["term"] = raw["product"].map({"A": "1 year", "B": "3 year"})
        return raw


# the term transitions of this test: of what renews, the share in units (and in USD) that renews to every term
TRANSITION_SHARES = {"1 year": {"1y": (0.8, 0.7), "2y": (0.2, 0.3), "3y": (0.0, 0.0)},
                     "3 year": {"1y": (0.4, 0.25), "2y": (0.0, 0.0), "3y": (0.6, 0.75)}}


class TransitionsSyntheticConfig(TermSyntheticConfig):
    """The same extract with the six columns of the term transitions: product A (1 year) renews 80 % of its units to 1
    year (70 % of the USD) and the rest to 2 years; product B (3 years) 40 % to 1 year (25 % of the USD), the rest to 3."""
    def read_raw(self) -> pd.DataFrame:
        raw = super().read_raw()
        for destination in ("1y", "2y", "3y"):
            units_share = raw["term"].map({term: shares[destination][0] for term, shares in TRANSITION_SHARES.items()})
            usd_share = raw["term"].map({term: shares[destination][1] for term, shares in TRANSITION_SHARES.items()})
            raw[f"total_renewed_units_to_{destination}"] = raw["total_renewed_units"] * units_share
            raw[f"total_renewed_usd_to_{destination}"] = raw["total_renewed_usd"] * usd_share
        return raw


TRANSITION_FIELDS = {f"renewed_{kind}_to_{destination}_col": f"total_renewed_{kind}_to_{destination}"
                     for kind in ("units", "usd") for destination in ("1y", "2y", "3y")}


def term_configuration(term_column: str = "term", with_transitions: bool = False):
    base = synthetic_with()
    arguments = {name: getattr(base, name) for name in base.__dataclass_fields__}
    arguments.update(business_mandatory_dims=list(base.business_mandatory_dims) + ["term"], term_column=term_column,
                     leveled_dims={"term": "ordinal"}, levels_path=os.path.join(tempfile.mkdtemp(), "levels.json"))
    if with_transitions:
        arguments.update(TRANSITION_FIELDS, term_transition_levels=[["product"], []], term_transition_min_renewed_units=1.0)
        return TransitionsSyntheticConfig(**arguments)
    return TermSyntheticConfig(**arguments)


def wiped_by_term(orchestrator) -> dict:
    """Units due in the extract and after the calendar, from 2027-09 on, per term."""
    fine = orchestrator.context["fine_table"]
    late = fine[fine["period"] >= pd.Period("2027-09", "M")]
    return late.groupby("term").agg(extract=("s0_vencen_unidades", "sum"), after=("total_tr_units", "sum")).to_dict("index")


def test_the_term_column() -> None:
    print("A · term in one column, leveled: only the 1-year pipeline not known yet is wiped")
    orchestrator = Orchestrator(term_configuration())
    console = console_of(lambda: orchestrator.run_step("preparation"))         # block 1: steps 1.1 to 1.7
    wiped = wiped_by_term(orchestrator)
    check(wiped["1 year"]["extract"] > 0 and wiped["1 year"]["after"] == 0
          and wiped["3 year"]["after"] == wiped["3 year"]["extract"] > 0,
          "the 1-year pipeline from 2027-09 is wiped, the 3-year one is kept")
    check("rows wiped per term: {'1 year':" in console, "step 1.3 says which terms it wiped")
    check({"term", "term_level_1"} <= set(orchestrator.context["fine_table"].columns),
          "the term keeps its name in every step; its generated level is next to it")
    results = {}
    console = console_of(lambda: results.update(Orchestrator(term_configuration()).run()))
    check("units at 0 units due:" in console and "wiped on purpose by the calendar" in console,
          "step 1.5 tells the units the calendar wiped on purpose from the ones at 0 in the extract")
    forecast = results["forecast"]["forecast"]
    projected = forecast[forecast["origen_pipeline"] == "proyectada"]
    check(len(projected) > 0 and set(projected["term"].astype(str)) == {"1 year"},
          "step 7.2 projects again only the 1-year licences")


def test_the_term_transitions() -> None:
    print("C · the term transitions: the projected renewals split by the term every licence renews to")
    results = {}
    configuration = term_configuration(with_transitions=True)
    console = console_of(lambda: results.update(Orchestrator(configuration).run()))
    forecast = results["forecast"]["forecast"]
    window = configuration.simulation_window
    renewing = forecast[forecast["period"].isin(window) & (forecast["origen_pipeline"] == "real") & (forecast["esperado_unidades"] > 0)]
    projected = forecast[(forecast["origen_pipeline"] == "proyectada") & (forecast["fila_maduracion"] == 0)]
    expected_units, expected_usd = 0.0, 0.0
    for term, block in renewing.groupby(renewing["term"].astype(str)):
        expected_units += block["esperado_unidades"].sum() * TRANSITION_SHARES[term]["1y"][0]
        expected_usd += block["esperado_usd"].sum() * TRANSITION_SHARES[term]["1y"][1]
    check(set(projected["term"].astype(str)) == {"1 year"}
          and abs(projected["total_tr_units"].sum() - expected_units) <= 1e-6 * max(expected_units, 1)
          and abs(projected["total_tr_usd"].sum() - expected_usd) <= 1e-6 * max(expected_usd, 1),
          f"the projected renewals are the expected renewals × the share that renews to 1 year, in units and in USD: "
          f"80 % / 70 % of the 1-year licences (the 20 % to 2 years is a gap) and 40 % / 25 % of the 3-year ones "
          f"({projected['total_tr_units'].sum():,.0f} units, ${projected['total_tr_usd'].sum():,.0f})")
    from_three_years = renewing[renewing["term"].astype(str) == "3 year"]
    check(len(from_three_years) > 0 and "term transitions" in console and "from licences of 1 year" in console
          and "3 years" in console,
          "the 3-year licences due in the window that renew to 1 year are projected too, and step 7.2 says it")
    table = results["forecast"]["term_transitions"]
    portfolio = table[table["dims_del_nivel"] == "cartera"].set_index("plazo_origen_anos")
    check(abs(portfolio.loc[1, "parte_unidades_a_2y"] - 0.2) < 1e-9 and abs(portfolio.loc[3, "parte_usd_a_1y"] - 0.25) < 1e-9
          and "SUPUESTO" in console and "the destination term is known" in console,
          "the shares are learned from the closed months (sff_transiciones_plazo), the assumption is said and checked")


def test_a_term_column_that_is_not_mandatory() -> None:
    print("B · a term_column that is not a mandatory dim stops when the Config is built")
    check_stops(lambda: term_configuration("term_level_2"), "must be one of the mandatory dims",
                "a term_column that is not a declared mandatory dim stops before any data is read")


if __name__ == "__main__":
    test_the_term_column()
    test_the_term_transitions()
    test_a_term_column_that_is_not_mandatory()
    finish()
