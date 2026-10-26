"""
test_1_6_series_routes.py — Step 1.6 on the synthetic raw: every series gets its coverage, route and
universe; nothing is dropped; the series table is written.

    python test_1_6_series_routes.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from block_1_preparation import build_series_routes, route_from_coverage
from test_helpers import check, console_of, count_status, finish
from test_1_5_forecast_units import units_synthetic


def run_step(units, configuration):
    frames = {}
    console = console_of(lambda: frames.setdefault("series", build_series_routes(units, configuration)))
    return frames["series"], console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_routes() -> None:
    print("A · coverage and route of every series")
    check(route_from_coverage({"entrenamiento", "examen", "proyeccion"}) == "predecible"
          and route_from_coverage({"entrenamiento"}) == "solo_historia"
          and route_from_coverage({"proyeccion"}) == "solo_futuro"
          and route_from_coverage({"examen", "proyeccion"}) == "predecible",
          "the route rule: history + future → predecible · only history → solo_historia · only future → solo_futuro")
    _, units, configuration = units_synthetic()
    series, console = run_step(units, configuration)
    check(len(series) == 17 and series["fs_id"].is_unique, "one row per series, none dropped")
    check(series["ruta"].value_counts().to_dict() == {"predecible": 15, "solo_futuro": 1, "solo_historia": 1},
          "15 predecible, 1 solo_futuro, 1 solo_historia")
    only_history = series[series["ruta"] == "solo_historia"].iloc[0]
    check(only_history["cobertura"] == "entrenamiento" and only_history["usd_por_predecir"] == 0,
          "the solo_historia series covers only training and has nothing to predict")
    check((series["meses_entrenamiento"] + series["meses_examen"] + series["meses_proyeccion"] == series["meses"]).all(), "the months per role add up to the months of the series")
    check(abs(series["usd_por_predecir"].sum() - units.loc[units["rol"] == "proyeccion", "total_tr_usd"].sum()) < 0.01,
          "the USD to predict is what falls due in the projection")
    check(count_status(console, "ok") == 2 and "2 checks: 2 ok" in console, "the 2 checks pass and are logged")
    check("universo" not in series.columns, "no universe column: the time_series rows were separated in step 1.2")


if __name__ == "__main__":
    test_the_routes()
    finish()
