"""
test_resumen_graficos.py — The summary for the business and the charts of a run.

A · the summary: on screen at the end of the run and at the top of the report, with every part, in business words
B · the charts: every PNG drawn in <output_folder>/graficos and linked from the report; a chart that cannot be drawn
    does not stop the run
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os

from main import run
from test_helpers import check, console_of, finish, synthetic_with


def run_synthetic():
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    return configuration, results, console


def test_the_summary(configuration, results, console) -> None:
    print("A · the summary for the business")
    section = console.split("RESUMEN PARA NEGOCIO ══")[1].split("FIN DEL RESUMEN PARA NEGOCIO")[0] if "RESUMEN PARA NEGOCIO ══" in console else ""
    check(all(part in section for part in ("LA PIPELINE", "LA PREDICCIÓN", "EL RESULTADO", "LA MADURACIÓN", "CUÁNTO FIARSE", "EL RUIDO", "OJO")),
          "on screen at the end: the pipeline, the prediction, the result, how much to trust it, the noise and the caveats")
    check("predicciones de la tasa" in section and "frente a la baseline" in section and "pp de media" in section,
          "it says how many predictions, the gain over the baseline and the binomial noise before and after")
    report = open(os.path.join(configuration.output_folder, "informe_sff.md"), encoding="utf-8").read()
    check("### Resumen para negocio" in report and "- LA PIPELINE" in report and report.count("\n## ") == 9,
          "the report opens with the summary, and keeps its nine chapters")
    check("LA BANDA" in section and "LO QUE HAY QUE VALIDAR" in section,
          "the summary says what the band is made of and which hypotheses do not hold on these data")


def test_the_charts(configuration, results, console) -> None:
    print("B · the charts")
    folder = os.path.join(configuration.output_folder, "graficos")
    expected = ["01_backtest_esquema", "02_examen_por_horizonte", "03_cascada_usd", "04_ruido_binomial", "06_serie_ejemplo",
                "07_tecnicas_serie_ejemplo", "08_forecast_mensual"]
    check(all(os.path.getsize(os.path.join(folder, f"{name}.png")) > 10_000 for name in expected),
          "every chart of a run without dated states is drawn (the maturation curve needs maturation_dims)")
    report = open(os.path.join(configuration.output_folder, "informe_sff.md"), encoding="utf-8").read()
    check(all(f"](graficos/{name}.png)" in report for name in expected), "the report links every chart")
    check("not drawn" not in console, "no chart failed")


if __name__ == "__main__":
    shared = run_synthetic()
    test_the_summary(*shared)
    test_the_charts(*shared)
    finish()
