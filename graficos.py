"""
graficos.py — The charts and diagrams of a run, as PNG files: how the forecast is built and how much to trust it.

Every chart answers one question, and is drawn from the results of the run (it computes nothing the steps did
not compute, except to aggregate them for the picture):

  01_backtest_esquema         how the backtest and the exam work: the calendar of the run (training, selection,
                              exam, projection) and, for one exam month, what every horizon sees and predicts
  02_examen_por_horizonte     the error of the monthly total in the exam, per horizon and method (the cascade from
                              the baseline to the framework, and the states predicted when the extract dates them)
  03_cascada_usd              the same in money: the mean error of a month in USD of the baseline, the ladder with
                              the default technique and the framework; the gain over the baseline
  04_ruido_binomial           the binomial noise before and after the improvements: the share of the money whose
                              rate is known within x pp, for the series alone and after the ladder and credibility
  05_curva_maduracion         the maturation curve of every state with its dates: the share of the final state
                              already visible k months before due, and the probability that a unit without the state ends in it
  06_serie_ejemplo            one forecast series from start to end: its own monthly rate, its composition's, the
                              exam predictions, the forecast with its band; and the maturation of its states month by month
  07_tecnicas_serie_ejemplo   every technique of that series' composition in the backtest: its error, the chosen one,
                              the default technique
  08_forecast_mensual         the renewed USD month by month: real history, forecast and its bands

Colours: the validated categorical order of the reference palette (blue, orange, aqua, violet) on a light
surface; identity is never colour alone (every line has its direct label or a legend).

Used by step 8.6 (the files go to <output_folder>/graficos and the report links them).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from config import (ACTIVE_FLAG_VALUES, CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, Config, DUE_UNITS_AT_DUE_COLUMN,
                    METHOD_BASELINE, METHOD_DEFAULT, METHOD_FRAMEWORK, PURPOSE_EXAM, PURPOSE_SELECTION, RATE_COLUMN,
                    SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TRUTH_ROLES, join_columns, negative_states)
from maturation import MaturationCurve, closed_rows_of
from prediction import band_quantiles, rate_band
from resumen_negocio import exam_error_of_months

# ─── the look ────────────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8f8d86"
GRID = "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
NEUTRAL_FILL = "#d9d8d2"
METHOD_STYLE = {METHOD_BASELINE: ("Baseline (su celda, últimos meses)", ORANGE, "-"),
                "raw": ("Serie sola", INK_MUTED, ":"),
                METHOD_DEFAULT: ("Escalera + técnica por defecto", AQUA, "-"),
                METHOD_FRAMEWORK: ("Framework", BLUE, "-"),
                "framework_honesto": ("Framework + maduración de los estados del cliente", VIOLET, "--")}
SHORT_LABEL = {METHOD_BASELINE: "Baseline\n(su celda)", METHOD_DEFAULT: "Escalera +\ntécnica por defecto",
               METHOD_FRAMEWORK: "Framework\n(técnica elegida)"}
ROLE_STYLE = {"entrenamiento": ("Entrenamiento", "#ecebe6"), "seleccion": ("Selección de la técnica", "#cfeee2"),
              "examen": ("Examen", "#d4e4f7"), "proyeccion": ("Proyección (forecast)", "#fbe0d4")}
FIGURE_DPI = 150
CHART_FOLDER = "graficos"


def new_figure(width: float = 10.0, height: float = 5.0, rows: int = 1, columns: int = 1, **options):
    figure, axes = plt.subplots(rows, columns, figsize=(width, height), facecolor=SURFACE, **options)
    for axis in np.atleast_1d(axes).ravel():
        style_axis(axis)
    return figure, axes


def style_axis(axis) -> None:
    """Recessive grid and axes; text in ink, never in the series colour."""
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def titles(figure, title: str, subtitle: str = "") -> None:
    figure.suptitle(title, x=0.01, y=0.99, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        figure.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY, wrap=True)


def save(figure, path: str, top: float = 0.86) -> str:
    """Lay the axes out below the title and the subtitle, then write the PNG."""
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def money_label(value: float) -> str:
    """$1.2M, $340k, $5,100; a negative amount as −$406 (the sign before the dollar)."""
    if not np.isfinite(value):
        return "—"
    sign, value = ("−" if value < 0 else ""), abs(value)
    if value >= 1e6:
        return f"{sign}${value / 1e6:,.1f}M"
    if value >= 1e4:
        return f"{sign}${value / 1e3:,.0f}k"
    return f"{sign}${value:,.0f}"


def month_position(months: list, month) -> int:
    return months.index(month)


# ═══════════════════════════════════════════════════════════════════════════════════
# 01 · HOW THE BACKTEST WORKS
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_backtest_scheme(results: dict, configuration: Config, path: str) -> str:
    """The calendar of the run as coloured spans, and for the last exam month T every horizon h as a line: what it
    sees (up to T − h, solid) and what it predicts (T, the dot). The selection months choose the technique; the
    exam months measure it without having seen them."""
    fine = results["fine_table"]
    period = configuration.period_col
    boundaries = configuration.calendar_boundaries()
    first_month, last_month = fine[period].min(), fine[period].max()
    months = list(pd.period_range(first_month, last_month, freq="M"))
    current, test_start = boundaries["current"], boundaries["test_start"]
    selection_start = test_start - configuration.backtest_selection_months
    spans = [("entrenamiento", first_month, selection_start - 1), ("seleccion", selection_start, test_start - 1),
             ("examen", test_start, current - 1), ("proyeccion", current, last_month)]
    horizons = sorted(configuration.backtest_horizons)
    target = current - 1

    figure, axis = new_figure(11, 2.6 + 0.45 * len(horizons))
    axis.grid(False)
    calendar_row = len(horizons) + 1
    legend_handles = []
    for role, start, end in spans:
        if end < start:
            continue
        label, colour = ROLE_STYLE[role]
        left, right = month_position(months, start), month_position(months, end) + 1
        axis.add_patch(Rectangle((left, calendar_row - 0.4), right - left, 0.8, facecolor=colour, edgecolor=SURFACE, linewidth=2))
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=colour, label=f"{label}: {start} – {end} ({right - left} meses)"))
    axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8.5, frameon=False)
    axis.axvline(month_position(months, current), color=INK_SECONDARY, linewidth=1, linestyle="--")
    axis.text(month_position(months, current) + 0.3, calendar_row + 0.5, f"mes en curso {current}", fontsize=8, color=INK_SECONDARY)
    for row, horizon in enumerate(reversed(horizons), start=1):
        origin = target - horizon
        if origin < first_month:
            continue
        seen_end = month_position(months, origin) + 1
        axis.plot([0, seen_end], [row, row], color=BLUE, linewidth=3, solid_capstyle="round")
        target_x = month_position(months, target) + 0.5
        axis.add_patch(FancyArrowPatch((seen_end, row), (target_x, row), arrowstyle="-|>", mutation_scale=12,
                                       color=INK_MUTED, linewidth=1, linestyle=":"))
        axis.plot(target_x, row, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        axis.text(-0.5, row, f"h = {horizon}", ha="right", va="center", fontsize=9, color=INK)
        axis.text(seen_end + 0.2, row + 0.22, f"ve hasta {origin}", fontsize=8, color=INK_SECONDARY)
    axis.text(month_position(months, target) + 0.5, 0.45, f"predice T = {target}", fontsize=8.5, color=INK, ha="right")
    axis.set_xlim(-4, len(months) + 1)
    axis.set_ylim(0.2, calendar_row + 1.6)
    axis.set_yticks([])
    tick_positions = [position for position, month in enumerate(months) if month.month == 1]
    axis.set_xticks(tick_positions, [str(months[position].year) for position in tick_positions])
    titles(figure, "Cómo se hace el backtest",
           f"Para un mes de examen T, cada horizonte h predice con lo que se sabía en T − h (línea) el mes T (punto). En los "
           f"{configuration.backtest_selection_months} meses de selección se elige la técnica de cada composición y tramo; "
           f"en los de examen se mide, sin haberlos visto.")
    return save(figure, path)


# ═══════════════════════════════════════════════════════════════════════════════════
# 02-03 · THE EXAM: PER HORIZON, AND IN MONEY
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_exam_by_horizon(results: dict, configuration: Config, path: str) -> str:
    """The mean |error| of the monthly total in the exam, per horizon, one line per method, labelled at its end."""
    summary = results["series_exam"]["summary"]
    figure, axis = new_figure(9, 5)
    horizons = sorted(summary["h"].unique())
    for method, (label, colour, line_style) in METHOD_STYLE.items():
        rows = summary[summary["metodo"] == method].set_index("h").reindex(horizons)
        if rows["error_total_medio"].isna().all():
            continue
        values = 100 * rows["error_total_medio"]
        axis.plot(horizons, values, line_style, color=colour, linewidth=2, marker="o", markersize=7,
                  markeredgecolor=SURFACE, markeredgewidth=1.5)
        axis.annotate(f"{label}  {values.iloc[-1]:.1f} %", (horizons[-1], values.iloc[-1]), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=9, color=INK)
    axis.set_xticks(horizons, [f"{horizon} mes" if horizon == 1 else f"{horizon} meses" for horizon in horizons])
    axis.set_ylabel("error medio del total de un mes (%)", color=INK_SECONDARY)
    axis.set_ylim(bottom=0)
    axis.set_xlim(horizons[0] - 0.5, horizons[-1] + (horizons[-1] - horizons[0]) * 0.9 + 1)
    titles(figure, "El examen: cuánto se yerra el total de un mes, según cuánto antes se predice",
           "Meses ya cerrados, predichos con lo que se sabía h meses antes. De la baseline (el cálculo que cualquiera haría) "
           "al framework, cada línea añade una pieza.")
    return save(figure, path)


def draw_cascade(results: dict, configuration: Config, path: str) -> str:
    """Per horizon, the mean |error| of a month in USD of the baseline, the ladder with the default technique and the
    framework, with the gain over the baseline written on the bars."""
    summary = results["series_exam"]["summary"]
    steps = [method for method in (METHOD_BASELINE, METHOD_DEFAULT, METHOD_FRAMEWORK) if method in set(summary["metodo"])]
    horizons = sorted(summary["h"].unique())
    figure, axes = new_figure(5.6 * len(horizons), 5.0, 1, len(horizons), sharey=True)
    for axis, horizon in zip(np.atleast_1d(axes), horizons):
        rows = summary[(summary["h"] == horizon)].set_index("metodo")
        values = [rows.loc[method, "error_usd_medio"] for method in steps]
        colours = [METHOD_STYLE[method][1] for method in steps]
        bars = axis.bar(range(len(steps)), values, color=colours, width=0.62, edgecolor=SURFACE, linewidth=2)
        baseline_value = rows.loc[METHOD_BASELINE, "error_usd_medio"] if METHOD_BASELINE in rows.index else np.nan
        for bar, method, value in zip(bars, steps, values):
            text = money_label(value)
            if method != METHOD_BASELINE and np.isfinite(baseline_value):
                difference = value - baseline_value
                text += f"\n{'+' if difference > 0 else ''}{money_label(difference)} vs baseline"
                consistency = rows.loc[method, "meses_mejor_que_baseline"]
                if np.isfinite(consistency):
                    text += f"\nmejor en {consistency:.0%} de meses"
            axis.text(bar.get_x() + bar.get_width() / 2, value, text, ha="center", va="bottom", fontsize=8, color=INK)
        axis.set_xticks(range(len(steps)), [SHORT_LABEL[method] for method in steps], fontsize=8.5)
        axis.set_title(f"a {horizon} mes" if horizon == 1 else f"a {horizon} meses", fontsize=10, color=INK, loc="left")
        axis.margins(y=0.35)
    np.atleast_1d(axes)[0].set_ylabel("error medio de un mes (USD)", color=INK_SECONDARY)
    np.atleast_1d(axes)[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: money_label(value)))
    titles(figure, "Lo que vale cada pieza, en dinero",
           "Error medio del total renovado de un mes en el examen. El precio es el mismo para todos: la diferencia es la de la tasa.")
    return save(figure, path)


# ═══════════════════════════════════════════════════════════════════════════════════
# 04 · THE BINOMIAL NOISE BEFORE AND AFTER
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_noise(results: dict, configuration: Config, path: str) -> str:
    """The share of the money to predict whose rate is known within x pp (90 %), for every series alone (its binomial
    error with its own support) and after the ladder and the credibility (z × the error of the estimate)."""
    estimate = results["series_estimate"]
    weights = estimate["usd_por_predecir"].fillna(0).to_numpy(dtype=float)
    alone = estimate["error_binomial_pp"].to_numpy(dtype=float)
    after = configuration.z * estimate["se_estimacion_pp"].to_numpy(dtype=float)
    figure, axis = new_figure(9, 5)
    limit = float(np.nanpercentile(np.concatenate([alone, after]), 97)) if np.isfinite(alone).any() else 30.0
    grid = np.linspace(0, max(limit, configuration.promise_pp * 2), 300)
    for values, label, colour, style in ((alone, "Cada serie sola", ORANGE, "-"), (after, "Tras la escalera y la credibilidad", BLUE, "-")):
        known = np.isfinite(values)
        total = weights.sum()
        shares = [weights[known & (values <= x)].sum() / total if total else 0 for x in grid]
        axis.plot(grid, 100 * np.array(shares), style, color=colour, linewidth=2.2)
        at_promise = weights[known & (values <= configuration.promise_pp)].sum() / total if total else 0
        axis.annotate(f"{label}: {at_promise:.0%}", (configuration.promise_pp, 100 * at_promise), xytext=(12, -16 if colour == ORANGE else 10),
                      textcoords="offset points", fontsize=9, color=INK)
        axis.plot(configuration.promise_pp, 100 * at_promise, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    axis.axvline(configuration.promise_pp, color=INK_SECONDARY, linestyle="--", linewidth=1)
    axis.text(configuration.promise_pp, 2, f"  ±{configuration.promise_pp:.0f} pp: la promesa", fontsize=8.5, color=INK_SECONDARY)
    axis.set_xlabel("error de la tasa por azar (pp, 90 %)", color=INK_SECONDARY)
    axis.set_ylabel("% del dinero por predecir con ese error o menos", color=INK_SECONDARY)
    axis.set_ylim(0, 102)
    axis.set_xlim(0, grid[-1])
    titles(figure, "El ruido binomial: cuánto mejora la tasa al apoyarse en series parecidas",
           "Una serie con pocos contratos al mes tiene una tasa que se mueve por azar. La escalera junta series casi iguales "
           "y la credibilidad mezcla con su referencia: el mismo dinero, con la tasa conocida con menos ruido.")
    return save(figure, path)


# ═══════════════════════════════════════════════════════════════════════════════════
# 05 · THE MATURATION CURVE
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_maturation_curve(results: dict, configuration: Config, path: str):
    """For every state with its dates: the share of the final state already visible k months before due, and the
    probability that a unit not in the state k months before ends in the state. None when no state carries its dates."""
    if not configuration.maturation_dims:
        return None
    closed = closed_rows_of(results["fine_table"], configuration)
    months = sorted(closed[configuration.period_col].unique())[-configuration.maturation_history_months:]
    history = closed[closed[configuration.period_col].isin(months)]
    figure, axes = new_figure(10, 4.6, 1, 2)
    for state, colour in zip(configuration.maturation_dims, (BLUE, VIOLET, AQUA)):
        table = MaturationCurve(history, state, configuration).portfolio_table()
        axes[0].plot(table["meses_antes"], 100 * table["visible_del_final"], color=colour, linewidth=2.2, marker="o", markersize=5)
        axes[0].annotate(state, (table["meses_antes"].iloc[-1], 100 * table["visible_del_final"].iloc[-1]), xytext=(6, 0),
                         textcoords="offset points", fontsize=9, color=INK)
        axes[1].plot(table["meses_antes"], 100 * table["p_tomar_estado"], color=colour, linewidth=2.2, marker="o", markersize=5)
    axes[0].set_title("Del estado final, cuánto se ve k meses antes", fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("% de las unidades que acabarán en el estado", color=INK_SECONDARY)
    axes[1].set_title("p(k): de lo que NO estaba en el estado k meses antes, cuánto acaba en él", fontsize=10, color=INK, loc="left")
    axes[1].set_ylabel("% de las unidades sin el estado", color=INK_SECONDARY)
    for axis in axes:
        axis.set_xlabel("meses antes del vencimiento (k)", color=INK_SECONDARY)
        axis.invert_xaxis()
        axis.set_ylim(bottom=0)
    titles(figure, "La maduración de los estados del cliente: cómo se va completando antes de vencer",
           f"Aprendida de los últimos {len(months)} meses cerrados (sus estados ya son finales). Para un mes futuro a k meses, "
           f"el forecast suma a lo visible hoy la parte p(k) de lo que aún no tiene el estado.")
    return save(figure, path)


# ─── 04.1 · the maturation of every state by region (drawn by step 4.1, shown on screen) ───
REGION_COLOURS = (BLUE, ORANGE, AQUA, VIOLET, "#c9a227", "#d6336c", "#5f8f2e", "#7a5c3e")


def draw_maturation_by_region(table: pd.DataFrame, box, configuration: Config, path: str) -> str:
    """One panel per negative state, one line per region (the first region level) and the portfolio in grey: the
    share of the final state already visible k months before due. A dated state shows its PAST curve (average of the
    cohorts the box learns from); an undated one shows the pipeline as seen TODAY over the final share of its region,
    dashed, because it is not a curve of the past. `table` is maturation_by_region of block_4_customer_states."""
    states = list(dict.fromkeys(table["estado"]))
    dated = set(configuration.maturation_dims)
    figure, axes = new_figure(4.6 * len(states), 5.4, 1, len(states), sharey=True)
    axes = np.atleast_1d(axes)
    regions = [region for region in table["region"].drop_duplicates() if region != "cartera"]
    for axis, state in zip(axes, states):
        view = "pasado" if state in dated else "hoy"
        of_state = table[(table["estado"] == state) & (table["vista"] == view)]
        for region, colour in zip(regions, REGION_COLOURS):
            line = of_state[of_state["region"] == region].sort_values("meses_antes")
            final = line["cuota_final"].iloc[0] if len(line) else np.nan
            axis.plot(line["meses_antes"], 100 * line["visible_del_final"], color=colour, linewidth=2.0,
                      linestyle="-" if view == "pasado" else "--", marker="o", markersize=3.5,
                      label=f"{region} (final {100 * final:.1f} %)")
        portfolio = of_state[of_state["region"] == "cartera"].sort_values("meses_antes")
        if len(portfolio):
            axis.plot(portfolio["meses_antes"], 100 * portfolio["visible_del_final"], color=INK_MUTED, linewidth=1.4,
                      linestyle=":", label=f"cartera (final {100 * portfolio['cuota_final'].iloc[0]:.1f} %)")
        subtitle = ("curva del pasado: media de las cohortes cerradas" if view == "pasado"
                    else "sin fechas en el extracto: la pipeline vista HOY")
        axis.set_title(f"{state}\n{subtitle}", fontsize=10, color=INK, loc="left")
        axis.set_xlabel("meses antes del vencimiento (k)", color=INK_SECONDARY)
        axis.invert_xaxis()
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    axes[0].set_ylabel("% del estado final ya visible", color=INK_SECONDARY)
    months = box.history_months
    titles(figure, "La maduración de los estados del cliente por región",
           f"Cuánto del estado final se ve k meses antes de vencer, por región. Las cohortes del pasado son "
           f"{months[0]}..{months[-1]}. Para los estados sin fechas (línea discontinua) el pasado no se ve: es lo que "
           f"se ve hoy en la pipeline sobre la cuota final de la región.")
    return save(figure, path, top=0.88)


# ─── 04.1 · the maturation curve explained: when the state is set, how it is predicted, how well ───
def draw_maturation_explained(explained: dict, threshold_pp: float, path: str) -> str:
    """One row per dated state, three panels: (1) WHEN the final state is set — the distribution f(k) of the months
    before due, as bars, and the share already visible F(h), as a line, with the month by which half is visible;
    (2) the pipeline of TODAY — per future month, the share in the state seen now and what the curve adds up to the
    expected final, next to the tope of the history; (3) HOW WELL — per horizon, the error of the final share in the
    exam, with the curve and with the states seen then, next to the threshold to use the curve. A state whose curve
    failed its exam says so in its title. explained: {state: the numbers of maturation_explained (step 4.1)}."""
    states = list(explained)
    if not states:
        return None
    figure, axes = new_figure(15, 4.6 * len(states), len(states), 3)
    axes = np.atleast_2d(axes)
    for row, state in enumerate(states):
        numbers = explained[state]
        when, today, exam = numbers["when"], numbers["today"], numbers["exam"]
        months = len(when) - 1

        # (1) when the state is set
        axis = axes[row, 0]
        axis.bar(when["meses_antes"], 100 * when["f"], color=VIOLET, alpha=0.35, width=0.8,
                 label="f(k): parte del estado final que se puso k meses antes")
        axis.plot(when["meses_antes"], 100 * when["F"], color=VIOLET, linewidth=2.4, marker="o", markersize=4,
                  label="F(h): parte del final ya visible h meses antes")
        half = when[when["F"] >= 0.5]["meses_antes"].max() if (when["F"] >= 0.5).any() else np.nan
        if np.isfinite(half):
            axis.axvline(half, color=INK_MUTED, linewidth=1, linestyle=":")
            axis.annotate(f"la mitad ya se ve\n{int(half)} meses antes", (half, 75), xytext=(-5, 0), textcoords="offset points",
                          fontsize=8.5, color=INK_SECONDARY, ha="right")
        axis.set_xticks(range(months + 1))
        axis.set_xticklabels([f"{k}+" if k == months else str(k) for k in range(months + 1)], fontsize=8)
        axis.invert_xaxis()
        axis.set_ylim(0, 135)
        axis.set_yticks(range(0, 101, 20))
        axis.set_xlabel("meses antes de vencer", color=INK_SECONDARY)
        axis.set_ylabel("% del estado final", color=INK_SECONDARY)
        axis.set_title(f"{state.upper()} · ¿cuándo se pone el estado?\n"
                       + (f"acaba en él el {100 * numbers['tope']:.1f} % de lo que vence" if numbers["used"]
                          else "LA CURVA NO SE USA: suspendió su examen"),
                       fontsize=10, color=INK if numbers["used"] else ORANGE, loc="left")
        axis.legend(fontsize=7.5, frameon=False, loc="upper left")

        # (2) the pipeline of today: seen now + what the curve adds
        axis = axes[row, 1]
        if len(today):
            labels = [str(month) for month in today["mes"]]
            positions = np.arange(len(today))
            axis.bar(positions, 100 * today["visto_hoy"], color=VIOLET, width=0.7, label="visto hoy en la pipeline")
            axis.bar(positions, 100 * today["anade_la_curva"], bottom=100 * today["visto_hoy"], color=VIOLET, alpha=0.3,
                     width=0.7, label="lo que añade la curva hasta vencer" if numbers["used"]
                     else "lo que añadiría la curva (no se usa: el estado toma su cuota histórica)")
            axis.axhline(100 * numbers["tope"], color=INK_MUTED, linewidth=1.2, linestyle="--",
                         label=f"tope de los últimos {len(numbers['history_months'])} meses cerrados")
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=60, fontsize=7.5)
        axis.set_ylabel(f"% de lo que vence, en {state}", color=INK_SECONDARY)
        axis.set_title("la pipeline de hoy, por mes de vencimiento:\nlo que ya se ve + lo que añade la curva", fontsize=10,
                       color=INK, loc="left")
        axis.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
        top_of_the_bars = float((today["visto_hoy"] + today["anade_la_curva"]).max()) if len(today) else 0.0
        axis.set_ylim(0, 100 * max(top_of_the_bars, numbers["tope"]) * 1.45 or 1)

        # (3) how well, by how far ahead
        axis = axes[row, 2]
        if exam is not None and len(exam):
            positions = np.arange(len(exam))
            seen_error = exam["error_vista_abs_pp"] if "error_vista_abs_pp" in exam else exam["error_vista_pp"].abs()
            for offset, values, colour, label in ((-0.18, exam["error_curva_abs_pp"], VIOLET, "con la curva"),
                                                  (0.18, seen_error, NEUTRAL_FILL, "sin curva: el estado visto entonces")):
                bars = axis.bar(positions + offset, values, width=0.36, color=colour, label=label)
                axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], fontsize=8, color=INK_SECONDARY, padding=2)
            axis.axhline(threshold_pp, color=ORANGE, linewidth=1, linestyle=":",
                         label=f"umbral para usar la curva ({threshold_pp:g} pp)")
            axis.set_ylim(0, max(float(seen_error.max()), float(exam["error_curva_abs_pp"].max()),
                                 threshold_pp) * 1.35)
            axis.set_xticks(positions)
            axis.set_xticklabels([f"{int(h)} {'mes' if int(h) == 1 else 'meses'}" for h in exam["h"]], fontsize=8.5)
        axis.set_xlabel("antelación de la predicción", color=INK_SECONDARY)
        axis.set_ylabel("error medio de la cuota final (pp)", color=INK_SECONDARY)
        axis.set_title("¿cuánto acierta según la antelación?\nerror de la cuota final en el examen (meses cerrados)",
                       fontsize=10, color=INK, loc="left")
        axis.legend(fontsize=7.5, frameon=False, loc="upper left")
    titles(figure, "La curva de maduración de los estados del cliente: cuándo se ponen, cómo se predicen y cuánto aciertan",
           f"Aprendida de las cohortes {explained[states[0]]['history_months'][0]}..{explained[states[0]]['history_months'][-1]} "
           f"(ponderadas por unidades). Para un mes "
           f"futuro, lo esperado al vencer = lo visto hoy + (lo que aún no está en el estado) × p(h), la probabilidad de "
           f"entrar en él en los h meses que faltan. El supuesto: las cohortes futuras maduran como estas.")
    figure.subplots_adjust(wspace=0.28, hspace=0.75)
    return save(figure, path, top=0.90 if len(states) > 1 else 0.84)


# ─── 05.1 · the acquisitions month by month: in total and by term ───
TERM_COLOURS = {1: BLUE, 2: AQUA, 3: VIOLET}
KIND_STYLE = {"real": "-", "estimado": ":", "simulado": "--"}


def draw_kinds(axis, positions: np.ndarray, values: np.ndarray, kinds: np.ndarray, colour: str, label: str) -> None:
    """One line whose pieces take the style of their kind (real solid, estimated dotted, simulated dashed); every piece
    starts at the last point of the one before, so the line has no gaps."""
    start = 0
    for end in range(1, len(kinds) + 1):
        if end == len(kinds) or kinds[end] != kinds[start]:
            piece = slice(max(start - 1, 0), end)
            axis.plot(positions[piece], values[piece], color=colour, linewidth=2 if kinds[start] == "real" else 1.8,
                      linestyle=KIND_STYLE.get(kinds[start], "-"), label=label if start == 0 else None)
            start = end


def draw_acquisitions_by_month(months: pd.DataFrame, current, target_usd, path: str) -> str:
    """Top: the acquisitions of every month (every term together), the total of every year written over it. Bottom: the
    same, one line per term. Solid: sold (read in the extract); dotted: sold but estimated (its term falls due after
    the extract); dashed: simulated (current month to December, up to the target when there is one)."""
    months = months.sort_values(["mes_venta", "plazo"])
    order = list(dict.fromkeys(months["mes_venta"]))
    positions = np.arange(len(order))
    total = months.groupby("mes_venta", sort=False)["usd"].sum().reindex(order)
    kind_of_month = months.groupby("mes_venta", sort=False)["tipo"].agg(
        lambda kinds: "simulado" if (kinds == "simulado").any() else "estimado" if (kinds == "estimado").any() else "real").reindex(order)
    figure, axes = new_figure(14, 8.5, 2, 1, sharex=True)
    draw_kinds(axes[0], positions, total.to_numpy(dtype=float), kind_of_month.to_numpy(), INK, "todos los plazos")
    axes[0].set_ylim(bottom=0)
    top = float(total.max()) if len(total) else 0.0
    for year in sorted({month.year for month in order}):
        in_year = [index for index, month in enumerate(order) if month.year == year]
        year_total = float(total.iloc[in_year].sum())
        sold = float(total.iloc[[index for index in in_year if order[index] < current]].sum())
        text = f"{year}: {money_label(year_total)}"
        if any(order[index] >= current for index in in_year):
            text = f"{year}: {money_label(sold)} vendido + {money_label(year_total - sold)} simulado = {money_label(year_total)}"
            if target_usd:
                text += f"\nobjetivo del equipo de adquisición: {money_label(target_usd)}"
        axes[0].annotate(text, (np.mean(in_year), top * 1.02), ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        axes[0].axvline(in_year[0] - 0.5, color=GRID, linewidth=1)
    axes[0].set_ylim(0, top * 1.25 if top > 0 else 1)
    axes[0].yaxis.set_major_formatter(lambda value, _: money_label(value))
    axes[0].set_title("todos los plazos (1, 2 y 3 años), USD por mes de venta", fontsize=10, color=INK, loc="left")
    for term, by_term in months.groupby("plazo"):
        by_term = by_term.set_index("mes_venta").reindex(order)
        draw_kinds(axes[1], positions, by_term["usd"].fillna(0.0).to_numpy(dtype=float), by_term["tipo"].fillna("real").to_numpy(),
                   TERM_COLOURS.get(int(term), INK_MUTED), f"{int(term)} {'año' if int(term) == 1 else 'años'}")
    axes[1].set_ylim(bottom=0)
    axes[1].yaxis.set_major_formatter(lambda value, _: money_label(value))
    axes[1].set_title("por plazo", fontsize=10, color=INK, loc="left")
    axes[1].legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
    step = 3
    axes[1].set_xticks(positions[::step])
    axes[1].set_xticklabels([str(month) for month in order][::step], rotation=60, fontsize=8)
    titles(figure, "La adquisición mes a mes: lo vendido y lo que se simula hasta diciembre",
           "Línea continua: vendido (leído en el extracto). Punteada: vendido pero estimado (su plazo vence después del "
           "extracto). Discontinua: simulado"
           + (" hasta el objetivo del equipo de adquisición" if target_usd else "") + ". Solo el plazo de 1 año vence en el "
           "horizonte; los de 2 y 3 años se cuentan para que el año cuadre.")
    return save(figure, path, top=0.9)


# ═══════════════════════════════════════════════════════════════════════════════════
# 06-07 · ONE FORECAST SERIES FROM START TO END, AND ITS TECHNIQUES
# ═══════════════════════════════════════════════════════════════════════════════════

def example_series(results: dict) -> str:
    """The forecast series with the most money to predict that has a composition and exam predictions."""
    estimate = results["series_estimate"]
    examined = set(results["series_exam"]["detail"][SERIES_ID_COLUMN])
    candidates = estimate[estimate[COMPOSITION_ID_COLUMN].notna() & estimate[SERIES_ID_COLUMN].isin(examined)]
    return candidates.sort_values("usd_por_predecir", ascending=False)[SERIES_ID_COLUMN].iloc[0]


def draw_example_series(results: dict, configuration: Config, series_id: str, path: str) -> str:
    """The series' own monthly rate, its composition's, the exam predictions at h = 1 (framework and baseline), the
    forecast rate with its band; below, for the future months of its GROUP (the series without its states: the neutral
    one and its twins with a state), every negative state today and expected at expiry (stage 4.1)."""
    period = configuration.period_col
    rated = results["rated_units"]
    own = rated[(rated[SERIES_ID_COLUMN] == series_id) & (rated[SYNTHETIC_COLUMN] == 0)
                & rated[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & rated[RATE_COLUMN].notna()]
    estimate = results["series_estimate"].set_index(SERIES_ID_COLUMN).loc[series_id]
    composition = results["pool_series"]
    composition = composition[(composition[COMPOSITION_ID_COLUMN] == estimate[COMPOSITION_ID_COLUMN]) & composition[RATE_COLUMN].notna()]
    detail = results["series_exam"]["detail"]
    exam = detail[(detail[SERIES_ID_COLUMN] == series_id) & (detail["h"] == min(configuration.backtest_horizons))]
    forecast = results["forecast"]["forecast"]
    future = forecast[(forecast[SERIES_ID_COLUMN] == series_id) & (forecast[configuration.pipeline_units_col] > 0)]
    units = configuration.pipeline_units_col
    # its group: the series without its states (the maturation moves units between the series of a group)
    states = [state for state in negative_states(configuration) if state in forecast.columns]
    group_columns = [column for column in configuration.rate_series_columns if column not in states]
    group_key = join_columns(forecast, group_columns)
    group_of_series = group_key[forecast[SERIES_ID_COLUMN] == series_id]
    group_future = forecast[group_key == group_of_series.iloc[0]] if len(group_of_series) else forecast.iloc[0:0]
    # the band of the SERIES in the month (its own units, its technique at its horizon), not the average of the bands
    # of its fine rows: a fine row has few units and its binomial band is far wider than the series'
    by_month = future.groupby(period).apply(lambda rows: pd.Series({
        "tasa": rows["esperado_unidades"].sum() / rows[DUE_UNITS_AT_DUE_COLUMN].sum(),
        "unidades": rows[units].sum(), "h": int(rows["h"].max()), "tecnica": rows["tecnica"].mode().iloc[0]}),
        include_groups=False)
    if len(by_month):
        q_low, q_high = band_quantiles(results["backtest"]["bands"], by_month["tecnica"], by_month["h"], configuration)
        by_month["baja"], by_month["alta"] = rate_band(by_month["tasa"], by_month["unidades"], q_low, q_high)
    if len(by_month):                                   # every month of the horizon: a month with nothing due breaks the line
        by_month = by_month.reindex(pd.period_range(by_month.index.min(), by_month.index.max(), freq="M"))
    by_month = by_month.rename_axis(period).reset_index()

    figure, axes = new_figure(11, 7.5, 2, 1, gridspec_kw={"height_ratios": [2.2, 1]})
    axis = axes[0]
    x_of = lambda months: [month.to_timestamp() for month in months]
    axis.plot(x_of(own[period]), 100 * own[RATE_COLUMN], color=INK_MUTED, linewidth=1, marker="o", markersize=3, label="la serie sola (su tasa de cada mes)")
    axis.plot(x_of(composition[period]), 100 * composition[RATE_COLUMN], color=AQUA, linewidth=2, label="su composición (la escalera)")
    for method, marker in ((METHOD_FRAMEWORK, "o"), (METHOD_BASELINE, "s")):
        rows = exam[exam["method"] == method]
        if len(rows):
            label, colour, _ = METHOD_STYLE[method]
            axis.plot(x_of(rows[period]), 100 * rows["pred_rate"], marker, color=colour, markersize=7, markeredgecolor=SURFACE,
                      linestyle="none", label=f"examen, {label.lower()}")
    if len(by_month):
        axis.fill_between(x_of(by_month[period]), 100 * by_month["baja"], 100 * by_month["alta"], color=BLUE, alpha=0.15, linewidth=0,
                          label="banda del forecast (90 %)")
        axis.plot(x_of(by_month[period]), 100 * by_month["tasa"], color=BLUE, linewidth=2.4, label="forecast de la tasa")
    axis.axvline(configuration.calendar_boundaries()["current"].to_timestamp(), color=INK_SECONDARY, linestyle="--", linewidth=1)
    axis.set_ylabel("tasa de renovación (%)", color=INK_SECONDARY)
    axis.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    same = estimate[COMPOSITION_ID_COLUMN] == series_id
    axis.set_title(textwrap.fill(f"{series_id}", 150) + "\n"
                   + ("predice con su propia historia (es su propia composición)" if same
                      else textwrap.fill(f"composición: {estimate[COMPOSITION_ID_COLUMN]}", 150))
                   + f"  ·  z = {estimate['z']:.2f}  ·  {money_label(estimate['usd_por_predecir'])} por predecir",
                   fontsize=8.5, color=INK, loc="left")

    # the states of the future months of its group: today (the data) and expected at expiry (the box of stage 4.1)
    axis = axes[1]
    shown_months = sorted(group_future[period].unique())[:12]
    width = 0.8 / max(len(states), 1)
    for position, (state, colour) in enumerate(zip(states, (BLUE, VIOLET, AQUA, ORANGE))):
        today, expected = [], []
        for month in shown_months:
            rows = group_future[group_future[period] == month]
            total = rows[units].sum()
            in_state = rows[state].isin(ACTIVE_FLAG_VALUES)
            today.append(100 * rows.loc[in_state, units].sum() / total if total else np.nan)
            expected.append(100 * rows.loc[in_state, DUE_UNITS_AT_DUE_COLUMN].sum() / total if total else np.nan)
        x = np.arange(len(shown_months)) + (position - (len(states) - 1) / 2) * width
        axis.bar(x, expected, width=width * 0.9, color=colour, alpha=0.35, edgecolor=SURFACE, label=f"{state}: esperado al vencer")
        axis.bar(x, today, width=width * 0.9, color=colour, edgecolor=SURFACE, label=f"{state}: visible hoy")
    axis.set_xticks(np.arange(len(shown_months)), [str(month) for month in shown_months], fontsize=8, rotation=45)
    axis.set_ylabel("% de lo que vence", color=INK_SECONDARY)
    axis.legend(fontsize=7.5, frameon=False, ncol=max(len(states), 1), loc="upper left")
    axis.set_title("Los estados del cliente de su grupo (la serie sin sus estados) en los meses futuros: lo que ya se ve y lo que la "
                   "caja de maduración espera al vencer", fontsize=9, color=INK, loc="left")
    titles(figure, "Una forecast series de principio a fin",
           "Su historia, la de su composición, cómo se predijo en el examen y su forecast con la banda; abajo, la maduración "
           "de sus estados mes a mes.")
    return save(figure, path, top=0.9)


def draw_techniques_of_series(results: dict, configuration: Config, series_id: str, path: str) -> str:
    """Every technique of the series' composition in the backtest, per horizon band: the mean |err_norm| in the
    selection months (bar) and in the exam (dot); the chosen one in blue, the default technique in orange."""
    composition_id = results["series_estimate"].set_index(SERIES_ID_COLUMN).loc[series_id, COMPOSITION_ID_COLUMN]
    predictions = results["backtest"]["predictions"]
    predictions = predictions[predictions[COMPOSITION_ID_COLUMN] == composition_id].assign(abs_norm=lambda frame: frame["err_norm"].abs())
    decision = results["backtest"]["decision"]
    chosen_of = decision[decision[COMPOSITION_ID_COLUMN] == composition_id].set_index("tramo_h")["tecnica"].to_dict()
    bands = list(configuration.horizon_bands)
    figure, axes = new_figure(5.2 * len(bands), 5, 1, len(bands), sharey=False)
    for axis, band in zip(np.atleast_1d(axes), bands):
        rows = predictions[predictions["tramo_h"] == band]
        selection = rows[rows["proposito"] == PURPOSE_SELECTION].groupby("tecnica")["abs_norm"].mean().sort_values()
        exam = rows[rows["proposito"] == PURPOSE_EXAM].groupby("tecnica")["abs_norm"].mean()
        colours = [BLUE if technique == chosen_of.get(band) else ORANGE if technique == configuration.default_technique else NEUTRAL_FILL
                   for technique in selection.index]
        positions = np.arange(len(selection))
        axis.barh(positions, selection.to_numpy(), color=colours, edgecolor=SURFACE, height=0.7)
        axis.plot(exam.reindex(selection.index).to_numpy(), positions, "o", color=INK, markersize=5)
        axis.set_yticks(positions, [f"{technique}  ← elegida" if technique == chosen_of.get(band) else
                                    f"{technique}  (por defecto)" if technique == configuration.default_technique else technique
                                    for technique in selection.index], fontsize=8.5)
        axis.invert_yaxis()
        axis.grid(True, axis="x", color=GRID)
        axis.grid(False, axis="y")
        axis.set_xlabel("error medio / ruido binomial (|err_norm|)", color=INK_SECONDARY)
        axis.set_title(f"tramo {band}", fontsize=10, color=INK, loc="left")
    titles(figure, f"Todas las técnicas de su composición ({composition_id})",
           "Barras: error en los meses de selección (eligen la técnica). Puntos: el mismo error en el examen. 1 = tan cerca como "
           "permite el azar. Una técnica sustituye a la técnica por defecto solo si le gana por un margen en los mismos meses.")
    return save(figure, path)


# ═══════════════════════════════════════════════════════════════════════════════════
# 08 · THE FORECAST MONTH BY MONTH
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_forecast_by_month(results: dict, configuration: Config, path: str) -> str:
    """The renewed USD of every month: the real history, the forecast, its band with every error in the same
    direction (linear) and its band with independent errors (quadrature)."""
    period = configuration.period_col
    fine = results["fine_table"]
    history = (fine[fine[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)].groupby(period)[configuration.renewed_usd_col].sum()
               .reset_index())
    by_month = exam_error_of_months(results, configuration)
    if by_month is None:
        by_month = results["forecast"]["by_month"].assign(error_usd=np.nan)
    figure, axis = new_figure(11, 5)
    x_of = lambda months: [month.to_timestamp() for month in months]
    axis.plot(x_of(history[period]), history[configuration.renewed_usd_col], color=INK, linewidth=1.8, label="renovado real")
    axis.fill_between(x_of(by_month[period]), by_month["esperado_usd"] - by_month["error_usd"], by_month["esperado_usd"] + by_month["error_usd"],
                      color=BLUE, alpha=0.18, linewidth=0, label="error típico del total, del examen (a su horizonte)")
    axis.plot(x_of(by_month[period]), by_month["esperado_usd"], color=BLUE, linewidth=2.4, label="forecast")
    axis.axvline(configuration.calendar_boundaries()["current"].to_timestamp(), color=INK_SECONDARY, linestyle="--", linewidth=1)
    axis.set_ylabel("USD renovado al mes", color=INK_SECONDARY)
    axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: money_label(value)))
    axis.legend(fontsize=8.5, frameon=False, loc="lower left")
    titles(figure, "El forecast mes a mes", "Lo renovado en los meses cerrados y lo esperado desde el mes en curso. La banda es "
           "el error medio del total de un mes en el examen, al horizonte de cada mes.")
    return save(figure, path)


# ═══════════════════════════════════════════════════════════════════════════════════
# ALL OF THEM
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_charts(results: dict, configuration: Config) -> dict:
    """Every chart of the run into <output_folder>/graficos; name → path (a chart that cannot be drawn is skipped,
    with the reason in its place)."""
    folder = os.path.join(configuration.output_folder or ".", CHART_FOLDER)
    os.makedirs(folder, exist_ok=True)
    drawn = {}
    series_id = None
    try:
        series_id = example_series(results)
    except (KeyError, IndexError):
        pass
    plan = [("01_backtest_esquema", lambda path: draw_backtest_scheme(results, configuration, path)),
            ("02_examen_por_horizonte", lambda path: draw_exam_by_horizon(results, configuration, path)),
            ("03_cascada_usd", lambda path: draw_cascade(results, configuration, path)),
            ("04_ruido_binomial", lambda path: draw_noise(results, configuration, path)),
            ("05_curva_maduracion", lambda path: draw_maturation_curve(results, configuration, path)),
            ("06_serie_ejemplo", lambda path: draw_example_series(results, configuration, series_id, path) if series_id else None),
            ("07_tecnicas_serie_ejemplo", lambda path: draw_techniques_of_series(results, configuration, series_id, path) if series_id else None),
            ("08_forecast_mensual", lambda path: draw_forecast_by_month(results, configuration, path))]
    for name, draw in plan:
        path = os.path.join(folder, f"{name}.png")
        try:
            if draw(path):
                drawn[name] = path
        except Exception as error:                     # a chart never stops the run: it says why it is missing
            plt.close("all")
            configuration.logger.doc(f"[8.6] chart {name} not drawn: {type(error).__name__}: {error}")
    return drawn
