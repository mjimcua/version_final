"""
sff_explorer.py — An interactive explorer of a run of the SFF, inside a notebook.

From the business answers down to every decision the framework took: the totals by year, the pipeline
of next year and where it comes from, any breakdown by dimension, every forecast series with its
history, exam, technique and forecast, how the ladder improved the support, how the backtest chose
the techniques, how well every method did in the exam, the uplift and the price increases, the states
still to come, and the parameters and checks of the run.

HOW TO USE IT (one notebook cell):

    from sff_explorer import SFFExplorer
    explorer = SFFExplorer.from_results(results, configuration)     # results = run(configuration)
    explorer.show()

or, in a later session, without running the framework again (it reads the sff_* tables of the run):

    explorer = SFFExplorer.from_sql(configuration)
    explorer.show()

As one HTML file to open in a browser or send (no notebook needed to read it):

    explorer.export_html("salida/sff_explorador.html", top_series=10)

Without widgets (or to put one page in a report), every page can be printed statically:

    explorer.show_page("negocio")            # negocio, series, ficha, escalera, backtest, examen, uplift, estados, parametros
    explorer.show_page("ficha", fs_id="EU|A|0|0|0|0|web")

Requirements: pandas, plotly and ipywidgets (all in Anaconda). If the charts do not appear inside the
tabs (some notebook front-ends), use figure_mode="html": SFFExplorer.from_results(..., figure_mode="html").

Design:
  · the explorer reads tables, it never recomputes the forecast: every number is the one the run wrote;
  · a table is loaded the first time a page needs it, and kept (from SQL, the core alone may take a minute);
  · every period is shown as the text YYYY-MM, whether it came from memory (pd.Period) or from SQL (text);
  · every page is a function that returns a list of items (titles, figures, tables); the widgets only
    choose the arguments and show the items, so the same pages work in a static report and in the tests.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import html

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError as error:                                   # the explorer cannot draw without plotly
    raise ImportError("sff_explorer needs plotly: pip install plotly") from error


# ─── where every table lives: in the results of run(), and in SQL ─────────────────
# logical name → (path inside the results dict, name of the table in SQL without the prefix)
TABLE_SOURCES = {
    "core":                     (("core",), "nucleo"),
    "series":                   (("forecast_series",), "forecast_series"),
    "legend":                   (("core_legend",), "nucleo_leyenda"),
    "pipeline_source":          (("power_bi", "pipeline_source"), "forecast_pipeline_source"),
    "business_summary":         (("forecast", "by_year"), "resumen_negocio"),
    "forecast_month":           (("forecast", "by_month"), "forecast_mes"),
    "forecast_total":           (("forecast_total",), "forecast_total"),
    "maturation":               (("forecast", "maturation"), "forecast_maduracion"),
    "exam_detail":              (("series_exam", "detail"), "series_exam_detail"),
    "exam_per_series":          (("series_exam", "per_series"), "series_exam"),
    "portfolio_exam":           (("series_exam", "by_month"), "examen_cartera"),
    "portfolio_exam_summary":   (("series_exam", "summary"), "examen_cartera_resumen"),
    "ladder_steps":             (("ladder", "steps"), "ladder_steps"),
    "ladder_summary":           (("ladder", "summary"), "ladder_summary"),
    "ladder_merges":            (("ladder", "merges"), "ladder_merges"),
    "composition_members":      (("ladder", "composition_members"), "composition_members"),
    "composition":              (("audit", "composition"), "composition"),
    "composition_techniques":   (("audit", "composition_techniques"), "composition_techniques"),
    "composition_forecast_all": (("audit", "composition_forecast_all"), "composition_forecast_all"),
    "credibility":              (("audit", "credibility"), "credibility"),
    "credibility_members":      (("audit", "credibility_members"), "credibility_members"),
    "series_dynamics":          (("audit", "series_dynamics"), "series_dynamics"),
    "pool_series":              (("pool_series",), "pool_serie"),
    "pool_reference":           (("pool_reference",), "pool_referencia"),
    "money_by_level":           (("money_by_level",), "niveles_riesgo"),
    "dimension_decision":       (("dimension_decision",), "decision_eta2"),
    "backtest_predictions":     (("backtest", "predictions"), "backtest_predicciones"),
    "technique_decision":       (("backtest", "decision"), "decision_tecnica"),
    "error_bands":              (("backtest", "bands"), "decision_bandas"),
    "exam_by_pool":             (("backtest", "exam_by_pool"), "backtest_examen"),
    "exam_total":               (("backtest", "exam_total"), "backtest_examen_total"),
    "uplift_cells":             (("uplift_cells",), "uplift_celda"),
    "uplift_backtest":          (("uplift_backtest",), "backtest_uplift"),
    "contract_check":           (("contract_check",), "uplift_contrato_check"),
    "uplift_homogeneity":       (("uplift_homogeneity",), "uplift_homogeneidad"),
    "price_monitor":            (("power_bi", "price_monitor"), "price_increase_monitor"),
    "price_calendar":           (("power_bi", "price_calendar"), "price_increase_calendar"),
    "price_increases":          (("price_increases",), None),          # only in memory (step 3.1 → step 7.2)
    "parameters":               (("power_bi", "parameters"), "parametros"),
    "validation":               (("validation",), "validacion"),
}

# the columns of the core the explorer reads (from SQL only these are loaded, besides the dimensions of the Config:
# the core has about a hundred columns and a million rows)
CORE_COLUMNS = ["period", "s1_4_fs_id", "forecast_status", "forecast_universe", "forecast_pipeline_source", "forecast_year",
                "forecast_to_renew_units", "forecast_to_renew_USD", "forecast_renewed_units", "forecast_renewed_USD",
                "forecast_renewed_USD_low", "forecast_renewed_USD_high", "forecast_renewed_USD_maturation_adj",
                "forecast_to_renew_units_at_due", "forecast_to_renew_USD_at_due", "forecast_maturation_units",
                "s7_2_vencen_unidades", "s7_2_vencen_usd", "s7_2_vencen_unidades_al_vencer", "s7_2_tasa", "s7_2_tasa_baja",
                "s7_2_tasa_alta", "s7_2_tecnica", "s7_2_confidence", "s3_1_uplift", "origen_fila"]

# the columns that hold a month: shown as the text YYYY-MM
PERIOD_COLUMNS = ["period", "mes", "mes_objetivo", "ultimo_mes_visto", "inicio", "fin_ciclo", "primer_mes", "ultimo_mes"]

# the pages, in order: (key, tab title)
PAGES = [("negocio", "1 · Negocio"), ("series", "2 · Forecast series"), ("ficha", "3 · Ficha de serie"),
         ("escalera", "4 · Escalera"), ("backtest", "5 · Backtest y técnica"), ("examen", "6 · Examen"),
         ("uplift", "7 · Uplift y precios"), ("estados", "8 · Estados del cliente"), ("parametros", "9 · Parámetros y validación")]

# colours: a validated categorical order, and the three states of the confidence (with their label, never alone)
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, BAND = "#14171c", "#78808b", "rgba(42, 120, 214, 0.18)"
CONFIDENCE_COLOURS = {"high": "#0ca30c", "medium": "#fab219", "low": "#d03b3b"}
CONFIDENCE_LABELS = {"high": "Alta", "medium": "Media", "low": "Baja"}
METHOD_LABELS = {"baseline": "Baseline (su celda, últimos meses)", "raw": "Serie sola (raw)",
                 "por_defecto": "Escalera + técnica por defecto", "framework": "Framework"}
METHOD_COLOURS = {"framework": SERIES_COLOURS[0], "raw": SERIES_COLOURS[6], "baseline": SERIES_COLOURS[1],
                  "por_defecto": SERIES_COLOURS[2]}
TECHNIQUE_LABELS = {"T15_level_seasonal": "Nivel + efecto del mes", "T11_holt_winters": "Holt-Winters",
                    "T2_mean": "Media de toda la historia", "T9_ses": "Suavizado exponencial simple",
                    "T4_ewma": "Media ponderada exponencial", "T12_theta": "Theta", "T10_holt_damped": "Holt amortiguado",
                    "T14_temporal_cred": "Credibilidad temporal", "T3_ma3": "Media de 3 meses (técnica por defecto)",
                    "T3_ma6": "Media de 6 meses"}
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
TABLE_ROWS_SHOWN = 25          # rows of a table on screen (the rest: filter, or export the table)
SERIES_OPTIONS_SHOWN = 400     # series in the selector of the card (by money); any other can be typed


# ═══════════════════════════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════════

def money(value) -> str:
    """$1.234.567 (Spanish grouping); — when empty."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return "$" + f"{value:,.0f}".replace(",", ".")


def millions(value) -> str:
    """$136,9M for the big numbers; $ and thousands below a million."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if abs(value) >= 1e6:
        return "$" + f"{value / 1e6:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") + "M"
    return money(value)


def integer(value) -> str:
    """1.234 (Spanish grouping); — when empty."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:,.0f}".replace(",", ".")


def percent(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{100 * value:.{digits}f}".replace(".", ",") + " %"


def points(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",") + " pp"


def as_period_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Every month column as the text YYYY-MM (pd.Period from memory, text from SQL)."""
    frame = frame.copy()
    for column_name in PERIOD_COLUMNS:
        if column_name in frame.columns:
            frame[column_name] = frame[column_name].map(lambda value: None if value is None or (isinstance(value, float)
                                                                                                and np.isnan(value))
                                                         else str(value)[:7])
    return frame


# ═══════════════════════════════════════════════════════════════════════════════════
# THE ITEMS OF A PAGE (a page is a list of them: the widgets and the static report show the same)
# ═══════════════════════════════════════════════════════════════════════════════════

class Title:
    """A heading of a page, with an optional line under it that says how to read what follows."""
    def __init__(self, text: str, note: str = "", level: int = 3):
        self.text, self.note, self.level = text, note, level

    def to_html(self) -> str:
        note = f"<div style='color:{MUTED};font-size:13px;margin-top:2px;max-width:900px'>{self.note}</div>" if self.note else ""
        return f"<h{self.level} style='margin:18px 0 4px 0;font-family:sans-serif'>{html.escape(self.text)}</h{self.level}>{note}"


class Note:
    """A paragraph: an answer in words, or a warning."""
    def __init__(self, text: str, kind: str = "answer"):
        self.text, self.kind = text, kind

    def to_html(self) -> str:
        border = {"answer": SERIES_COLOURS[0], "warning": CONFIDENCE_COLOURS["medium"], "info": MUTED}[self.kind]
        return (f"<div style='border-left:3px solid {border};padding:8px 12px;margin:8px 0;max-width:980px;"
                f"font-family:sans-serif;font-size:14px;line-height:1.5;background:rgba(127,127,127,0.06)'>{self.text}</div>")


class Cards:
    """A row of key figures: (label, value, sub-line)."""
    def __init__(self, cards: list):
        self.cards = cards

    def to_html(self) -> str:
        cells = "".join(
            f"<div style='border:1px solid rgba(127,127,127,0.3);border-radius:8px;padding:10px 12px;min-width:150px;flex:1'>"
            f"<div style='font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:{MUTED}'>{html.escape(label)}</div>"
            f"<div style='font-size:22px;font-weight:600;margin:2px 0'>{value}</div>"
            f"<div style='font-size:12px;color:{MUTED}'>{sub}</div></div>"
            for label, value, sub in self.cards)
        return f"<div style='display:flex;flex-wrap:wrap;gap:10px;margin:8px 0;font-family:sans-serif'>{cells}</div>"


class Table:
    """A table on screen: the first rows, already formatted; how many there are in total."""
    def __init__(self, frame: pd.DataFrame, caption: str = "", rows: int = TABLE_ROWS_SHOWN, formats: dict = None):
        self.frame, self.caption, self.rows, self.formats = frame, caption, rows, formats or {}

    def to_html(self) -> str:
        if self.frame is None or self.frame.empty:
            return f"<div style='color:{MUTED};font-family:sans-serif;margin:6px 0'>(sin filas)</div>"
        shown = self.frame.head(self.rows).copy()
        for column_name, formatter in self.formats.items():
            if column_name in shown.columns:
                shown[column_name] = shown[column_name].map(formatter)
        more = (f"<div style='color:{MUTED};font-size:12px'>{len(self.frame):,} filas; se muestran {len(shown)}</div>"
                if len(self.frame) > len(shown) else "")
        caption = f"<div style='font-size:13px;color:{MUTED};margin:4px 0'>{self.caption}</div>" if self.caption else ""
        table_html = shown.to_html(index=False, escape=True, na_rep="—", border=0)
        style = ("<style>.sffx table{border-collapse:collapse;font-family:sans-serif;font-size:12.5px}"
                 ".sffx th{text-align:left;padding:5px 9px;border-bottom:2px solid rgba(127,127,127,0.35);font-size:11px;"
                 "text-transform:uppercase;letter-spacing:0.04em}"
                 ".sffx td{padding:4px 9px;border-bottom:1px solid rgba(127,127,127,0.18);white-space:nowrap}</style>")
        return f"{style}{caption}<div class='sffx' style='overflow-x:auto;max-width:100%'>{table_html}</div>{more}"


def layout(figure: go.Figure, title: str = "", height: int = 380, y_title: str = "", y_format: str = None) -> go.Figure:
    """The common look of every chart: white template, one y axis, the legend on top, hover on the x."""
    figure.update_layout(template="plotly_white", title=dict(text=title, x=0, font=dict(size=15)), height=height + 50,
                         margin=dict(l=60, r=20, t=50 if title else 20, b=40),
                         legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0),
                         font=dict(family="sans-serif", size=12))
    # a few months as text (YYYY-MM) read as categories: as dates plotly would put weekly ticks between them
    months = {str(value) for trace in figure.data if getattr(trace, "x", None) is not None and trace.orientation != "h"
              for value in trace.x if value is not None}
    if months and len(months) <= 24 and all(len(value) == 7 and value[4] == "-" for value in months):
        figure.update_xaxes(type="category", categoryorder="category ascending")
    if figure.layout.hovermode is None:                     # a chart that set its own hover keeps it
        figure.update_layout(hovermode="x unified")
    figure.update_yaxes(title_text=y_title, gridcolor="rgba(127,127,127,0.18)", zeroline=False)
    figure.update_xaxes(gridcolor="rgba(127,127,127,0.10)")
    if y_format:
        figure.update_yaxes(tickformat=y_format)
    return figure


# ═══════════════════════════════════════════════════════════════════════════════════
# THE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════════

class SFFExplorer:
    """The tables of one run and the pages that read them. See the module docstring."""

    def __init__(self, configuration, results: dict = None, engine=None, figure_mode: str = "show"):
        self.configuration = configuration
        self.results = results
        self.engine = engine
        self.figure_mode = figure_mode
        self.cache = {}
        self.page_errors = []          # (page, error) of every page the widget could not build

    # ─── the two ways in ───────────────────────────────────────────────────────
    @classmethod
    def from_results(cls, results: dict, configuration, figure_mode: str = "show") -> "SFFExplorer":
        """From the dict that run(configuration) returns: everything in memory, nothing read again."""
        return cls(configuration, results=results, figure_mode=figure_mode)

    @classmethod
    def from_sql(cls, configuration, figure_mode: str = "show") -> "SFFExplorer":
        """From the sff_* tables the run wrote (configuration.sql_engine, table_prefix, sql_schema)."""
        return cls(configuration, engine=configuration.sql_engine, figure_mode=figure_mode)

    # ─── the tables ────────────────────────────────────────────────────────────
    def table(self, name: str) -> pd.DataFrame:
        """A table of the run by its logical name (TABLE_SOURCES), loaded once; empty when the run has none."""
        if name in self.cache:
            return self.cache[name]
        path, sql_name = TABLE_SOURCES[name]
        frame = None
        if self.results is not None:
            value = self.results
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            frame = value if isinstance(value, pd.DataFrame) else None
        elif self.engine is not None and sql_name:
            frame = self.read_sql(sql_name, self.core_columns() if name == "core" else None)
        frame = as_period_text(frame) if frame is not None else pd.DataFrame()
        self.cache[name] = frame
        return frame

    def core_columns(self) -> list:
        """The columns of the core the pages read: CORE_COLUMNS and the dimensions a total can be broken down by."""
        configuration = self.configuration
        dimensions = (list(configuration.business_mandatory_dims) + [f"{name}_level_1" for name in configuration.leveled_dims]
                      + list(configuration.extra_renovacion) + list(configuration.extra_revalorizacion)
                      + list(configuration.structural_timevarying_dims))
        return list(dict.fromkeys(CORE_COLUMNS + dimensions))

    def read_sql(self, table_name: str, columns: list = None):
        """One sff_ table from SQL (only `columns`, those it has, when given); None when it does not exist."""
        import sqlalchemy
        physical = f"{getattr(self.configuration, 'table_prefix', 'sff_')}{table_name}"
        schema = getattr(self.configuration, "sql_schema", None)
        try:
            if columns is not None:
                existing = {column["name"] for column in sqlalchemy.inspect(self.engine).get_columns(physical, schema=schema)}
                columns = [column_name for column_name in columns if column_name in existing]
            return pd.read_sql_table(physical, self.engine, schema=schema, columns=columns)
        except Exception:                                          # a table this run did not write
            return None

    def has(self, name: str) -> bool:
        return not self.table(name).empty

    # ─── what several pages share ──────────────────────────────────────────────
    def pipeline_core(self) -> pd.DataFrame:
        """The rows of the core with a forecast series (the pipeline universe)."""
        core = self.table("core")
        if "forecast_universe" in core.columns:
            return core[core["forecast_universe"] == "pipeline"]
        return core

    def source_labels(self) -> dict:
        """forecast_pipeline_source → (business label, order)."""
        sources = self.table("pipeline_source")
        if sources.empty:
            return {}
        return {row.forecast_pipeline_source: (row.source_label, int(row.source_order)) for row in sources.itertuples()}

    def source_colour(self, source: str) -> str:
        """The colour of an origin of the pipeline: by its order in the report, the same in every chart."""
        order = self.source_labels().get(source, (source, 0))[1]
        return SERIES_COLOURS[(order - 1) % len(SERIES_COLOURS)] if order else MUTED

    def series_by_money(self) -> pd.DataFrame:
        series = self.table("series")
        return series.sort_values("s7_2_esperado_usd_total", ascending=False, na_position="last")

    def breakdown_dimensions(self) -> list:
        """The dimensions a total can be broken down by: those of the Config that the core carries."""
        configuration, core = self.configuration, self.table("core")
        candidates = (list(configuration.business_mandatory_dims) + [f"{name}_level_1" for name in configuration.leveled_dims]
                      + list(configuration.extra_renovacion) + list(configuration.extra_revalorizacion)
                      + list(configuration.structural_timevarying_dims) + ["forecast_pipeline_source", "s7_2_confidence"])
        return [name for name in dict.fromkeys(candidates) if name in core.columns]

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1 · THE BUSINESS
    # ═══════════════════════════════════════════════════════════════════════════
    def page_business(self, dimension: str = None, year: int = None) -> list:
        """How 2026 ends, what pipeline 2027 has, how 2027 ends; month by month; any breakdown."""
        items = [Title("Las tres preguntas de negocio", "Renovado real de los meses cerrados + esperado de los futuros. "
                       "Banda lineal: todos los errores en la misma dirección (el peor caso); cuadratura: errores "
                       "independientes. La verdad está entre las dos.", level=2)]
        summary = self.table("business_summary")
        core = self.table("core")
        years = sorted(int(value) for value in core["forecast_year"].dropna().unique()) if "forecast_year" in core.columns else []
        future_years = [value for value in years if (core[core["forecast_year"] == value]["forecast_status"] == "forecast").any()]
        if not summary.empty and future_years:
            cards = []
            for value in future_years[:2]:
                row = summary[summary["ano"] == value]
                if row.empty:
                    continue
                row = row.iloc[0]
                time_series_part = sum(float(row.get(column_name, 0.0)) for column_name in
                                       ("ingresos_ts_real_usd", "ingresos_ts_usd", "esperado_ts_reentrada_usd"))
                band = row.get("error_examen_usd", row["banda_cuadratura_usd"])
                cards.append((f"Total {value}", millions(row["total_usd"]),
                              f"real {millions(row['renovado_real_usd'])} + esperado {millions(row['esperado_usd'])}"
                              + (f" + time_series {millions(time_series_part)}" if time_series_part else "")
                              + f" · ± {millions(band)} ({'examen' if 'error_examen_usd' in row else 'cuadratura'})"))
            if len(future_years) > 1:
                next_year = future_years[1]
                due = self.pipeline_core()
                due = due[(due["forecast_year"] == next_year)]
                labels = self.source_labels()
                by_source = due.groupby("forecast_pipeline_source")["forecast_to_renew_USD"].sum()
                parts = " · ".join(f"{labels.get(source, (source, 0))[0]} {millions(value)}" for source, value in by_source.items() if value)
                cards.append((f"Pipeline que vence en {next_year}", millions(by_source.sum()), parts))
            items.append(Cards(cards))
            shown = summary.copy()
            money_columns = [column for column in shown.columns if column != "ano"]
            items.append(Table(shown, "Por año (pipeline; paso 7.2 · sff_resumen_negocio)",
                               formats={column: millions for column in money_columns}))
        total = self.table("forecast_total")
        if not total.empty:
            items.append(Table(total, "Por año y origen, con el universo time_series (paso 7.3 · sff_forecast_total)",
                               formats={"usd_vence": millions, "usd_renovado": millions,
                                        "unidades_vencen": lambda value: f"{value:,.0f}".replace(",", "."),
                                        "unidades_renovadas": lambda value: f"{value:,.0f}".replace(",", ".")}))
        items.append(self.figure_business_months())
        items.append(self.figure_business_rates())
        items.extend(self.acquisition_check())
        if dimension:
            items.extend(self.business_breakdown(dimension, year or (future_years[0] if future_years else None)))
        return items

    def figure_business_months(self) -> go.Figure:
        """Renewed USD month by month, by origin of the pipeline, with the quadrature band of the forecast."""
        core = self.table("core")
        labels = self.source_labels()
        frame = core.groupby(["period", "forecast_pipeline_source"])["forecast_renewed_USD"].sum().reset_index()
        months = sorted(frame["period"].unique())
        recent = [month for month in months if month >= f"{int(min(core['forecast_year'].max(), 9999)) - 2}-01"]
        frame = frame[frame["period"].isin(recent)]
        figure = go.Figure()
        order = sorted(frame["forecast_pipeline_source"].unique(), key=lambda source: labels.get(source, (source, 99))[1])
        for source in order:
            part = frame[frame["forecast_pipeline_source"] == source]
            if part["forecast_renewed_USD"].abs().sum() == 0:
                continue
            figure.add_bar(x=part["period"], y=part["forecast_renewed_USD"], name=labels.get(source, (source, 0))[0],
                           marker_color=self.source_colour(source))
        months_table = self.table("forecast_month")
        if not months_table.empty:
            # every month between the first and the last: a month without forecast breaks the band instead of joining it
            all_months = [str(month) for month in pd.period_range(months_table["period"].min(), months_table["period"].max(), freq="M")]
            months_table = months_table.set_index("period").reindex(all_months).reset_index().rename(columns={"index": "period"})
            figure.add_scatter(x=months_table["period"], y=months_table["banda_cuadratura_alta"], mode="lines", connectgaps=False,
                               line=dict(color=INK, width=1, dash="dot"), name="Banda (cuadratura)")
            figure.add_scatter(x=months_table["period"], y=months_table["banda_cuadratura_baja"], mode="lines", connectgaps=False,
                               line=dict(color=INK, width=1, dash="dot"), showlegend=False)
        figure.update_layout(barmode="stack")
        return layout(figure, "Renovado por mes: real y esperado, por origen de la pipeline", 420, "USD")

    def figure_business_rates(self) -> go.Figure:
        """The renewal rate in units and in USD, month by month: real in the closed months, expected after."""
        core = self.pipeline_core()
        grouped = core.groupby(["period", "forecast_status"])[["forecast_to_renew_units", "forecast_renewed_units",
                                                                "forecast_to_renew_USD", "forecast_renewed_USD"]].sum().reset_index()
        grouped = grouped[grouped["forecast_to_renew_units"] > 0]
        grouped["tasa_unidades"] = grouped["forecast_renewed_units"] / grouped["forecast_to_renew_units"]
        grouped["tasa_usd"] = grouped["forecast_renewed_USD"] / grouped["forecast_to_renew_USD"]
        figure = go.Figure()
        for status, dash, label in (("actual", "solid", "real"), ("forecast", "dot", "esperada")):
            part = grouped[grouped["forecast_status"] == status]
            figure.add_scatter(x=part["period"], y=part["tasa_unidades"], name=f"Tasa en unidades ({label})",
                               line=dict(color=SERIES_COLOURS[0], dash=dash, width=2))
            figure.add_scatter(x=part["period"], y=part["tasa_usd"], name=f"Tasa en USD ({label})",
                               line=dict(color=SERIES_COLOURS[1], dash=dash, width=2))
        return layout(figure, "Tasa de renovación de la pipeline: unidades y USD", 360, "", ".0%")

    def acquisition_check(self) -> list:
        """The simulated acquisitions against what was sold in the same months a year before: an acquisition sold in m
        is the 1-year acquisition pipeline due in m + 12. The ratio should be the recent growth, not several times it."""
        configuration, core = self.configuration, self.pipeline_core()
        acquisition, term = configuration.acquisition_column, configuration.term_column
        window = configuration.simulation_window
        if not acquisition or not window or acquisition not in core.columns:
            return []
        acquisitions = core[core[acquisition].astype(str).isin([str(value) for value in configuration.acquisition_values])]
        if term and term in acquisitions.columns:
            acquisitions = acquisitions[acquisitions[term].astype(str) == str(configuration.one_year_term_value)]
        rows = []
        for value in configuration.acquisition_values:
            of_value = acquisitions[acquisitions[acquisition].astype(str) == str(value)]
            for month in window:
                sold_before = of_value[(of_value["period"] == str(month)) & (of_value["forecast_pipeline_source"] != "pipeline_simulada")]
                simulated = of_value[(of_value["period"] == str(month + 12)) & (of_value["forecast_pipeline_source"] == "pipeline_simulada")]
                before_units = sold_before["forecast_to_renew_units"].sum()
                rows.append({acquisition: value, "mes_venta": str(month), "unidades_ano_anterior": before_units,
                             "usd_ano_anterior": sold_before["forecast_to_renew_USD"].sum(),
                             "unidades_simuladas": simulated["forecast_to_renew_units"].sum(),
                             "usd_simulado": simulated["forecast_to_renew_USD"].sum(),
                             "ratio_unidades": simulated["forecast_to_renew_units"].sum() / before_units if before_units else np.nan})
        table = pd.DataFrame(rows)
        return [Title("Adquisición simulada frente a lo vendido un año antes",
                      "Una venta de m vence en m + 12. Lo simulado debería ser lo vendido un año antes por el crecimiento "
                      f"reciente (aviso fuera de {configuration.acquisition_ratio_low:g}–{configuration.acquisition_ratio_high:g}×)."),
                Table(table, formats={"unidades_ano_anterior": integer, "unidades_simuladas": integer, "usd_ano_anterior": money,
                                      "usd_simulado": money, "ratio_unidades": lambda value: "—" if pd.isna(value) else f"{value:.2f}×".replace(".", ",")})]

    def business_breakdown(self, dimension: str, year: int) -> list:
        """One dimension of one year: what falls due, what renews or is expected, the rates and the band."""
        core = self.pipeline_core()
        core = core[core["forecast_year"] == year]
        if core.empty:
            return [Note(f"No hay pipeline en {year}.", "info")]
        grouped = core.groupby(dimension).agg(vence_usd=("forecast_to_renew_USD", "sum"),
                                              renovado_usd=("forecast_renewed_USD", "sum"),
                                              banda_baja=("forecast_renewed_USD_low", "sum"),
                                              banda_alta=("forecast_renewed_USD_high", "sum"),
                                              vence_unidades=("forecast_to_renew_units", "sum"),
                                              renovadas_unidades=("forecast_renewed_units", "sum")).reset_index()
        real = core[core["forecast_status"] == "actual"].groupby(dimension)["forecast_renewed_USD"].sum()
        grouped["de_ello_real_usd"] = grouped[dimension].map(real).fillna(0.0)
        grouped["tasa_usd"] = grouped["renovado_usd"] / grouped["vence_usd"].where(grouped["vence_usd"] > 0)
        grouped["tasa_unidades"] = grouped["renovadas_unidades"] / grouped["vence_unidades"].where(grouped["vence_unidades"] > 0)
        grouped["peso"] = grouped["renovado_usd"] / grouped["renovado_usd"].sum()
        grouped = grouped.sort_values("renovado_usd", ascending=False)
        top = grouped.head(20)
        figure = go.Figure()
        figure.add_bar(x=top[dimension].astype(str), y=top["vence_usd"], name="Vence", marker_color="rgba(120,128,139,0.35)")
        figure.add_bar(x=top[dimension].astype(str), y=top["renovado_usd"], name="Renovado + esperado",
                       marker_color=SERIES_COLOURS[0],
                       error_y=dict(type="data", symmetric=False, array=top["banda_alta"] - top["renovado_usd"],
                                    arrayminus=top["renovado_usd"] - top["banda_baja"], color=INK, thickness=1))
        figure.update_layout(barmode="group")
        return [Title(f"{year} por {dimension}", "Las 20 de más dinero. La banda es la lineal (suma de las de sus filas)."),
                layout(figure, "", 380, "USD"),
                Table(grouped[[dimension, "vence_usd", "renovado_usd", "de_ello_real_usd", "banda_baja", "banda_alta",
                               "tasa_usd", "tasa_unidades", "peso"]], rows=40,
                      formats={"vence_usd": money, "renovado_usd": money, "de_ello_real_usd": money, "banda_baja": money,
                               "banda_alta": money, "tasa_usd": percent, "tasa_unidades": percent, "peso": percent})]

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2 · THE FORECAST SERIES
    # ═══════════════════════════════════════════════════════════════════════════
    def series_table(self, confidence: list = None, risk_levels: list = None, text: str = "") -> pd.DataFrame:
        """The forecast series, by money, filtered by confidence, risk level and a piece of their id."""
        series = self.series_by_money()
        if confidence:
            series = series[series["s7_2_confidence"].isin(confidence)]
        if risk_levels:
            series = series[series["s2_3_nivel_riesgo"].isin(risk_levels)]
        if text:
            series = series[series["s1_4_fs_id"].astype(str).str.contains(text, case=False, regex=False)]
        total = self.table("series")["s7_2_esperado_usd_total"].sum()
        shown = pd.DataFrame({
            "forecast_serie": series["s1_4_fs_id"], "esperado_usd": series["s7_2_esperado_usd_total"],
            "peso": series["s7_2_esperado_usd_total"] / total if total else np.nan,
            "confianza": series["s7_2_confidence"].map(CONFIDENCE_LABELS), "vencen_al_mes": series["s1_7_n_propio"],
            "nivel_riesgo": series["s2_3_nivel_riesgo"], "soporte_composicion": series["s2_2_stage3_support"],
            "tecnica": series["s2_6_tecnica_corto"].map(lambda value: TECHNIQUE_LABELS.get(value, value)),
            "error_examen_pp": series["s7_1_exam_mae_pp"], "dentro_intervalo": series["s7_1_exam_coverage"],
            "error_sobre_ruido": series["s7_1_exam_error_over_noise"], "mejora_sobre_raw_pp": series["s7_1_improvement_mae_pp"]})
        return shown.reset_index(drop=True)

    def page_series(self, confidence: list = None, risk_levels: list = None, text: str = "") -> list:
        series = self.table("series")
        shown = self.series_table(confidence, risk_levels, text)
        total = series["s7_2_esperado_usd_total"].sum()
        money_by_confidence = {level: float((series["s7_2_esperado_usd_total"] * series[f"s7_2_pct_usd_{level}"]).sum())
                               for level in ("high", "medium", "low")}
        coverage = series["s7_1_exam_in_band"].sum() / max(series["s7_1_exam_predictions"].sum(), 1)
        raw_coverage = series["s7_1_raw_in_band"].sum() / max(series["s7_1_raw_predictions"].sum(), 1)
        items = [Title("Las forecast series", "Cada número del forecast es de una forecast serie. Ordenadas por lo que "
                       "se espera renovar; filtra por confianza, nivel de riesgo o por un trozo de su id.", level=2),
                 Cards([("Forecast series", f"{len(series):,}".replace(",", "."), f"{len(shown):,} con el filtro".replace(",", ".")),
                        ("Esperado", millions(total), "suma de las series"),
                        ("Dinero con confianza alta", percent(money_by_confidence["high"] / total if total else np.nan, 0),
                         f"media {percent(money_by_confidence['medium'] / total if total else np.nan, 0)} · "
                         f"baja {percent(money_by_confidence['low'] / total if total else np.nan, 0)}"),
                        ("Dentro del intervalo (examen)", percent(coverage, 0), f"serie sola: {percent(raw_coverage, 0)}")]),
                 Table(shown, formats={"esperado_usd": money, "peso": lambda value: percent(value, 2),
                                       "error_examen_pp": points, "dentro_intervalo": lambda value: percent(value, 0),
                                       "error_sobre_ruido": lambda value: "—" if pd.isna(value) else f"{value:.2f}".replace(".", ","),
                                       "mejora_sobre_raw_pp": points, "vencen_al_mes": lambda value: f"{value:,.0f}".replace(",", "."),
                                       "soporte_composicion": lambda value: f"{value:,.0f}".replace(",", ".")})]
        # where the money is and how well it was predicted
        plotted = series[series["s7_2_esperado_usd_total"] > 0].copy()
        if text or confidence or risk_levels:
            plotted = plotted[plotted["s1_4_fs_id"].isin(shown["forecast_serie"])]
        figure = go.Figure()
        for level in ("high", "medium", "low"):
            part = plotted[plotted["s7_2_confidence"] == level]
            figure.add_scatter(x=part["s7_2_esperado_usd_total"], y=part["s7_1_exam_mae_pp"], mode="markers",
                               name=CONFIDENCE_LABELS[level], text=part["s1_4_fs_id"],
                               marker=dict(color=CONFIDENCE_COLOURS[level], size=8, line=dict(color="white", width=1)),
                               hovertemplate="%{text}<br>esperado $%{x:,.0f}<br>error examen %{y:.1f} pp<extra></extra>")
        figure.update_xaxes(type="log", title_text="Esperado USD (escala log)")
        figure.update_layout(hovermode="closest")
        items.append(layout(figure, "Dónde está el dinero y cuánto se equivocó en el examen", 420, "Error medio del examen (pp)"))
        by_level = (series.groupby("s2_3_nivel_riesgo")
                    .agg(series=("s1_4_fs_id", "size"), esperado_usd=("s7_2_esperado_usd_total", "sum"),
                         en_intervalo=("s7_1_exam_in_band", "sum"), predicciones=("s7_1_exam_predictions", "sum"),
                         error_abs=("s7_1_exam_abs_err_units", "sum"), reales=("s7_1_exam_real_units", "sum")).reset_index())
        by_level["dentro_intervalo"] = by_level["en_intervalo"] / by_level["predicciones"].where(by_level["predicciones"] > 0)
        by_level["wape_examen"] = by_level["error_abs"] / by_level["reales"].where(by_level["reales"] > 0)
        by_level["peso"] = by_level["esperado_usd"] / total if total else np.nan
        items.append(Table(by_level[["s2_3_nivel_riesgo", "series", "esperado_usd", "peso", "dentro_intervalo", "wape_examen"]],
                           "Por nivel de riesgo: cómo predice cada serie (A propio · A3 reforzado · B prestado · C credibilidad · D sin historia)",
                           formats={"esperado_usd": money, "peso": percent, "dentro_intervalo": percent, "wape_examen": percent}))
        return items

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3 · THE CARD OF ONE SERIES (the seven questions)
    # ═══════════════════════════════════════════════════════════════════════════
    def page_card(self, fs_id: str = None, exam_horizon: int = 1) -> list:
        series = self.table("series")
        if fs_id is None:
            fs_id = self.series_by_money()["s1_4_fs_id"].iloc[0]
        found = series[series["s1_4_fs_id"] == fs_id]
        if found.empty:
            return [Note(f"La forecast serie «{html.escape(str(fs_id))}» no está en la dimensión.", "warning")]
        row = found.iloc[0]
        composition_id = row["s2_2_stage3_id"]
        total = series["s7_2_esperado_usd_total"].sum()
        core = self.pipeline_core()
        rows = core[core["s1_4_fs_id"] == fs_id]
        future = rows[rows["forecast_status"] == "forecast"]
        items = [Title(f"Forecast serie {fs_id}", f"Composición con la que predice: {composition_id}", level=2)]
        items.append(Cards([
            ("Esperado", money(row["s7_2_esperado_usd_total"]),
             f"{money(future['forecast_renewed_USD_low'].sum())} – {money(future['forecast_renewed_USD_high'].sum())}"),
            ("Peso en la cartera", percent(row["s7_2_esperado_usd_total"] / total if total else np.nan, 2), "del esperado de las series"),
            ("Vencen al mes", integer(row["s1_7_n_propio"]), f"tasa histórica {percent(row['s1_7_tasa_propia'])}"),
            ("Cómo predice", str(row["s2_3_nivel_riesgo"]), f"composición de {integer(row['s2_2_stage3_support'])} · error ±{points(row['s2_3_se_prediccion_pp'])}"),
            ("Error en el examen", points(row["s7_1_exam_mae_pp"]),
             f"ruido {points(row['s7_1_exam_noise_pp'])} · error/ruido {row['s7_1_exam_error_over_noise']:.2f}".replace(".", ",")
             if pd.notna(row["s7_1_exam_error_over_noise"]) else "sin examen"),
            ("Dentro del intervalo", f"{row['s7_1_exam_in_band']:.0f} / {row['s7_1_exam_predictions']:.0f}" if pd.notna(row["s7_1_exam_predictions"]) else "—",
             f"serie sola: {row['s7_1_raw_in_band']:.0f} / {row['s7_1_raw_predictions']:.0f}" if pd.notna(row["s7_1_raw_predictions"]) else ""),
            ("Confianza", CONFIDENCE_LABELS.get(row["s7_2_confidence"], row["s7_2_confidence"]),
             f"alta {percent(row['s7_2_pct_usd_high'], 0)} · media {percent(row['s7_2_pct_usd_medium'], 0)} · baja {percent(row['s7_2_pct_usd_low'], 0)}")]))
        items.append(Note(self.card_story(row, rows, future)))
        items.append(self.figure_card_history(fs_id, rows))
        items.append(self.figure_card_exam(fs_id, exam_horizon))
        items.extend(self.card_techniques(composition_id))
        items.append(self.figure_card_money(future))
        items.extend(self.card_ladder(fs_id, composition_id))
        items.append(Title("Las filas futuras", "Una fila por registro del núcleo: origen, técnica, unidades hoy y al vencer (la "
                                                "maduración de los estados del cliente), tasa y su banda, uplift, esperado y ajuste por estados."))
        detail = future[["period", "origen_fila", "forecast_pipeline_source", "s7_2_tecnica", "s7_2_vencen_unidades",
                         "s7_2_vencen_unidades_al_vencer", "forecast_to_renew_USD",
                         "s7_2_tasa", "s7_2_tasa_baja", "s7_2_tasa_alta", "s3_1_uplift", "forecast_renewed_USD",
                         "forecast_renewed_USD_low", "forecast_renewed_USD_high", "forecast_renewed_USD_maturation_adj",
                         "s7_2_confidence"]].sort_values("period")
        items.append(Table(detail, rows=60, formats={"forecast_to_renew_USD": money, "forecast_renewed_USD": money,
                                                     "forecast_renewed_USD_low": money, "forecast_renewed_USD_high": money,
                                                     "forecast_renewed_USD_maturation_adj": money, "s7_2_tasa": percent,
                                                     "s7_2_tasa_baja": percent, "s7_2_tasa_alta": percent,
                                                     "s3_1_uplift": lambda value: f"{value:.3f}".replace(".", ","),
                                                     "s7_2_vencen_unidades": lambda value: f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."),
                                                     "s7_2_vencen_unidades_al_vencer": lambda value: f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")}))
        return items

    def card_story(self, row, rows: pd.DataFrame, future: pd.DataFrame) -> str:
        """The seven answers in words, with the numbers of this series."""
        history = rows[rows["forecast_status"] == "actual"]
        due = history["forecast_to_renew_units"].sum()
        rate = history["forecast_renewed_units"].sum() / due if due else np.nan
        technique = TECHNIQUE_LABELS.get(row["s2_6_tecnica_corto"], row["s2_6_tecnica_corto"])
        lines = [f"<b>1 · Qué es.</b> Se esperan {money(row['s7_2_esperado_usd_total'])}; le vencen "
                 f"{integer(row['s1_7_n_propio'])} licencias al mes.",
                 f"<b>2 · Cómo se ha comportado.</b> Renueva el {percent(rate)} de las unidades"
                 + (f"; su variación es {row['s2_5_series_phi']:.1f} veces la del azar (φ) y su patrón anual de "
                    f"{points(row['s2_5_series_amplitude_pp'])}".replace(".", ",") if pd.notna(row.get("s2_5_series_phi")) else "") + ".",
                 f"<b>3 · Con qué datos predice.</b> Nivel {row['s2_3_nivel_riesgo']}: "
                 + ("predice con su propia historia." if str(row["s2_3_nivel_riesgo"]).startswith("A")
                    else f"se apoya en su composición ({integer(row['s2_2_stage3_support'])} licencias al mes)."),
                 f"<b>4 · Con qué técnica.</b> {technique}: error de {points(row['s2_6_examen_err_pp_corto'])} a un mes en el examen, "
                 f"frente a {points(row['s2_6_examen_por_defecto_err_pp_corto'])} de la media de 3 meses.",
                 f"<b>5 · Cuánto acertó.</b> {row['s7_1_exam_in_band']:.0f} de {row['s7_1_exam_predictions']:.0f} predicciones dentro del "
                 f"intervalo; error medio {points(row['s7_1_exam_mae_pp'])} frente a un ruido de {points(row['s7_1_exam_noise_pp'])}."
                 if pd.notna(row["s7_1_exam_predictions"]) else "<b>5 · Cuánto acertó.</b> No tiene meses de examen.",
                 f"<b>6 · Qué predice.</b> {money(future['forecast_renewed_USD'].sum())} de {money(future['forecast_to_renew_USD_at_due'].sum())} "
                 f"que le vencerán (hoy {money(future['forecast_to_renew_USD'].sum())}: la diferencia es la maduración de los estados del cliente), entre {money(future['forecast_renewed_USD_low'].sum())} y {money(future['forecast_renewed_USD_high'].sum())}.",
                 f"<b>7 · Confianza.</b> {CONFIDENCE_LABELS.get(row['s7_2_confidence'], row['s7_2_confidence'])} "
                 f"(alta en el {percent(row['s7_2_pct_usd_high'], 0)} de su dinero)."]
        return "<br>".join(lines)

    def figure_card_history(self, fs_id: str, rows: pd.DataFrame) -> go.Figure:
        """Real rate month by month, the exam predictions (1 month before) with their interval, the forecast with its band."""
        figure = go.Figure()
        history = rows[rows["forecast_status"] == "actual"].groupby("period")[["forecast_to_renew_units", "forecast_renewed_units"]].sum()
        history = history[history["forecast_to_renew_units"] > 0]
        figure.add_scatter(x=history.index, y=history["forecast_renewed_units"] / history["forecast_to_renew_units"],
                           name="Real", line=dict(color=INK, width=2))
        future = rows[rows["forecast_status"] == "forecast"].copy()
        if not future.empty:
            future["_baja"] = future["s7_2_vencen_unidades_al_vencer"] * future["s7_2_tasa_baja"]
            future["_alta"] = future["s7_2_vencen_unidades_al_vencer"] * future["s7_2_tasa_alta"]
            monthly = future.groupby(["period", "forecast_pipeline_source"])[["s7_2_vencen_unidades_al_vencer", "forecast_renewed_units",
                                                                              "_baja", "_alta"]].sum().reset_index()
            labels = self.source_labels()
            for index, (source, part) in enumerate(monthly.groupby("forecast_pipeline_source")):
                part = part.sort_values("period")
                units = part["s7_2_vencen_unidades_al_vencer"].where(part["s7_2_vencen_unidades_al_vencer"] > 0)
                figure.add_scatter(x=part["period"], y=part["_alta"] / units, mode="lines", line=dict(width=0),
                                   showlegend=False, hoverinfo="skip")
                figure.add_scatter(x=part["period"], y=part["_baja"] / units, mode="lines", line=dict(width=0),
                                   fill="tonexty", fillcolor=BAND, name=f"Banda · {labels.get(source, (source, 0))[0]}")
                figure.add_scatter(x=part["period"], y=part["forecast_renewed_units"] / units, mode="lines+markers",
                                   name=f"Forecast · {labels.get(source, (source, 0))[0]}",
                                   line=dict(color=self.source_colour(source), width=2))
        exam = self.table("exam_detail")
        exam = exam[(exam["fs_id"] == fs_id) & (exam["method"] == "framework") & (exam["h"] == 1)]
        if not exam.empty:
            figure.add_scatter(x=exam["period"], y=exam["pred_rate"], mode="markers", name="Examen (1 mes antes)",
                               marker=dict(color=SERIES_COLOURS[0], size=9, line=dict(color="white", width=1.5)),
                               error_y=dict(type="data", symmetric=False, array=exam["band_high"] - exam["pred_rate"],
                                            arrayminus=exam["pred_rate"] - exam["band_low"], color=SERIES_COLOURS[0]))
        return layout(figure, "Tasa de renovación: real, examen y forecast con su banda", 400, "", ".0%")

    def figure_card_exam(self, fs_id: str, horizon: int) -> go.Figure:
        """Every method of the exam against the real rate, for one horizon."""
        exam = self.table("exam_detail")
        exam = exam[(exam["fs_id"] == fs_id) & (exam["h"] == horizon)]
        figure = go.Figure()
        if exam.empty:
            return layout(figure, f"Sin examen a {horizon} meses", 200)
        real = exam.drop_duplicates("period").sort_values("period")
        figure.add_scatter(x=real["period"], y=real["real_rate"], name="Real", mode="lines+markers",
                           line=dict(color=INK, width=3), marker=dict(symbol="line-ew-open", size=18, color=INK))
        for method, part in exam.groupby("method"):
            part = part.sort_values("period")
            error = None
            if part["band_low"].notna().any():
                error = dict(type="data", symmetric=False, array=part["band_high"] - part["pred_rate"],
                             arrayminus=part["pred_rate"] - part["band_low"], color=METHOD_COLOURS.get(method, MUTED))
            figure.add_scatter(x=part["period"], y=part["pred_rate"], mode="markers", name=METHOD_LABELS.get(method, method),
                               marker=dict(color=METHOD_COLOURS.get(method, MUTED), size=10), error_y=error,
                               customdata=np.stack([part["err_pp"], part["noise_pp"]], axis=1),
                               hovertemplate="%{y:.1%} · error %{customdata[0]:.1f} pp · ruido %{customdata[1]:.1f} pp")
        figure.update_xaxes(type="category")
        return layout(figure, f"El examen a {horizon} mes{'es' if horizon > 1 else ''} vista: cada método frente a lo real", 360, "", ".0%")

    def card_techniques(self, composition_id: str) -> list:
        techniques = self.table("composition_techniques")
        techniques = techniques[techniques["composition_id"] == composition_id]
        if techniques.empty:
            return [Note("Su composición no tiene técnicas probadas (por debajo del suelo de soporte o sin composición).", "info")]
        figure = go.Figure()
        first_band = sorted(techniques["tramo_h"].unique())[0]
        order = (techniques[techniques["tramo_h"] == first_band].sort_values("exam_err_pp", ascending=False)["tecnica"]
                 .map(lambda value: TECHNIQUE_LABELS.get(value, value)).tolist())
        for index, (band, part) in enumerate(techniques.groupby("tramo_h")):
            names = part["tecnica"].map(lambda value: TECHNIQUE_LABELS.get(value, value))
            figure.add_bar(y=names, x=part["exam_err_pp"], orientation="h", name=f"Error del examen · {band}",
                           marker=dict(color=SERIES_COLOURS[index], opacity=[1.0 if chosen else 0.35 for chosen in part["is_chosen"]]),
                           text=["★ elegida" if chosen else "" for chosen in part["is_chosen"]], textposition="outside",
                           hovertemplate="%{y}: %{x:.2f} pp<extra>" + band + "</extra>")
        figure.update_yaxes(categoryorder="array", categoryarray=order)
        figure.update_layout(barmode="group", hovermode="closest")
        shown = techniques[["tramo_h", "tecnica", "status", "months_available", "months_required", "selection_err_pp",
                            "exam_err_pp", "rank", "is_chosen"]].sort_values(["tramo_h", "rank"])
        return [Title("Las técnicas de su composición", "★ = la elegida. Se elige por el error en los meses de selección "
                      "(rank) y se juzga en el examen, que no se usa para elegir."),
                layout(figure, "Error del examen de cada técnica (pp); la opaca es la elegida en cada tramo", 80 + 26 * len(techniques), "", None),
                Table(shown, rows=40, formats={"selection_err_pp": points, "exam_err_pp": points})]

    def figure_card_money(self, future: pd.DataFrame) -> go.Figure:
        figure = go.Figure()
        if future.empty:
            return layout(figure, "Sin filas futuras", 200)
        labels = self.source_labels()
        monthly = future.groupby(["period", "forecast_pipeline_source"])[["forecast_to_renew_USD", "forecast_renewed_USD",
                                                                          "forecast_renewed_USD_low", "forecast_renewed_USD_high"]].sum().reset_index()
        due = monthly.groupby("period")["forecast_to_renew_USD"].sum()
        figure.add_bar(x=due.index, y=due.values, name="Vence", marker_color="rgba(120,128,139,0.30)")
        for source, part in monthly.groupby("forecast_pipeline_source"):
            figure.add_bar(x=part["period"], y=part["forecast_renewed_USD"], name=f"Esperado · {labels.get(source, (source, 0))[0]}",
                           marker_color=self.source_colour(source),
                           error_y=dict(type="data", symmetric=False, array=part["forecast_renewed_USD_high"] - part["forecast_renewed_USD"],
                                        arrayminus=part["forecast_renewed_USD"] - part["forecast_renewed_USD_low"], color=INK, thickness=1))
        figure.update_layout(barmode="overlay")
        return layout(figure, "Lo que vence y lo que se espera renovar, con su banda", 360, "USD")

    def card_ladder(self, fs_id: str, composition_id: str) -> list:
        """How it got its support: its ladder steps and merges, who is in its composition, its credibility."""
        items = [Title("Cómo consiguió su soporte", "Su grupo en cada paso de la escalera; si una fusión mejoró (sesgo frente a error); "
                       "quién entra en la tasa de su composición.")]
        steps = self.table("ladder_steps")
        steps = steps[steps["fs_id"] == fs_id]
        if not steps.empty:
            items.append(Table(steps.sort_values("ladder_step"), rows=20))
        merges = self.table("ladder_merges")
        merges = merges[merges["fs_id"] == fs_id] if not merges.empty else merges
        if not merges.empty:
            items.append(Table(merges, "Sus fusiones", formats={"own_rate": percent, "group_rate": percent, "bias_pp": points,
                                                                "error_alone_pp": points, "error_merged_pp": points}))
        members = self.table("composition_members")
        members = members[members["composition_id"] == composition_id]
        if len(members) > 1:
            series = self.table("series").set_index("s1_4_fs_id")
            members = members.assign(vencen_al_mes=members["fs_id"].map(series["s1_7_n_propio"]),
                                     tasa_propia=members["fs_id"].map(series["s1_7_tasa_propia"]),
                                     esperado_usd=members["fs_id"].map(series["s7_2_esperado_usd_total"]))
            items.append(Table(members.sort_values("vencen_al_mes", ascending=False),
                               "Su composición: uses = predice con ella · lends = solo presta su historia",
                               formats={"tasa_propia": percent, "esperado_usd": money}))
            items.append(self.figure_composition(composition_id, fs_id))
        dynamics = self.table("series_dynamics")
        dynamics = dynamics[dynamics["fs_id"] == fs_id]
        if not dynamics.empty:
            items.append(Table(dynamics, "Su dinámica propia (φ, tendencia, estacionalidad) frente a la de su composición"))
        return items

    def figure_composition(self, composition_id: str, fs_id: str = None) -> go.Figure:
        """The monthly rate of a composition, and of one of its series."""
        pool = self.table("pool_series")
        pool = pool[pool["composition_id"] == composition_id].sort_values("period")
        figure = go.Figure()
        figure.add_scatter(x=pool["period"], y=pool["tasa"], name=f"Composición {composition_id}", line=dict(color=SERIES_COLOURS[0], width=2))
        if fs_id:
            rows = self.pipeline_core()
            rows = rows[(rows["s1_4_fs_id"] == fs_id) & (rows["forecast_status"] == "actual")]
            monthly = rows.groupby("period")[["forecast_to_renew_units", "forecast_renewed_units"]].sum()
            monthly = monthly[monthly["forecast_to_renew_units"] > 0]
            figure.add_scatter(x=monthly.index, y=monthly["forecast_renewed_units"] / monthly["forecast_to_renew_units"],
                               name="La serie sola", line=dict(color=SERIES_COLOURS[1], width=1.5, dash="dot"))
        return layout(figure, "La tasa con la que predice (su composición) frente a la de la serie sola", 340, "", ".0%")

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 4 · THE LADDER: HOW THE SERIES GOT THEIR SUPPORT
    # ═══════════════════════════════════════════════════════════════════════════
    def page_ladder(self, composition_id: str = None) -> list:
        items = [Title("La escalera: cómo se mejoró el soporte de las forecast series",
                       "Una serie con menos de 30 licencias al mes no se puede predecir sola: se junta paso a paso con series "
                       "parecidas (signo, extras, colapso de mandatory) hasta tener soporte. 271 da un error de ±5 pp en un mes.", level=2)]
        summary = self.table("ladder_summary")
        if not summary.empty:
            figure = go.Figure()
            figure.add_bar(x=summary["step_name"], y=summary["pct_usd_floor"], name="USD en grupos con soporte ≥ 30",
                           marker_color=SERIES_COLOURS[0])
            figure.add_bar(x=summary["step_name"], y=summary["pct_usd_own_rate"], name="USD en grupos con soporte ≥ 271",
                           marker_color=SERIES_COLOURS[2])
            figure.update_layout(barmode="group")
            items.append(layout(figure, "Paso a paso: el dinero que ya tiene soporte suficiente", 340, "% del USD", ".0%"))
            items.append(Table(summary, "Paso a paso (sff_ladder_summary)", formats={"pct_usd_floor": percent, "pct_usd_own_rate": percent}))
        series = self.table("series")
        stage_columns = [f"s2_2_stage{stage}_support" for stage in range(4) if f"s2_2_stage{stage}_support" in series.columns]
        if stage_columns:
            weights = series["s1_6_usd_por_predecir"].fillna(0.0)
            rows = []
            for column_name in stage_columns:
                support = series[column_name]
                rows.append({"etapa": column_name.replace("s2_2_", "").replace("_support", ""),
                             "series_bajo_30": int((support < 30).sum()), "series_bajo_271": int((support < 271).sum()),
                             "usd_bajo_30": float(weights[support < 30].sum() / weights.sum()) if weights.sum() else np.nan,
                             "usd_desde_271": float(weights[support >= 271].sum() / weights.sum()) if weights.sum() else np.nan,
                             "soporte_mediano": float(support.median())})
            stages = pd.DataFrame(rows)
            figure = go.Figure()
            figure.add_bar(x=stages["etapa"], y=stages["usd_desde_271"], name="USD con soporte ≥ 271 (va sola)", marker_color=SERIES_COLOURS[0])
            figure.add_bar(x=stages["etapa"], y=stages["usd_bajo_30"], name="USD con soporte < 30", marker_color=SERIES_COLOURS[1])
            figure.update_layout(barmode="group")
            items.append(layout(figure, "El dinero según el soporte con el que predice, etapa a etapa", 340, "% del USD por predecir", ".0%"))
            items.append(Table(stages, formats={"usd_bajo_30": percent, "usd_desde_271": percent}))
        levels = self.table("money_by_level")
        if not levels.empty:
            items.append(Table(levels, "El dinero por nivel de riesgo (sff_niveles_riesgo)", formats={"usd_por_predecir": money, "pct_usd": percent, "error_pp": points}))
        merges = self.table("ladder_merges")
        if not merges.empty:
            figure = go.Figure()
            for improves, part in merges.groupby("improves"):
                figure.add_scatter(x=part["error_alone_pp"], y=part["error_merged_pp"], mode="markers",
                                   name="mejora" if improves else "no mejora", text=part["fs_id"] + " → " + part["group_id"].astype(str),
                                   marker=dict(color=CONFIDENCE_COLOURS["high"] if improves else CONFIDENCE_COLOURS["low"], size=8),
                                   hovertemplate="%{text}<br>sola %{x:.1f} pp · junta %{y:.1f} pp<extra></extra>")
            limit = float(np.nanmax([merges["error_alone_pp"].max(), merges["error_merged_pp"].max(), 1.0]))
            figure.add_scatter(x=[0, limit], y=[0, limit], mode="lines", line=dict(color=MUTED, dash="dot"), name="igual")
            figure.update_layout(hovermode="closest")
            items.append(layout(figure, "Cada fusión: el error de la serie sola frente al de junta (sesgo incluido)", 380,
                                "Error junta (pp)"))
            items.append(Table(merges.sort_values("error_alone_pp", ascending=False), "Las fusiones (sff_ladder_merges)",
                               formats={"own_rate": percent, "group_rate": percent, "bias_pp": points,
                                        "error_alone_pp": points, "error_merged_pp": points}))
        decision = self.table("dimension_decision")
        if not decision.empty:
            items.append(Table(decision, "Cuánto separa cada dimensión la tasa y el orden de colapso (paso 2.1)"))
        items.extend(self.composition_detail(composition_id))
        return items

    def compositions_by_money(self) -> pd.DataFrame:
        composition = self.table("composition")
        if composition.empty:
            return composition
        composition = composition[composition["is_composition"] == 1] if "is_composition" in composition.columns else composition
        return composition.sort_values("usd_por_predecir", ascending=False)

    def composition_detail(self, composition_id: str = None) -> list:
        compositions = self.compositions_by_money()
        if compositions.empty:
            return []
        if composition_id is None:
            composition_id = compositions["composition_id"].iloc[0]
        row = compositions[compositions["composition_id"] == composition_id]
        items = [Title(f"Composición {composition_id}", "Quién forma su tasa y cómo se predice.")]
        if not row.empty:
            items.append(Table(row, formats={"rate": percent, "ref_rate": percent, "estimated_rate": percent,
                                             "usd_por_predecir": money, "expected_usd_high": money,
                                             "expected_usd_medium": money, "expected_usd_low": money}))
        members = self.table("composition_members")
        items.append(Table(members[members["composition_id"] == composition_id], "Sus series"))
        items.append(self.figure_composition(composition_id))
        return items

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 5 · THE BACKTEST AND THE CHOICE OF THE TECHNIQUE
    # ═══════════════════════════════════════════════════════════════════════════
    def page_backtest(self, composition_id: str = None, horizon: int = 1) -> list:
        items = [Title("El backtest: qué técnica predice cada composición",
                       "En los meses de selección se prueban todas las técnicas con lo que se sabía antes; gana la de menor error "
                       "normalizado si bate a la técnica por defecto (media de 3 meses) por el margen. El examen, posterior, la juzga.", level=2)]
        decision = self.table("technique_decision")
        reference = self.table("pool_reference")
        if not decision.empty:
            money_of = reference.set_index("composition_id")["usd_por_predecir"] if not reference.empty else pd.Series(dtype=float)
            decision = decision.assign(usd_por_predecir=decision["composition_id"].map(money_of).fillna(0.0))
            wins = (decision.groupby(["tramo_h", "tecnica"]).agg(composiciones=("composition_id", "size"),
                                                                 usd_por_predecir=("usd_por_predecir", "sum")).reset_index())
            figure = go.Figure()
            for index, (band, part) in enumerate(wins.groupby("tramo_h")):
                figure.add_bar(x=part["tecnica"].map(lambda value: TECHNIQUE_LABELS.get(value, value)), y=part["usd_por_predecir"],
                               name=band, marker_color=SERIES_COLOURS[index])
            figure.update_layout(barmode="group")
            items.append(layout(figure, "El dinero que predice cada técnica, por tramo de horizonte", 380, "USD por predecir"))
            origins = decision.groupby(["tramo_h", "tecnica_origen"]).agg(composiciones=("composition_id", "size"),
                                                                          usd_por_predecir=("usd_por_predecir", "sum")).reset_index()
            items.append(Table(origins, "Por qué ganó: campeón (batió a la técnica por defecto) · por defecto · sin soporte (no se juzga)",
                               formats={"usd_por_predecir": money}))
        exam = self.table("exam_by_pool")
        if not exam.empty:
            figure = go.Figure()
            for index, (band, part) in enumerate(exam.groupby("tramo_h")):
                figure.add_scatter(x=part["por_defecto_err_pp_medio"], y=part["elegida_err_pp_medio"], mode="markers", name=band,
                                   text=part["composition_id"] + " · " + part["tecnica"].astype(str),
                                   marker=dict(color=SERIES_COLOURS[index], size=8),
                                   hovertemplate="%{text}<br>por defecto %{x:.2f} pp · elegida %{y:.2f} pp<extra></extra>")
            limit = float(np.nanmax([exam["por_defecto_err_pp_medio"].max(), exam["elegida_err_pp_medio"].max(), 1.0]))
            figure.add_scatter(x=[0, limit], y=[0, limit], mode="lines", line=dict(color=MUTED, dash="dot"), name="igual")
            figure.update_layout(hovermode="closest")
            items.append(layout(figure, "En el examen: el error de la elegida frente al de la técnica por defecto (debajo de la diagonal, la elegida gana)",
                                400, "Error de la elegida (pp)"))
        total = self.table("exam_total")
        if not total.empty:
            figure = go.Figure()
            for h_value, part in total.groupby("h"):
                part = part.sort_values("mes_objetivo")
                figure.add_scatter(x=part["mes_objetivo"], y=part["elegida_error_pct"], name=f"Elegida · h={h_value}", mode="lines+markers")
                figure.add_scatter(x=part["mes_objetivo"], y=part["por_defecto_error_pct"], name=f"Retador · h={h_value}", mode="lines+markers",
                                   line=dict(dash="dot"))
            items.append(layout(figure, "Error del TOTAL de renovaciones en cada mes de examen", 340, "Error (%)", ".1%"))
        bands = self.table("error_bands")
        if not bands.empty:
            items.append(Table(bands, "Las bandas: cuantiles del error normalizado por técnica y horizonte (de ahí sale el intervalo)"))
        items.extend(self.backtest_detail(composition_id, horizon))
        return items

    def backtest_detail(self, composition_id: str = None, horizon: int = 1) -> list:
        predictions = self.table("backtest_predictions")
        if predictions.empty:
            return []
        if composition_id is None:
            compositions = self.compositions_by_money()
            composition_id = compositions["composition_id"].iloc[0] if not compositions.empty else predictions["composition_id"].iloc[0]
        part = predictions[(predictions["composition_id"] == composition_id) & (predictions["h"] == horizon)]
        decision = self.table("technique_decision")
        chosen = decision[decision["composition_id"] == composition_id]
        items = [Title(f"El backtest de {composition_id} a h = {horizon}",
                       "Cada técnica predice cada mes objetivo con lo que se sabía h meses antes; círculos = meses de selección, "
                       "rombos = examen.")]
        if part.empty:
            return items + [Note("Sin predicciones para esta composición y horizonte.", "info")]
        figure = go.Figure()
        real = part.drop_duplicates("mes_objetivo").sort_values("mes_objetivo")
        figure.add_scatter(x=real["mes_objetivo"], y=real["tasa_real"], name="Real", line=dict(color=INK, width=3))
        chosen_names = set(chosen["tecnica"])
        for index, (technique, rows) in enumerate(part.groupby("tecnica")):
            rows = rows.sort_values("mes_objetivo")
            important = technique in chosen_names or technique == getattr(self.configuration, "default_technique", "T3_ma3")
            figure.add_scatter(x=rows["mes_objetivo"], y=rows["tasa_pred"], mode="lines+markers",
                               name=TECHNIQUE_LABELS.get(technique, technique) + (" ★" if technique in chosen_names else ""),
                               visible=True if important else "legendonly",
                               marker=dict(symbol=["diamond" if purpose != "seleccion" else "circle" for purpose in rows["proposito"]], size=7),
                               line=dict(width=2 if important else 1, color=SERIES_COLOURS[index % len(SERIES_COLOURS)]))
        items.append(layout(figure, "", 380, "Tasa", ".0%"))
        items.append(Note("Haz clic en la leyenda para ver las demás técnicas.", "info"))
        items.append(Table(chosen, "La decisión de esta composición", formats={"err_pp_seleccion": points}))
        techniques = self.table("composition_techniques")
        items.append(Table(techniques[techniques["composition_id"] == composition_id].sort_values(["tramo_h", "rank"]),
                           "Todas las técnicas: estado, error de selección y de examen, puesto", rows=40,
                           formats={"selection_err_pp": points, "exam_err_pp": points}))
        return items

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 6 · THE EXAM: HOW WELL EVERY METHOD PREDICTED
    # ═══════════════════════════════════════════════════════════════════════════
    def page_exam(self, horizon: int = 1) -> list:
        items = [Title("El examen: de la baseline al framework, en los mismos meses",
                       "Meses de examen predichos con lo que se sabía h meses antes. Error/ruido ≈ 1 es el límite: lo que queda "
                       "es azar binomial.", level=2)]
        summary = self.table("portfolio_exam_summary")
        if not summary.empty:
            items.append(Table(summary, "Por método y horizonte", formats={"error_total_medio": percent, "sesgo_total_medio": percent,
                                                                          "wape_series": percent, "en_intervalo": percent,
                                                                          "error_vs_ruido": lambda value: "—" if pd.isna(value) else f"{value:.2f}".replace(".", ",")}))
        monthly = self.table("portfolio_exam")
        if not monthly.empty:
            part = monthly[monthly["h"] == horizon].sort_values("mes")
            figure = go.Figure()
            figure.add_scatter(x=part["mes"], y=part["renovadas_reales"], name="Real", line=dict(color=INK, width=3))
            for method in ("framework", "por_defecto", "raw", "baseline"):
                if f"{method}_pred" in part.columns:
                    figure.add_scatter(x=part["mes"], y=part[f"{method}_pred"], name=METHOD_LABELS[method], mode="lines+markers",
                                       line=dict(color=METHOD_COLOURS[method], dash="solid" if method == "framework" else "dot"))
            items.append(layout(figure, f"Renovaciones totales del examen a h = {horizon}: cada método frente a lo real", 360, "Unidades"))
        per_series = self.table("exam_per_series")
        if not per_series.empty:
            figure = go.Figure()
            for method, colour in (("framework", METHOD_COLOURS["framework"]), ("raw", METHOD_COLOURS["raw"])):
                values = per_series[f"{method}_error_over_noise"].dropna()
                figure.add_histogram(x=values.clip(upper=5), name=METHOD_LABELS[method], marker_color=colour, opacity=0.65, nbinsx=40)
            figure.add_vline(x=1, line_dash="dot", line_color=INK)
            figure.update_layout(barmode="overlay", hovermode="closest")
            items.append(layout(figure, "Error / ruido por forecast serie (cortado en 5): a la izquierda de 1, el error es el del azar",
                                340, "Forecast series"))
            figure = go.Figure()
            series = self.table("series").set_index("s1_4_fs_id")
            weights = per_series["fs_id"].map(series["s7_2_esperado_usd_total"]).fillna(0.0)
            figure.add_scatter(x=per_series["raw_mae_pp"], y=per_series["framework_mae_pp"], mode="markers", text=per_series["fs_id"],
                               marker=dict(size=np.clip(np.sqrt(weights / max(weights.max(), 1.0)) * 30, 4, 30), color=SERIES_COLOURS[0], opacity=0.6),
                               hovertemplate="%{text}<br>sola %{x:.1f} pp · framework %{y:.1f} pp<extra></extra>", name="series")
            limit = float(np.nanmax([per_series["raw_mae_pp"].max(), per_series["framework_mae_pp"].max(), 1.0]))
            figure.add_scatter(x=[0, limit], y=[0, limit], mode="lines", line=dict(color=MUTED, dash="dot"), name="igual")
            figure.update_layout(hovermode="closest")
            items.append(layout(figure, "Cada serie: su error sola frente al del framework (tamaño = dinero; debajo de la diagonal mejora)",
                                420, "Error del framework (pp)"))
        series = self.table("series")
        by_confidence = (series.groupby("s7_2_confidence").agg(series=("s1_4_fs_id", "size"), predicciones=("s7_1_exam_predictions", "sum"),
                                                             en_intervalo=("s7_1_exam_in_band", "sum"), error_abs=("s7_1_exam_abs_err_units", "sum"),
                                                             reales=("s7_1_exam_real_units", "sum")).reset_index())
        by_confidence["dentro_intervalo"] = by_confidence["en_intervalo"] / by_confidence["predicciones"].where(by_confidence["predicciones"] > 0)
        by_confidence["wape"] = by_confidence["error_abs"] / by_confidence["reales"].where(by_confidence["reales"] > 0)
        items.append(Table(by_confidence, "¿La confianza dice la verdad? Acierto del examen por etiqueta de confianza",
                           formats={"dentro_intervalo": percent, "wape": percent}))
        return items

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 7 · THE UPLIFT AND THE PRICE INCREASES
    # ═══════════════════════════════════════════════════════════════════════════
    def price_groups(self) -> list:
        monitor = self.table("price_monitor")
        if monitor.empty:
            return []
        return list(monitor.groupby("price_group")["renewed_USD"].sum().sort_values(ascending=False).index)

    def page_uplift(self, price_group: str = None) -> list:
        items = [Title("El uplift y las subidas de precio",
                       "Uplift = lo que paga quien renueva / lo que valía lo que vencía. Estadístico: el de la ventana reciente sin "
                       "las subidas detectadas; contrato: 1 / (1 − descuento).", level=2)]
        cells = self.table("uplift_cells")
        if not cells.empty:
            by_origin = cells.groupby("uplift_origen").agg(celdas=("uplift_cell_id", "size"), uplift_medio=("uplift", "mean"),
                                                           renovadores=("renovadores", "sum")).reset_index()
            items.append(Table(by_origin, "Celdas por origen del uplift (propia ≥ suelo · padre · celda mandatory · global)",
                               formats={"uplift_medio": lambda value: f"{value:.3f}".replace(".", ",")}))
            figure = go.Figure()
            figure.add_histogram(x=cells["uplift"], nbinsx=50, marker_color=SERIES_COLOURS[0], name="celdas")
            figure.update_layout(hovermode="closest")
            items.append(layout(figure, "El uplift de las celdas", 300, "Celdas"))
        for name, caption in (("uplift_backtest", "El examen del uplift: cada vía en los mismos renovadores (paso 3.3)"),
                              ("contract_check", "La regla de contrato frente a lo observado, por tramo de descuento"),
                              ("uplift_homogeneity", "Homogeneidad: lo que valían los renovadores frente a la media de su celda")):
            frame = self.table(name)
            if not frame.empty:
                items.append(Table(frame, caption, rows=30))
        increases = self.table("price_increases")
        calendar = self.table("price_calendar")
        if not increases.empty:
            items.append(Table(increases, "Subidas en juego hoy: mes del ciclo, meses futuros que aún la llevan (paso 3.1)"))
        if not calendar.empty:
            items.append(Table(calendar, "Calendario de subidas detectadas (paso 8.4)"))
        elif not increases.empty or self.has("price_monitor"):
            items.append(Note("No se ha detectado ninguna subida de precio confirmada.", "info"))
        groups = self.price_groups()
        if groups:
            price_group = price_group or groups[0]
            monitor = self.table("price_monitor")
            part = monitor[monitor["price_group"] == price_group].sort_values("period")
            figure = go.Figure()
            figure.add_scatter(x=part["period"], y=part["uplift"], name="Uplift (todos)", line=dict(color=SERIES_COLOURS[1]))
            if part["uplift_isolated"].notna().any():
                figure.add_scatter(x=part["period"], y=part["uplift_isolated"], name="Uplift isolated (detector)", line=dict(color=SERIES_COLOURS[0], width=2.5))
            figure.add_scatter(x=part["period"], y=part["uplift_previous_12"], name="Media de los 12 meses previos", line=dict(color=MUTED, dash="dot"))
            flagged = part[part["price_increase_flag"] == 1]
            figure.add_scatter(x=flagged["period"], y=flagged["detector"], mode="markers", name="Inicio de subida",
                               marker=dict(symbol="star", size=14, color=CONFIDENCE_COLOURS["low"]))
            above = part[part["above_threshold"] == 1]
            figure.add_scatter(x=above["period"], y=above["detector"], mode="markers", name="Por encima del umbral",
                               marker=dict(symbol="circle-open", size=11, color=CONFIDENCE_COLOURS["medium"]))
            items.append(Title(f"Grupo de precio {price_group}", "El detector: un escalón ≥ umbral frente a sus 12 meses previos, "
                               "mantenido varios meses, es una subida; su efecto dura un ciclo."))
            items.append(layout(figure, "", 380, "Uplift"))
        return items

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 8 · THE STATES STILL TO COME
    # ═══════════════════════════════════════════════════════════════════════════
    def page_states(self) -> list:
        items = [Title("Los estados del cliente que aún no han llegado (dormant, softcancel, sin instalar)",
                       "Hoy: la foto del extracto. Esperada: tras el reparto a su proporción histórica. Final: la proporción "
                       "histórica del mismo mes. Un estado solo crece: hoy por encima de su final es un estado prematuro.", level=2)]
        maturation = self.table("maturation")
        if maturation.empty:
            return items + [Note("Esta ejecución no reparte los estados del cliente (distribute_marks desactivado).", "info")]
        states = [column_name[:-4] for column_name in maturation.columns if column_name.endswith("_hoy") and column_name != "neutro_hoy"]
        figure = go.Figure()
        for index, state in enumerate(states):
            colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
            figure.add_scatter(x=maturation["period"], y=maturation[f"{state}_hoy"], name=f"{state} hoy", line=dict(color=colour, dash="dot"))
            figure.add_scatter(x=maturation["period"], y=maturation[f"{state}_esperado"], name=f"{state} esperada", line=dict(color=colour, width=2.5))
            if f"{state}_final" in maturation.columns:
                figure.add_scatter(x=maturation["period"], y=maturation[f"{state}_final"], name=f"{state} final (histórica)",
                                   line=dict(color=colour, width=1), mode="markers", marker=dict(symbol="line-ew-open", size=14, color=colour))
        items.append(layout(figure, "Cada estado: hoy, esperado y final, mes a mes", 420, "% de las unidades que vencen", ".0%"))
        figure = go.Figure()
        figure.add_bar(x=maturation["period"], y=maturation["ajuste_usd"], marker_color=SERIES_COLOURS[1], name="Ajuste por estados")
        items.append(layout(figure, "Lo que el reparto de estados resta al forecast, mes a mes", 300, "USD"))
        shown_columns = ["period", "meses_vista"] + [f"{state}_{moment}" for state in states for moment in ("hoy", "esperado", "final")
                                                     if f"{state}_{moment}" in maturation.columns] + ["tasa_base", "tasa_final", "ajuste_usd", "ajuste_pct"]
        formats = {column_name: percent for column_name in shown_columns if column_name.endswith(("_hoy", "_esperado", "_final")) or column_name.startswith("tasa")}
        formats.update({"ajuste_usd": money, "ajuste_pct": percent})
        items.append(Table(maturation[shown_columns], rows=40, formats=formats))
        return items

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 9 · THE PARAMETERS AND THE CHECKS
    # ═══════════════════════════════════════════════════════════════════════════
    def page_parameters(self, only_changed: bool = True, text: str = "") -> list:
        items = [Title("Parámetros y validación de la ejecución", "Todo número que decide algo está en la Config; aquí, su valor "
                       "en esta ejecución y los pasos que lo leen. La leyenda describe cada columna del núcleo.", level=2)]
        validation = self.table("validation")
        if not validation.empty:
            items.append(Table(validation, "La validación final (paso 8.5)"))
        parameters = self.table("parameters")
        if not parameters.empty:
            shown = parameters[parameters["cambiado"] == 1] if only_changed else parameters
            if text:
                mask = shown.apply(lambda row: text.lower() in " ".join(str(value).lower() for value in row.values), axis=1)
                shown = shown[mask]
            items.append(Table(shown, "Parámetros" + (" con valor propio en esta ejecución" if only_changed else ""), rows=200))
        legend = self.table("legend")
        if not legend.empty and text:
            mask = legend.apply(lambda row: text.lower() in " ".join(str(value).lower() for value in row.values), axis=1)
            items.append(Table(legend[mask], f"Columnas del núcleo y de la dimensión que contienen «{html.escape(text)}»", rows=100))
        return items

    # ═══════════════════════════════════════════════════════════════════════════
    # SHOWING THE PAGES
    # ═══════════════════════════════════════════════════════════════════════════
    def page(self, key: str, **arguments) -> list:
        builders = {"negocio": self.page_business, "series": self.page_series, "ficha": self.page_card,
                    "escalera": self.page_ladder, "backtest": self.page_backtest, "examen": self.page_exam,
                    "uplift": self.page_uplift, "estados": self.page_states, "parametros": self.page_parameters}
        return builders[key](**arguments)

    def display_items(self, items: list) -> None:
        """Show the items of a page in the current cell (or inside an ipywidgets Output)."""
        from IPython.display import HTML, display
        for item in items:
            if isinstance(item, go.Figure):
                if self.figure_mode == "html":
                    display(HTML(item.to_html(include_plotlyjs="cdn", full_html=False, config=PLOTLY_CONFIG)))
                else:
                    item.show(config=PLOTLY_CONFIG)
            elif item is not None:
                display(HTML(item.to_html()))

    def show_page(self, key: str, **arguments) -> None:
        """One page, statically, in the current cell."""
        self.display_items(self.page(key, **arguments))

    def show(self):
        """The whole explorer: one tab per page, with its controls; returns the widget."""
        import ipywidgets as widgets
        from IPython.display import display

        tabs, builders = [], {}

        def page_with_controls(key, controls: list, read_arguments):
            output = widgets.Output()

            def refresh(*_):
                with output:
                    output.clear_output(wait=True)
                    try:
                        self.display_items(self.page(key, **read_arguments()))
                    except Exception as error:                           # show it in the tab, do not break the explorer
                        self.page_errors.append((key, f"{type(error).__name__}: {error}"))
                        print(f"No se pudo construir la página: {type(error).__name__}: {error}")
            for control in controls:
                if hasattr(control, "observe"):
                    control.observe(refresh, names="value")
            builders[key] = refresh
            return widgets.VBox([widgets.HBox(controls, layout=widgets.Layout(flex_flow="row wrap", gap="12px")), output])

        # 1 · business: a dimension and a year to break down
        core = self.table("core")
        years = sorted(int(value) for value in core["forecast_year"].dropna().unique()) if "forecast_year" in core.columns else []
        dimension = widgets.Dropdown(options=[("—", None)] + [(name, name) for name in self.breakdown_dimensions()], description="Desglose")
        year = widgets.Dropdown(options=years, value=years[-1] if years else None, description="Año")
        tabs.append(page_with_controls("negocio", [dimension, year], lambda: {"dimension": dimension.value, "year": year.value}))

        # 2 · series: confidence, risk level, text
        series = self.table("series")
        confidence = widgets.SelectMultiple(options=[(CONFIDENCE_LABELS.get(value, value), value) for value in ["high", "medium", "low"]],
                                            description="Confianza", rows=3)
        risk = widgets.SelectMultiple(options=sorted(series["s2_3_nivel_riesgo"].dropna().unique()), description="Riesgo", rows=4)
        search = widgets.Text(description="Buscar", placeholder="trozo del fs_id (p. ej. EU|A)")
        tabs.append(page_with_controls("series", [confidence, risk, search],
                                       lambda: {"confidence": list(confidence.value), "risk_levels": list(risk.value), "text": search.value}))

        # 3 · the card: the series (by money; any other can be typed) and the horizon of the exam
        ordered = self.series_by_money()
        options = [str(value) for value in ordered["s1_4_fs_id"].head(SERIES_OPTIONS_SHOWN)]
        fs_box = widgets.Combobox(options=options, value=options[0] if options else "", description="Serie", ensure_option=False,
                                  placeholder="escribe o elige un fs_id", layout=widgets.Layout(width="520px"))
        horizons = list(getattr(self.configuration, "backtest_horizons", [1, 6]))
        exam_h = widgets.Dropdown(options=horizons, value=horizons[0], description="Examen h")
        card_tab = page_with_controls("ficha", [fs_box, exam_h], lambda: {"fs_id": fs_box.value or None, "exam_horizon": exam_h.value})
        tabs.append(card_tab)

        # 4 · ladder and 5 · backtest: a composition
        compositions = self.compositions_by_money()
        composition_options = [str(value) for value in compositions["composition_id"].head(SERIES_OPTIONS_SHOWN)] if not compositions.empty else []
        ladder_box = widgets.Combobox(options=composition_options, value=composition_options[0] if composition_options else "",
                                      description="Composición", ensure_option=False, layout=widgets.Layout(width="520px"))
        tabs.append(page_with_controls("escalera", [ladder_box], lambda: {"composition_id": ladder_box.value or None}))
        backtest_box = widgets.Combobox(options=composition_options, value=composition_options[0] if composition_options else "",
                                        description="Composición", ensure_option=False, layout=widgets.Layout(width="520px"))
        backtest_h = widgets.Dropdown(options=sorted(self.table("backtest_predictions")["h"].unique()) if self.has("backtest_predictions") else [1],
                                      description="h")
        tabs.append(page_with_controls("backtest", [backtest_box, backtest_h], lambda: {"composition_id": backtest_box.value or None,
                                                                                       "horizon": backtest_h.value}))
        # 6 · exam: the horizon
        exam_page_h = widgets.Dropdown(options=horizons, value=horizons[0], description="h")
        tabs.append(page_with_controls("examen", [exam_page_h], lambda: {"horizon": exam_page_h.value}))
        # 7 · uplift: a price group
        groups = self.price_groups()
        group_box = widgets.Dropdown(options=groups[:SERIES_OPTIONS_SHOWN] or [None], description="Grupo precio", layout=widgets.Layout(width="520px"))
        tabs.append(page_with_controls("uplift", [group_box], lambda: {"price_group": group_box.value}))
        # 8 · states
        tabs.append(page_with_controls("estados", [], lambda: {}))
        # 9 · parameters
        only_changed = widgets.Checkbox(value=True, description="Solo los cambiados")
        parameter_text = widgets.Text(description="Buscar", placeholder="parámetro, paso o columna del núcleo")
        tabs.append(page_with_controls("parametros", [only_changed, parameter_text],
                                       lambda: {"only_changed": only_changed.value, "text": parameter_text.value}))

        tab = widgets.Tab(children=tabs)
        for index, (_, title) in enumerate(PAGES):
            tab.set_title(index, title)
        drawn = set()

        def draw_selected(*_):
            key = PAGES[tab.selected_index][0]
            if key not in drawn:                                       # a page is built the first time it is opened
                drawn.add(key)
                builders[key]()
        tab.observe(draw_selected, names="selected_index")
        header = widgets.HTML(self.header_html())
        explorer_widget = widgets.VBox([header, tab])
        display(explorer_widget)
        draw_selected()
        return explorer_widget

    def export_html(self, file_path: str, top_series: int = 5, offline: bool = True) -> str:
        """Every page, statically, in one HTML file to open in a browser or send: the business, the series, the card of
        the top_series series with the most money, the ladder, the backtest, the exam, the uplift, the states and the
        parameters. offline=True embeds plotly once (about 4 MB, opens without internet); False loads it from its CDN."""
        sections, plotly_written = [], False
        pages = [("negocio", {}), ("series", {})]
        pages += [("ficha", {"fs_id": fs_id}) for fs_id in self.series_by_money()["s1_4_fs_id"].head(top_series)]
        pages += [("escalera", {}), ("backtest", {}), ("examen", {}), ("uplift", {}), ("estados", {}), ("parametros", {})]
        titles = dict(PAGES)
        for key, arguments in pages:
            parts = [f"<h1 style='font-family:sans-serif;border-top:1px solid #ccc;padding-top:18px'>{titles[key]}</h1>"]
            for item in self.page(key, **arguments):
                if isinstance(item, go.Figure):
                    include = (True if offline else "cdn") if not plotly_written else False
                    parts.append(item.to_html(include_plotlyjs=include, full_html=False, config=PLOTLY_CONFIG))
                    plotly_written = True
                elif item is not None:
                    parts.append(item.to_html())
            sections.append("\n".join(parts))
        document = ("<!doctype html><html><head><meta charset='utf-8'><title>SFF · explorador</title></head>"
                    "<body style='max-width:1200px;margin:0 auto;padding:16px'>" + self.header_html() + "\n".join(sections)
                    + "</body></html>")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(document)
        return file_path

    def header_html(self) -> str:
        source = "los resultados en memoria" if self.results is not None else "las tablas SQL de la ejecución"
        current = getattr(self.configuration, "current_month", "—")
        return (f"<div style='font-family:sans-serif;margin:4px 0 8px 0'><b style='font-size:18px'>SFF · explorador de la ejecución</b>"
                f"<span style='color:{MUTED};margin-left:12px'>mes en curso {current} · leído de {source} · cada pestaña se construye "
                f"al abrirla</span></div>")
