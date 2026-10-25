"""
perfil_para_sintetico.py — The profile of an extract, printed on screen, to build a synthetic that looks like it.

Only AGGREGATES are printed (shares, rates, quantiles, monthly totals): no row, no licence, no customer. The
output is meant to be read (or photographed) and handed over: with it the synthetic (synthetic_real.py) is
calibrated against the shape of the real extract without the extract leaving its machine.

    from perfil_para_sintetico import print_profile
    raw = configuration.read_raw()                 # the same raw the run reads
    print_profile(raw, configuration)              # every section; or sections=["P05", "P07"]

Every section starts with "=== PERFIL Pnn · title ===" so it can be found in the console. Run it once on the real
extract and once on the synthetic (python perfil_para_sintetico.py) and the two outputs compare line by line.

    P01  the calendar: months, units and USD due and renewed, per month (the booked part of the last ones)
    P02  the size of the series: how concentrated the money is (forecast series and business series)
    P03  every business dimension: share of units and USD, unit rate, USD rate, price of a due unit
    P04  the crossings that move the rate most: renewal type × net_new, term × renewal type
    P05  the states: share and rate of every combination, by renewal type and net_new, and month by month
    P06  the discount: unknown by partner, the buckets by renewal type and net_new, values per forecast unit
    P07  the dynamics of the rate: the portfolio month by month, and per series φ, autocorrelation and slope
    P08  the uplift: by discount bucket, by net_new, month by month, the isolated renewals
    P09  the price groups: how many, their volume, their uplift month by month (the big ones)
    P10  the cohorts: what falls due in t against what renewed (and was acquired) in t − term
    P11  acquisitions and partners, month by month and year by year
    P12  the future: what is due and what is already booked, by month ahead
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, parse_month

# ─── named constants ─────────────────────────────────────────────────────────────
TOP_VALUES = 15                    # values of a dimension shown (the rest summed as "(resto)")
SIZE_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
MIN_UNITS_FOR_DYNAMICS = 30        # a series enters P07 when its typical month has at least this many units due
MIN_MONTHS_FOR_DYNAMICS = 18
DISCOUNT_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
PRICE_GROUPS_SHOWN = 8
PANDAS_DISPLAY = {"display.max_rows": 80, "display.max_columns": 20, "display.width": 160}


# ═══════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════

def section(code: str, title: str) -> None:
    print(f"\n=== PERFIL {code} · {title} ===")


def show(frame: pd.DataFrame, decimals: int = 3) -> None:
    with pd.option_context(*[item for pair in PANDAS_DISPLAY.items() for item in pair]):
        print(frame.round(decimals).to_string())


def ratio(numerator, denominator):
    return numerator / denominator.where(denominator != 0) if isinstance(denominator, pd.Series) else (
        numerator / denominator if denominator else np.nan)


class Columns:
    """The column names of the extract, read from the Config (so the script runs on the real one and on the synthetic)."""
    def __init__(self, raw: pd.DataFrame, configuration: Config):
        self.period = configuration.period_col
        self.due = configuration.pipeline_units_col
        self.due_usd = configuration.pipeline_usd_col
        self.renewed = configuration.renewed_units_col
        self.renewed_usd = configuration.renewed_usd_col
        self.base_usd = configuration.renewed_pipeline_usd_col
        self.isolated_usd = configuration.isolated_renewed_usd_col
        self.isolated_base = configuration.isolated_renewed_pipeline_usd_col
        self.discount = configuration.discount_value_column
        self.acquisition = configuration.acquisition_column
        self.term = configuration.term_column
        self.states = [column for column in configuration.structural_timevarying_dims if column in raw.columns]
        self.mandatory = [column for column in configuration.business_mandatory_dims if column in raw.columns]
        self.extras = [column for column in configuration.extra_renovacion if column in raw.columns]
        self.price_dims = [column for column in configuration.price_group_dims if column in raw.columns]


def add_helpers(raw: pd.DataFrame, columns: Columns, configuration: Config) -> pd.DataFrame:
    """The month as a period, the calendar role (closed / future), the state combination and the discount bucket."""
    frame = raw.copy()
    frame["_month"] = frame[columns.period].map(parse_month)
    current = configuration.calendar_boundaries()["current"]
    frame["_closed"] = frame["_month"] < current
    frame["_months_ahead"] = (frame["_month"] - current).map(lambda offset: offset.n) + 1
    if columns.states:
        frame["_states"] = frame[columns.states].astype(int).astype(str).agg("".join, axis=1)
        names = {}
        for code in frame["_states"].unique():
            in_state = [state for state, flag in zip(columns.states, code) if flag == "1"]
            names[code] = "+".join(in_state) if in_state else "neutral"
        frame["_states"] = frame["_states"].map(names)
    if columns.discount:
        buckets = pd.cut(frame[columns.discount], DISCOUNT_EDGES, right=False,
                         labels=[f"{int(low * 100):02d}-{int(low * 100) + 10}%" for low in DISCOUNT_EDGES[:-1]])
        frame["_bucket"] = buckets.astype(str).where(frame[columns.discount].notna(), "desconocido")
    return frame


def rate_table(frame: pd.DataFrame, by, columns: Columns, top: int = None) -> pd.DataFrame:
    """Share of units and USD due, unit rate, USD rate and price of a due unit, by `by` (closed months)."""
    grouped = frame.groupby(by, dropna=False)[[columns.due, columns.due_usd, columns.renewed, columns.renewed_usd]].sum()
    if top and len(grouped) > top:
        ordered = grouped.sort_values(columns.due_usd, ascending=False)
        rest = ordered.iloc[top:].sum().to_frame("(resto)").T
        grouped = pd.concat([ordered.iloc[:top], rest])
    table = pd.DataFrame({"cuota_unid": grouped[columns.due] / grouped[columns.due].sum(),
                          "cuota_usd": grouped[columns.due_usd] / grouped[columns.due_usd].sum(),
                          "tasa_unid": ratio(grouped[columns.renewed], grouped[columns.due]),
                          "tasa_usd": ratio(grouped[columns.renewed_usd], grouped[columns.due_usd]),
                          "precio_unidad": ratio(grouped[columns.due_usd], grouped[columns.due])})
    return table


# ═══════════════════════════════════════════════════════════════════════════════════
# THE SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════════

def profile_calendar(frame, columns, configuration):
    section("P01", "el calendario: lo que vence y lo que renueva, mes a mes")
    monthly = frame.groupby("_month")[[columns.due, columns.due_usd, columns.renewed, columns.renewed_usd]].sum()
    monthly["filas"] = frame.groupby("_month").size()
    monthly["tasa_unid"] = ratio(monthly[columns.renewed], monthly[columns.due])
    monthly["tasa_usd"] = ratio(monthly[columns.renewed_usd], monthly[columns.due_usd])
    monthly = monthly.rename(columns={columns.due: "vencen_unid", columns.due_usd: "vencen_usd",
                                      columns.renewed: "renov_unid", columns.renewed_usd: "renov_usd"})
    print(f"mes en curso del Config: {configuration.calendar_boundaries()['current']} · meses {monthly.index.min()}..{monthly.index.max()}")
    show(monthly[["filas", "vencen_unid", "vencen_usd", "renov_unid", "tasa_unid", "tasa_usd"]], 3)


def profile_sizes(frame, columns):
    section("P02", "el tamaño de las series: lo concentrado que está el dinero (meses cerrados)")
    closed = frame[frame["_closed"]]
    for label, keys in (("forecast series (todas las dims, estados del cliente y extras)", columns.mandatory + columns.states + columns.extras),
                        ("series de negocio (solo las obligatorias)", columns.mandatory)):
        per_month = closed.groupby(keys + ["_month"], dropna=False)[[columns.due, columns.due_usd]].sum()
        typical = per_month[columns.due].groupby(level=list(range(len(keys)))).median()
        money = per_month[columns.due_usd].groupby(level=list(range(len(keys)))).sum().sort_values(ascending=False)
        months = per_month.groupby(level=list(range(len(keys)))).size()
        print(f"\n{label}: {len(typical):,} series")
        quantiles = pd.DataFrame({"unidades_mes_tipico": typical.quantile(SIZE_QUANTILES),
                                  "meses_con_algo": months.quantile(SIZE_QUANTILES)})
        show(quantiles, 1)
        cumulative = money.cumsum() / money.sum()
        shares = {f"top {share:.1%} de las series": float(cumulative.iloc[max(int(len(money) * share) - 1, 0)])
                  for share in (0.001, 0.01, 0.05, 0.10, 0.25, 0.50)}
        print("cuota del USD que vence en las series más grandes:", {key: round(value, 3) for key, value in shares.items()})
        for floor in (30, 100, 271, 1000):
            above = typical[typical >= floor].index
            print(f"  series con mes típico ≥ {floor}: {len(above):,} · llevan el {money.reindex(above).sum() / money.sum():.1%} del USD")
    rows_per_unit = closed.groupby(columns.mandatory + columns.states + columns.extras + ["_month"], dropna=False).size()
    print(f"\nfilas finas por forecast unit: media {rows_per_unit.mean():.2f} · p90 {rows_per_unit.quantile(0.9):.0f} · máx {rows_per_unit.max()}")


def profile_dimensions(frame, columns):
    section("P03", "cada dimensión: cuota, tasa y precio de una unidad que vence (meses cerrados)")
    closed = frame[frame["_closed"]]
    for column in columns.mandatory + columns.extras + ([columns.acquisition] if columns.acquisition and columns.acquisition not in columns.extras else []):
        print(f"\n· {column} ({closed[column].nunique()} valores)")
        show(rate_table(closed, column, columns, top=TOP_VALUES))


def profile_crossings(frame, columns):
    section("P04", "los cruces que más mueven la tasa (meses cerrados)")
    closed = frame[frame["_closed"]]
    pairs = []
    renewal_type = next((column for column in columns.mandatory if "renewal_type" in column), None)
    if renewal_type and columns.acquisition:
        pairs.append([renewal_type, columns.acquisition])
    if renewal_type and columns.term:
        pairs.append([columns.term, renewal_type])
    for extra in columns.extras:
        if columns.acquisition and extra != columns.acquisition:
            pairs.append([columns.acquisition, extra])
    for pair in pairs:
        print(f"\n· {' × '.join(pair)}")
        show(rate_table(closed, pair, columns))


def profile_states(frame, columns):
    section("P05", "los estados del cliente: cada combinación, por tipo de renovación y net_new, y mes a mes")
    if not columns.states:
        print("sin estados del cliente en el Config")
        return
    closed = frame[frame["_closed"]]
    print("· toda la cartera (meses cerrados: estados finales)")
    show(rate_table(closed, "_states", columns))
    renewal_type = next((column for column in columns.mandatory if "renewal_type" in column), None)
    for column in [renewal_type, columns.acquisition]:
        if column:
            print(f"\n· por {column}: cuota de unidades de cada combinación dentro del valor, y su tasa de unidades")
            grouped = closed.groupby([column, "_states"])[[columns.due, columns.renewed]].sum()
            share = grouped[columns.due] / grouped[columns.due].groupby(level=0).transform("sum")
            show(pd.DataFrame({"cuota_unid": share, "tasa_unid": ratio(grouped[columns.renewed], grouped[columns.due])}).unstack(1), 3)
    print("\n· mes a mes: cuota de unidades con estado (en los meses futuros, lo que el extracto ya ve hoy)")
    monthly = frame.groupby("_month").apply(lambda month: pd.Series(
        {state: (month[columns.due] * month[state]).sum() / month[columns.due].sum() for state in columns.states}))
    monthly["meses_vista"] = frame.groupby("_month")["_months_ahead"].first()
    show(monthly, 3)


def profile_discount(frame, columns):
    section("P06", "el descuento: desconocido por partner, tramos por tipo de renovación y net_new")
    if not columns.discount:
        print("sin columna de descuento en el Config")
        return
    unknown = frame[columns.discount].isna()
    print(f"desconocido: {unknown.mean():.1%} de las filas · {frame.loc[unknown, columns.due].sum() / frame[columns.due].sum():.1%} "
          f"de las unidades · {frame.loc[unknown, columns.due_usd].sum() / frame[columns.due_usd].sum():.1%} del USD que vence")
    partner = next((column for column in columns.mandatory if "partner" in column), None)
    if partner:
        by_partner = frame.groupby(partner).apply(lambda rows: pd.Series({
            "cuota_usd": rows[columns.due_usd].sum() / frame[columns.due_usd].sum(),
            "desconocido_usd": rows.loc[rows[columns.discount].isna(), columns.due_usd].sum() / rows[columns.due_usd].sum()}))
        print("\n· por partner")
        show(by_partner.sort_values("cuota_usd", ascending=False).head(TOP_VALUES))
    renewal_type = next((column for column in columns.mandatory if "renewal_type" in column), None)
    for column in [renewal_type, columns.acquisition]:
        if column:
            known = frame[~unknown]
            grouped = known.groupby([column, "_bucket"])[columns.due].sum()
            print(f"\n· tramo del descuento conocido por {column} (cuota de unidades dentro del valor)")
            show((grouped / grouped.groupby(level=0).transform("sum")).unstack(1).fillna(0.0), 3)
    closed = frame[frame["_closed"]]
    print("\n· tasa y precio por tramo (meses cerrados)")
    show(rate_table(closed, "_bucket", columns))
    values_per_unit = frame.groupby(columns.mandatory + columns.states + columns.extras + ["_month"], dropna=False)[columns.discount].nunique(dropna=False)
    print(f"\nvalores distintos de descuento por forecast unit: media {values_per_unit.mean():.2f} · p90 {values_per_unit.quantile(0.9):.0f}")


def profile_dynamics(frame, columns):
    section("P07", "la dinámica de la tasa: la cartera mes a mes y cada serie con soporte")
    closed = frame[frame["_closed"]]
    keys = columns.mandatory + columns.states + columns.extras
    per_month = closed.groupby(keys + ["_month"], dropna=False)[[columns.due, columns.renewed]].sum().reset_index()
    per_month["_series"] = per_month[keys].astype(str).agg("|".join, axis=1)
    rows = []
    for _, series in per_month.groupby("_series"):
        if len(series) < MIN_MONTHS_FOR_DYNAMICS or series[columns.due].median() < MIN_UNITS_FOR_DYNAMICS:
            continue
        series = series.sort_values("_month")
        rate = (series[columns.renewed] / series[columns.due]).to_numpy()
        mean_rate = series[columns.renewed].sum() / series[columns.due].sum()
        binomial_variance = np.mean(mean_rate * (1 - mean_rate) / series[columns.due].to_numpy())
        residual = rate - pd.Series(rate).rolling(6, min_periods=3, center=True).mean().to_numpy()
        months = np.arange(len(rate))
        rows.append({"unidades_mes": series[columns.due].median(), "tasa": mean_rate,
                     "phi": np.var(rate, ddof=1) / binomial_variance if binomial_variance > 0 else np.nan,
                     "sd_cambio_mensual_pp": 100 * np.std(np.diff(rate), ddof=1),
                     "autocorr_1": pd.Series(rate).autocorr(1),
                     "autocorr_1_residuo": pd.Series(residual).autocorr(1),
                     "pendiente_pp_ano": 100 * 12 * np.polyfit(months, rate, 1)[0]})
    dynamics = pd.DataFrame(rows)
    print(f"series con mes típico ≥ {MIN_UNITS_FOR_DYNAMICS} y ≥ {MIN_MONTHS_FOR_DYNAMICS} meses: {len(dynamics):,}")
    if len(dynamics):
        show(dynamics.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T[["count", "mean", "10%", "25%", "50%", "75%", "90%"]], 3)
        dynamics["tamano"] = pd.cut(dynamics["unidades_mes"], [0, 100, 300, 1000, np.inf], labels=["30-100", "100-300", "300-1000", ">1000"])
        print("\n· por tamaño: mediana de φ, del cambio mensual y de la autocorrelación")
        show(dynamics.groupby("tamano", observed=True)[["phi", "sd_cambio_mensual_pp", "autocorr_1", "pendiente_pp_ano"]].median(), 3)


def profile_uplift(frame, columns):
    section("P08", "el uplift: Σ renovado USD / Σ base de los renovadores")
    if not columns.base_usd:
        print("sin base exacta del uplift en el Config")
        return
    closed = frame[frame["_closed"] & (frame[columns.renewed] > 0) & (frame[columns.base_usd] > 0)]

    def uplift_by(by):
        grouped = closed.groupby(by, dropna=False)[[columns.renewed_usd, columns.base_usd]].sum()
        table = pd.DataFrame({"cuota_usd": grouped[columns.renewed_usd] / grouped[columns.renewed_usd].sum(),
                              "uplift": grouped[columns.renewed_usd] / grouped[columns.base_usd]})
        if columns.isolated_usd and columns.isolated_base:
            isolated = closed.groupby(by, dropna=False)[[columns.isolated_usd, columns.isolated_base]].sum()
            table["uplift_isolated"] = isolated[columns.isolated_usd] / isolated[columns.isolated_base].where(isolated[columns.isolated_base] > 0)
            table["cuota_isolated"] = isolated[columns.isolated_base] / grouped[columns.base_usd]
        return table
    for label, by in (("tramo de descuento", "_bucket"), (columns.acquisition, columns.acquisition), ("estados", "_states")):
        if by and by in closed.columns:
            print(f"\n· por {label}")
            show(uplift_by(by))
    print("\n· mes a mes (toda la cartera)")
    show(uplift_by("_month"))


def profile_price_groups(frame, columns):
    section("P09", "los grupos de precio: cuántos, su volumen y el uplift mes a mes de los grandes")
    if not columns.price_dims or not columns.base_usd:
        print("sin price_group_dims o sin base exacta del uplift")
        return
    closed = frame[frame["_closed"] & (frame[columns.renewed] > 0) & (frame[columns.base_usd] > 0)].copy()
    closed["_group"] = closed[columns.price_dims].astype(str).agg("|".join, axis=1)
    monthly = closed.groupby(["_group", "_month"])[[columns.renewed, columns.renewed_usd, columns.base_usd]].sum()
    renewals = monthly[columns.renewed].groupby(level=0).median().sort_values(ascending=False)
    print(f"{len(renewals)} grupos · renovaciones por mes (mediana de cada grupo):")
    show(renewals.quantile(SIZE_QUANTILES).to_frame("renovadas_mes"), 0)
    print(f"grupos con mediana ≥ 100 renovaciones al mes: {(renewals >= 100).sum()}")
    largest = renewals.index[:PRICE_GROUPS_SHOWN]
    uplift = (monthly[columns.renewed_usd] / monthly[columns.base_usd]).unstack(0)[largest]
    uplift.columns = [f"G{position + 1}" for position in range(len(largest))]
    print("\n· uplift mes a mes de los grupos más grandes (G1 = el de más renovaciones; los nombres, abajo)")
    show(uplift, 3)
    for position, name in enumerate(largest):
        print(f"  G{position + 1} = {name}")


def profile_cohorts(frame, columns):
    section("P10", "las cohortes: lo que vence en t frente a lo que se renovó y adquirió en t − plazo")
    if not columns.term:
        print("sin term_column en el Config")
        return
    monthly = frame.groupby([columns.term, "_month"] + ([columns.acquisition] if columns.acquisition else []))[
        [columns.due, columns.renewed]].sum()
    for term in frame[columns.term].dropna().unique():
        months = 12 * int("".join(character for character in str(term) if character.isdigit()) or 1)
        rows = monthly.xs(term, level=columns.term)
        due_total = rows[columns.due].groupby(level="_month").sum()
        renewed_total = rows[columns.renewed].groupby(level="_month").sum()
        bought_month = pd.PeriodIndex([month - months for month in due_total.index], freq="M")
        comparison = pd.DataFrame({"vence_t": due_total,
                                   "renovado_t_menos_plazo": renewed_total.reindex(bought_month).to_numpy()}, index=due_total.index)
        if columns.acquisition:
            for value in rows.index.get_level_values(columns.acquisition).unique():
                comparison[f"vence_t_{value}"] = rows[columns.due].xs(value, level=columns.acquisition).reindex(comparison.index)
        comparison["vence_t / renovado_t-plazo"] = comparison["vence_t"] / comparison["renovado_t_menos_plazo"]
        retention = [column for column in comparison.columns if column.startswith("vence_t_Retention")]
        if retention:
            comparison["retencion_t / renovado_t-plazo"] = comparison[retention[0]] / comparison["renovado_t_menos_plazo"]
        comparison = comparison[comparison["renovado_t_menos_plazo"] > 0]
        print(f"\n· plazo {term} ({months} meses): {len(comparison)} meses con su cohorte en el extracto")
        show(comparison.tail(24), 2)


def profile_acquisitions_and_partners(frame, columns):
    section("P11", "adquisiciones y partners")
    if columns.acquisition:
        print("· unidades que vencen por valor de net_new, mes a mes")
        show(frame.groupby(["_month", columns.acquisition])[columns.due].sum().unstack(1).fillna(0), 0)
    partner = next((column for column in columns.mandatory if "partner" in column), None)
    if partner:
        print("\n· unidades que vencen por partner y año")
        show(frame.groupby([frame["_month"].map(lambda month: month.year), partner])[columns.due].sum().unstack(1).fillna(0), 0)


def profile_future(frame, columns):
    section("P12", "el futuro: lo que vence y lo ya contabilizado, por meses vista")
    future = frame[~frame["_closed"]]
    monthly = future.groupby(["_month", "_months_ahead"])[[columns.due, columns.due_usd, columns.renewed, columns.renewed_usd]].sum()
    monthly["renovado_ya_contabilizado"] = ratio(monthly[columns.renewed], monthly[columns.due])
    if columns.term:
        one_year = future[future[columns.term].astype(str).str.startswith("1")]
        monthly["cuota_1_ano"] = (one_year.groupby(["_month", "_months_ahead"])[columns.due].sum() / monthly[columns.due])
    show(monthly, 3)


SECTIONS = {"P01": profile_calendar, "P02": profile_sizes, "P03": profile_dimensions, "P04": profile_crossings,
            "P05": profile_states, "P06": profile_discount, "P07": profile_dynamics, "P08": profile_uplift,
            "P09": profile_price_groups, "P10": profile_cohorts, "P11": profile_acquisitions_and_partners,
            "P12": profile_future}
NEEDS_CONFIGURATION = {"P01"}


def print_profile(raw: pd.DataFrame, configuration: Config, sections: list = None) -> None:
    """Print every section (or the ones asked for) of the profile of `raw`, with the column names of `configuration`."""
    columns = Columns(raw, configuration)
    frame = add_helpers(raw, columns, configuration)
    print(f"PERFIL DEL EXTRACTO · {len(frame):,} filas · {frame[columns.period].nunique()} meses")
    for code, function in SECTIONS.items():
        if sections and code not in sections:
            continue
        if code in NEEDS_CONFIGURATION:
            function(frame, columns, configuration)
        else:
            function(frame, columns)


if __name__ == "__main__":
    # on the synthetic that looks like the real extract: the same sections, to compare with the real ones
    from main import like_real_configuration
    configuration = like_real_configuration()
    print_profile(configuration.read_raw(), configuration)
