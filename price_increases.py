"""
price_increases.py — The price increases of every price group, read from the closed months.

A tariff changes by region and product (price_group_dims), at a fixed exchange rate. When it goes up,
the renewals of ONE cycle pay the new tariff against the old one: the uplift of the group steps up for
price_cycle_months months, and then the step fades (a cycle later, everyone due already bought at the
new tariff). The detector reads that step in the uplift of the ISOLATED renewals (the normal renewal
process, no retention discounts in it) or, without them, in the plain uplift.

Shared by three steps, so the three read the same increases:
  step 3.1   the statistical uplift of the window WITHOUT the increases (its months inside a cycle are
            deflated), and the increases still active at the current month, with their cycle
  step 3.3   the same, as of the first exam month (no peeking), to judge the uplift in the exam
  step 8.4   the monitor and the calendar of the increases, for Power BI (from the core)
Step 7.2 applies the factor of the increases still active to the future months inside their cycle
(increase_factor, which reads nothing from the Config).

The functions:
  price_monitor               per price group and closed month: the uplifts, the step, the detection
  detect_increases            the detection of one price group
  increase_calendar           one row per increase detected: start, step, uplift before and in the cycle
  active_increases            the increases whose cycle touches the uplift window or the future, with
                              their size, their factor and the month of the cycle at a given month
  pending_steps               the groups whose last closed months are above the threshold in a run
                              still too short to be confirmed (an increase that may have just started)
  increase_factor             the factor of every row: the product of (1 + size) of the increases of
                              its group whose cycle contains its month; 1 elsewhere
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CORE_ISOLATED_BAND_PREFIX, CORE_ISOLATED_PIPELINE_UNITS, CORE_ISOLATED_RENEWED_PIPELINE_USD,
                    CORE_ISOLATED_RENEWED_UNITS, CORE_ISOLATED_RENEWED_USD, CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR,
                    CORE_RENEWED_PIPELINE_USD, ISOLATED_RATIO_BANDS, Config, join_columns, weighted_ratio_spread)


PORTFOLIO_GROUP = "cartera"          # the only price group when no price_group_dims are declared


# ─── the price group of a row ────────────────────────────────────────────────────
def price_group_columns(frame: pd.DataFrame, configuration: Config) -> list:
    """The price_group_dims that the frame carries."""
    return [column_name for column_name in configuration.price_group_dims if column_name in frame.columns]


def price_group_of(frame: pd.DataFrame, group_dims: list) -> pd.Series:
    """The price group of every row: its price dims joined; the portfolio when there are none."""
    if group_dims:
        return join_columns(frame, group_dims)
    return pd.Series(PORTFOLIO_GROUP, index=frame.index)


# ─── the monitor: per price group and closed month ───────────────────────────────
def price_monitor(closed: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per price group and closed pipeline month: the uplift of everyone, the uplift of the ISOLATED
    renewals (the normal renewal process, no retention event), and the detection of price increases over it.

    INPUT:   the closed pipeline rows with something due, with the names of the core (the core itself, filtered
             by step 8.4, or step_14_uplift.closed_view_of_fine_table). The closed-month measures are read IF PRESENT, by their
             fixed core names (CORE_* in config.py): the exact uplift base and the isolated measures. Each metric
             needs only its own columns, so a missing one blanks only the metrics that use it. Each uplift uses
             its exact base when it is there and the approximation renewed units × the row's due price when not.
    OUTPUT:  one row per price group and period: uplift, uplift_isolated, share_due_isolated, rate_units_isolated,
             desv_isolated (the std deviation of the ratio between isolated licences), share_isolated_near_100
             (their value in [0.95, 1.05)), share_isolated_ge110 (at or above 1.10), uplift_previous_12 (average
             of the previous 12 months, NaN with fewer than price_min_reference_months), uplift_step (the month over
             that average, − 1), above_threshold (step ≥ price_step_threshold), price_increase_flag (1 in the FIRST
             month of a run of ≥ price_persistence_months months above the threshold: an increase is a step that
             stays, one noisy month is not), in_increase_cycle (a flagged month in this or the previous
             price_cycle_months − 1 months) and months_since_increase. A run still too short at the end of the
             history is not flagged yet: it is confirmed when its months close (pending_steps lists it).
    RULES:   the detector runs on uplift_isolated when it can be computed (no retention discounts in it), on the
             plain uplift otherwise; attrs["detector_series"] says which. A month of a group with fewer than
             price_group_min_renewed_units renewals is left out of the detection (noise, not a price).
    """
    closed = closed.copy()
    closed["_precio_fila"] = closed["forecast_to_renew_USD"] / closed["forecast_to_renew_units"]
    closed["_base_aproximada"] = closed["forecast_renewed_units"] * closed["_precio_fila"]
    present = set(closed.columns)
    if CORE_ISOLATED_RENEWED_UNITS in present:
        closed["_base_aproximada_isolated"] = closed[CORE_ISOLATED_RENEWED_UNITS] * closed["_precio_fila"]

    # the price groups: a tariff changes by region and product, so every group is watched on its own
    group_dims = price_group_columns(closed, configuration)
    closed["price_group"] = price_group_of(closed, group_dims)
    dims_of_group = (closed.drop_duplicates("price_group").set_index("price_group")[group_dims]
                     if group_dims else pd.DataFrame(index=pd.Index([PORTFOLIO_GROUP], name="price_group")))
    summed = closed.groupby(["price_group", "period"]).sum(numeric_only=True)
    monitor = pd.DataFrame(index=summed.index)
    monitor["renewed_units"] = summed["forecast_renewed_units"]
    uplift_base = summed[CORE_RENEWED_PIPELINE_USD] if CORE_RENEWED_PIPELINE_USD in present else summed["_base_aproximada"]
    monitor["renewed_USD"] = summed["forecast_renewed_USD"]          # the two sums, so Power BI can add groups up
    monitor["uplift_base_USD"] = uplift_base
    monitor["uplift"] = summed["forecast_renewed_USD"] / uplift_base.where(uplift_base > 0)

    # the isolated renewals: each metric with the columns it needs, and nothing else
    monitor["share_due_isolated"] = np.nan
    monitor["rate_units_isolated"] = np.nan
    monitor["uplift_isolated"] = np.nan
    if CORE_ISOLATED_PIPELINE_UNITS in present:
        monitor["share_due_isolated"] = summed[CORE_ISOLATED_PIPELINE_UNITS] / summed["forecast_to_renew_units"]
        if CORE_ISOLATED_RENEWED_UNITS in present:
            due_isolated = summed[CORE_ISOLATED_PIPELINE_UNITS]
            monitor["rate_units_isolated"] = summed[CORE_ISOLATED_RENEWED_UNITS] / due_isolated.where(due_isolated > 0)
    isolated_base = (summed[CORE_ISOLATED_RENEWED_PIPELINE_USD] if CORE_ISOLATED_RENEWED_PIPELINE_USD in present
                     else summed["_base_aproximada_isolated"] if CORE_ISOLATED_RENEWED_UNITS in present else None)
    if CORE_ISOLATED_RENEWED_USD in present and isolated_base is not None:
        monitor["uplift_isolated"] = summed[CORE_ISOLATED_RENEWED_USD] / isolated_base.where(isolated_base > 0)

    # the dispersion of the isolated renewals, licence by licence (from the sums the extract builds per licence)
    monitor["desv_isolated"] = np.nan
    monitor["share_isolated_near_100"] = np.nan
    monitor["share_isolated_ge110"] = np.nan
    if CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR in present and CORE_ISOLATED_RENEWED_PIPELINE_USD in present:
        monitor["desv_isolated"] = weighted_ratio_spread(summed[CORE_ISOLATED_RENEWED_PIPELINE_USD],
                                                         summed[CORE_ISOLATED_RENEWED_USD],
                                                         summed[CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR])
    band_columns = {short_name: CORE_ISOLATED_BAND_PREFIX + short_name for _, short_name, _, _ in ISOLATED_RATIO_BANDS}
    if all(column in present for column in band_columns.values()):
        band_total = summed[list(band_columns.values())].sum(axis=1)
        near = summed[band_columns["095_100"]] + summed[band_columns["100_105"]]
        high = summed[band_columns["110_120"]] + summed[band_columns["ge120"]]
        monitor["share_isolated_near_100"] = near / band_total.where(band_total > 0)
        monitor["share_isolated_ge110"] = high / band_total.where(band_total > 0)

    # the detector: per price group, the chosen series against the average of its previous 12 months; a month
    # with too few renewals in its group is noise, not a price: it is left out of the detection
    detector_series = "uplift_isolated" if monitor["uplift_isolated"].notna().any() else "uplift"
    # the volume of the detector: the isolated renewals when it reads them and they are there, every renewal otherwise
    volume = (summed[CORE_ISOLATED_RENEWED_UNITS] if detector_series == "uplift_isolated" and CORE_ISOLATED_RENEWED_UNITS in present
              else summed["forecast_renewed_units"])
    monitor["volumen_detector"] = volume.to_numpy()
    monitor["detector"] = monitor[detector_series].where(monitor["volumen_detector"] >= configuration.price_group_min_renewed_units)
    monitor = monitor.reset_index().sort_values(["price_group", "period"]).reset_index(drop=True)
    detected = [detect_increases(group_rows, configuration) for _, group_rows in monitor.groupby("price_group", sort=False)]
    monitor = pd.concat(detected).sort_index()
    monitor = monitor.merge(dims_of_group.reset_index(), on="price_group", how="left")
    monitor.attrs["detector_series"] = detector_series
    return monitor


def detect_increases(group_rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The detection of one price group, month by month (its rows sorted by period): the step against the
    average of its previous 12 months, the months above the threshold, the increases (a run of
    price_persistence_months above it starts one), the cycle and the months since the increase."""
    rows = group_rows.copy()
    detector = rows["detector"]
    rows["uplift_previous_12"] = detector.shift(1).rolling(window=12, min_periods=configuration.price_min_reference_months).mean()
    rows["uplift_step"] = detector / rows["uplift_previous_12"] - 1
    above = ((rows["uplift_step"] >= configuration.price_step_threshold) & rows["uplift_previous_12"].notna()).to_numpy()
    rows["above_threshold"] = above.astype(int)
    flags = np.zeros(len(rows), dtype=int)
    position = 0
    while position < len(above):
        if not above[position]:
            position += 1
            continue
        run_end = position
        while run_end < len(above) and above[run_end]:
            run_end += 1
        if run_end - position >= configuration.price_persistence_months:
            flags[position] = 1                      # the increase starts where the run starts
        position = run_end
    rows["price_increase_flag"] = flags
    rows["in_increase_cycle"] = (rows["price_increase_flag"].rolling(window=configuration.price_cycle_months, min_periods=1)
                                 .max().astype(int).to_numpy())
    months_since, last_flagged_position = [], None
    for position, flagged in enumerate(rows["price_increase_flag"]):
        if flagged:
            last_flagged_position = position
        months_since.append(position - last_flagged_position if last_flagged_position is not None else np.nan)
    rows["months_since_increase"] = months_since
    return rows


def increase_calendar(monitor: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The increases detected, one row per price group and start month: its size (the step), the uplift
    before (the previous 12 months) and during the cycle, the months of the cycle already seen and the
    renewed USD in them (what the increase touched), biggest first."""
    rows = []
    for _, start in monitor[monitor["price_increase_flag"] == 1].iterrows():
        same_group = monitor[monitor["price_group"] == start["price_group"]]
        in_cycle = same_group[(same_group["period"] >= start["period"])
                              & (same_group["period"] < start["period"] + configuration.price_cycle_months)]
        row = {"price_group": start["price_group"], "inicio": start["period"],
               "escalon": start["uplift_step"], "uplift_antes": start["uplift_previous_12"],
               "uplift_ciclo": in_cycle["detector"].mean(), "meses_vistos": int(in_cycle["detector"].notna().sum()),
               "usd_renovado_ciclo": in_cycle["renewed_USD"].sum()}
        for column_name in configuration.price_group_dims:
            if column_name in start.index:
                row[column_name] = start[column_name]
        rows.append(row)
    columns = (["price_group", "inicio"] + [c for c in configuration.price_group_dims if c in monitor.columns]
               + ["escalon", "uplift_antes", "uplift_ciclo", "meses_vistos", "usd_renovado_ciclo"])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("usd_renovado_ciclo", ascending=False).reset_index(drop=True)


# ─── what the uplift needs: the increases still in play, and the ones not confirmed yet ───
ACTIVE_INCREASE_COLUMNS = ["price_group", "inicio", "fin_ciclo", "escalon", "tamano", "factor", "meses_vistos",
                           "mes_del_ciclo", "meses_futuros_con_efecto", "usd_renovado_ciclo"]


def active_increases(calendar: pd.DataFrame, as_of, window_months: int, configuration: Config) -> pd.DataFrame:
    """The increases whose cycle touches the uplift window or the months from `as_of` on.

    INPUT:   the calendar of increases (increase_calendar of a monitor built ONLY with the months before as_of),
             the first month that is not history (the current month, or the first exam month) and the window
             of the uplift in months (0 = all the history: every increase ever detected is in play).
    OUTPUT:  one row per increase: price_group, its dims, inicio, fin_ciclo (inicio + cycle − 1), escalon (the step
             of its first month), tamano (the SIZE used: the uplift of the cycle months seen over the uplift
             before, − 1; the first step when the cycle has no month seen), factor (1 + tamano), meses_vistos,
             mes_del_ciclo (the month of the cycle that `as_of` is: 1 = the first; above price_cycle_months = the
             effect is over), meses_futuros_con_efecto (months from as_of still inside the cycle) and the renewed
             USD the cycle touched.
    RULES:   the size of the whole cycle seen, not the step of its first month: one month is noisier than the mean
             of several. An increase with no size (no reference) is left out: it cannot be removed nor applied.
    """
    group_dims = [column_name for column_name in configuration.price_group_dims if column_name in calendar.columns]
    if not len(calendar):
        return pd.DataFrame(columns=ACTIVE_INCREASE_COLUMNS[:2] + group_dims + ACTIVE_INCREASE_COLUMNS[2:])
    rows = calendar.copy()
    cycle = configuration.price_cycle_months
    rows["fin_ciclo"] = rows["inicio"] + (cycle - 1)
    if window_months:
        first_month_of_window = as_of - window_months
        rows = rows[rows["fin_ciclo"] >= first_month_of_window]
    size_of_cycle = rows["uplift_ciclo"] / rows["uplift_antes"] - 1
    rows["tamano"] = size_of_cycle.where(size_of_cycle.notna(), rows["escalon"])
    rows = rows[rows["tamano"].notna()].copy()
    rows["factor"] = 1 + rows["tamano"]
    rows["mes_del_ciclo"] = [(as_of - start).n + 1 for start in rows["inicio"]]
    rows["meses_futuros_con_efecto"] = [max((end - as_of).n + 1, 0) for end in rows["fin_ciclo"]]
    columns = ACTIVE_INCREASE_COLUMNS[:2] + group_dims + ACTIVE_INCREASE_COLUMNS[2:]
    return rows[columns].sort_values("usd_renovado_ciclo", ascending=False).reset_index(drop=True)


def pending_steps(monitor: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The groups whose LAST closed months are above the threshold, in a run still shorter than
    price_persistence_months: an increase that may have just started and cannot be confirmed yet.
    One row per group: price_group, its dims, inicio (the first month of the run), meses_por_encima,
    escalon_medio (the mean step of the run) and the renewed USD of those months."""
    group_dims = [column_name for column_name in configuration.price_group_dims if column_name in monitor.columns]
    columns = ["price_group"] + group_dims + ["inicio", "meses_por_encima", "escalon_medio", "usd_renovado"]
    if not len(monitor):
        return pd.DataFrame(columns=columns)
    last_month = monitor["period"].max()
    rows = []
    for group, group_rows in monitor.groupby("price_group", sort=False):
        group_rows = group_rows.sort_values("period")
        if group_rows["period"].iloc[-1] != last_month:
            continue                                  # the group has nothing in the last closed month: no run is open
        above = group_rows["above_threshold"].to_numpy()
        run_length = 0
        while run_length < len(above) and above[len(above) - 1 - run_length]:
            run_length += 1
        if 0 < run_length < configuration.price_persistence_months:
            run = group_rows.tail(run_length)
            row = {"price_group": group, "inicio": run["period"].iloc[0], "meses_por_encima": run_length,
                   "escalon_medio": float(run["uplift_step"].mean()), "usd_renovado": float(run["renewed_USD"].sum())}
            for column_name in group_dims:
                row[column_name] = run[column_name].iloc[0]
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("usd_renovado", ascending=False).reset_index(drop=True)


def increase_factor(groups: pd.Series, periods: pd.Series, increases: pd.DataFrame) -> np.ndarray:
    """The factor of every row: the product of the factors of the increases of its price group whose cycle
    (inicio..fin_ciclo) contains its month; 1 for every other row. Reads nothing from the Config.

    INPUT:   the price group and the period of every row (aligned), and the increases (active_increases).
    OUTPUT:  a float array, one per row. Every increase looks only at the rows of its own group (indexed once)."""
    factor = np.ones(len(groups), dtype=float)
    if not len(increases) or not len(groups):
        return factor
    positions_of_group = pd.Series(np.arange(len(groups))).groupby(groups.to_numpy()).indices
    month_ordinals = np.array([month.ordinal for month in periods], dtype=np.int64)
    for _, increase in increases.iterrows():
        positions = positions_of_group.get(increase["price_group"])
        if positions is None:
            continue
        months = month_ordinals[positions]
        inside = (months >= increase["inicio"].ordinal) & (months <= increase["fin_ciclo"].ordinal)
        factor[positions[inside]] *= float(increase["factor"])
    return factor
